###############################################################################
# SimSiam for UVNet (drop-in replacement for the SimCLR version)
###############################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import uvnet.encoders

from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_mutual_info_score, accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline

import uvnet  # ensure this resolves to your local package/module
import numpy as np

# ----------------------------- SimSiam Loss ----------------------------------
class SimSiamLoss(nn.Module):
    """
    SimSiam loss: L = 0.5 * [D(p1, z2) + D(p2, z1)], where
    D(p, z) = - cos_sim( p, stopgrad(z) ).
    """
    def __init__(self):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=1)

    def forward(self, p1, z2, p2, z1):
        z1 = z1.detach()
        z2 = z2.detach()
        # normalize as in SimSiam paper/code
        p1 = F.normalize(p1, dim=1)
        p2 = F.normalize(p2, dim=1)
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        loss = -0.5 * (self.cos(p1, z2).mean() + self.cos(p2, z1).mean())
        return loss


# ------------------------ UVNet + Projector + Predictor ----------------------
class UVNetContrastiveLearner(nn.Module):
    def __init__(self,
                 latent_dim,
                 crv_in_channels=3,
                 crv_emb_dim=64,
                 srf_emb_dim=64,
                 graph_emb_dim=128,
                 dropout=0.3,
                 out_dim=128):
        """
        UVNet for SimSiam. Adds a predictor head on top of the projector.
        """
        super().__init__()
        self.crv_in_channels = crv_in_channels
        self.curv_encoder = uvnet.encoders.UVNetCurveEncoder(in_channels=crv_in_channels, output_dims=crv_emb_dim)
        self.surf_encoder = uvnet.encoders.UVNetSurfaceEncoder(in_channels=4, output_dims=srf_emb_dim)
        self.graph_encoder = uvnet.encoders.UVNetGraphEncoder(srf_emb_dim, crv_emb_dim, graph_emb_dim)

        # --- Projector (MLP) ---
        # Use BN and bottleneck as in SimSiam (3-layer MLP is common).
        self.projection_layer = nn.Sequential(
            nn.Linear(latent_dim, latent_dim, bias=False),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, latent_dim, bias=False),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim, affine=False)  # BN without affine is typical in SimSiam repo
        )

        # --- Predictor (2-layer MLP) ---
        hidden_dim = out_dim // 2 if out_dim >= 64 else max(32, out_dim)  # small bottleneck
        self.predictor = nn.Sequential(
            nn.Linear(out_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)  # with bias
        )

    def _encode_graph(self, bg):
        # Select channels: XYZ + mask for nodes; XYZ for edges
        nfeat = bg.ndata["x"][:, [0, 1, 2, 6], :, :]   # [N_nodes, 4, 10, 10]
        efeat = bg.edata["x"][:, :self.crv_in_channels, :]  # [N_edges, 3, 10]
        crv_feat = self.curv_encoder(efeat)            # [N_edges, crv_emb_dim]
        srf_feat = self.surf_encoder(nfeat)            # [N_nodes, srf_emb_dim]
        node_emb, graph_emb = self.graph_encoder(bg, srf_feat, crv_feat)  # graph_emb: [B, latent_dim]
        return graph_emb

    def forward(self, bg, return_pred: bool = False):
        """
        If return_pred=False (default): returns (proj, global_emb)
        If return_pred=True: returns (proj, pred, global_emb)
        """
        global_emb = self._encode_graph(bg)                    # [B, latent_dim]
        z = self.projection_layer(global_emb)                  # [B, out_dim]
        z = F.normalize(z, p=2, dim=-1)                        # normalized projector output for retrieval
        if not return_pred:
            return z, global_emb
        p = self.predictor(z)                                  # [B, out_dim]
        return z, p, global_emb


# --------------------------- Lightning Module --------------------------------
class Contrastive5(pl.LightningModule):
    """
    PyTorch Lightning module to train/test SimSiam with UVNet.
    """
    def __init__(self, latent_dim=128, out_dim=128):
        super().__init__()
        self.save_hyperparameters()
        self.loss_fn = SimSiamLoss()
        self.model = UVNetContrastiveLearner(latent_dim=latent_dim, out_dim=out_dim)

    def _permute_graph_data_channels(self, graph):
        graph.ndata["x"] = graph.ndata["x"].permute(0, 3, 1, 2)
        graph.edata["x"] = graph.edata["x"].permute(0, 2, 1)
        return graph

    def step(self, batch, batch_idx):
        # Two graph views (same topology, different augmentations)
        graph1, graph2 = batch["graph"], batch["graph2"]
        graph1 = self._permute_graph_data_channels(graph1)
        graph2 = self._permute_graph_data_channels(graph2)

        # Forward both views with predictor
        z1, p1, _ = self.model(graph1, return_pred=True)   # z1, p1: [B, out_dim]
        z2, p2, _ = self.model(graph2, return_pred=True)   # z2, p2: [B, out_dim]

        # SimSiam loss
        loss = self.loss_fn(p1, z2, p2, z1)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        # SimSiam typically uses SGD with momentum & cosine schedule, but Adam also works.
        optimizer = torch.optim.Adam(self.parameters(), weight_decay=1e-4)
        return optimizer

    # ------------------------ Downstream eval utils --------------------------
    @torch.no_grad()
    def get_embeddings_from_dataloader(self, dataloader):
        """
        Unchanged external contract:
        returns {"embeddings": embeddings, "labels": labels, "outputs": outs, "filenames": filenames}
        where `outs` are the normalized projector outputs (z) used for retrieval.
        """
        self.eval()
        embeddings, outs, labels, filenames = [], [], [], []
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            bg = batch["graph"].to(self.device)
            bg = self._permute_graph_data_channels(bg)
            proj, emb = self.model(bg)             # proj=z (normalized), emb=global_emb
            outs.append(proj.detach().cpu().numpy())
            embeddings.append(emb.detach().cpu().numpy())
            if "label" in batch:
                label = batch["label"]
                labels.append(label.squeeze(-1).detach().cpu().numpy())
            filenames.extend(batch["filename"])
        outs = np.concatenate(outs)
        embeddings = np.concatenate(embeddings)
        labels = np.concatenate(labels) if len(labels) > 0 else None
        data_count = len(dataloader.dataset)
        assert len(embeddings) == data_count and embeddings.ndim == 2
        assert len(outs) == data_count and outs.ndim == 2
        if labels is not None:
            assert len(labels) == data_count and labels.ndim == 1
        assert len(filenames) == data_count
        return {"embeddings": embeddings, "labels": labels, "outputs": outs, "filenames": filenames}

    @torch.no_grad()
    def clustering(self, data, num_clusters=26, n_init=100, standardize=False):
        if standardize:
            scaler = StandardScaler().fit(data["embeddings"])
            embeddings = scaler.transform(data["embeddings"].copy())
        else:
            embeddings = data["embeddings"]
        kmeans = KMeans(init="random", n_clusters=num_clusters, n_init=n_init, max_iter=100000)
        print(f"Fitting K-Means with {num_clusters} clusters...")
        kmeans.fit(embeddings)
        pred_labels = kmeans.labels_
        score = adjusted_mutual_info_score(data["labels"], pred_labels)
        km_acc = accuracy_score(data["labels"], pred_labels)
        print(f"Clustering ACC on test set: {km_acc * 100.0:2.3f}")
        return score

    @torch.no_grad()
    def linear_svm_classification(self, train_data, test_data, max_iter=100000):
        clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=max_iter, tol=1e-3))
        print("Training Linear SVM...")
        ret = clf.fit(train_data["embeddings"], train_data["labels"])
        pred_labels = clf.predict(test_data["embeddings"])
        return accuracy_score(test_data["labels"], pred_labels)
