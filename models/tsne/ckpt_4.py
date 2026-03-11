import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from tqdm import tqdm
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, adjusted_mutual_info_score
from sklearn.cluster import KMeans
import uvnet.encoders


###############################################################################
# Encoder with Projection Layer (your definition)
###############################################################################
class UVNetContrastiveLearner(nn.Module):
    def __init__(self,
                 latent_dim,
                 crv_in_channels=3,
                 crv_emb_dim=64,
                 srf_emb_dim=64,
                 graph_emb_dim=128,
                 dropout=0.3,
                 out_dim=128):
        super().__init__()
        self.crv_in_channels = crv_in_channels
        self.curv_encoder = uvnet.encoders.UVNetCurveEncoder(in_channels=crv_in_channels,
                                                             output_dims=crv_emb_dim)
        self.surf_encoder = uvnet.encoders.UVNetSurfaceEncoder(in_channels=4,
                                                               output_dims=srf_emb_dim)
        self.graph_encoder = uvnet.encoders.UVNetGraphEncoder(srf_emb_dim, crv_emb_dim,
                                                              graph_emb_dim)

        self.projection_layer = nn.Sequential(
            nn.Linear(latent_dim, latent_dim, bias=False),
            nn.BatchNorm1d(latent_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim, bias=False),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, out_dim, bias=False)
        )

    def forward(self, bg):
        nfeat = bg.ndata["x"][:, [0, 1, 2, 6], :, :]  # XYZ + trimming mask
        efeat = bg.edata["x"][:, :self.crv_in_channels, :]  # XYZ
        crv_feat = self.curv_encoder(efeat)
        srf_feat = self.surf_encoder(nfeat)
        node_emb, graph_emb = self.graph_encoder(bg, srf_feat, crv_feat)
        global_emb = graph_emb
        projection_out = self.projection_layer(global_emb)
        projection_out = F.normalize(projection_out, p=2, dim=-1)
        return projection_out, global_emb


###############################################################################
# BYOL-style training module using UVNetContrastiveLearner
###############################################################################
class Contrastive4(pl.LightningModule):
    """
    BYOL-like retrieval model:
      - Online encoder: UVNetContrastiveLearner
      - Target encoder: EMA copy of online encoder
      - No predictor layer (projection only)
      - Symmetric cosine similarity loss
    """

    def __init__(self,
                 latent_dim=128,
                 out_dim=128,
                 crv_in_channels=3,
                 momentum=0.99,
                 lr=1e-3,
                 weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.encoder_online = UVNetContrastiveLearner(latent_dim=latent_dim,
                                                      out_dim=out_dim,
                                                      crv_in_channels=crv_in_channels)
        self.encoder_target = UVNetContrastiveLearner(latent_dim=latent_dim,
                                                      out_dim=out_dim,
                                                      crv_in_channels=crv_in_channels)

        # initialize target as a copy of online
        for p_o, p_t in zip(self.encoder_online.parameters(),
                            self.encoder_target.parameters()):
            p_t.data.copy_(p_o.data)
            p_t.requires_grad = False

        self.momentum = momentum
        self.lr = lr
        self.weight_decay = weight_decay

    @torch.no_grad()
    def _update_target(self):
        """EMA update for target encoder."""
        for p_o, p_t in zip(self.encoder_online.parameters(),
                            self.encoder_target.parameters()):
            p_t.data.mul_(self.momentum).add_(p_o.data, alpha=(1.0 - self.momentum))

    @staticmethod
    def _neg_cosine_similarity(p, z):
        p = F.normalize(p, dim=-1)
        z = F.normalize(z, dim=-1)
        return 2 - 2 * (p * z).sum(dim=-1).mean()

    def _permute_graph_data_channels(self, graph):
        graph.ndata["x"] = graph.ndata["x"].permute(0, 3, 1, 2)
        graph.edata["x"] = graph.edata["x"].permute(0, 2, 1)
        return graph

    def step(self, batch):
        g1, g2 = batch["graph"], batch["graph2"]
        g1 = self._permute_graph_data_channels(g1).to(self.device)
        g2 = self._permute_graph_data_channels(g2).to(self.device)

        # Online encoder
        z1_o, _ = self.encoder_online(g1)
        z2_o, _ = self.encoder_online(g2)

        # Target encoder (EMA)
        with torch.no_grad():
            self.encoder_target.eval()
            z1_t, _ = self.encoder_target(g1)
            z2_t, _ = self.encoder_target(g2)

        # Symmetric cosine similarity loss
        loss = 0.5 * (self._neg_cosine_similarity(z1_o, z2_t) + self._neg_cosine_similarity(z2_o, z1_t))
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("train_loss", loss, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("val_loss", loss, on_epoch=True, sync_dist=True)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        self._update_target()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    # ---------- downstream utilities ----------
    @torch.no_grad()
    def get_embeddings_from_dataloader(self, dataloader):
        """
        Extract latent graph embeddings (pre-projection) from online encoder.
        """
        self.eval()
        embeddings, labels, filenames = [], [], []
        for batch in tqdm(dataloader, desc="Extracting embeddings (BYOLRetrieval)"):
            bg = batch["graph"].to(self.device)
            bg = self._permute_graph_data_channels(bg)
            _, graph_emb = self.encoder_online(bg)
            embeddings.append(graph_emb.detach().cpu().numpy())
            if "label" in batch:
                labels.append(batch["label"].squeeze(-1).cpu().numpy())
            filenames.extend(batch.get("filename", []))

        embeddings = np.concatenate(embeddings)
        labels = np.concatenate(labels) if len(labels) > 0 else None
        return {"embeddings": embeddings, "labels": labels, "filenames": filenames}

    @torch.no_grad()
    def linear_svm_classification(self, train_data, test_data, max_iter=100000):
        clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=max_iter, tol=1e-3))
        clf.fit(train_data["embeddings"], train_data["labels"])
        pred_labels = clf.predict(test_data["embeddings"])
        return accuracy_score(test_data["labels"], pred_labels)

    @torch.no_grad()
    def clustering(self, data, num_clusters=26, n_init=100, standardize=False):
        if standardize:
            scaler = StandardScaler().fit(data["embeddings"])
            embeddings = scaler.transform(data["embeddings"])
        else:
            embeddings = data["embeddings"]
        kmeans = KMeans(init="random", n_clusters=num_clusters, n_init=n_init, max_iter=100000)
        print(f"Fitting K-Means with {num_clusters} clusters...")
        kmeans.fit(embeddings)
        pred_labels = kmeans.labels_
        score = adjusted_mutual_info_score(data["labels"], pred_labels)
        acc = accuracy_score(data["labels"], pred_labels)
        print(f"Clustering ACC on test set: {acc * 100.0:2.3f}")
        return score
