##This is the byol version of our code. This use projection and predition layer which are not compartible for retrieval application. The performance is very poor.

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
import uvnet.encoders
import dgl


###############################################################################
# Self-supervised model (BYOL)
###############################################################################

class UVNetBYOLEncoder(nn.Module):
    """
    UV-Net backbone + projector MLP for BYOL.
    - Backbone: UVNetCurve/Surface + UVNetGraph -> graph_emb (B, graph_emb_dim)
    - Bottleneck: graph_emb_dim -> latent_dim (optional)
    - Projector: MLP -> out_dim (z)
    """
    def __init__(self,
                 latent_dim=128,
                 out_dim=128,
                 crv_in_channels=3,
                 crv_emb_dim=64,
                 srf_emb_dim=64,
                 graph_emb_dim=128,
                 dropout=0.3):
        super().__init__()
        self.crv_in_channels = crv_in_channels
        self.curv_encoder = uvnet.encoders.UVNetCurveEncoder(in_channels=crv_in_channels, output_dims=crv_emb_dim)
        self.surf_encoder = uvnet.encoders.UVNetSurfaceEncoder(in_channels=4, output_dims=srf_emb_dim)
        self.graph_encoder = uvnet.encoders.UVNetGraphEncoder(srf_emb_dim, crv_emb_dim, graph_emb_dim)

        # optional bottleneck to latent_dim
        if latent_dim != graph_emb_dim:
            self.bottleneck = nn.Linear(graph_emb_dim, latent_dim, bias=False)
            nn.init.kaiming_uniform_(self.bottleneck.weight.data)
        else:
            self.bottleneck = nn.Identity()

        # projector: similar to your SimCLR head
        self.projector = nn.Sequential(nn.Linear(latent_dim, latent_dim, bias=False), 
                                       nn.BatchNorm1d(latent_dim), 
                                       nn.Dropout(dropout), 
                                       nn.ReLU(), 
                                       nn.Linear(latent_dim, latent_dim, bias=False), 
                                       nn.BatchNorm1d(latent_dim), 
                                       nn.ReLU(), 
                                       nn.Linear(latent_dim, out_dim, bias=False))

    def encode_graph(self, bg):
        """
        Returns:
            graph_emb (B, graph_emb_dim)  - pre-projector
        """
        nfeat = bg.ndata["x"][:, [0, 1, 2, 6], :, :]       # XYZ + trimming mask
        efeat = bg.edata["x"][:, :self.crv_in_channels, :] # XYZ
        crv_feat = self.curv_encoder(efeat)
        srf_feat = self.surf_encoder(nfeat)
        _, graph_emb = self.graph_encoder(bg, srf_feat, crv_feat)
        return graph_emb

    def forward(self, bg):
        """
        Returns:
            z (B, out_dim)            - L2-normalized projector output
            emb (B, latent_dim)       - pre-projector latent (after bottleneck)
        """
        graph_emb = self.encode_graph(bg)                  # [B, graph_emb_dim]
        emb = graph_emb      #emb = self.bottleneck(graph_emb)                   # [B, latent_dim]
        z = self.projector(emb)                            # [B, out_dim]
        z = F.normalize(z, p=2, dim=-1)
        return z, emb


class BYOL(pl.LightningModule):
    """
    BYOL for UV-Net graphs:
    - Online: encoder_online (backbone+projector) + predictor
    - Target: encoder_target (backbone+projector)   [EMA from online]
    Loss: symmetric negative cosine between predictor(q) and stop-grad target(z)
    """
    def __init__(self,
                 latent_dim=128,
                 out_dim=128,
                 crv_in_channels=3,
                 momentum=0.996,
                 lr=1e-3,
                 weight_decay=1e-4,
                 predictor_hidden=2048):
        super().__init__()
        self.save_hyperparameters()

        self.encoder_online = UVNetBYOLEncoder(latent_dim=latent_dim, out_dim=out_dim, crv_in_channels=crv_in_channels)
        self.encoder_target = UVNetBYOLEncoder(latent_dim=latent_dim, out_dim=out_dim, crv_in_channels=crv_in_channels)

        # initialize target with online params, then freeze target grads
        for p_o, p_t in zip(self.encoder_online.parameters(), self.encoder_target.parameters()):
            p_t.data.copy_(p_o.data)
            p_t.requires_grad = False

        # predictor: MLP out_dim -> out_dim
        hidden = out_dim if predictor_hidden is None else predictor_hidden
        self.predictor = nn.Sequential(nn.Linear(out_dim, hidden, bias=False), 
                                       nn.BatchNorm1d(hidden), 
                                       nn.ReLU(inplace=True), 
                                       nn.Linear(hidden, out_dim, bias=True))

        self.momentum = momentum
        self.lr = lr
        self.weight_decay = weight_decay

    @torch.no_grad()
    def _update_target(self):
        """EMA update: θ_t ← m*θ_t + (1−m)*θ_o"""
        for p_o, p_t in zip(self.encoder_online.parameters(), self.encoder_target.parameters()):
            p_t.data.mul_(self.momentum).add_(p_o.data, alpha=(1.0 - self.momentum))

    @staticmethod
    def _neg_cosine_similarity(p, z):
        # p, z: (B, D); p is predictor output, z is target projector output (stop-grad)
        p = F.normalize(p, dim=-1)
        z = F.normalize(z, dim=-1)
        return 2 - 2 * (p * z).sum(dim=-1)

    def _permute_graph_data_channels(self, graph):
        graph.ndata["x"] = graph.ndata["x"].permute(0, 3, 1, 2)
        graph.edata["x"] = graph.edata["x"].permute(0, 2, 1)
        return graph

    def forward(self, graph):
        """For embedding extraction: return backbone latent (pre-projector) from online encoder."""
        graph = self._permute_graph_data_channels(graph)
        # get latent emb from online backbone
        with torch.no_grad():
            _, emb = self.encoder_online(graph)
        return emb

    def step(self, batch):
        # two augmented graph views
        g1, g2 = batch["graph"], batch["graph2"]
        g1 = self._permute_graph_data_channels(g1).to(self.device)
        g2 = self._permute_graph_data_channels(g2).to(self.device)

        # Online branch: z_o, then predictor q
        g = dgl.batch([g1, g2])
        # g = torch.cat((g1, g2), dim = 0)
        z_o, _ = self.encoder_online(g)     # (B, D)
        # p1 = self.predictor(z_o)
        p_1, p_2 = z_o.chunk(2, dim = 0)

        # Target branch (no grad). Keep target in eval mode to freeze BN stats.
        
        with torch.no_grad():
            self.encoder_target.eval()
            z_t, _ = self.encoder_target(g) # (B, D)    #Added
            t_1, t_2 = z_t.detach().chunk(2, dim = 0)

        # Symmetric BYOL loss
        assert not t_1.requires_grad and not t_2.requires_grad
        loss = self._neg_cosine_similarity(p_1, t_2.detach()) + self._neg_cosine_similarity(p_2, t_1.detach())
    
        return loss.mean()

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss
    
    def optimizer_step(self, *args, **kwargs):
        # run the default optimizer step
        out = super().optimizer_step(*args, **kwargs)
        # EMA update AFTER the step
        self._update_target()
        return out

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        # EMA update AFTER optimizer step
        self._update_target()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    # ---------- (optional) utilities for downstream eval ----------
    @torch.no_grad()
    def get_embeddings_from_dataloader(self, dataloader):
        """
        Extract embeddings from the ONLINE encoder.
        Returns:
          - embeddings: backbone latent (pre-projector) for downstream linear eval
          - outputs: projector outputs z (useful for clustering)
        """
        self.eval()
        embeddings, outs, labels, filenames = [], [], [], []
        for batch in tqdm(dataloader, desc="Extracting embeddings (BYOL online)"):
            bg = batch["graph"].to(self.device)
            bg = self._permute_graph_data_channels(bg)
            z, emb = self.encoder_online(bg)
            outs.append(z.detach().cpu().numpy())
            embeddings.append(emb.detach().cpu().numpy())
            if "label" in batch:
                label = batch["label"]
                labels.append(label.squeeze(-1).detach().cpu().numpy())
            filenames.extend(batch.get("filename", []))
        outs = np.concatenate(outs)
        embeddings = np.concatenate(embeddings)
        if len(labels) > 0:
            labels = np.concatenate(labels)
        else:
            labels = None
        data_count = len(dataloader.dataset)
        assert len(embeddings) == data_count and embeddings.ndim == 2
        assert len(outs) == data_count and outs.ndim == 2
        if labels is not None:
            assert len(labels) == data_count and labels.ndim == 1
        assert len(filenames) == data_count or len(filenames) == 0
        return {"embeddings": embeddings, "labels": labels, "outputs": outs, "filenames": filenames}

    @torch.no_grad()
    def linear_svm_classification(self, train_data, test_data, max_iter=100000):
        clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=max_iter, tol=1e-3))
        _ = clf.fit(train_data["embeddings"], train_data["labels"])
        pred_labels = clf.predict(test_data["embeddings"])
        return accuracy_score(test_data["labels"], pred_labels)

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
