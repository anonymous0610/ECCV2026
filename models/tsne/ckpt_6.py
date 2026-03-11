###############################################################################
# SwAV for UVNet (drop-in replacement for SimCLR)
# - 2-crop SwAV (graph, graph2) with online prototypes and Sinkhorn assignments
# - Keeps your model I/O and eval utilities compatible with the original code
###############################################################################

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_mutual_info_score, accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline

import uvnet  # ensure this resolves to your local package/module


# ----------------------------- Utils -----------------------------------------

def l2_normalize(t: torch.Tensor, dim: int = 1, eps: float = 1e-6):
    return t / (t.norm(p=2, dim=dim, keepdim=True) + eps)


@torch.no_grad()
def distributed_concat(tensor: torch.Tensor):
    """Concatenate tensors from all processes (no grad). Falls back to single-proc."""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return tensor, None  # None means no split is needed
    tensors_gather = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor)
    return torch.cat(tensors_gather, dim=0), tensor.shape[0]


@torch.no_grad()
def sinkhorn_knopp(logits: torch.Tensor, n_iters: int = 3, epsilon: float = 0.05):
    """
    Sinkhorn-Knopp on assignment logits.
    logits: [B, K] (features-to-prototypes similarities / epsilon)
    Returns Q: [B, K] doubly-stochastic codes (rows sum to 1).
    """
    Q = torch.exp(logits / epsilon).t()  # [K, B]
    Q /= torch.sum(Q, dim=0, keepdim=True) + 1e-12

    K, B = Q.shape
    for _ in range(n_iters):
        # normalize rows
        Q /= torch.sum(Q, dim=1, keepdim=True) + 1e-12
        Q /= K
        # normalize cols
        Q /= torch.sum(Q, dim=0, keepdim=True) + 1e-12
        Q /= B

    Q = (Q / torch.sum(Q, dim=0, keepdim=True)).t().contiguous()  # [B, K]
    return Q


# ------------------------ UVNet + Projector + Prototypes ---------------------

class UVNetSwAV(nn.Module):
    def __init__(self,
                 latent_dim,
                 crv_in_channels=3,
                 crv_emb_dim=64,
                 srf_emb_dim=64,
                 graph_emb_dim=128,
                 dropout=0.3,
                 out_dim=128,
                 n_prototypes=300,
                 normalize_prototypes=True):
                # Note: out_dim is the projection size used for SwAV
        super().__init__()
        self.crv_in_channels = crv_in_channels
        self.curv_encoder = uvnet.encoders.UVNetCurveEncoder(
            in_channels=crv_in_channels, output_dims=crv_emb_dim
        )
        self.surf_encoder = uvnet.encoders.UVNetSurfaceEncoder(
            in_channels=4, output_dims=srf_emb_dim
        )
        self.graph_encoder = uvnet.encoders.UVNetGraphEncoder(
            srf_emb_dim, crv_emb_dim, graph_emb_dim
        )

        # --- Projector (3-layer MLP as in SwAV) ---
        self.projection_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim, bias=False),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, latent_dim, bias=False),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim, affine=False)  # no affine on last BN
        )

        # --- Prototypes (learned cluster centers) ---
        self.prototypes = nn.Linear(out_dim, n_prototypes, bias=False)
        self.normalize_prototypes = normalize_prototypes

        # (optional) warm-up: freeze prototype updates for first few iters/epochs in Lightning

    def _encode_graph(self, bg):
        nfeat = bg.ndata["x"][:, [0, 1, 2, 6], :, :]        # [N_nodes, 4, 10, 10]
        efeat = bg.edata["x"][:, :self.crv_in_channels, :]  # [N_edges, 3, 10]
        crv_feat = self.curv_encoder(efeat)
        srf_feat = self.surf_encoder(nfeat)
        node_emb, graph_emb = self.graph_encoder(bg, srf_feat, crv_feat)  # [B, latent_dim]
        return graph_emb

    def forward(self, bg, return_logits=False):
        """
        Returns:
          z: [B, D] L2-normalized projector output for retrieval
          emb: [B, latent_dim] backbone embedding
          (optionally) logits: [B, K] similarity to prototypes (no temperature applied)
        """
        emb = self._encode_graph(bg)
        z = self.projection_head(emb)                 # [B, D]
        z = l2_normalize(z, dim=1)                    # normalized features

        # normalize prototype weights (weight normalization along dim=1)
        if self.normalize_prototypes:
            with torch.no_grad():
                w = self.prototypes.weight.data
                self.prototypes.weight.data.copy_(l2_normalize(w, dim=1))

        if not return_logits:
            return z, emb

        logits = self.prototypes(z)  # [B, K]
        return z, emb, logits


# ------------------------------ SwAV Loss ------------------------------------

class SwAVLoss(nn.Module):
    """
    Swapped assignment prediction loss for two views:
      L = 0.5 * [ CE(q1, p2) + CE(q2, p1) ]
    where:
      - q* are Sinkhorn codes from logits of each view (no grad through Sinkhorn),
      - p* are softmax probabilities over prototypes from the *other* view.
    """
    def __init__(self, temperature: float = 0.1, sinkhorn_iters: int = 3, sinkhorn_epsilon: float = 0.05):
        super().__init__()
        self.temperature = temperature
        self.sinkhorn_iters = sinkhorn_iters
        self.sinkhorn_epsilon = sinkhorn_epsilon

    @torch.no_grad()
    def _codes(self, logits: torch.Tensor):
        # Optionally gather across DDP before Sinkhorn to balance assignments globally
        all_logits, local_bs = distributed_concat(logits)
        Q_all = sinkhorn_knopp(all_logits, n_iters=self.sinkhorn_iters, epsilon=self.sinkhorn_epsilon)  # [B_all, K]
        if local_bs is None:
            return Q_all  # single-process
        # slice this process' chunk
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        start = rank * local_bs
        end = start + local_bs
        return Q_all[start:end, :]

    def forward(self, logits_1: torch.Tensor, logits_2: torch.Tensor):
        # logits_* are similarities to prototypes (no temp)
        with torch.no_grad():
            q1 = self._codes(logits_1).detach()  # [B, K]
            q2 = self._codes(logits_2).detach()  # [B, K]

        p1 = F.softmax(logits_1 / self.temperature, dim=1)  # [B, K]
        p2 = F.softmax(logits_2 / self.temperature, dim=1)

        # swapped prediction: CE(q1, p2) + CE(q2, p1)
        loss_12 = -(q1 * torch.log(p2 + 1e-12)).sum(dim=1).mean()
        loss_21 = -(q2 * torch.log(p1 + 1e-12)).sum(dim=1).mean()
        return 0.5 * (loss_12 + loss_21)


# --------------------------- Lightning Module --------------------------------

class Contrastive6(pl.LightningModule):
    """
    PyTorch Lightning module to train/test SwAV with UVNet encoders.

    Key hparams:
      - n_prototypes: number of prototype vectors (clusters)
      - temperature: softmax temperature for swapped prediction
      - sinkhorn_iters / epsilon: assignment solver parameters
      - proto_freeze_epochs: freeze prototype gradients for warm-up (optional, improves stability)
    """
    def __init__(self,
                 latent_dim=128,
                 out_dim=128,
                 n_prototypes=300,
                 temperature=0.1,
                 sinkhorn_iters=3,
                 sinkhorn_epsilon=0.05,
                 proto_freeze_epochs: int = 0,
                 lr=1e-3,
                 weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = UVNetSwAV(
            latent_dim=latent_dim,
            out_dim=out_dim,
            n_prototypes=n_prototypes,
            normalize_prototypes=True
        )
        self.criterion = SwAVLoss(temperature=temperature,
                                  sinkhorn_iters=sinkhorn_iters,
                                  sinkhorn_epsilon=sinkhorn_epsilon)
        self.lr = lr
        self.weight_decay = weight_decay
        self.proto_freeze_epochs = proto_freeze_epochs

    def _permute_graph_data_channels(self, graph):
        graph.ndata["x"] = graph.ndata["x"].permute(0, 3, 1, 2)
        graph.edata["x"] = graph.edata["x"].permute(0, 2, 1)
        return graph

    def on_train_epoch_start(self):
        # Optionally freeze prototypes for warm-up (as in SwAV)
        requires_grad = not (self.current_epoch < self.proto_freeze_epochs)
        for p in self.model.prototypes.parameters():
            p.requires_grad = requires_grad

    def step(self, batch, batch_idx):
        # Two graph views with identical topology but different augmentations
        g1, g2 = batch["graph"], batch["graph2"]
        g1 = self._permute_graph_data_channels(g1)
        g2 = self._permute_graph_data_channels(g2)

        # Forward both views and get prototype logits (no temperature inside)
        z1, _, logits1 = self.model(g1, return_logits=True)   # z1: [B, D], logits1: [B, K]
        z2, _, logits2 = self.model(g2, return_logits=True)   # z2: [B, D], logits2: [B, K]

        # SwAV swapped assignment loss
        loss = self.criterion(logits1, logits2)
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
        # The official repo often uses LARS/SGD + cosine decay; Adam works fine here.
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    # ------------------------ Downstream eval utils --------------------------
    @torch.no_grad()
    def get_embeddings_from_dataloader(self, dataloader):
        """
        Returns:
          {"embeddings": backbone embeddings, "labels": ..., "outputs": projector z, "filenames": ...}
        `outputs` are normalized projector features (z), i.e., what you'd use for retrieval.
        """
        self.eval()
        embeddings, outs, labels, filenames = [], [], [], []
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            bg = batch["graph"].to(self.device)
            bg = self._permute_graph_data_channels(bg)
            z, emb = self.model(bg)  # z: [B, D] normalized projector; emb: [B, latent_dim]
            outs.append(z.detach().cpu().numpy())
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
