import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import (normalized_mutual_info_score, adjusted_mutual_info_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
import uvnet.encoders






###############################################################################
# Self-supervised model
###############################################################################

class VICRegLoss(pl.LightningModule):
    """
    VICReg Loss (Variance–Invariance–Covariance Regularization)
    Adapted for graph-level UV-Net embeddings.
    """

    def __init__(self, lambda_inv=25.0, lambda_var=25.0, lambda_cov=1.0, eps=1e-4, gamma=1.0):
        super().__init__()
        self.lambda_inv = lambda_inv
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov
        self.eps = eps
        self.gamma = gamma

    def invariance_loss(self, z1, z2):
        return F.mse_loss(z1, z2)

    def variance_loss(self, z):
        std = torch.sqrt(z.var(dim=0) + self.eps)
        penalty = F.relu(self.gamma - std)
        return penalty.mean()

    def covariance_loss(self, z):
        n, d = z.size()
        z = z - z.mean(dim=0)
        cov = (z.T @ z) / (n - 1)
        off_diag = cov.flatten()[:-1].view(d - 1, d + 1)[:, 1:].flatten()
        return (off_diag.pow(2).sum()) / d

    def forward(self, z1, z2):
        inv_loss = self.invariance_loss(z1, z2)
        var_loss = self.variance_loss(z1) + self.variance_loss(z2)
        cov_loss = self.covariance_loss(z1) + self.covariance_loss(z2)

        loss = (
            self.lambda_inv * inv_loss
            + self.lambda_var * var_loss
            + self.lambda_cov * cov_loss
        )
        return loss


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
        UVNetContrastivelearner
        """
        super().__init__()
        self.crv_in_channels = crv_in_channels
        self.curv_encoder = uvnet.encoders.UVNetCurveEncoder(in_channels=crv_in_channels, output_dims=crv_emb_dim)
        self.surf_encoder = uvnet.encoders.UVNetSurfaceEncoder(in_channels=4, output_dims=srf_emb_dim)
        self.graph_encoder = uvnet.encoders.UVNetGraphEncoder(srf_emb_dim, crv_emb_dim, graph_emb_dim)
        # self.fc_layers = nn.Sequential(
        #     nn.Linear(graph_emb_dim, latent_dim, bias=False),
        #     nn.BatchNorm1d(latent_dim),
        #     nn.ReLU(),
        #     nn.Linear(latent_dim, latent_dim),
        # )
        self.projection_layer = nn.Sequential(nn.Linear(latent_dim, latent_dim, bias=False),
                                              nn.BatchNorm1d(latent_dim),
                                              nn.Dropout(dropout),
                                              nn.ReLU(),
                                              nn.Linear(latent_dim, latent_dim, bias=False),
                                              nn.BatchNorm1d(latent_dim),
                                              nn.ReLU(),
                                              nn.Linear(latent_dim, out_dim, bias=False))

    def forward(self, bg):
        # We only use point coordinates & mask in contrastive experiments
        # TODO(pradeep): expose a better way for the user to select these channels
        nfeat = bg.ndata["x"][:, [0, 1, 2, 6], :, :]  # XYZ+mask channels --> [:, (X, Y, Z, trimming_mask), :, :] --> [8430, 4, 10, 10]
        efeat = bg.edata["x"][:, :self.crv_in_channels, :]  # XYZ channels without Tangent --> [:, :3, :] --> [:, (X, Y, Z), :] --> [48700, 3, 10]
        crv_feat = self.curv_encoder(efeat)    # [48700, 3, 10]-->[48700, 64] <-- [no_nodes, crv_emb_dim]
        srf_feat = self.surf_encoder(nfeat)    # [8430, 4, 10, 10]-->[8430, 64] <-- [no_edges, srf_emb_dim]
        node_emb, graph_emb = self.graph_encoder(bg, srf_feat, crv_feat)  # node_emb=[8430, 128], graph_emb=[256, 128]
        global_emb = graph_emb  # self.fc_layers(graph_emb)  #[256, 128]
        projection_out = self.projection_layer(global_emb)  # [256, 128] --> [256, 64]
        projection_out = F.normalize(projection_out, p=2, dim=-1)  # [256, 64] --> [256, 64]

        return projection_out, global_emb
    
    '''
    print(type(bg)) --> <class 'dgl.heterograph.DGLHeteroGraph'>
    print(type(bg.ndata)) --> <class 'dgl.view.HeteroNodeDataView'>
    print(type(bg.edata)) --> <class 'dgl.view.HeteroEdgeDataView'>

    print(bg) --> Graph(num_nodes=8430, num_edges=48700, 
                        ndata_schemes={'x': Scheme(shape=(7, 10, 10), dtype=torch.float32)}
                        edata_schemes={'x': Scheme(shape=(6, 10), dtype=torch.float32)})
                        
    print(bg.ndata["x"].shape) --> torch.Size([8430, 7, 10, 10])
            torch.Size([:, 7, :, :]) --> [:, (X, Y, Z, X_Normal, Y_Normal, Z_Normal, trimming_mask), :, :]
            
    print(bg.edata["x"].shape) --> torch.Size([48700, 6, 10])
            torch.Size([48700, 6, 10]) --> [:, (X, Y, Z, X_tangent, Y_tangent, Z_tangent), :]
    '''


class VICReg(pl.LightningModule):
    """
    PyTorch Lightning module for VICReg self-supervised learning on UV-Net.
    """

    def __init__(self,
                 latent_dim=128,
                 out_dim=128,
                 lambda_inv=25.0,
                 lambda_var=25.0,
                 lambda_cov=1.0):
        super().__init__()
        self.save_hyperparameters()
        self.loss_fn = VICRegLoss(lambda_inv, lambda_var, lambda_cov)
        self.model = UVNetContrastiveLearner(latent_dim=latent_dim, out_dim=out_dim)

    def _permute_graph_data_channels(self, graph):
        graph.ndata["x"] = graph.ndata["x"].permute(0, 3, 1, 2)
        graph.edata["x"] = graph.edata["x"].permute(0, 2, 1)
        return graph

    def step(self, batch, batch_idx):
        graph1, graph2 = batch["graph"], batch["graph2"]
        graph1 = self._permute_graph_data_channels(graph1)
        graph2 = self._permute_graph_data_channels(graph2)

        proj1, _ = self.model(graph1)
        proj2, _ = self.model(graph2)

        loss = self.loss_fn(proj1, proj2)
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
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)
        return optimizer

    @torch.no_grad()
    def get_embeddings_from_dataloader(self, dataloader):
        self.eval()
        embeddings, outs, labels, filenames = [], [], [], []
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            bg = batch["graph"].to(self.device)
            bg = self._permute_graph_data_channels(bg)
            proj, emb = self.model(bg)
            outs.append(proj.detach().cpu().numpy())
            embeddings.append(emb.detach().cpu().numpy())
            if "label" in batch:
                labels.append(batch["label"].squeeze(-1).detach().cpu().numpy())
            filenames.extend(batch["filename"])
        outs = np.concatenate(outs)
        embeddings = np.concatenate(embeddings)
        labels = np.concatenate(labels) if len(labels) > 0 else None
        return {"embeddings": embeddings, "labels": labels, "outputs": outs, "filenames": filenames}
