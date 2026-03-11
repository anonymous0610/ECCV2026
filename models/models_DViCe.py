import numpy as np
import pytorch_lightning as pl
import torch
import math
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

def mask_correlated_samples(batch_size, device=torch.device("cpu")):
    N = 2 * batch_size
    mask = torch.ones((N, N), dtype=bool, device=device)
    mask = mask.fill_diagonal_(0)
    for i in range(batch_size):
        mask[i, batch_size + i] = 0
        mask[batch_size + i, i] = 0
    return mask

class NTXentLossAdaptiveProb(pl.LightningModule):
    def __init__(self, 
                 temperature=0.1, 
                 batch_size=256, 
                 p_target=0.65,        # desired avg positive probability
                 k=0.025,               # controller gain (small) 
                 ema_decay=0.98,       # smoothing for p 
                 tau_min=0.05, 
                 tau_max=2.0):
        super().__init__()
        # track log(tau) for stable multiplicative updates
        self.register_buffer("log_tau", torch.tensor(float(torch.log(torch.tensor(temperature)))))
        self.p_target = p_target
        self.k = k
        self.ema_decay = ema_decay
        self.tau_min = tau_min
        self.tau_max = tau_max

        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

        # mask starts on CPU; moved to device at runtime
        mask = mask_correlated_samples(batch_size, torch.device("cpu"))
        self.register_buffer("mask", mask)

        # running avg of positive probability
        self.register_buffer("p_ema", torch.tensor(0.0))
        self.register_buffer("has_p_ema", torch.tensor(False))

    @property
    def tau(self):
        return self.log_tau.exp().clamp_(min=self.tau_min, max=self.tau_max)

    def _update_tau_from_prob(self, p_batch_mean, device):
        # EMA for smoother signal
        if not bool(self.has_p_ema.item()):
            self.p_ema = p_batch_mean.detach().to(device)
            self.has_p_ema = torch.tensor(True, device=device)
        else:
            self.p_ema = self.ema_decay * self.p_ema + (1 - self.ema_decay) * p_batch_mean.detach().to(device)

        # proportional control in log-space: log_tau <- log_tau + k * (p_ema - p_target)
        error = self.p_ema - self.p_target
        with torch.no_grad():
            self.log_tau += self.k * error
            # clamp τ by clamping logτ
            self.log_tau.clamp_(min=torch.log(torch.tensor(self.tau_min, device=device)),
                                max=torch.log(torch.tensor(self.tau_max, device=device)))

    def forward(self, z_i, z_j):
        device = z_i.device
        batch_size = z_i.size(0)
        N = 2 * batch_size

        mask = self.mask.to(device)
        z = torch.cat([z_i, z_j], dim=0)  # [2B, D]

        # similarities and logits with current tau
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0))  # [2B,2B]
        logits = sim / self.tau  # current temperature

        # positives are across the diagonal offset by ±B
        sim_i_j = torch.diag(logits,  batch_size)
        sim_j_i = torch.diag(logits, -batch_size)
        positives = torch.cat([sim_i_j, sim_j_i], dim=0).reshape(N, 1)

        negatives = logits[mask].reshape(N, -1)
        all_logits = torch.cat([positives, negatives], dim=1)  # class 0 is positive
        labels = torch.zeros(N, dtype=torch.long, device=device)

        # ---- compute loss
        loss = self.criterion(all_logits, labels) / N

        # ---- compute positive probability under current τ (no grad into τ update)
        with torch.no_grad():
            # softmax over [pos | negs]
            probs = all_logits.softmax(dim=1)
            p_pos = probs[:, 0]
            p_mean = p_pos.mean()
            self._update_tau_from_prob(p_mean, device)

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



# # Upto Exp-7
class Contrastive(pl.LightningModule):
    """
    PyTorch Lightning module to train/test the contrastive learning model.
    """

    def __init__(self, latent_dim=128, out_dim=128, batch_size=256, temperature=0.1, momentum=0.99):
        super().__init__()
        self.save_hyperparameters()
        self.hparams.base_momentum = momentum
        #self.loss_fn = NTXentLoss(temperature=temperature)
        self.loss_fn = NTXentLossAdaptiveProb(temperature=temperature)    #EXP-26, EXP-30, 31, 32
        #self.loss_fn = NTXentLossAdaptiveEntropy(temperature=temperature)  #EXP-27
        #self.loss_fn = NTXentLossHybridProb(temperature=temperature)      #EXP-28
        #self.loss_fn = NTXentLossHybridEntropy(temperature=temperature)   #EXP-29
        #self.loss_fn = GlobalNTXentLoss(temperature=temperature, batch_size=256, lambda_comp=0.2, lambda_exp=0.3)
        #self.loss_fn = GlobalNTXentLoss(temperature=temperature, lambda_comp=0.00, lambda_exp=0.1)
        self.model_1 = UVNetContrastiveLearner(latent_dim=latent_dim, out_dim=out_dim)
        self.model_2 = UVNetContrastiveLearner(latent_dim=latent_dim, out_dim=out_dim)
        self.model_3 = UVNetContrastiveLearner(latent_dim=latent_dim, out_dim=out_dim)
        # Momentum update: not trained directly
        for param_1, param_2 in zip(self.model_1.parameters(), self.model_2.parameters()):
            param_2.data = momentum * param_2.data + (1. - momentum) * param_1.data
        for param_1, param_3 in zip(self.model_1.parameters(), self.model_3.parameters()):
            param_3.data = momentum * param_3.data + (1. - momentum) * param_1.data
            
    def _permute_graph_data_channels(self, graph):
        graph.ndata["x"] = graph.ndata["x"].permute(0, 3, 1, 2)
        graph.edata["x"] = graph.edata["x"].permute(0, 2, 1)
        return graph
    
    @staticmethod
    def _neg_cosine_similarity(p, z):
        p = F.normalize(p, dim=-1)
        z = F.normalize(z, dim=-1)
        return 2 - 2 * (p * z).sum(dim=-1).mean()

    def step(self, batch, batch_idx):
        graph1, graph2 = batch["graph"], batch["graph2"]
        graph1 = self._permute_graph_data_channels(graph1)
        graph2 = self._permute_graph_data_channels(graph2)
        with torch.no_grad():
            proj2, _ = self.model_2(graph2)
            # proj2_1, _ = self.model_2(graph1)
            proj3, _ = self.model_3(graph2)
        proj1, _ = self.model_1(graph1)
        loss_BYOL = self._neg_cosine_similarity(proj1, proj3.detach())
        return self.loss_fn(proj1, proj2.detach()) + loss_BYOL.mean(), proj1, proj3.detach()

    def training_step(self, batch, batch_idx):
        loss, proj1, proj3 = self.step(batch, batch_idx)
        self.momentum_update(proj1, proj3)
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("lr", lr, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _, _ = self.step(batch, batch_idx)  # <-- FIX
        # self._update_target()
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    @torch.no_grad()
    def momentum_update(self, proj_online=None, proj_momentum=None):
        # t = int(self.global_step_counter.item())
        t = int(self.global_step)  # safer across PL versions

        if hasattr(self.trainer, "estimated_stepping_batches"):
            T = self.trainer.estimated_stepping_batches
        elif hasattr(self.trainer, "num_training_batches") and self.trainer.num_training_batches is not None:
            T = self.trainer.num_training_batches * self.trainer.max_epochs
        elif self.trainer.max_steps and self.trainer.max_steps > 0:
            T = self.trainer.max_steps
        else:
            T = 100000

        base_m = self.hparams.base_momentum
        cosine_m = 1.0 - (1.0 - base_m) * (0.5 * (1.0 + math.cos(math.pi * t / T)))

        if proj_online is not None and proj_momentum is not None:
            proj_online = F.normalize(proj_online, dim=-1)
            proj_momentum = F.normalize(proj_momentum, dim=-1)
            sim = (proj_online * proj_momentum).sum(dim=-1)
            var = torch.var(sim.detach())
            if not hasattr(self, "_var_ema"):
                self._var_ema = var.clone()
            else:
                self._var_ema = 0.98 * self._var_ema + 0.02 * var
            ratio = (var / (self._var_ema + 1e-8)).clamp(0.5, 2.0)
            m = (cosine_m * (2.0 - ratio)).clamp(0.9 * base_m, 0.9999)
        else:
            m = cosine_m

        m_scalar = m.mean().detach().item() if torch.is_tensor(m) else float(m)
        for param_q, param_k in zip(self.model_1.parameters(), self.model_3.parameters()):
            param_k.data = m_scalar * param_k.data + (1. - m_scalar) * param_q.data

        self.log("train/momentum_coeff", m_scalar, on_step=True, on_epoch=False, sync_dist=True)
    
    @torch.no_grad()
    def _update_target(self):
        """EMA update: θ_t ← m*θ_t + (1−m)*θ_o"""
        momentum = self.hparams.base_momentum
        for p_o, p_t in zip(self.model_1.parameters(), self.model_2.parameters()):
            p_t.data.mul_(momentum).add_(p_o.data, alpha=(1.0 - momentum))

    #Main code
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), weight_decay=1e-4)
        return optimizer

    # def configure_optimizers(self):
    #     # === Compute scaled learning rate ===
    #     base_lr = 0.001
    #     batch_size = self.hparams.batch_size if "batch_size" in self.hparams else 512
    #     lr = (batch_size / 256) * base_lr  # linear scaling rule
        
    #     print(f"Using learning rate: {lr:.6f} (batch_size={batch_size})")
        
    #     optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)

    #     # === Warmup + Cosine Decay ===
    #     def lr_lambda(current_epoch):
    #         warmup_epochs = 20
    #         max_epochs = self.trainer.max_epochs if self.trainer else 100
    #         final_lr_scale = 0.1 / lr  # ensures lr_end = 0.002

    #         if current_epoch < warmup_epochs:
    #             # Linear warmup from 0 → 1
    #             return current_epoch / float(warmup_epochs)
    #         else:
    #             # Cosine decay from 1 → final_lr_scale
    #             progress = (current_epoch - warmup_epochs) / float(max_epochs - warmup_epochs)
    #             cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    #             return final_lr_scale + (1 - final_lr_scale) * cosine_decay

    #     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    #     return [optimizer], [scheduler]


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
        # added by me: 
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

    @torch.no_grad()
    def get_embeddings_from_dataloader(self, dataloader):
        self.eval()
        embeddings = []
        outs = []
        labels = []
        filenames = []
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            bg = batch["graph"].to(self.device)
            bg = self._permute_graph_data_channels(bg)
            proj, emb = self.model_1(bg)
            #proj, emb = self.model(bg)                   #         -> n_data -->(surf_encoder)--> surf_feat -->
                                                          # bg --> |                                            |->(graph_encoder)--> emb[256, 128] -->(proj_head)--> proj[256, 64]
                                                          #         -> e_data -->(curv_encoder)--> curv_feat -->
            outs.append(proj.detach().cpu().numpy())      # List of numpy arrays of shape (batch_size, 64)
            embeddings.append(emb.detach().cpu().numpy()) # List of numpy arrays of shape (batch_size, 128)
            if "label" in batch:
                label = batch["label"]                    # label-->letter labels = [batch_size]    A/a-> 0, B/b --> 1, C/c --> 2, ...
                labels.append(label.squeeze(-1).detach().cpu().numpy())
            filenames.extend(batch["filename"])           # filenames-->letter location = [batch_size]    p_Marko One_lower --> which signifies the categories i.e. p(letter) Marko(style) One(texture) lower(case)
        outs = np.concatenate(outs)
        embeddings = np.concatenate(embeddings)
        if len(labels) > 0:
            labels = np.concatenate(labels)
        else:
            labels = None
        data_count = len(dataloader.dataset)
        assert len(embeddings) == data_count, f"{embeddings.shape}, {data_count}"
        assert len(embeddings.shape) == 2, f"{embeddings.shape}"
        assert len(outs) == data_count, f"{outs.shape}, {data_count}"
        assert len(outs.shape) == 2, f"{outs.shape}"
        if labels is not None:
            assert len(labels) == data_count
            assert len(labels.shape) == 1, f"{labels.shape}"
            assert len(labels.shape) == 1, f"{labels.shape}"
        assert len(filenames) == data_count, f"{len(filenames)}, {data_count}"
        return {"embeddings": embeddings, "labels": labels, "outputs": outs, "filenames": filenames}   # embeddings=[19404, 128], labels=[19404], outs=projections=[19404, 64], filenames=[19404]

