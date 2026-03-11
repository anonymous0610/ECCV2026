import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import math
from scipy.spatial.distance import cdist
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import (normalized_mutual_info_score, adjusted_mutual_info_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import uvnet.encoders
from tqdm import tqdm
import numpy as np
import torchmetrics



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
                 p_target=0.65,        
                 k=0.025,               
                 ema_decay=0.98,       
                 tau_min=0.05, 
                 tau_max=2.0,
                 queue_size=4096,
                 out_dim=128):
        super().__init__()
        # ---- tau controller
        self.register_buffer("log_tau", torch.tensor(float(torch.log(torch.tensor(temperature)))))
        self.p_target = p_target
        self.k = k
        self.ema_decay = ema_decay
        self.tau_min = tau_min
        self.tau_max = tau_max

        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

        # ---- in-batch mask
        mask = mask_correlated_samples(batch_size, torch.device("cpu"))
        self.register_buffer("mask", mask)

        # ---- running avg for tau control
        self.register_buffer("p_ema", torch.tensor(0.0))
        self.register_buffer("has_p_ema", torch.tensor(False))

        # ---- queue for negatives (MoCo-style)
        self.register_buffer("queue", torch.randn(out_dim, queue_size))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @property
    def tau(self):
        return self.log_tau.exp().clamp_(min=self.tau_min, max=self.tau_max)

    # -----------------------------
    # Queue ops
    # -----------------------------
    @torch.no_grad()
    def dequeue_and_enqueue(self, keys):
        """Enqueue new keys and dequeue oldest ones."""
        if not torch.distributed.is_initialized():
            keys_all = keys
        else:
            keys_all = concat_all_gather(keys)

        batch_size = keys_all.shape[0]
        K = self.queue.shape[1]
        ptr = int(self.queue_ptr)

        assert K % batch_size == 0, "Queue size must be divisible by batch size"

        # replace entries
        self.queue[:, ptr:ptr + batch_size] = keys_all.T
        ptr = (ptr + batch_size) % K
        self.queue_ptr[0] = ptr

    # -----------------------------
    # Tau update from probability
    # -----------------------------
    def _update_tau_from_prob(self, p_batch_mean, device):
        if not bool(self.has_p_ema.item()):
            self.p_ema = p_batch_mean.detach().to(device)
            self.has_p_ema = torch.tensor(True, device=device)
        else:
            self.p_ema = self.ema_decay * self.p_ema + (1 - self.ema_decay) * p_batch_mean.detach().to(device)

        error = self.p_ema - self.p_target
        with torch.no_grad():
            self.log_tau += self.k * error
            self.log_tau.clamp_(min=torch.log(torch.tensor(self.tau_min, device=device)),
                                max=torch.log(torch.tensor(self.tau_max, device=device)))

    # -----------------------------
    # Forward with queue negatives
    # -----------------------------
    def forward(self, z_i, z_j):
        device = z_i.device
        batch_size = z_i.size(0)
        N = 2 * batch_size

        mask = self.mask.to(device)
        z = torch.cat([z_i, z_j], dim=0)  # [2B, D]

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0))  # [2B,2B]
        logits = sim / self.tau

        # positives
        sim_i_j = torch.diag(logits,  batch_size)
        sim_j_i = torch.diag(logits, -batch_size)
        positives = torch.cat([sim_i_j, sim_j_i], dim=0).reshape(N, 1)

        # in-batch negatives
        negatives_inbatch = logits[mask].reshape(N, -1)

        # queue negatives
        if self.queue is not None:
            q_neg = torch.einsum("nc,ck->nk", [z, self.queue.clone().detach()]) / self.tau
            negatives = torch.cat([negatives_inbatch, q_neg], dim=1)
        else:
            negatives = negatives_inbatch

        # final logits
        all_logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(N, dtype=torch.long, device=device)

        loss = self.criterion(all_logits, labels) / N

        with torch.no_grad():
            probs = all_logits.softmax(dim=1)
            p_pos = probs[:, 0]
            p_mean = p_pos.mean()
            self._update_tau_from_prob(p_mean, device)

        return loss


# -----------------------------
# Distributed gather helper
# -----------------------------
@torch.no_grad()
def concat_all_gather(tensor):
    if not torch.distributed.is_initialized():
        return tensor
    tensors_gather = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    return torch.cat(tensors_gather, dim=0)



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
        self.curv_encoder = uvnet.encoders.UVNetCurveEncoder(in_channels=crv_in_channels, output_dims=crv_emb_dim)
        self.surf_encoder = uvnet.encoders.UVNetSurfaceEncoder(in_channels=4, output_dims=srf_emb_dim)
        self.graph_encoder = uvnet.encoders.UVNetGraphEncoder(srf_emb_dim, crv_emb_dim, graph_emb_dim)

        self.projection_layer = nn.Sequential(nn.Linear(latent_dim, latent_dim, bias=False),
                                              nn.BatchNorm1d(latent_dim),
                                              nn.Dropout(dropout),
                                              nn.ReLU(),
                                              nn.Linear(latent_dim, latent_dim, bias=False),
                                              nn.BatchNorm1d(latent_dim),
                                              nn.ReLU(),
                                              nn.Linear(latent_dim, out_dim, bias=False))

    def forward(self, bg):
        nfeat = bg.ndata["x"][:, [0, 1, 2, 6], :, :]  
        efeat = bg.edata["x"][:, :self.crv_in_channels, :]  
        crv_feat = self.curv_encoder(efeat)
        srf_feat = self.surf_encoder(nfeat)
        node_emb, graph_emb = self.graph_encoder(bg, srf_feat, crv_feat)
        global_emb = graph_emb  
        projection_out = self.projection_layer(global_emb)
        projection_out = F.normalize(projection_out, p=2, dim=-1)
        return projection_out, global_emb

class Contrastive(pl.LightningModule):
    def __init__(self, latent_dim=128, out_dim=128, batch_size=256, temperature=0.1, base_momentum=0.99):
        super().__init__()
        self.save_hyperparameters()
        self.loss_fn = NTXentLossAdaptiveProb(temperature=temperature,
                                              batch_size=batch_size,
                                              out_dim=out_dim)    
        self.model_1 = UVNetContrastiveLearner(latent_dim=latent_dim, out_dim=out_dim)
        self.model_2 = UVNetContrastiveLearner(latent_dim=latent_dim, out_dim=out_dim)

        # initialize model_2 with model_1 params
        self.model_2.load_state_dict(self.model_1.state_dict())
        for p in self.model_2.parameters():
            p.requires_grad = False

        # global step counter for scheduling
        self.register_buffer("global_step_counter", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def momentum_update(self):
        """MoCo-style momentum scheduling (cosine schedule)."""
        t = int(self.global_step_counter.item())

        # ---- total training steps
        if hasattr(self.trainer, "estimated_stepping_batches"):
            T = self.trainer.estimated_stepping_batches
        elif hasattr(self.trainer, "num_training_batches") and self.trainer.num_training_batches is not None:
            T = self.trainer.num_training_batches * self.trainer.max_epochs
        elif self.trainer.max_steps and self.trainer.max_steps > 0:
            T = self.trainer.max_steps
        else:
            T = 100000  # fallback constant

        base_m = self.hparams.base_momentum
        m = 1.0 - (1.0 - base_m) * (0.5 * (1.0 + math.cos(math.pi * t / T)))

        for param_q, param_k in zip(self.model_1.parameters(), self.model_2.parameters()):
            param_k.data = m * param_k.data + (1. - m) * param_q.data



    def _permute_graph_data_channels(self, graph):
        graph.ndata["x"] = graph.ndata["x"].permute(0, 3, 1, 2)
        graph.edata["x"] = graph.edata["x"].permute(0, 2, 1)
        return graph

    def step(self, batch, batch_idx):
        graph1, graph2 = batch["graph"], batch["graph2"]
        graph1 = self._permute_graph_data_channels(graph1)
        graph2 = self._permute_graph_data_channels(graph2)

        # Momentum encoder (keys)
        with torch.no_grad():
            proj2, _ = self.model_2(graph2)
            # enqueue keys into loss_fn queue
            self.loss_fn.dequeue_and_enqueue(proj2)

        # Online encoder (queries)
        proj1, _ = self.model_1(graph1)

        return self.loss_fn(proj1, proj2)

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        # self.momentum_update()
        self.global_step_counter += 1
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), weight_decay=1e-4)
        return optimizer

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
        assert len(embeddings) == data_count
        assert len(embeddings.shape) == 2
        assert len(outs) == data_count
        assert len(outs.shape) == 2
        if labels is not None:
            assert len(labels) == data_count
            assert len(labels.shape) == 1
        assert len(filenames) == data_count

        return {"embeddings": embeddings, "labels": labels, "outputs": outs, "filenames": filenames}
