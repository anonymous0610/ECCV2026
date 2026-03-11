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
import copy  #<==added






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



# NT-Xent with probability-controlled τ (same as Exp-26) but different hyperparams adjusted due to the temp=0.1 not 0.5
# NT-Xent loss function used in Exp-30, 31, 32
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

class DINOHead(nn.Module):  #<==added
    def __init__(self, in_dim, out_prototypes, norm_last_layer=True):  #<==added
        super().__init__()  #<==added
        self.prototypes = nn.Linear(in_dim, out_prototypes, bias=False)  #<==added
        if norm_last_layer:  #<==added
            self.prototypes.weight.data = F.normalize(self.prototypes.weight.data, dim=1)  #<==added

    def forward(self, z):  #<==added
        # normalize features and weights for cosine-classifier style logits  #<==added
        w = F.normalize(self.prototypes.weight, dim=1)  # [K, D]  #<==added
        z = F.normalize(z, dim=1)  # [B, D]  #<==added
        logits = z @ w.t()  # [B, K]  #<==added
     

# ====================== DINO w/ Momentum Teacher + Prototypes + Adaptive Temps ======================  #<==added
class DINOWithPrototypes(pl.LightningModule):  #<==added
    """                                                                                                  #<==added
    DINO-style predictive distillation over your UVNetContrastiveLearner backbones.                      #<==added
    - Momentum teacher (EMA)                                                                              #<==added
    - Prototype heads (student + teacher)                                                                 #<==added
    - Centering to avoid collapse                                                                         #<==added
    - Adaptive temperatures for BOTH student and teacher (entropy-controlled)                             #<==added
    """                                                                                                   #<==added
    def __init__(self,                                                                                   #<==added
                 latent_dim=128,                                                                          #<==added
                 out_dim=128,                                                                             #<==added
                 num_prototypes=256,                                                                      #<==added
                 base_momentum=0.996,                                                                     #<==added
                 center_momentum=0.9,                                                                     #<==added
                 student_temp_init=0.1,                                                                   #<==added
                 teacher_temp_init=0.07,                                                                  #<==added
                 tau_min=0.05, tau_max=2.0,                                                               #<==added
                 entropy_target_student=0.60,                                                             #<==added
                 entropy_target_teacher=0.70,                                                             #<==added
                 tau_k=0.05,                                                                              #<==added
                 tau_ema_decay=0.98,                                                                      #<==added
                 weight_decay=1e-4):                                                                      #<==added
        super().__init__()                                                                                #<==added
        self.save_hyperparameters()                                                                       #<==added

        # --- Backbones (student & teacher) ------------------------------------------------------------  #<==added
        self.student = UVNetContrastiveLearner(latent_dim=latent_dim, out_dim=out_dim)                    #<==added
        self.teacher = UVNetContrastiveLearner(latent_dim=latent_dim, out_dim=out_dim)                    #<==added

        # Initialize teacher as student (EMA teacher starts equal)                                       #<==added
        for ps, pt in zip(self.student.parameters(), self.teacher.parameters()):                          #<==added
            pt.data.copy_(ps.data)                                                                        #<==added
            pt.requires_grad = False                                                                      #<==added

        # --- Prototype heads (linear over normalized projections) -------------------------------------  #<==added
        self.student_head = nn.Linear(out_dim, num_prototypes, bias=False)                                #<==added
        self.teacher_head = nn.Linear(out_dim, num_prototypes, bias=False)                                #<==added
        self.teacher_head.weight.data.copy_(self.student_head.weight.data)                                #<==added
        for p in self.teacher_head.parameters():                                                          #<==added
            p.requires_grad = False                                                                       #<==added

        # --- Center buffer for teacher outputs (to avoid collapse) ------------------------------------  #<==added
        self.register_buffer("center", torch.zeros(1, num_prototypes))                                    #<==added
        self.center_momentum = center_momentum                                                            #<==added

        # --- Adaptive temperature (student + teacher) -------------------------------------------------  #<==added
        self.register_buffer("log_tau_student", torch.log(torch.tensor(float(student_temp_init))))         #<==added
        self.register_buffer("log_tau_teacher", torch.log(torch.tensor(float(teacher_temp_init))))         #<==added
        self.tau_min, self.tau_max = tau_min, tau_max                                                     #<==added
        self.entropy_target_student = entropy_target_student                                              #<==added
        self.entropy_target_teacher = entropy_target_teacher                                              #<==added
        self.tau_k = tau_k                                                                                #<==added
        self.tau_ema_decay = tau_ema_decay                                                                #<==added
        self.register_buffer("entropy_ema_s", torch.tensor(0.0))                                          #<==added
        self.register_buffer("entropy_ema_t", torch.tensor(0.0))                                          #<==added
        self.register_buffer("has_entropy_s", torch.tensor(False))                                        #<==added
        self.register_buffer("has_entropy_t", torch.tensor(False))                                        #<==added

        # --- Momentum scheduling ----------------------------------------------------------------------  #<==added
        self.base_momentum = base_momentum                                                                #<==added

        # --- Optim settings ---------------------------------------------------------------------------  #<==added
        self.weight_decay = weight_decay                                                                   #<==added

    # ====================== Utilities =================================================================  #<==added
    @property                                                                                             #<==added
    def tau_student(self):                                                                                #<==added
        return self.log_tau_student.exp().clamp(self.tau_min, self.tau_max)                               #<==added

    @property                                                                                             #<==added
    def tau_teacher(self):                                                                                #<==added
        return self.log_tau_teacher.exp().clamp(self.tau_min, self.tau_max)                               #<==added

    def _permute_graph_data_channels(self, graph):                                                        #<==added
        graph.ndata["x"] = graph.ndata["x"].permute(0, 3, 1, 2)                                           #<==added
        graph.edata["x"] = graph.edata["x"].permute(0, 2, 1)                                              #<==added
        return graph                                                                                       #<==added

    @torch.no_grad()                                                                                      #<==added
    def _momentum_update_teacher(self, m):                                                                #<==added
        # EMA update for backbone                                                                          #<==added
        for ps, pt in zip(self.student.parameters(), self.teacher.parameters()):                          #<==added
            pt.data = pt.data * m + ps.data * (1. - m)                                                    #<==added
        # EMA update for prototype head                                                                    #<==added
        for ps, pt in zip(self.student_head.parameters(), self.teacher_head.parameters()):                #<==added
            pt.data = pt.data * m + ps.data * (1. - m)                                                    #<==added

    def _cosine_momentum(self):                                                                           #<==added
        # Safe estimation of total steps                                                                   #<==added
        if hasattr(self.trainer, "estimated_stepping_batches") and self.trainer.estimated_stepping_batches:  #<==added
            T = self.trainer.estimated_stepping_batches                                                    #<==added
        elif getattr(self.trainer, "num_training_batches", None) is not None and getattr(self.trainer, "max_epochs", None):  #<==added
            T = self.trainer.num_training_batches * self.trainer.max_epochs                                #<==added
        elif getattr(self.trainer, "max_steps", 0) > 0:                                                    #<==added
            T = self.trainer.max_steps                                                                     #<==added
        else:                                                                                              #<==added
            T = 100000                                                                                     #<==added
        t = max(0, int(self.global_step))                                                                  #<==added
        m = 1.0 - (1.0 - self.base_momentum) * (0.5 * (1.0 + math.cos(math.pi * t / T)))                  #<==added
        return float(m)                                                                                    #<==added

    @torch.no_grad()                                                                                      #<==added
    def _update_center(self, probs_teacher):                                                               #<==added
        """EMA center update using mean teacher probabilities (across both views)."""                      #<==added
        batch_center = probs_teacher.mean(dim=0, keepdim=True)                                             #<==added
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)      #<==added

    def _update_tau_from_entropy(self, entropy, max_entropy, for_teacher=True):                           #<==added
        """Entropy-proportional controller in log-temperature space."""                                    #<==added
        ent_norm = (entropy / max_entropy).clamp(0, 1)                                                     #<==added
        if for_teacher:                                                                                    #<==added
            target, ema, flag = self.entropy_target_teacher, self.entropy_ema_t, self.has_entropy_t       #<==added
            logtau = self.log_tau_teacher                                                                  #<==added
        else:                                                                                              #<==added
            target, ema, flag = self.entropy_target_student, self.entropy_ema_s, self.has_entropy_s       #<==added
            logtau = self.log_tau_student                                                                  #<==added
        if not bool(flag.item()):                                                                          #<==added
            ema = ent_norm.detach()                                                                        #<==added
            flag = torch.tensor(True, device=entropy.device)                                               #<==added
        else:                                                                                              #<==added
            ema = self.tau_ema_decay * ema + (1 - self.tau_ema_decay) * ent_norm.detach()                  #<==added
        error = ema - target                                                                               #<==added
        with torch.no_grad():                                                                              #<==added
            logtau += self.tau_k * error                                                                   #<==added
        if for_teacher:                                                                                    #<==added
            self.entropy_ema_t, self.has_entropy_t, self.log_tau_teacher = ema, flag, logtau               #<==added
        else:                                                                                              #<==added
            self.entropy_ema_s, self.has_entropy_s, self.log_tau_student = ema, flag, logtau               #<==added

    # ====================== Forward passes ==============================================================  #<==added
    def _student_logits(self, g):                                                                          #<==added
        g = self._permute_graph_data_channels(g)                                                           #<==added
        proj, _ = self.student(g)                                                                          #<==added
        # return self.student_head(proj)                                                                     #<==added
        return proj

    @torch.no_grad()                                                                                      #<==added
    def _teacher_probs(self, g):                                                                           #<==added
        g = self._permute_graph_data_channels(g)                                                           #<==added
        proj, _ = self.teacher(g)                                                                          #<==added
        # logits = self.teacher_head(proj)                                                                   #<==added
        logits = proj
        logits = (logits - self.center) / self.tau_teacher                                                 #<==added
        return F.softmax(logits, dim=1)                                                                    #<==added

    # ====================== Training / Validation steps =================================================  #<==added
    def training_step(self, batch, batch_idx):                                                             #<==added
        g1, g2 = batch["graph"], batch["graph2"]                                                           #<==added

        # student forward                                                                                   #<==added
        s1 = self._student_logits(g1)                                                                      #<==added
        s2 = self._student_logits(g2)                                                                      #<==added

        # teacher forward (EMA, no grad)                                                                    #<==added
        with torch.no_grad():                                                                               #<==added
            self.teacher.eval()                                                                             #<==added
            t1 = self._teacher_probs(g1)                                                                    #<==added
            t2 = self._teacher_probs(g2)                                                                    #<==added

        # DINO loss: cross-view distillation                                                                #<==added
        # student log-probs with adaptive student temperature                                               #<==added
        ls1 = F.log_softmax(s1 / self.tau_student, dim=1)                                                   #<==added
        ls2 = F.log_softmax(s2 / self.tau_student, dim=1)                                                   #<==added
        loss12 = -(t1 * ls2).sum(dim=1).mean()                                                              #<==added
        loss21 = -(t2 * ls1).sum(dim=1).mean()                                                              #<==added
        loss = (loss12 + loss21) * 0.5                                                                      #<==added

        # center update                                                                                      #<==added
        with torch.no_grad():                                                                               #<==added
            self._update_center(torch.cat([t1, t2], dim=0))                                                 #<==added

        # adaptive τ updates via entropy                                                                     #<==added
        with torch.no_grad():                                                                               #<==added
            # teacher entropy                                                                                #<==added
            probs_t_all = torch.cat([t1, t2], dim=0)                                                        #<==added
            ent_t = -(probs_t_all * (probs_t_all + 1e-12).log()).sum(dim=1).mean()                          #<==added
            max_ent_t = torch.log(torch.tensor(probs_t_all.size(1), device=self.device))                    #<==added
            self._update_tau_from_entropy(ent_t, max_ent_t, for_teacher=True)                               #<==added
            # student entropy                                                                                #<==added
            probs_s_all = torch.cat([F.softmax(s1 / self.tau_student, dim=1),                               #<==added
                                     F.softmax(s2 / self.tau_student, dim=1)], dim=0)                       #<==added
            ent_s = -(probs_s_all * (probs_s_all + 1e-12).log()).sum(dim=1).mean()                          #<==added
            max_ent_s = torch.log(torch.tensor(probs_s_all.size(1), device=self.device))                    #<==added
            self._update_tau_from_entropy(ent_s, max_ent_s, for_teacher=False)                              #<==added

        # teacher momentum update                                                                            #<==added
        with torch.no_grad():                                                                               #<==added
            m = self._cosine_momentum()                                                                     #<==added
            self._momentum_update_teacher(m)                                                                 #<==added

        # logs                                                                                               #<==added
        self.log("train/loss", loss, on_step=False, on_epoch=True, sync_dist=True)                          #<==added
        self.log("train/tau_teacher", self.tau_teacher.item(), on_step=True, on_epoch=False, sync_dist=True) #<==added
        self.log("train/tau_student", self.tau_student.item(), on_step=True, on_epoch=False, sync_dist=True) #<==added
        self.log("train/center_mean", self.center.mean().item(), on_step=True, on_epoch=False, sync_dist=True) #<==added
        self.log("train/momentum", m, on_step=True, on_epoch=False, sync_dist=True)                         #<==added
        self.log("train/entropy_teacher", ent_t.item(), on_step=True, on_epoch=False, sync_dist=True)       #<==added
        self.log("train/entropy_student", ent_s.item(), on_step=True, on_epoch=False, sync_dist=True)       #<==added

        return loss                                                                                         #<==added

    def validation_step(self, batch, batch_idx):                                                            #<==added
        g1, g2 = batch["graph"], batch["graph2"]                                                           #<==added
        s1 = self._student_logits(g1)                                                                      #<==added
        s2 = self._student_logits(g2)                                                                      #<==added
        with torch.no_grad():                                                                               #<==added
            t1 = self._teacher_probs(g1)                                                                    #<==added
            t2 = self._teacher_probs(g2)                                                                    #<==added
        ls1 = F.log_softmax(s1 / self.tau_student, dim=1)                                                   #<==added
        ls2 = F.log_softmax(s2 / self.tau_student, dim=1)                                                   #<==added
        loss12 = -(t1 * ls2).sum(dim=1).mean()                                                              #<==added
        loss21 = -(t2 * ls1).sum(dim=1).mean()                                                              #<==added
        loss = 0.5 * (loss12 + loss21)                                                                      #<==added
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)                            #<==added
        return loss                                                                                         #<==added

    # ====================== Optimizer ===================================================================  #<==added
    def configure_optimizers(self):                                                                         #<==added
        # Optimize only the student + student_head                                                          #<==added
        params = list(self.student.parameters()) + list(self.student_head.parameters())                     #<==added
        optimizer = torch.optim.Adam(params, weight_decay=self.weight_decay)                                #<==added
        return optimizer                                                                                    #<==added
# ========================================================================================================  #<==added


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
            proj, emb = self.student(bg)
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

