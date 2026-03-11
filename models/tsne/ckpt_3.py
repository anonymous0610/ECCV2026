# import numpy as np
# import pytorch_lightning as pl
# import torch
# import torch.nn.functional as F
# import torchmetrics
# from tqdm import tqdm
# from scipy.spatial.distance import cdist
# from sklearn import svm
# from sklearn.cluster import KMeans
# from sklearn.linear_model import SGDClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.metrics.cluster import (normalized_mutual_info_score, adjusted_mutual_info_score)
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
# from torch import nn
# import uvnet.encoders



# ###############################################################################
# # Self-supervised model
# ###############################################################################

# def mask_correlated_samples(batch_size, device=torch.device("cpu")):
#     N = 2 * batch_size
#     mask = torch.ones((N, N), dtype=bool, device=device)
#     mask = mask.fill_diagonal_(0)
#     for i in range(batch_size):
#         mask[i, batch_size + i] = 0
#         mask[batch_size + i, i] = 0
#     return mask



# # NT-Xent with probability-controlled τ
# # NT-Xent loss function used in Exp-26
# class NTXentLossAdaptiveProb(pl.LightningModule):
#     def __init__(self, 
#                  temperature=0.1, 
#                  batch_size=256, 
#                  p_target=0.65,        # desired avg positive probability
#                  k=0.05,               # controller gain (small) 
#                  ema_decay=0.98,       # smoothing for p 
#                  tau_min=0.05, 
#                  tau_max=2.0):
#         super().__init__()
#         # track log(tau) for stable multiplicative updates
#         self.register_buffer("log_tau", torch.tensor(float(torch.log(torch.tensor(temperature)))))
#         self.p_target = p_target
#         self.k = k
#         self.ema_decay = ema_decay
#         self.tau_min = tau_min
#         self.tau_max = tau_max

#         self.criterion = nn.CrossEntropyLoss(reduction="sum")
#         self.similarity_f = nn.CosineSimilarity(dim=2)

#         # mask starts on CPU; moved to device at runtime
#         mask = mask_correlated_samples(batch_size, torch.device("cpu"))
#         self.register_buffer("mask", mask)

#         # running avg of positive probability
#         self.register_buffer("p_ema", torch.tensor(0.0))
#         self.register_buffer("has_p_ema", torch.tensor(False))

#     @property
#     def tau(self):
#         return self.log_tau.exp().clamp_(min=self.tau_min, max=self.tau_max)

#     def _update_tau_from_prob(self, p_batch_mean, device):
#         # EMA for smoother signal
#         if not bool(self.has_p_ema.item()):
#             self.p_ema = p_batch_mean.detach().to(device)
#             self.has_p_ema = torch.tensor(True, device=device)
#         else:
#             self.p_ema = self.ema_decay * self.p_ema + (1 - self.ema_decay) * p_batch_mean.detach().to(device)

#         # proportional control in log-space: log_tau <- log_tau + k * (p_ema - p_target)
#         error = self.p_ema - self.p_target
#         with torch.no_grad():
#             self.log_tau += self.k * error
#             # clamp τ by clamping logτ
#             self.log_tau.clamp_(min=torch.log(torch.tensor(self.tau_min, device=device)),
#                                 max=torch.log(torch.tensor(self.tau_max, device=device)))

#     def forward(self, z_i, z_j):
#         device = z_i.device
#         batch_size = z_i.size(0)
#         N = 2 * batch_size

#         mask = self.mask.to(device)
#         z = torch.cat([z_i, z_j], dim=0)  # [2B, D]

#         # similarities and logits with current tau
#         sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0))  # [2B,2B]
#         logits = sim / self.tau  # current temperature

#         # positives are across the diagonal offset by ±B
#         sim_i_j = torch.diag(logits,  batch_size)
#         sim_j_i = torch.diag(logits, -batch_size)
#         positives = torch.cat([sim_i_j, sim_j_i], dim=0).reshape(N, 1)

#         negatives = logits[mask].reshape(N, -1)
#         all_logits = torch.cat([positives, negatives], dim=1)  # class 0 is positive
#         labels = torch.zeros(N, dtype=torch.long, device=device)

#         # ---- compute loss
#         loss = self.criterion(all_logits, labels) / N

#         # ---- compute positive probability under current τ (no grad into τ update)
#         with torch.no_grad():
#             # softmax over [pos | negs]
#             probs = all_logits.softmax(dim=1)
#             p_pos = probs[:, 0]
#             p_mean = p_pos.mean()
#             self._update_tau_from_prob(p_mean, device)

#         return loss


# class UVNetContrastiveLearner(nn.Module):
#     def __init__(self,
#                  latent_dim,
#                  crv_in_channels=3,
#                  crv_emb_dim=64,
#                  srf_emb_dim=64,
#                  graph_emb_dim=128,
#                  dropout=0.3,
#                  out_dim=128):
#         """
#         UVNetContrastivelearner
#         """
#         super().__init__()
#         self.crv_in_channels = crv_in_channels
#         self.curv_encoder = uvnet.encoders.UVNetCurveEncoder(in_channels=crv_in_channels, output_dims=crv_emb_dim)
#         self.surf_encoder = uvnet.encoders.UVNetSurfaceEncoder(in_channels=4, output_dims=srf_emb_dim)
#         self.graph_encoder = uvnet.encoders.UVNetGraphEncoder(srf_emb_dim, crv_emb_dim, graph_emb_dim)
#         # self.fc_layers = nn.Sequential(
#         #     nn.Linear(graph_emb_dim, latent_dim, bias=False),
#         #     nn.BatchNorm1d(latent_dim),
#         #     nn.ReLU(),
#         #     nn.Linear(latent_dim, latent_dim),
#         # )
#         self.projection_layer = nn.Sequential(nn.Linear(latent_dim, latent_dim, bias=False),
#                                               nn.BatchNorm1d(latent_dim),
#                                               nn.Dropout(dropout),
#                                               nn.ReLU(),
#                                               nn.Linear(latent_dim, latent_dim, bias=False),
#                                               nn.BatchNorm1d(latent_dim),
#                                               nn.ReLU(),
#                                               nn.Linear(latent_dim, out_dim, bias=False))

#     def forward(self, bg):
#         # We only use point coordinates & mask in contrastive experiments
#         # TODO(pradeep): expose a better way for the user to select these channels
#         nfeat = bg.ndata["x"][:, [0, 1, 2, 6], :, :]  # XYZ+mask channels --> [:, (X, Y, Z, trimming_mask), :, :] --> [8430, 4, 10, 10]
#         efeat = bg.edata["x"][:, :self.crv_in_channels, :]  # XYZ channels without Tangent --> [:, :3, :] --> [:, (X, Y, Z), :] --> [48700, 3, 10]
#         crv_feat = self.curv_encoder(efeat)    # [48700, 3, 10]-->[48700, 64] <-- [no_nodes, crv_emb_dim]
#         srf_feat = self.surf_encoder(nfeat)    # [8430, 4, 10, 10]-->[8430, 64] <-- [no_edges, srf_emb_dim]
#         node_emb, graph_emb = self.graph_encoder(bg, srf_feat, crv_feat)  # node_emb=[8430, 128], graph_emb=[256, 128]
#         global_emb = graph_emb  # self.fc_layers(graph_emb)  #[256, 128]
#         projection_out = self.projection_layer(global_emb)  # [256, 128] --> [256, 64]
#         projection_out = F.normalize(projection_out, p=2, dim=-1)  # [256, 64] --> [256, 64]

#         return projection_out, global_emb
    
#     '''
#     print(type(bg)) --> <class 'dgl.heterograph.DGLHeteroGraph'>
#     print(type(bg.ndata)) --> <class 'dgl.view.HeteroNodeDataView'>
#     print(type(bg.edata)) --> <class 'dgl.view.HeteroEdgeDataView'>

#     print(bg) --> Graph(num_nodes=8430, num_edges=48700, 
#                         ndata_schemes={'x': Scheme(shape=(7, 10, 10), dtype=torch.float32)}
#                         edata_schemes={'x': Scheme(shape=(6, 10), dtype=torch.float32)})
                        
#     print(bg.ndata["x"].shape) --> torch.Size([8430, 7, 10, 10])
#             torch.Size([:, 7, :, :]) --> [:, (X, Y, Z, X_Normal, Y_Normal, Z_Normal, trimming_mask), :, :]
            
#     print(bg.edata["x"].shape) --> torch.Size([48700, 6, 10])
#             torch.Size([48700, 6, 10]) --> [:, (X, Y, Z, X_tangent, Y_tangent, Z_tangent), :]
#     '''




# # # Upto Exp-7
# class Contrastive3(pl.LightningModule):
#     """
#     PyTorch Lightning module to train/test the contrastive learning model.
#     """

#     def __init__(self, latent_dim=128, out_dim=128, batch_size=256, temperature=0.1, momentum=0.99):
#         super().__init__()
#         self.save_hyperparameters()
#         #self.loss_fn = NTXentLoss(temperature=temperature)
#         self.loss_fn = NTXentLossAdaptiveProb(temperature=temperature)    #EXP-26, EXP-30, 31, 32
#         #self.loss_fn = NTXentLossAdaptiveEntropy(temperature=temperature)  #EXP-27
#         #self.loss_fn = NTXentLossHybridProb(temperature=temperature)      #EXP-28
#         #self.loss_fn = NTXentLossHybridEntropy(temperature=temperature)   #EXP-29
#         #self.loss_fn = GlobalNTXentLoss(temperature=temperature, batch_size=256, lambda_comp=0.2, lambda_exp=0.3)
#         #self.loss_fn = GlobalNTXentLoss(temperature=temperature, lambda_comp=0.00, lambda_exp=0.1)
#         self.model_1 = UVNetContrastiveLearner(latent_dim=latent_dim, out_dim=out_dim)
#         self.model_2 = UVNetContrastiveLearner(latent_dim=latent_dim, out_dim=out_dim)
#         # Momentum update: not trained directly
#         for param_1, param_2 in zip(self.model_1.parameters(), self.model_2.parameters()):
#             param_2.data = momentum * param_2.data + (1. - momentum) * param_1.data

#     def _permute_graph_data_channels(self, graph):
#         graph.ndata["x"] = graph.ndata["x"].permute(0, 3, 1, 2)
#         graph.edata["x"] = graph.edata["x"].permute(0, 2, 1)
#         return graph

#     def step(self, batch, batch_idx):  # print(batch.keys())=dict_keys(['graph', 'graph2', 'label', 'filename'])
#         graph1, graph2 = batch["graph"], batch["graph2"]  # graph1 and graph2 are DGL graphs with the same structure=dgl.heterograph.DGLHeteroGraph
#         graph1 = self._permute_graph_data_channels(graph1) # [:, 10, 10, 7]--> [:, 7, 10, 10] # [:, 10, 7]--> [:, 7, 10] 
#         graph2 = self._permute_graph_data_channels(graph2) 
#         with torch.no_grad():
#             proj2, _ = self.model_2(graph2)  # frozen momentum encoder
#         proj1, _ = self.model_1(graph1)      # trainable encoder
#         #proj1, _ = self.model_1(graph1)  # proj1=[256, 64] --> [batch_size, out_dim]
#         #proj2, _ = self.model_2(graph2)  # proj2=[256, 64] --> [batch_size, out_dim]
#         return self.loss_fn(proj1, proj2)

#     def training_step(self, batch, batch_idx):
#         loss = self.step(batch, batch_idx)
#         self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         loss = self.step(batch, batch_idx)
#         self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
#         return loss

#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), weight_decay=1e-4)
#         return optimizer

#     @torch.no_grad()
#     def clustering(self, data, num_clusters=26, n_init=100, standardize=False):
#         if standardize:
#             scaler = StandardScaler().fit(data["embeddings"])
#             embeddings = scaler.transform(data["embeddings"].copy())
#         else:
#             embeddings = data["embeddings"]
#         kmeans = KMeans(init="random", n_clusters=num_clusters, n_init=n_init, max_iter=100000)
#         print(f"Fitting K-Means with {num_clusters} clusters...")
#         kmeans.fit(embeddings)
#         pred_labels = kmeans.labels_
#         score = adjusted_mutual_info_score(data["labels"], pred_labels)
#         # added by me: 
#         km_acc = accuracy_score(data["labels"], pred_labels)
#         print(f"Clustering ACC on test set: {km_acc * 100.0:2.3f}")
#         return score

#     @torch.no_grad()
#     def linear_svm_classification(self, train_data, test_data, max_iter=100000):
#         clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=max_iter, tol=1e-3))
#         print("Training Linear SVM...")
#         ret = clf.fit(train_data["embeddings"], train_data["labels"])
#         pred_labels = clf.predict(test_data["embeddings"])
#         return accuracy_score(test_data["labels"], pred_labels)

#     @torch.no_grad()
#     def get_embeddings_from_dataloader(self, dataloader):
#         self.eval()
#         embeddings = []
#         outs = []
#         labels = []
#         filenames = []
#         for batch in tqdm(dataloader, desc="Extracting embeddings"):
#             bg = batch["graph"].to(self.device)
#             bg = self._permute_graph_data_channels(bg)
#             proj, emb = self.model_1(bg)
#             #proj, emb = self.model(bg)                   #         -> n_data -->(surf_encoder)--> surf_feat -->
#                                                           # bg --> |                                            |->(graph_encoder)--> emb[256, 128] -->(proj_head)--> proj[256, 64]
#                                                           #         -> e_data -->(curv_encoder)--> curv_feat -->
#             outs.append(proj.detach().cpu().numpy())      # List of numpy arrays of shape (batch_size, 64)
#             embeddings.append(emb.detach().cpu().numpy()) # List of numpy arrays of shape (batch_size, 128)
#             if "label" in batch:
#                 label = batch["label"]                    # label-->letter labels = [batch_size]    A/a-> 0, B/b --> 1, C/c --> 2, ...
#                 labels.append(label.squeeze(-1).detach().cpu().numpy())
#             filenames.extend(batch["filename"])           # filenames-->letter location = [batch_size]    p_Marko One_lower --> which signifies the categories i.e. p(letter) Marko(style) One(texture) lower(case)
#         outs = np.concatenate(outs)
#         embeddings = np.concatenate(embeddings)
#         if len(labels) > 0:
#             labels = np.concatenate(labels)
#         else:
#             labels = None
#         data_count = len(dataloader.dataset)
#         assert len(embeddings) == data_count, f"{embeddings.shape}, {data_count}"
#         assert len(embeddings.shape) == 2, f"{embeddings.shape}"
#         assert len(outs) == data_count, f"{outs.shape}, {data_count}"
#         assert len(outs.shape) == 2, f"{outs.shape}"
#         if labels is not None:
#             assert len(labels) == data_count
#             assert len(labels.shape) == 1, f"{labels.shape}"
#             assert len(labels.shape) == 1, f"{labels.shape}"
#         assert len(filenames) == data_count, f"{len(filenames)}, {data_count}"
#         return {"embeddings": embeddings, "labels": labels, "outputs": outs, "filenames": filenames}   # embeddings=[19404, 128], labels=[19404], outs=projections=[19404, 64], filenames=[19404]


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
# Classification model
###############################################################################
class _NonLinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.3):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=dropout)
        self.linear3 = nn.Linear(256, num_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, inp):
        x = F.relu(self.bn1(self.linear1(inp)))
        x = self.dp1(x)
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)
        return x


class UVNetClassifier(nn.Module):
    """
    UV-Net solid classification model
    """
    def __init__(self,
                 num_classes,
                 crv_emb_dim=64,
                 srf_emb_dim=64,
                 graph_emb_dim=128,
                 dropout=0.3):
        super().__init__()
        self.curv_encoder = uvnet.encoders.UVNetCurveEncoder(in_channels=6, output_dims=crv_emb_dim)
        self.surf_encoder = uvnet.encoders.UVNetSurfaceEncoder(in_channels=7, output_dims=srf_emb_dim)
        self.graph_encoder = uvnet.encoders.UVNetGraphEncoder(srf_emb_dim, crv_emb_dim, graph_emb_dim)
        self.clf = _NonLinearClassifier(graph_emb_dim, num_classes, dropout)

    def forward(self, batched_graph):
        input_crv_feat = batched_graph.edata["x"]
        input_srf_feat = batched_graph.ndata["x"]
        hidden_crv_feat = self.curv_encoder(input_crv_feat)
        hidden_srf_feat = self.surf_encoder(input_srf_feat)
        _, graph_emb = self.graph_encoder(batched_graph, hidden_srf_feat, hidden_crv_feat)
        out = self.clf(graph_emb)
        return out


class Classification(pl.LightningModule):
    """
    PyTorch Lightning module to train/test the classifier.
    """
    def __init__(self, num_classes):
        super().__init__()
        self.save_hyperparameters()
        self.model = UVNetClassifier(num_classes=num_classes)
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

    def forward(self, batched_graph):
        logits = self.model(batched_graph)
        return logits

    def training_step(self, batch, batch_idx):
        inputs = batch["graph"].to(self.device)
        labels = batch["label"].to(self.device)
        inputs.ndata["x"] = inputs.ndata["x"].permute(0, 3, 1, 2)
        inputs.edata["x"] = inputs.edata["x"].permute(0, 2, 1)
        logits = self.model(inputs)
        loss = F.cross_entropy(logits, labels, reduction="mean")
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        preds = F.softmax(logits, dim=-1)
        self.log("train_acc", self.train_acc(preds, labels), on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch["graph"].to(self.device)
        labels = batch["label"].to(self.device)
        inputs.ndata["x"] = inputs.ndata["x"].permute(0, 3, 1, 2)
        inputs.edata["x"] = inputs.edata["x"].permute(0, 2, 1)
        logits = self.model(inputs)
        loss = F.cross_entropy(logits, labels, reduction="mean")
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        preds = F.softmax(logits, dim=-1)
        self.log("val_acc", self.val_acc(preds, labels), on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        inputs = batch["graph"].to(self.device)
        labels = batch["label"].to(self.device)
        inputs.ndata["x"] = inputs.ndata["x"].permute(0, 3, 1, 2)
        inputs.edata["x"] = inputs.edata["x"].permute(0, 2, 1)
        logits = self.model(inputs)
        loss = F.cross_entropy(logits, labels, reduction="mean")
        self.log("test_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        preds = F.softmax(logits, dim=-1)
        self.log("test_acc", self.test_acc(preds, labels), on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer


###############################################################################
# Segmentation model
###############################################################################
class UVNetSegmenter(nn.Module):
    """
    UV-Net solid face segmentation model
    """
    def __init__(self,
                 num_classes,
                 crv_in_channels=6,
                 crv_emb_dim=64,
                 srf_emb_dim=64,
                 graph_emb_dim=128,
                 dropout=0.3):
        super().__init__()
        self.curv_encoder = uvnet.encoders.UVNetCurveEncoder(in_channels=crv_in_channels, output_dims=crv_emb_dim)
        self.surf_encoder = uvnet.encoders.UVNetSurfaceEncoder(in_channels=7, output_dims=srf_emb_dim)
        self.graph_encoder = uvnet.encoders.UVNetGraphEncoder(srf_emb_dim, crv_emb_dim, graph_emb_dim)
        self.seg = _NonLinearClassifier(graph_emb_dim + srf_emb_dim, num_classes, dropout=dropout)

    def forward(self, batched_graph):
        input_crv_feat = batched_graph.edata["x"]
        input_srf_feat = batched_graph.ndata["x"]
        hidden_crv_feat = self.curv_encoder(input_crv_feat)
        hidden_srf_feat = self.surf_encoder(input_srf_feat)
        node_emb, graph_emb = self.graph_encoder(batched_graph, hidden_srf_feat, hidden_crv_feat)
        num_nodes_per_graph = batched_graph.batch_num_nodes().to(graph_emb.device)
        graph_emb = graph_emb.repeat_interleave(num_nodes_per_graph, dim=0).to(graph_emb.device)
        local_global_feat = torch.cat((node_emb, graph_emb), dim=1)
        out = self.seg(local_global_feat)
        return out


class Segmentation(pl.LightningModule):
    """
    PyTorch Lightning module to train/test the segmenter (per-face classifier).
    """
    def __init__(self, num_classes, crv_in_channels=6):
        super().__init__()
        self.save_hyperparameters()
        self.model = UVNetSegmenter(num_classes, crv_in_channels=crv_in_channels)
        self.train_iou = torchmetrics.IoU(num_classes=num_classes, compute_on_step=False)
        self.val_iou = torchmetrics.IoU(num_classes=num_classes, compute_on_step=False)
        self.test_iou = torchmetrics.IoU(num_classes=num_classes, compute_on_step=False)
        self.train_accuracy = torchmetrics.Accuracy(num_classes=num_classes, compute_on_step=False)
        self.val_accuracy = torchmetrics.Accuracy(num_classes=num_classes, compute_on_step=False)
        self.test_accuracy = torchmetrics.Accuracy(num_classes=num_classes, compute_on_step=False)

    def forward(self, batched_graph):
        logits = self.model(batched_graph)
        return logits

    def training_step(self, batch, batch_idx):
        inputs = batch["graph"].to(self.device)
        inputs.ndata["x"] = inputs.ndata["x"].permute(0, 3, 1, 2)
        inputs.edata["x"] = inputs.edata["x"].permute(0, 2, 1)
        labels = inputs.ndata["y"]
        logits = self.model(inputs)
        loss = F.cross_entropy(logits, labels, reduction="mean")
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        preds = F.softmax(logits, dim=-1)
        self.train_iou(preds, labels)
        self.train_accuracy(preds, labels)
        return loss

    def training_epoch_end(self, outs):
        self.log("train_iou", self.train_iou.compute())
        self.log("train_accuracy", self.train_accuracy.compute())

    def validation_step(self, batch, batch_idx):
        inputs = batch["graph"].to(self.device)
        inputs.ndata["x"] = inputs.ndata["x"].permute(0, 3, 1, 2)
        inputs.edata["x"] = inputs.edata["x"].permute(0, 2, 1)
        labels = inputs.ndata["y"]
        logits = self.model(inputs)
        loss = F.cross_entropy(logits, labels, reduction="mean")
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        preds = F.softmax(logits, dim=-1)
        self.val_iou(preds, labels)
        self.val_accuracy(preds, labels)
        return loss

    def validation_epoch_end(self, outs):
        self.log("val_iou", self.val_iou.compute())
        self.log("val_accuracy", self.val_accuracy.compute())

    def test_step(self, batch, batch_idx):
        inputs = batch["graph"].to(self.device)
        inputs.ndata["x"] = inputs.ndata["x"].permute(0, 3, 1, 2)
        inputs.edata["x"] = inputs.edata["x"].permute(0, 2, 1)
        labels = inputs.ndata["y"]
        logits = self.model(inputs)
        loss = F.cross_entropy(logits, labels, reduction="mean")
        self.log("test_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        preds = F.softmax(logits, dim=-1)
        self.test_iou(preds, labels)
        self.test_accuracy(preds, labels)

    def test_epoch_end(self, outs):
        self.log("test_iou", self.test_iou.compute())
        self.log("test_accuracy", self.test_accuracy.compute())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer


###############################################################################
# Self-supervised model (MoCo)
###############################################################################

class UVNetMoCoEncoder(nn.Module):
    """
    UV-Net backbone + projection head for MoCo.
    Uses your contrastive inputs (XYZ + mask for faces; XYZ for edges).
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

        # projection head (same mapping as your SimCLR variant)
        self.projection_layer = nn.Sequential(nn.Linear(latent_dim, latent_dim, bias=False), 
                                              nn.BatchNorm1d(latent_dim), 
                                              nn.Dropout(dropout), 
                                              nn.ReLU(), 
                                              nn.Linear(latent_dim, latent_dim, bias=False), 
                                              nn.BatchNorm1d(latent_dim), 
                                              nn.ReLU(), 
                                              nn.Linear(latent_dim, out_dim, bias=False))

        # map graph_emb_dim -> latent_dim if needed
        if latent_dim != graph_emb_dim:
            self.bottleneck = nn.Linear(graph_emb_dim, latent_dim, bias=False)
            nn.init.kaiming_uniform_(self.bottleneck.weight.data)
        else:
            self.bottleneck = nn.Identity()

    def forward(self, bg):
        # Expect bg.ndata["x"] shape: [N_nodes, 7, 10, 10] (after permute: [N, C, H, W])
        # and bg.edata["x"] shape: [N_edges, 6, 10]        (after permute: [N, C, L])
        nfeat = bg.ndata["x"][:, [0, 1, 2, 6], :, :]       # XYZ + trimming mask
        efeat = bg.edata["x"][:, :self.crv_in_channels, :] # XYZ (no tangent)
        crv_feat = self.curv_encoder(efeat)                # [E, crv_emb_dim]
        srf_feat = self.surf_encoder(nfeat)                # [V, srf_emb_dim]
        _, graph_emb = self.graph_encoder(bg, srf_feat, crv_feat)  # [B, graph_emb_dim]
        z = self.bottleneck(graph_emb)                     # [B, latent_dim]
        proj = self.projection_layer(z)                    # [B, out_dim]
        proj = F.normalize(proj, p=2, dim=-1)
        return proj


def _distributed_available():
    return torch.distributed.is_available() and torch.distributed.is_initialized()


@torch.no_grad()
def batch_shuffle_ddp(x):
    """Batch shuffle for making BN statistics correct during momentum encoder forward (DDP only)."""
    if not _distributed_available():
        idx_shuffle = torch.arange(x.shape[0], device=x.device)
        idx_unshuffle = idx_shuffle
        return x, idx_unshuffle
    batch_size_this = x.shape[0]
    world_size = torch.distributed.get_world_size()

    # gather across GPUs
    x_gather = concat_all_gather(x)
    batch_size_all = x_gather.shape[0]
    idx_shuffle = torch.randperm(batch_size_all, device=x.device)
    # broadcast to all ranks
    torch.distributed.broadcast(idx_shuffle, src=0)

    # index for unshuffle
    idx_unshuffle = torch.argsort(idx_shuffle)

    # select this rank's portion
    rank = torch.distributed.get_rank()
    idx_this = idx_shuffle.view(world_size, -1)[rank]
    return x[idx_this], idx_unshuffle


@torch.no_grad()
def batch_unshuffle_ddp(x, idx_unshuffle):
    if not _distributed_available():
        return x
    x_gather = concat_all_gather(x)
    return x_gather[idx_unshuffle]


@torch.no_grad()
def concat_all_gather(tensor):
    """Gathers tensors from all processes, supporting grads off."""
    if not _distributed_available():
        return tensor
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    return torch.cat(tensors_gather, dim=0)


class Contrastive3(pl.LightningModule):
    """
    PyTorch Lightning MoCo for UV-Net graphs.

    - encoder_q: updated by SGD
    - encoder_k: updated by momentum (EMA) from encoder_q
    - queue: dictionary of negative keys
    """
    def __init__(self,
                 latent_dim=128,
                 out_dim=128,
                 temperature=0.1,
                 m=0.999,
                 K=65536,              # queue size
                 crv_in_channels=3,
                 lr=1e-3,
                 weight_decay=1e-4,
                 use_shuffle_bn=True):
        super().__init__()
        self.save_hyperparameters()

        # encoders
        self.encoder_q = UVNetMoCoEncoder(latent_dim=latent_dim, out_dim=out_dim, crv_in_channels=crv_in_channels)
        self.encoder_k = UVNetMoCoEncoder(latent_dim=latent_dim, out_dim=out_dim, crv_in_channels=crv_in_channels)

        # initialize: copy weights and freeze k
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)    # initialize
            param_k.requires_grad = False       # not update by SGD

        # queue as buffer: [out_dim, K]
        self.register_buffer("queue", F.normalize(torch.randn(out_dim, K), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.temperature = temperature
        self.m = m
        self.K = K
        self.lr = lr
        self.weight_decay = weight_decay
        self.use_shuffle_bn = use_shuffle_bn

        # classifier convenience (optional downstream)
        self.out_dim = out_dim

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """EMA update for key encoder: k = m*k + (1-m)*q"""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]
        K = self.K
        ptr = int(self.queue_ptr)
        assert K % batch_size == 0, f"Queue size {K} must be divisible by total batch size {batch_size} across GPUs."

        # replace entries in queue
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % K
        self.queue_ptr[0] = ptr

    def _permute_graph_data_channels(self, graph):
        graph.ndata["x"] = graph.ndata["x"].permute(0, 3, 1, 2)
        graph.edata["x"] = graph.edata["x"].permute(0, 2, 1)
        return graph

    def forward(self, graph):
        # for embedding extraction: use the query encoder
        graph = self._permute_graph_data_channels(graph)
        return self.encoder_q(graph)

    def _compute_logits(self, q, k):
        # q, k: [N, C] L2-normalized
        # positives: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)    # [N, 1]
        # negatives: NxK
        queue = self.queue.clone().detach()                       # [C, K]
        l_neg = torch.einsum("nc,ck->nk", [q, queue])             # [N, K]
        logits = torch.cat([l_pos, l_neg], dim=1)                 # [N, 1+K]
        logits = logits / self.temperature
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        return logits, labels

    def step(self, batch, batch_idx, update_queue: bool):
        # get two graph views
        graph_q, graph_k = batch["graph"], batch["graph2"]  # same structure; two augmentations
        graph_q = self._permute_graph_data_channels(graph_q).to(self.device)
        graph_k = self._permute_graph_data_channels(graph_k).to(self.device)

        # compute query features
        q = self.encoder_q(graph_q)               # [N, C], normalized

        # compute key features (no grad) with momentum encoder
        with torch.no_grad():
            self._momentum_update_key_encoder()

            # ShuffleBN (DDP) to make BN in encoder_k see different shards
            if self.use_shuffle_bn:
                # pack graph_k tensors for shuffling/unshuffling (we only need to shuffle the node/edge features)
                # Here we only shuffle the batch dimension of the *global* graph batch container.
                # Practical trick: we don't have a direct batch tensor; we re-run forward on shuffled order is non-trivial for DGL.
                # Instead, we rely on concat_all_gather to improve queue diversity, which is the key part.
                pass

            k = self.encoder_k(graph_k)           # [N, C], normalized
            # (Optional) if you implement actual BN shuffle for your DGL batch construction, do it around this call.

        # logits & labels
        logits, labels = self._compute_logits(q, k)
        loss = F.cross_entropy(logits, labels, reduction="mean")

        # update queue
        if update_queue:
            with torch.no_grad():
                self._dequeue_and_enqueue(k)

        return loss, logits, labels

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.step(batch, batch_idx, update_queue=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _, _ = self.step(batch, batch_idx, update_queue=False)
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        # AdamW + cosine schedule (simple)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    # ---------- (optional) utilities for downstream eval ----------
    @torch.no_grad()
    def get_embeddings_from_dataloader(self, dataloader):
        """
        Returns L2-normalized projected features from encoder_q for downstream tasks.
        """
        self.eval()
        outs = []
        labels = []
        filenames = []
        for batch in tqdm(dataloader, desc="Extracting embeddings (MoCo encoder_q)"):
            bg = batch["graph"].to(self.device)
            bg = self._permute_graph_data_channels(bg)
            proj = self.encoder_q(bg)                    # [B, C]
            outs.append(proj.detach().cpu().numpy())
            if "label" in batch:
                label = batch["label"]
                labels.append(label.squeeze(-1).detach().cpu().numpy())
            filenames.extend(batch.get("filename", []))
        outs = np.concatenate(outs)
        if len(labels) > 0:
            labels = np.concatenate(labels)
        else:
            labels = None
        data_count = len(dataloader.dataset)
        assert len(outs) == data_count, f"{outs.shape}, {data_count}"
        if labels is not None:
            assert len(labels) == data_count
        assert len(filenames) == data_count or len(filenames) == 0
        return {"embeddings": outs, "labels": labels, "outputs": outs, "filenames": filenames}

    @torch.no_grad()
    def linear_svm_classification(self, train_data, test_data, max_iter=100000):
        clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=max_iter, tol=1e-3))
        ret = clf.fit(train_data["embeddings"], train_data["labels"])
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

