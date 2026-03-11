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
# # Classification model
# ###############################################################################
# class _NonLinearClassifier(nn.Module):
#     def __init__(self, input_dim, num_classes, dropout=0.3):
#         """
#         A 3-layer MLP with linear outputs

#         Args:
#             input_dim (int): Dimension of the input tensor 
#             num_classes (int): Dimension of the output logits
#             dropout (float, optional): Dropout used after each linear layer. Defaults to 0.3.
#         """
#         super().__init__()
#         self.linear1 = nn.Linear(input_dim, 512, bias=False)
#         self.bn1 = nn.BatchNorm1d(512)
#         self.dp1 = nn.Dropout(p=dropout)
#         self.linear2 = nn.Linear(512, 256, bias=False)
#         self.bn2 = nn.BatchNorm1d(256)
#         self.dp2 = nn.Dropout(p=dropout)
#         self.linear3 = nn.Linear(256, num_classes)

#         for m in self.modules():
#             self.weights_init(m)

#     def weights_init(self, m):
#         if isinstance(m, nn.Linear):
#             torch.nn.init.kaiming_uniform_(m.weight.data)
#             if m.bias is not None:
#                 m.bias.data.fill_(0.0)

#     def forward(self, inp):
#         """
#         Forward pass

#         Args:
#             inp (torch.tensor): Inputs features to be mapped to logits
#                                 (batch_size x input_dim)

#         Returns:
#             torch.tensor: Logits (batch_size x num_classes)
#         """
#         x = F.relu(self.bn1(self.linear1(inp)))
#         x = self.dp1(x)
#         x = F.relu(self.bn2(self.linear2(x)))
#         x = self.dp2(x)
#         x = self.linear3(x)
#         return x


# class UVNetClassifier(nn.Module):
#     """
#     UV-Net solid classification model
#     """

#     def __init__(self,
#                  num_classes,
#                  crv_emb_dim=64,
#                  srf_emb_dim=64,
#                  graph_emb_dim=128,
#                  dropout=0.3):
#         """
#         Initialize the UV-Net solid classification model
        
#         Args:
#             num_classes (int): Number of classes to output
#             crv_emb_dim (int, optional): Embedding dimension for the 1D edge UV-grids. Defaults to 64.
#             srf_emb_dim (int, optional): Embedding dimension for the 2D face UV-grids. Defaults to 64.
#             graph_emb_dim (int, optional): Embedding dimension for the graph. Defaults to 128.
#             dropout (float, optional): Dropout for the final non-linear classifier. Defaults to 0.3.
#         """
#         super().__init__()
#         self.curv_encoder = uvnet.encoders.UVNetCurveEncoder(in_channels=6, output_dims=crv_emb_dim)
#         self.surf_encoder = uvnet.encoders.UVNetSurfaceEncoder(in_channels=7, output_dims=srf_emb_dim)
#         self.graph_encoder = uvnet.encoders.UVNetGraphEncoder(srf_emb_dim, crv_emb_dim, graph_emb_dim)
#         self.clf = _NonLinearClassifier(graph_emb_dim, num_classes, dropout)

#     def forward(self, batched_graph):
#         """
#         Forward pass

#         Args:
#             batched_graph (dgl.Graph): A batched DGL graph containing the face 2D UV-grids in node features
#                                        (ndata['x']) and 1D edge UV-grids in the edge features (edata['x']).

#         Returns:
#             torch.tensor: Logits (batch_size x num_classes)
#         """
#         # Input features
#         input_crv_feat = batched_graph.edata["x"]
#         input_srf_feat = batched_graph.ndata["x"]
#         # Compute hidden edge and face features
#         hidden_crv_feat = self.curv_encoder(input_crv_feat)
#         hidden_srf_feat = self.surf_encoder(input_srf_feat)
#         # Message pass and compute per-face(node) and global embeddings
#         # Per-face embeddings are ignored during solid classification
#         _, graph_emb = self.graph_encoder(batched_graph, hidden_srf_feat, hidden_crv_feat)
#         # Map to logits
#         out = self.clf(graph_emb)
#         return out


# class Classification(pl.LightningModule):
#     """
#     PyTorch Lightning module to train/test the classifier.
#     """

#     def __init__(self, num_classes):
#         """
#         Args:
#             num_classes (int): Number of per-solid classes in the dataset
#         """
#         super().__init__()
#         self.save_hyperparameters()
#         self.model = UVNetClassifier(num_classes=num_classes)
#         self.train_acc = torchmetrics.Accuracy()
#         self.val_acc = torchmetrics.Accuracy()
#         self.test_acc = torchmetrics.Accuracy()

#     def forward(self, batched_graph):
#         logits = self.model(batched_graph)
#         return logits

#     def training_step(self, batch, batch_idx):
#         inputs = batch["graph"].to(self.device)
#         labels = batch["label"].to(self.device)
#         inputs.ndata["x"] = inputs.ndata["x"].permute(0, 3, 1, 2)
#         inputs.edata["x"] = inputs.edata["x"].permute(0, 2, 1)
#         logits = self.model(inputs)
#         loss = F.cross_entropy(logits, labels, reduction="mean")
#         self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
#         preds = F.softmax(logits, dim=-1)
#         self.log("train_acc", self.train_acc(preds, labels), on_step=False, on_epoch=True, sync_dist=True)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         inputs = batch["graph"].to(self.device)
#         labels = batch["label"].to(self.device)
#         inputs.ndata["x"] = inputs.ndata["x"].permute(0, 3, 1, 2)
#         inputs.edata["x"] = inputs.edata["x"].permute(0, 2, 1)
#         logits = self.model(inputs)
#         loss = F.cross_entropy(logits, labels, reduction="mean")
#         self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
#         preds = F.softmax(logits, dim=-1)
#         self.log("val_acc", self.val_acc(preds, labels), on_step=False, on_epoch=True, sync_dist=True)
#         return loss

#     def test_step(self, batch, batch_idx):
#         inputs = batch["graph"].to(self.device)
#         labels = batch["label"].to(self.device)
#         inputs.ndata["x"] = inputs.ndata["x"].permute(0, 3, 1, 2)
#         inputs.edata["x"] = inputs.edata["x"].permute(0, 2, 1)
#         logits = self.model(inputs)
#         loss = F.cross_entropy(logits, labels, reduction="mean")
#         self.log("test_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
#         preds = F.softmax(logits, dim=-1)
#         self.log("test_acc", self.test_acc(preds, labels), on_step=False, on_epoch=True, sync_dist=True)

#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters())
#         return optimizer








# ###############################################################################
# # Segmentation model
# ###############################################################################
# class UVNetSegmenter(nn.Module):
#     """
#     UV-Net solid face segmentation model
#     """

#     def __init__(self,
#                  num_classes,
#                  crv_in_channels=6,
#                  crv_emb_dim=64,
#                  srf_emb_dim=64,
#                  graph_emb_dim=128,
#                  dropout=0.3):
#         """
#         Initialize the UV-Net solid face segmentation model

#         Args:
#             num_classes (int): Number of classes to output per-face
#             crv_in_channels (int, optional): Number of input channels for the 1D edge UV-grids
#             crv_emb_dim (int, optional): Embedding dimension for the 1D edge UV-grids. Defaults to 64.
#             srf_emb_dim (int, optional): Embedding dimension for the 2D face UV-grids. Defaults to 64.
#             graph_emb_dim (int, optional): Embedding dimension for the graph. Defaults to 128.
#             dropout (float, optional): Dropout for the final non-linear classifier. Defaults to 0.3.
#         """
#         super().__init__()
#         # A 1D convolutional network to encode B-rep edge geometry represented as 1D UV-grids
#         self.curv_encoder = uvnet.encoders.UVNetCurveEncoder(in_channels=crv_in_channels, output_dims=crv_emb_dim)
#         # A 2D convolutional network to encode B-rep face geometry represented as 2D UV-grids
#         self.surf_encoder = uvnet.encoders.UVNetSurfaceEncoder(in_channels=7, output_dims=srf_emb_dim)
#         # A graph neural network that message passes face and edge features
#         self.graph_encoder = uvnet.encoders.UVNetGraphEncoder(srf_emb_dim, crv_emb_dim, graph_emb_dim)
#         # A non-linear classifier that maps face embeddings to face logits
#         self.seg = _NonLinearClassifier(graph_emb_dim + srf_emb_dim, num_classes, dropout=dropout)

#     def forward(self, batched_graph):
#         """
#         Forward pass

#         Args:
#             batched_graph (dgl.Graph): A batched DGL graph containing the face 2D UV-grids in node features
#                                        (ndata['x']) and 1D edge UV-grids in the edge features (edata['x']).

#         Returns:
#             torch.tensor: Logits (total_nodes_in_batch x num_classes)
#         """
#         # Input features
#         input_crv_feat = batched_graph.edata["x"]
#         input_srf_feat = batched_graph.ndata["x"]
#         # Compute hidden edge and face features
#         hidden_crv_feat = self.curv_encoder(input_crv_feat)
#         hidden_srf_feat = self.surf_encoder(input_srf_feat)
#         # Message pass and compute per-face(node) and global embeddings
#         node_emb, graph_emb = self.graph_encoder(batched_graph, hidden_srf_feat, hidden_crv_feat)
#         # Repeat the global graph embedding so that it can be
#         # concatenated to the per-node embeddings
#         num_nodes_per_graph = batched_graph.batch_num_nodes().to(graph_emb.device)
#         graph_emb = graph_emb.repeat_interleave(num_nodes_per_graph, dim=0).to(graph_emb.device)
#         local_global_feat = torch.cat((node_emb, graph_emb), dim=1)
#         # Map to logits
#         out = self.seg(local_global_feat)
#         return out


# class Segmentation(pl.LightningModule):
#     """
#     PyTorch Lightning module to train/test the segmenter (per-face classifier).
#     """

#     def __init__(self, num_classes, crv_in_channels=6):
#         """
#         Args:
#             num_classes (int): Number of per-face classes in the dataset
#         """
#         super().__init__()
#         self.save_hyperparameters()
#         self.model = UVNetSegmenter(num_classes, crv_in_channels=crv_in_channels)
#         # Setting compute_on_step = False to compute "part IoU"
#         # This is because we want to compute the IoU on the entire dataset
#         # at the end to account for rare classes, rather than within each batch
#         self.train_iou = torchmetrics.IoU(num_classes=num_classes, compute_on_step=False)
#         self.val_iou = torchmetrics.IoU(num_classes=num_classes, compute_on_step=False)
#         self.test_iou = torchmetrics.IoU(num_classes=num_classes, compute_on_step=False)

#         self.train_accuracy = torchmetrics.Accuracy(num_classes=num_classes, compute_on_step=False)
#         self.val_accuracy = torchmetrics.Accuracy(num_classes=num_classes, compute_on_step=False)
#         self.test_accuracy = torchmetrics.Accuracy(num_classes=num_classes, compute_on_step=False)

#     def forward(self, batched_graph):
#         logits = self.model(batched_graph)
#         return logits

#     def training_step(self, batch, batch_idx):
#         inputs = batch["graph"].to(self.device)
#         inputs.ndata["x"] = inputs.ndata["x"].permute(0, 3, 1, 2)
#         inputs.edata["x"] = inputs.edata["x"].permute(0, 2, 1)
#         labels = inputs.ndata["y"]
#         logits = self.model(inputs)
#         loss = F.cross_entropy(logits, labels, reduction="mean")
#         self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
#         preds = F.softmax(logits, dim=-1)
#         self.train_iou(preds, labels)
#         self.train_accuracy(preds, labels)
#         return loss

#     def training_epoch_end(self, outs):
#         self.log("train_iou", self.train_iou.compute())
#         self.log("train_accuracy", self.train_accuracy.compute())

#     def validation_step(self, batch, batch_idx):
#         inputs = batch["graph"].to(self.device)
#         inputs.ndata["x"] = inputs.ndata["x"].permute(0, 3, 1, 2)
#         inputs.edata["x"] = inputs.edata["x"].permute(0, 2, 1)
#         labels = inputs.ndata["y"]
#         logits = self.model(inputs)
#         loss = F.cross_entropy(logits, labels, reduction="mean")
#         self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
#         preds = F.softmax(logits, dim=-1)
#         self.val_iou(preds, labels)
#         self.val_accuracy(preds, labels)
#         return loss

#     def validation_epoch_end(self, outs):
#         self.log("val_iou", self.val_iou.compute())
#         self.log("val_accuracy", self.val_accuracy.compute())

#     def test_step(self, batch, batch_idx):
#         inputs = batch["graph"].to(self.device)
#         inputs.ndata["x"] = inputs.ndata["x"].permute(0, 3, 1, 2)
#         inputs.edata["x"] = inputs.edata["x"].permute(0, 2, 1)
#         labels = inputs.ndata["y"]
#         logits = self.model(inputs)
#         loss = F.cross_entropy(logits, labels, reduction="mean")
#         self.log("test_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
#         preds = F.softmax(logits, dim=-1)
#         self.test_iou(preds, labels)
#         self.test_accuracy(preds, labels)

#     def test_epoch_end(self, outs):
#         self.log("test_iou", self.test_iou.compute())
#         self.log("test_accuracy", self.test_accuracy.compute())

#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters())
#         return optimizer








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


# class NTXentLoss(pl.LightningModule):
#     def __init__(self, temperature=0.5, batch_size=256):
#         super().__init__()
#         self.temperature = temperature
#         self.criterion = nn.CrossEntropyLoss(reduction="sum")
#         self.similarity_f = nn.CosineSimilarity(dim=2)
#         mask = mask_correlated_samples(batch_size, self.device)
#         self.register_buffer("mask", mask)

#     def forward(self, z_i, z_j):                          # z_i=[256, 64] and z_j=[256, 64] are the output of projection layer
#         """
#         We do not sample negative examples explicitly.
#         Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples
#         within a minibatch as negative examples.
#         """
#         batch_size = z_i.shape[0]
#         N = 2 * batch_size
#         mask = self.mask  # self.mask_correlated_samples(batch_size) # mask=[512, 512] with 1's and 0's

#         z = torch.cat((z_i, z_j), dim=0)                 # z=[512, 64]

#         sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature    # z.unsqueeze(0)= [1, 512, 64] and z.unsqueeze(1)=[512, 1, 64] and sim= [512, 512]

#         sim_i_j = torch.diag(sim, batch_size)            # sim_i_j = [256]
#         sim_j_i = torch.diag(sim, -batch_size)           # sim_j_i = [256]

#         positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)   # positive_samples = [512, 1] 
#         negative_samples = sim[mask].reshape(N, -1)                             # negative_samples = [512, 510] with 510 negative samples for each positive sample

#         labels = torch.zeros(N, device=positive_samples.device, dtype=torch.long)  # labels = [512]
#         logits = torch.cat((positive_samples, negative_samples), dim=1)            # logits = [512, 511] with 1 positive sample and 510 negative samples for each positive sample
#         loss = self.criterion(logits, labels)
#         loss /= N
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


# class Contrastive1(pl.LightningModule):
#     """
#     PyTorch Lightning module to train/test the contrastive learning model.
#     """

#     def __init__(self, latent_dim=128, out_dim=128, temperature=0.1):
#         super().__init__()
#         self.save_hyperparameters()
#         self.loss_fn = NTXentLoss(temperature=temperature)
#         self.model = UVNetContrastiveLearner(latent_dim=latent_dim, out_dim=out_dim)

#     def _permute_graph_data_channels(self, graph):
#         graph.ndata["x"] = graph.ndata["x"].permute(0, 3, 1, 2)
#         graph.edata["x"] = graph.edata["x"].permute(0, 2, 1)
#         return graph

#     def step(self, batch, batch_idx):  # print(batch.keys())=dict_keys(['graph', 'graph2', 'label', 'filename'])
#         graph1, graph2 = batch["graph"], batch["graph2"]  # graph1 and graph2 are DGL graphs with the same structure=dgl.heterograph.DGLHeteroGraph
#         graph1 = self._permute_graph_data_channels(graph1) # [:, 10, 10, 7]--> [:, 7, 10, 10] # [:, 10, 7]--> [:, 7, 10] 
#         graph2 = self._permute_graph_data_channels(graph2) 
#         proj1, _ = self.model(graph1)  # proj1=[256, 64] --> [batch_size, out_dim]
#         proj2, _ = self.model(graph2)  # proj2=[256, 64] --> [batch_size, out_dim]
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
#             proj, emb = self.model(bg)                    #         -> n_data -->(surf_encoder)--> surf_feat -->
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
# Classification model
###############################################################################
class _NonLinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.3):
        """
        A 3-layer MLP with linear outputs

        Args:
            input_dim (int): Dimension of the input tensor 
            num_classes (int): Dimension of the output logits
            dropout (float, optional): Dropout used after each linear layer. Defaults to 0.3.
        """
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
        """
        Forward pass

        Args:
            inp (torch.tensor): Inputs features to be mapped to logits
                                (batch_size x input_dim)

        Returns:
            torch.tensor: Logits (batch_size x num_classes)
        """
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
        """
        Initialize the UV-Net solid classification model
        
        Args:
            num_classes (int): Number of classes to output
            crv_emb_dim (int, optional): Embedding dimension for the 1D edge UV-grids. Defaults to 64.
            srf_emb_dim (int, optional): Embedding dimension for the 2D face UV-grids. Defaults to 64.
            graph_emb_dim (int, optional): Embedding dimension for the graph. Defaults to 128.
            dropout (float, optional): Dropout for the final non-linear classifier. Defaults to 0.3.
        """
        super().__init__()
        self.curv_encoder = uvnet.encoders.UVNetCurveEncoder(in_channels=6, output_dims=crv_emb_dim)
        self.surf_encoder = uvnet.encoders.UVNetSurfaceEncoder(in_channels=7, output_dims=srf_emb_dim)
        self.graph_encoder = uvnet.encoders.UVNetGraphEncoder(srf_emb_dim, crv_emb_dim, graph_emb_dim)
        self.clf = _NonLinearClassifier(graph_emb_dim, num_classes, dropout)

    def forward(self, batched_graph):
        """
        Forward pass

        Args:
            batched_graph (dgl.Graph): A batched DGL graph containing the face 2D UV-grids in node features
                                       (ndata['x']) and 1D edge UV-grids in the edge features (edata['x']).

        Returns:
            torch.tensor: Logits (batch_size x num_classes)
        """
        # Input features
        input_crv_feat = batched_graph.edata["x"]
        input_srf_feat = batched_graph.ndata["x"]
        # Compute hidden edge and face features
        hidden_crv_feat = self.curv_encoder(input_crv_feat)
        hidden_srf_feat = self.surf_encoder(input_srf_feat)
        # Message pass and compute per-face(node) and global embeddings
        # Per-face embeddings are ignored during solid classification
        _, graph_emb = self.graph_encoder(batched_graph, hidden_srf_feat, hidden_crv_feat)
        # Map to logits
        out = self.clf(graph_emb)
        return out


class Classification(pl.LightningModule):
    """
    PyTorch Lightning module to train/test the classifier.
    """

    def __init__(self, num_classes):
        """
        Args:
            num_classes (int): Number of per-solid classes in the dataset
        """
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
        """
        Initialize the UV-Net solid face segmentation model

        Args:
            num_classes (int): Number of classes to output per-face
            crv_in_channels (int, optional): Number of input channels for the 1D edge UV-grids
            crv_emb_dim (int, optional): Embedding dimension for the 1D edge UV-grids. Defaults to 64.
            srf_emb_dim (int, optional): Embedding dimension for the 2D face UV-grids. Defaults to 64.
            graph_emb_dim (int, optional): Embedding dimension for the graph. Defaults to 128.
            dropout (float, optional): Dropout for the final non-linear classifier. Defaults to 0.3.
        """
        super().__init__()
        # A 1D convolutional network to encode B-rep edge geometry represented as 1D UV-grids
        self.curv_encoder = uvnet.encoders.UVNetCurveEncoder(in_channels=crv_in_channels, output_dims=crv_emb_dim)
        # A 2D convolutional network to encode B-rep face geometry represented as 2D UV-grids
        self.surf_encoder = uvnet.encoders.UVNetSurfaceEncoder(in_channels=7, output_dims=srf_emb_dim)
        # A graph neural network that message passes face and edge features
        self.graph_encoder = uvnet.encoders.UVNetGraphEncoder(srf_emb_dim, crv_emb_dim, graph_emb_dim)
        # A non-linear classifier that maps face embeddings to face logits
        self.seg = _NonLinearClassifier(graph_emb_dim + srf_emb_dim, num_classes, dropout=dropout)

    def forward(self, batched_graph):
        """
        Forward pass

        Args:
            batched_graph (dgl.Graph): A batched DGL graph containing the face 2D UV-grids in node features
                                       (ndata['x']) and 1D edge UV-grids in the edge features (edata['x']).

        Returns:
            torch.tensor: Logits (total_nodes_in_batch x num_classes)
        """
        # Input features
        input_crv_feat = batched_graph.edata["x"]
        input_srf_feat = batched_graph.ndata["x"]
        # Compute hidden edge and face features
        hidden_crv_feat = self.curv_encoder(input_crv_feat)
        hidden_srf_feat = self.surf_encoder(input_srf_feat)
        # Message pass and compute per-face(node) and global embeddings
        node_emb, graph_emb = self.graph_encoder(batched_graph, hidden_srf_feat, hidden_crv_feat)
        # Repeat the global graph embedding so that it can be
        # concatenated to the per-node embeddings
        num_nodes_per_graph = batched_graph.batch_num_nodes().to(graph_emb.device)
        graph_emb = graph_emb.repeat_interleave(num_nodes_per_graph, dim=0).to(graph_emb.device)
        local_global_feat = torch.cat((node_emb, graph_emb), dim=1)
        # Map to logits
        out = self.seg(local_global_feat)
        return out


class Segmentation(pl.LightningModule):
    """
    PyTorch Lightning module to train/test the segmenter (per-face classifier).
    """

    def __init__(self, num_classes, crv_in_channels=6):
        """
        Args:
            num_classes (int): Number of per-face classes in the dataset
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = UVNetSegmenter(num_classes, crv_in_channels=crv_in_channels)
        # Setting compute_on_step = False to compute "part IoU"
        # This is because we want to compute the IoU on the entire dataset
        # at the end to account for rare classes, rather than within each batch
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
class Contrastive1(pl.LightningModule):
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

