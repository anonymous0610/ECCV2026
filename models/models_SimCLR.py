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


class NTXentLoss(pl.LightningModule):
    def __init__(self, temperature=0.5, batch_size=256):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)
        mask = mask_correlated_samples(batch_size, self.device)
        self.register_buffer("mask", mask)

    def forward(self, z_i, z_j):                          # z_i=[256, 64] and z_j=[256, 64] are the output of projection layer
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples
        within a minibatch as negative examples.
        """
        batch_size = z_i.shape[0]
        N = 2 * batch_size
        mask = self.mask  # self.mask_correlated_samples(batch_size) # mask=[512, 512] with 1's and 0's

        z = torch.cat((z_i, z_j), dim=0)                 # z=[512, 64]

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature    # z.unsqueeze(0)= [1, 512, 64] and z.unsqueeze(1)=[512, 1, 64] and sim= [512, 512]

        sim_i_j = torch.diag(sim, batch_size)            # sim_i_j = [256]
        sim_j_i = torch.diag(sim, -batch_size)           # sim_j_i = [256]

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)   # positive_samples = [512, 1] 
        negative_samples = sim[mask].reshape(N, -1)                             # negative_samples = [512, 510] with 510 negative samples for each positive sample

        labels = torch.zeros(N, device=positive_samples.device, dtype=torch.long)  # labels = [512]
        logits = torch.cat((positive_samples, negative_samples), dim=1)            # logits = [512, 511] with 1 positive sample and 510 negative samples for each positive sample
        loss = self.criterion(logits, labels)
        loss /= N
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


class Contrastive(pl.LightningModule):
    """
    PyTorch Lightning module to train/test the contrastive learning model.
    """

    def __init__(self, latent_dim=128, out_dim=128, temperature=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.loss_fn = NTXentLoss(temperature=temperature)
        self.model = UVNetContrastiveLearner(latent_dim=latent_dim, out_dim=out_dim)

    def _permute_graph_data_channels(self, graph):
        graph.ndata["x"] = graph.ndata["x"].permute(0, 3, 1, 2)
        graph.edata["x"] = graph.edata["x"].permute(0, 2, 1)
        return graph

    def step(self, batch, batch_idx):  # print(batch.keys())=dict_keys(['graph', 'graph2', 'label', 'filename'])
        graph1, graph2 = batch["graph"], batch["graph2"]  # graph1 and graph2 are DGL graphs with the same structure=dgl.heterograph.DGLHeteroGraph
        graph1 = self._permute_graph_data_channels(graph1) # [:, 10, 10, 7]--> [:, 7, 10, 10] # [:, 10, 7]--> [:, 7, 10] 
        graph2 = self._permute_graph_data_channels(graph2) 
        proj1, _ = self.model(graph1)  # proj1=[256, 64] --> [batch_size, out_dim]
        proj2, _ = self.model(graph2)  # proj2=[256, 64] --> [batch_size, out_dim]
        return self.loss_fn(proj1, proj2)

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
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
            proj, emb = self.model(bg)                    #         -> n_data -->(surf_encoder)--> surf_feat -->
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
