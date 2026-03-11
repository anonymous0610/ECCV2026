from torch.utils.data import Dataset, DataLoader
from torch import FloatTensor
import dgl
from dgl.data.utils import load_graphs
from datasets import util
from tqdm import tqdm
from abc import abstractmethod
import random
import torch


class BaseDataset(Dataset):
    @staticmethod
    @abstractmethod
    def num_classes():
        pass

    def load_graphs(self, file_paths, center_and_scale=True):
        self.data = []
        for fn in tqdm(file_paths, desc="Loading graphs"):
            if not fn.exists():
                continue
            sample = self.load_one_graph(fn)  #sample=Dict={"graph":  , "filename":  }
            if sample is None:
                continue
            if sample["graph"].edata["x"].size(0) == 0:
                # Catch the case of graphs with no edges
                continue
            self.data.append(sample)  #self.data=List[Dict1, Dict2,..]
        if center_and_scale:
            self.center_and_scale()
        self.convert_to_float32()

    def load_one_graph(self, file_path):
        graph = load_graphs(str(file_path))[0][0]  #load_graphs returns a tuple of list of graph #([DGL_Graph]).[0][0]=DGL_Graph
        # graph = dgl.to_simple(graph, return_counts=False)  # remove parallel edges
        sample = {"graph": graph, "filename": file_path.stem}
        return sample

        '''
        print(sample) --> {'graph': Graph(num_nodes=41, 
                                          num_edges=242, 
                                          ndata_schemes={'x': Scheme(shape=(10, 10, 7), dtype=torch.float32)}
                                          edata_schemes={'x': Scheme(shape=(10, 6), dtype=torch.float32)}), 
                           'filename': 'a_Gafata_lower'}

        print(self.data) --> [sample1, sample2, ...] where each sample is a DGL Graph object
        '''

    def center_and_scale_graph(self, graph):  #graph.ndata["x"].shape=[41, 10, 10, 7], graph.edata["x"].shape=[242, 10, 6]
        graph.ndata["x"], center, scale = util.center_and_scale_uvgrid(graph.ndata["x"], return_center_scale=True)
        graph.edata["x"][..., :3] -= center
        graph.edata["x"][..., :3] *= scale
        return graph

    def jitter_graph(self, graph):
        noise = random.uniform(-0.2, 0.2)
        graph.edata["x"][..., :3] += noise
        graph.edata["x"][..., :3] += noise
        return graph

    def center_and_scale(self):
        for i in range(len(self.data)):
            self.data[i]["graph"] = self.center_and_scale_graph(self.data[i]["graph"])

    def convert_to_float32(self):
        for i in range(len(self.data)):
            self.data[i]["graph"].ndata["x"] = self.data[i]["graph"].ndata["x"].type(FloatTensor)
            self.data[i]["graph"].edata["x"] = self.data[i]["graph"].edata["x"].type(FloatTensor)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.random_rotate:
            rotation = util.get_random_rotation()
            sample["graph"].ndata["x"] = util.rotate_uvgrid(sample["graph"].ndata["x"], rotation)
            sample["graph"].edata["x"] = util.rotate_uvgrid(sample["graph"].edata["x"], rotation)
        return sample

    def _collate(self, batch):
        batched_graph = dgl.batch([sample["graph"] for sample in batch])
        batched_filenames = [sample["filename"] for sample in batch]
        return {"graph": batched_graph, "filename": batched_filenames}

    def get_dataloader(self, batch_size=128, shuffle=True, num_workers=0, drop_last=True):
        return DataLoader(self, 
                          batch_size=batch_size, 
                          shuffle=shuffle, 
                          collate_fn=self._collate, 
                          num_workers=num_workers,  # Can be set to non-zero on Linux 
                          drop_last=drop_last)


class BaseContrastiveDataset(BaseDataset):
    def __init__(self, split, prob_full_graph):
        assert split in ("train", "val", "test", "selective")
        assert 0 <= prob_full_graph <= 1
        self.split = split
        self.prob_full_graph = prob_full_graph  # Probability of using the full graph as such as a view

    def apply_transformation(self, graph):
        all_transformations = ("sub_graph", "sub_graph_2hops", "drop_nodes", "drop_edges")
        transformation_type = random.choice(all_transformations)
        if transformation_type == "sub_graph":
            return self.get_subgraph(graph, num_nodes=1, hops=1, normalize=False)
        elif transformation_type == "sub_graph_2hops":
            return self.get_subgraph(graph, num_nodes=1, hops=2, normalize=False)
        elif transformation_type == "drop_nodes":
            graph2 = graph.clone()
            nodes_to_drop = []
            for i in range(graph2.number_of_nodes()):
                if random.random() <= 0.4:
                    nodes_to_drop.append(i)
            graph2.remove_nodes(nodes_to_drop)
            return graph2
        elif transformation_type == "drop_edges":
            graph2 = graph.clone()
            edges_to_drop = []
            for i in range(graph2.number_of_edges()):
                if random.random() <= 0.4:
                    edges_to_drop.append(i)
            graph2.remove_edges(edges_to_drop)
            return graph2

    def _collate(self, batch):
        collated = {"graph": dgl.batch([x["graph"] for x in batch]),
                    "graph2": dgl.batch([x["graph2"] for x in batch]),
                    "label": torch.cat([x["label"] for x in batch], dim=0),
                    "filename": [x["filename"] for x in batch]}
        return collated

    def __getitem__(self, idx):                     # idx=4864 (random)

        # Get the graph and filename from the data list at the given index
        graph_and_filename = self.data[idx]
        filename = graph_and_filename["filename"]    # filename= 'g_Playbill_upper'
        graph = graph_and_filename["graph"]
        
        # Apply transformations to the graph to create two views which is graph and graph2 for contrastive learning
        graph2 = self.apply_transformation(graph.clone())
        if self.split == "train" and random.uniform(0, 1) > self.prob_full_graph:
            graph = self.apply_transformation(graph.clone())
        else:
            graph = graph.clone()
        
        # Remove the _ID fields from the graph data and edge data of graph and graph2
        graph.ndata.pop("_ID", None)
        graph.edata.pop("_ID", None)
        graph2.ndata.pop("_ID", None)
        graph2.edata.pop("_ID", None)

        # Add noise to the graph and graph2
        graph = self.jitter_graph(graph)
        graph2 = self.jitter_graph(graph2)

        return {"graph": graph, 
                "graph2": graph2, 
                "label": self.labels[idx],
                "filename": filename}

    def get_dataloader(self, batch_size=128, shuffle=True, num_workers=0, drop_last=True):
        return DataLoader(self, 
                          batch_size=batch_size, 
                          shuffle=shuffle, 
                          collate_fn=self._collate, 
                          num_workers=num_workers, 
                          drop_last=drop_last)

    def get_subgraph(self, graph, num_nodes=1, hops=2, normalize=False):              # hops=1/2
        assert num_nodes >= 1
        assert hops >= 0
        # node_idx = random.sample(graph.nodes().cpu().numpy().tolist(), 1)
        subgraph_nodes = [random.sample(graph.nodes().cpu().numpy().tolist(), 1)[0]]  # random.sample() selects one node from the graph and output in the form of list
        for _ in range(hops):
            neighbors = []
            for idx in subgraph_nodes:
                # out_edges returns a tuple of src and dst nodes
                neighbors.extend(graph.out_edges(idx)[1].cpu().numpy().tolist())      # List all outgoing edges of that node in tuple format
            for ngh in neighbors:
                subgraph_nodes.append(ngh)                                            # Add all the neighbors of the selected node to the subgraph_nodes list for the given hops
        subgraph_nodes = list(set(subgraph_nodes))                                    # Remove duplicates from the list of nodes
        subgraph = graph.subgraph(subgraph_nodes)
        # subgraph.copy_from_parent()
        if normalize:
            subgraph = self.center_and_scale_graph(subgraph)
        return subgraph
    

# for contrastive training on fusiongallery

class BaseContrastiveDatasetFusionGallery(BaseDataset):
    def __init__(self, split, prob_full_graph):
        assert split in ("train", "val", "test")
        assert 0 <= prob_full_graph <= 1
        self.split = split
        self.prob_full_graph = prob_full_graph  # Probability of using the full graph as such as a view

    def apply_transformation(self, graph):
        all_transformations = ("sub_graph", "sub_graph_2hops", "drop_nodes", "drop_edges")
        transformation_type = random.choice(all_transformations)
        if transformation_type == "sub_graph":
            return self.get_subgraph(graph, num_nodes=1, hops=1, normalize=False)
        elif transformation_type == "sub_graph_2hops":
            return self.get_subgraph(graph, num_nodes=1, hops=2, normalize=False)
        elif transformation_type == "drop_nodes":
            graph2 = graph.clone()
            nodes_to_drop = []
            for i in range(graph2.number_of_nodes()):
                if random.random() <= 0.4:
                    nodes_to_drop.append(i)
            graph2.remove_nodes(nodes_to_drop)
            return graph2
        elif transformation_type == "drop_edges":
            graph2 = graph.clone()
            edges_to_drop = []
            for i in range(graph2.number_of_edges()):
                if random.random() <= 0.4:
                    edges_to_drop.append(i)
            graph2.remove_edges(edges_to_drop)
            return graph2

    def _collate(self, batch):
        collated = {"graph": dgl.batch([x["graph"] for x in batch]),
                    "graph2": dgl.batch([x["graph2"] for x in batch]),
                    "filename": [x["filename"] for x in batch]}
        return collated

    def __getitem__(self, idx):
        graph_and_filename = self.data[idx]
        filename = graph_and_filename["filename"]
        graph = graph_and_filename["graph"]
        graph2 = self.apply_transformation(graph.clone())
        if self.split == "train" and random.uniform(0, 1) > self.prob_full_graph:  # 10% of times, they use graph1 as it is without transformation. 
            graph = self.apply_transformation(graph.clone())
        else:
            graph = graph.clone()
        graph.ndata.pop("_ID", None)
        graph.edata.pop("_ID", None)
        graph2.ndata.pop("_ID", None)
        graph2.edata.pop("_ID", None)
        graph = self.jitter_graph(graph) # adds noise to the graph
        graph2 = self.jitter_graph(graph2)
        return {"graph": graph, "graph2": graph2, "filename": filename} # for fusiongallery

    def get_dataloader(self, batch_size=128, shuffle=True, num_workers=0, drop_last=True):
        return DataLoader(self, 
                          batch_size=batch_size, 
                          shuffle=shuffle, 
                          collate_fn=self._collate, 
                          num_workers=num_workers, 
                          drop_last=drop_last)

    def get_subgraph(self, graph, num_nodes=1, hops=2, normalize=False):
        assert num_nodes >= 1
        assert hops >= 0
        # node_idx = random.sample(graph.nodes().cpu().numpy().tolist(), 1)
        subgraph_nodes = [random.sample(graph.nodes().cpu().numpy().tolist(), 1)[0]]
        for _ in range(hops):
            neighbors = []
            for idx in subgraph_nodes:
                # out_edges returns a tuple of src and dst nodes
                neighbors.extend(graph.out_edges(idx)[1].cpu().numpy().tolist())
            for ngh in neighbors:
                subgraph_nodes.append(ngh)
        subgraph_nodes = list(set(subgraph_nodes))
        subgraph = graph.subgraph(subgraph_nodes)
        # subgraph.copy_from_parent()
        if normalize:
            subgraph = self.center_and_scale_graph(subgraph)
        return subgraph