import torch
from torch import nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import NNConv
from dgl.nn.pytorch.glob import MaxPooling


############-----------Convolutional Layers-----------############
def _conv1d(in_channels, out_channels, kernel_size=3, padding=0, bias=False):
    """
    Helper function to create a 1D convolutional layer with batchnorm and LeakyReLU activation

    Args:
        in_channels (int): Input channels
        out_channels (int): Output channels
        kernel_size (int, optional): Size of the convolutional kernel. Defaults to 3.
        padding (int, optional): Padding size on each side. Defaults to 0.
        bias (bool, optional): Whether bias is used. Defaults to False.

    Returns:
        nn.Sequential: Sequential contained the Conv1d, BatchNorm1d and LeakyReLU layers
    """
    return nn.Sequential(nn.Conv1d(in_channels, 
                                   out_channels, 
                                   kernel_size=kernel_size, 
                                   padding=padding, 
                                   bias=bias),
                        nn.BatchNorm1d(out_channels),
                        nn.LeakyReLU())


#2D Convolution
def _conv2d(in_channels, out_channels, kernel_size, padding=0, bias=False):
    """
    Helper function to create a 2D convolutional layer with batchnorm and LeakyReLU activation

    Args:
        in_channels (int): Input channels
        out_channels (int): Output channels
        kernel_size (int, optional): Size of the convolutional kernel. Defaults to 3.
        padding (int, optional): Padding size on each side. Defaults to 0.
        bias (bool, optional): Whether bias is used. Defaults to False.

    Returns:
        nn.Sequential: Sequential contained the Conv2d, BatchNorm2d and LeakyReLU layers
    """
    return nn.Sequential(nn.Conv2d(in_channels,
                                   out_channels,
                                   kernel_size=kernel_size,
                                   padding=padding,
                                   bias=bias),
                        nn.BatchNorm2d(out_channels),
                        nn.LeakyReLU())


def _fc(in_features, out_features, bias=False):
    return nn.Sequential(nn.Linear(in_features, out_features, bias=bias),
                         nn.BatchNorm1d(out_features),
                         nn.LeakyReLU())

############-----------Convolutional Layers-----------############




#################----------------MLP---------------################
class _MLP(nn.Module):
    def __init__(self, 
                 num_layers, 
                 input_dim, 
                 hidden_dim, 
                 output_dim):
        """
        MLP with linear output
        Args:
            num_layers (int): The number of linear layers in the MLP
            input_dim (int): Input feature dimension
            hidden_dim (int): Hidden feature dimensions for all hidden layers
            output_dim (int): Output feature dimension

        Raises:
            ValueError: If the given number of layers is <1
        """
        super(_MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("Number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            # TODO: this could move inside the above loop
            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):                                           # x=[8430, 64]
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):                     # i=0
                h = F.relu(self.batch_norms[i](self.linears[i](h)))  # [8430, 64]-->[8430, 64]
            return self.linears[-1](h)                               # [8430, 64]-->[8430, 64]

#################----------------MLP---------------################




##############-------------Curve_Encoder-------------##############
class UVNetCurveEncoder(nn.Module):
    def __init__(self, in_channels=6, output_dims=64):
        """
        This is the 1D convolutional network that extracts features from the B-rep edge
        geometry described as 1D UV-grids (see Section 3.2, Curve & surface convolution
        in paper)

        Args:
            in_channels (int, optional): Number of channels in the edge UV-grids. By default
                                         we expect 3 channels for point coordinates and 3 for
                                         curve tangents. Defaults to 6.
            output_dims (int, optional): Output curve embedding dimension. Defaults to 64.
        """
        super(UVNetCurveEncoder, self).__init__()
        self.in_channels = in_channels
        self.conv1 = _conv1d(in_channels, 64, kernel_size=3, padding=1, bias=False)
        self.conv2 = _conv1d(64, 128, kernel_size=3, padding=1, bias=False)
        self.conv3 = _conv1d(128, 256, kernel_size=3, padding=1, bias=False)
        self.final_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = _fc(256, output_dims, bias=False)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):                       # x=[48700, 3, 10]
        assert x.size(1) == self.in_channels
        batch_size = x.size(0)
        x = self.conv1(x)                       # [48700, 3, 10]--> [48700, 64, 10]
        x = self.conv2(x)                       # [48700, 64, 10]--> [48700, 128, 10]
        x = self.conv3(x)                       # [48700, 128, 10]--> [48700, 256, 10]
        x = self.final_pool(x)                  # [48700, 256, 10]--> [48700, 256, 1]
        x = x.view(batch_size, -1)              # [48700, 256, 1]--> [48700, 256]
        x = self.fc(x)                          # [48700, 256]--> [48700, 64]
        return x

##############-------------Curve_Encoder-------------##############



##############------------Surface_Encoder------------##############
class UVNetSurfaceEncoder(nn.Module):
    def __init__(self,
                 in_channels=7,
                 output_dims=64):
        """
        This is the 2D convolutional network that extracts features from the B-rep face
        geometry described as 2D UV-grids (see Section 3.2, Curve & surface convolution
        in paper)

        Args:
            in_channels (int, optional): Number of channels in the edge UV-grids. By default
                                         we expect 3 channels for point coordinates and 3 for
                                         surface normals and 1 for the trimming mask. Defaults
                                         to 7.
            output_dims (int, optional): Output surface embedding dimension. Defaults to 64.
        """
        super(UVNetSurfaceEncoder, self).__init__()
        self.in_channels = in_channels
        self.conv1 = _conv2d(in_channels, 64, 3, padding=1, bias=False)
        self.conv2 = _conv2d(64, 128, 3, padding=1, bias=False)
        self.conv3 = _conv2d(128, 256, 3, padding=1, bias=False)
        self.final_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = _fc(256, output_dims, bias=False)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):                      # [8430, 4, 10, 10]
        assert x.size(1) == self.in_channels
        batch_size = x.size(0)
        x = self.conv1(x)                      # [8430, 4, 10, 10] --> [8430, 64, 10, 10]
        x = self.conv2(x)                      # [8430, 64, 10, 10] --> [8430, 128, 10, 10]
        x = self.conv3(x)                      # [8430, 128, 10, 10] --> [8430, 256, 10, 10]
        x = self.final_pool(x)                 # [8430, 256, 10, 10] --> [8430, 256, 1, 1]
        x = x.view(batch_size, -1)             # [8430, 256, 1, 1] --> [8430, 256]
        x = self.fc(x)                         # [8430, 256] --> [8430, 64]
        return x

##############------------Surface_Encoder------------##############





###########----------Edge_Conv_in_Graph_Encoder----------###########
class _EdgeConv(nn.Module):
    def __init__(self,
                 edge_feats,
                 out_feats,
                 node_feats,
                 num_mlp_layers=2,
                 hidden_mlp_dim=64):
        """
        This module implements Eq. 2 from the paper where the edge features are
        updated using the node features at the endpoints.

        Args:
            edge_feats (int): Input edge feature dimension
            out_feats (int): Output feature deimension
            node_feats (int): Input node feature dimension
            num_mlp_layers (int, optional): Number of layers used in the MLP. Defaults to 2.
            hidden_mlp_dim (int, optional): Hidden feature dimension in the MLP. Defaults to 64.
        """
        super(_EdgeConv, self).__init__()
        self.proj = _MLP(1, node_feats, hidden_mlp_dim, edge_feats)
        self.mlp = _MLP(num_mlp_layers, edge_feats, hidden_mlp_dim, out_feats)
        self.batchnorm = nn.BatchNorm1d(out_feats)
        self.eps = torch.nn.Parameter(torch.FloatTensor([0.0]))

    def forward(self, graph, nfeat, efeat):
        src, dst = graph.edges()                                     # graph.edges()=tuple=(sorce_twnsor, destination_tensor)=(torch.Size([48700]), torch.Size([48700]))
        proj1, proj2 = self.proj(nfeat[src]), self.proj(nfeat[dst])  # [48700, 64]-->[48700, 64]
        agg = proj1 + proj2                                          # agg=[48700, 64]
        h = self.mlp((1 + self.eps) * efeat + agg)                   # self.eps is a learnable parameter = torch.Size([1]) = [0.0]
        h = F.leaky_relu(self.batchnorm(h))
        return h
###########----------Edge_Conv_in_Graph_Encoder----------###########




###########-----------Node_Conv_in_Graph_Encoder-----------###########
class _NodeConv(nn.Module):
    def __init__(self,
                 node_feats,
                 out_feats,
                 edge_feats,
                 num_mlp_layers=2,
                 hidden_mlp_dim=64):
        """
        This module implements Eq. 1 from the paper where the node features are
        updated using the neighboring node and edge features.

        Args:
            node_feats (int): Input edge feature dimension
            out_feats (int): Output feature deimension
            node_feats (int): Input node feature dimension
            num_mlp_layers (int, optional): Number of layers used in the MLP. Defaults to 2.
            hidden_mlp_dim (int, optional): Hidden feature dimension in the MLP. Defaults to 64.
        """
        super(_NodeConv, self).__init__()
        self.gconv = NNConv(in_feats=node_feats,
                            out_feats=out_feats,
                            edge_func=nn.Linear(edge_feats, node_feats * out_feats),
                            aggregator_type="sum",
                            bias=False)
        self.batchnorm = nn.BatchNorm1d(out_feats)
        self.mlp = _MLP(num_mlp_layers, node_feats, hidden_mlp_dim, out_feats)
        self.eps = torch.nn.Parameter(torch.FloatTensor([0.0]))

    def forward(self, graph, nfeat, efeat):
        h = (1 + self.eps) * nfeat          # self.eps is a learnable parameter = torch.Size([1]) = [0.0]
        h = self.gconv(graph, h, efeat)     # [8430, 64]--> [8430, 64]
        h = self.mlp(h)
        h = F.leaky_relu(self.batchnorm(h)) # [8430, 64]-->[8430, 64]
        return h

###########-----------Node_Conv_in_Graph_Encoder-----------###########




###############-------------Graph_Encoder-------------###############
class UVNetGraphEncoder(nn.Module):
    def __init__(self,
                 input_dim, 
                 input_edge_dim,
                 output_dim,
                 hidden_dim=64,
                 learn_eps=True,
                 num_layers=3,
                 num_mlp_layers=2):
        """
        This is the graph neural network used for message-passing features in the
        face-adjacency graph.  (see Section 3.2, Message passing in paper)

        Args:
            input_dim ([type]): [description]
            input_edge_dim ([type]): [description]
            output_dim ([type]): [description]
            hidden_dim (int, optional): [description]. Defaults to 64.
            learn_eps (bool, optional): [description]. Defaults to True.
            num_layers (int, optional): [description]. Defaults to 3.
            num_mlp_layers (int, optional): [description]. Defaults to 2.
        """
        super(UVNetGraphEncoder, self).__init__()
        self.num_layers = num_layers
        self.learn_eps = learn_eps

        # List of layers for node and edge feature message passing
        self.node_conv_layers = torch.nn.ModuleList()
        self.edge_conv_layers = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            node_feats = input_dim if layer == 0 else hidden_dim
            edge_feats = input_edge_dim if layer == 0 else hidden_dim
            self.node_conv_layers.append(_NodeConv(node_feats=node_feats,
                                                   out_feats=hidden_dim,
                                                   edge_feats=edge_feats,
                                                   num_mlp_layers=num_mlp_layers,
                                                   hidden_mlp_dim=hidden_dim))
            self.edge_conv_layers.append(_EdgeConv(edge_feats=edge_feats,
                                                   out_feats=hidden_dim,
                                                   node_feats=node_feats,
                                                   num_mlp_layers=num_mlp_layers,
                                                   hidden_mlp_dim=hidden_dim))

        # Linear function for graph poolings of output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()

        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(nn.Linear(input_dim, output_dim))
            else:
                self.linears_prediction.append(nn.Linear(hidden_dim, output_dim))

        self.drop1 = nn.Dropout(0.3)
        self.drop = nn.Dropout(0.5)
        self.pool = MaxPooling()

    def forward(self, g, h, efeat):                 # g=DGLHeteroGraph, h=[8430, 64], efeat=[48700, 64]
        hidden_rep = [h]
        he = efeat

        for i in range(self.num_layers - 1):        # i=0,1,2
            # Update node features
            h = self.node_conv_layers[i](g, h, he)  # [8430, 64]-->[8430, 64]
            # Update edge features
            he = self.edge_conv_layers[i](g, h, he) # [48700, 64]-->[48700, 64]
            hidden_rep.append(h)

        out = hidden_rep[-1]  # hidden_rep=[torch.Size([8430, 64]), torch.Size([8430, 64]), torch.Size([8430, 64])] # out=[8430, 64]
        out = self.drop1(out) # [8430, 64]-->[8430, 64]
        score_over_layer = 0

        # Perform pooling over all nodes in each graph in every layer
        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(g, h)                                          # pooled_h-->[batch_size, n_features] = [256, 64]
            score_over_layer += self.drop(self.linears_prediction[i](pooled_h)) # score_over_layer-->[256, 128]

        return out, score_over_layer

###############-------------Graph_Encoder-------------###############