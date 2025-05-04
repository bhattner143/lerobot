import torch
# import torch_scatter
import functools
import collections
import torch.nn as nn

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.mlp_model import LazyMLPBlock
from model.graph_attention_model import GraphAttentionBlock


# define edge and graph
EdgeSet = collections.namedtuple('EdgeSet', ['name', 'features', 'senders', 'receivers'])
GraphSet = collections.namedtuple('Graph', ['node_features', 'edge_set'])

class GATModel(nn.Module):
    """
    Basic Graph Attention Neural Network
    """
    def __init__(self, output_size, latent_size, num_layers, message_passing_steps):
        super(GATModel, self).__init__()
        
        # hyper-parameters
        self._output_size = output_size  # 3
        self._latent_size = latent_size  # 128
        self._num_layers = num_layers  # 2
        self._message_passing_steps = message_passing_steps  # 15
        
        # define encoder, updater, decoder
        self.encoder = Encoder(make_mlp=self._make_mlp)
        self.updater = Updater(make_mlp=self._make_mlp, output_size=self._latent_size, message_passing_steps=self._message_passing_steps)
        self.decoder = Decoder(make_mlp=functools.partial(self._make_mlp, layer_norm=False), output_size=self._output_size)

    def _make_mlp(self, output_size, layer_norm=True):
        """
            Build one MLP with output_size using lazy MLP block.
        """
        # assign output_sizes
        if type(output_size) == int:
            output_sizes = [self._latent_size] * self._num_layers + [output_size]
        elif type(output_size) == list:
            output_sizes = output_size
        else:
            raise ValueError('Invalid output_size type')
        
        # construct MLP according to output_sizes
        network = LazyMLPBlock(output_sizes)
        
        # add norm layer
        if layer_norm:
            network = nn.Sequential(network, nn.LayerNorm(normalized_shape=output_sizes[-1]))
        return network

    # forward GAT structure
    def forward(self, graph, image_feature):
        """
            Encode, Update, Decode Cloth Graph.
        """
        # encode template_graph and image_feature into latent_graph
        latent_graph = self.encoder(graph, image_feature)
        
        # update latent_graph with attention message passing
        latent_graph = self.updater(latent_graph)
        
        # decode latent_graph to mesh_position
        return self.decoder(latent_graph)

class Encoder(nn.Module):
    """
        Encode template node, edge features and image features into latent_graph
    """
    def __init__(self, make_mlp):
        super().__init__()
        
        self.node_encoder = make_mlp([256, 128])
        self.edge_encoder = make_mlp([256, 128])

    def forward(self, graph, image_feature):
        
        # get batch_size
        batch_size = image_feature.shape[0]
        
        # get number and dimension of graph node features
        node_num, node_dim = graph.node_features.shape
        
        # encode graph node_features and image_features (batch_size, node_num, 3 + 512) to latent node_features (batch_size, node_num, 128)
        node_feature = graph.node_features.view(1, node_num, node_dim).expand(batch_size, -1, -1)
        node_image_feature = image_feature.view(batch_size, 1, image_feature.shape[1]).expand(-1, node_num, -1)
        node_latents = self.node_encoder(torch.cat([node_feature, node_image_feature], -1))

        # encode graph edge features (batch_size, edge_num, 4) to latent edge features (batch_size, edge_num, 128)
        edge_num, edge_dim = graph.edge_set.features.shape
        edge_latents = self.edge_encoder(graph.edge_set.features.view(1, edge_num, edge_dim).expand(batch_size, -1, -1))
        
        # return encoded feature graph
        return GraphSet(node_latents, graph.edge_set._replace(features=edge_latents))
    
class Updater(nn.Module):
    """
        Update feature graph with N times of Attention Message Passing
    """
    def __init__(self, make_mlp, output_size, message_passing_steps):
        super().__init__()
        # stack GraphAttentionBlock with the number of message_passing_steps
        self._submodules_ordered_dict = collections.OrderedDict()
        for index in range(message_passing_steps):
            self._submodules_ordered_dict[str(index)] = GraphAttentionBlock(make_mlp=make_mlp, output_size=output_size)
        self.submodules = nn.Sequential(self._submodules_ordered_dict)

    def forward(self, graph):
        return self.submodules(graph)
    
class Decoder(nn.Module):
    """
        Decoder latent_graph to mesh_position
    """
    def __init__(self, make_mlp, output_size):
        super().__init__()
        self.node_decoder = make_mlp(output_size)

    def forward(self, graph):
        return self.node_decoder(graph.node_features)
    
