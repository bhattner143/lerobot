import torch
import torch.nn as nn
import torch_scatter
import collections

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.attention_model import AttentionBlock
from model.mlp_model import LazyMLPBlock

EdgeSet = collections.namedtuple('EdgeSet', ['name', 'features', 'senders', 'receivers'])
GraphSet = collections.namedtuple('Graph', ['node_features', 'edge_set'])

class GraphAttentionBlock(nn.Module):
    """
        Basic Graph Attention Block with residual connections
    """
    def __init__(self, make_mlp, output_size):
        super(GraphAttentionBlock, self).__init__()
        
        # construct MLP models for edge and node
        self.edge_model = make_mlp(output_size)
        self.node_model = make_mlp(output_size)
        
        # construct attention weight model
        self.attention_model = AttentionBlock()

    def _update_edge_features(self, node_features, edge_set):
        """
            Aggregate node features, apply MLP edge function.
        """
        
        # get node_sender_features, node_receiver_features, edge_set.features
        node_sender_features = torch.index_select(input=node_features, dim=1, index=edge_set.senders)
        node_receiver_features = torch.index_select(input=node_features, dim=1, index=edge_set.receivers)
        features = [node_sender_features, node_receiver_features, edge_set.features]
        
        # update edge_message_features = edge_MLP([sender_node_features, receiver_node_features, edge_features])
        return self.edge_model(torch.cat(features, -1))

    def _update_node_features(self, node_features, edge_set):
        """
            Aggregate edge features, apply node function.
        """
        features = [node_features]
        
        # get learnable attention_weights
        attention_weights = self.attention_model(edge_set.features, edge_set.receivers)
        
        # get attention_message_features: attention_weights * edge_message_features
        features.append(torch_scatter.scatter_add(torch.mul(edge_set.features, attention_weights), edge_set.receivers, dim=1))
        
        # update node_message_features: node_MLP([receiver_node_features, attention_message_features])
        return self.node_model(torch.cat(features, -1))

    def forward(self, graph, residual=True):
        """
            Update Latent Graph with Attention Message Passing.
        """

        # apply edge functions: update edge_features
        updated_edge_features = self._update_edge_features(graph.node_features, graph.edge_set)
        new_edge_set = graph.edge_set._replace(features=updated_edge_features)

        # apply node functions: update node_features
        new_node_features = self._update_node_features(graph.node_features, new_edge_set)

        # apply residual change
        if residual:
            new_node_features += graph.node_features
            new_edge_set = new_edge_set._replace(features=new_edge_set.features + graph.edge_set.features)
        
        # return updated latent graph
        return GraphSet(new_node_features, new_edge_set)
    
if __name__ == "__main__":
    
    # Hyperparameters and dimensions.
    batch_size = 1
    num_nodes = 4
    node_feature_dim = 6   # Number of features per node.
    num_edges = 6

    #  IMPORTANT: Set edge_feature_dim to match the output size of the edge model (here 6) for residual addition.
    edge_feature_dim = 6   
    output_size = 6        # Use output_size 6 for both nodes and edges.
    
    # Create dummy node features (shape: [batch_size, num_nodes, node_feature_dim]).
    node_features = torch.randn(batch_size, num_nodes, node_feature_dim)
    
    # Create dummy edge features (shape: [batch_size, num_edges, edge_feature_dim]).
    edge_features = torch.randn(batch_size, num_edges, edge_feature_dim)
    
    # Define sender and receiver indices.
    # These indices refer to nodes in dimension 1 of node_features.
    senders = torch.tensor([0, 1, 2, 2, 3, 3], dtype=torch.long)
    receivers = torch.tensor([1, 2, 0, 3, 1, 0], dtype=torch.long)
    
    # Create the EdgeSet with a given name.
    edge_set = EdgeSet(name="edge_set_1", features=edge_features, senders=senders, receivers=receivers)
    
    # Construct the graph as a named tuple.
    graph = GraphSet(node_features=node_features, edge_set=edge_set)
    
    # Define a simple make_mlp function that returns a LazyMLPBlock.
    def make_mlp(output_size):
        # For example, return a two-layer MLP with a hidden layer of size 10
        # and a final output layer projecting to 'output_size' dimensions.
        return LazyMLPBlock([10, output_size])
    
    # Instantiate the GraphAttentionBlock.
    model = GraphAttentionBlock(make_mlp, output_size)
    
    # Perform a forward pass through the graph attention block.
    updated_graph = model(graph, residual=True)
    
    # Print the shapes of the original and updated node and edge features.
    print("Original node features shape:", node_features.shape)
    print("Updated node features shape:", updated_graph.node_features.shape)
    print("Original edge features shape:", edge_features.shape)
    print("Updated edge features shape:", updated_graph.edge_set.features.shape)