import torch
import torch_scatter
import functools
import collections
import torch.nn as nn
import matplotlib.pyplot as plt
import pdb

# define edge and graph
EdgeSet = collections.namedtuple('EdgeSet', ['name', 'features', 'senders', 'receivers'])
GraphSet = collections.namedtuple('Graph', ['node_features', 'edge_set'])
        
    # ------------------- Graph Attention Neural Network Model ------------------- #

class GATModel(nn.Module):
    """
    Basic Graph Attention Neural Network (GATModel)
    
    This class defines the main architecture of the Graph Attention Network, including the encoder,
    updater (where attention-based message passing occurs), and decoder components.

    Attributes:
        _output_size (int): The size of the output features (e.g., output dimension of the decoder).
        _latent_size (int): The size of the latent features used throughout the network.
        _num_layers (int): The number of layers in each MLP (Multi-Layer Perceptron).
        _message_passing_steps (int): The number of message passing iterations in the updater.

        encoder (Encoder): The encoder module that processes input graph and image features.
        updater (Updater): The updater module that performs message passing using attention.
        decoder (Decoder): The decoder module that generates the final output from the updated graph.

    Methods:
        __init__: Initializes the GATModel with given hyperparameters.
        _make_mlp: Creates an MLP network with specified output sizes and layer normalization.
        forward: Defines the forward pass through the GAT model.
    """

    def __init__(self, output_size, latent_size, num_layers, message_passing_steps):
        """
        Initialize the GATModel with hyperparameters.

        Parameters:
            output_size (int): The size of the output features (e.g., 3 for 3D coordinates).
            latent_size (int): The size of the latent features (e.g., 128).
            num_layers (int): The number of layers in each MLP.
            message_passing_steps (int): The number of message passing iterations.
        """
        super(GATModel, self).__init__()

        # Hyperparameters
        self._output_size = output_size          # Output feature size, e.g., 3 for (x, y, z) coordinates
        self._latent_size = latent_size          # Latent feature size used in the network
        self._num_layers = num_layers            # Number of layers in the MLPs
        self._message_passing_steps = message_passing_steps  # Number of message passing iterations

        # Define the encoder, updater (message passer), and decoder modules
        # The encoder processes input graph and image features into latent representations
        self.encoder = Encoder(make_mlp=self._make_mlp)

        # The updater performs message passing using attention mechanisms
        self.updater = Updater(
            make_mlp=self._make_mlp,
            output_size=self._latent_size,
            message_passing_steps=self._message_passing_steps
        )

        # The decoder processes the updated latent graph to produce final outputs
        # Layer normalization is disabled in the decoder's MLP
        self.decoder = Decoder(
            make_mlp=functools.partial(self._make_mlp, layer_norm=False),
            output_size=self._output_size
        )

    
    def _make_mlp(self, output_size, layer_norm=True):
        """
        Build an MLP (Multi-Layer Perceptron) network.

        Parameters:
            output_size (int or list): The size(s) of the output features. Can be an integer or a list of integers.
            layer_norm (bool): Whether to apply layer normalization after the MLP.

        Returns:
            network (nn.Module): The constructed MLP network.
        """
        # Assign output sizes for the MLP layers
        if type(output_size) == int:
            # Create a list of sizes: [latent_size, ..., latent_size, output_size]
            # This creates an MLP with self._num_layers hidden layers(2) of size self._latent_size (128) and an output layer of size output_size (128).
            output_sizes = [self._latent_size] * self._num_layers + [output_size] #Updater-->[128,128,128] Decoder -->[128,128,3]
        elif type(output_size) == list:
            # Use the provided list of output sizes directly
            output_sizes = output_size #Decoder MLPs ,two layers [256,128]
        else:
            raise ValueError('Invalid output_size type')

        # Construct the MLP network according to the specified output sizes
        network = LazyMLPBlock(output_sizes)

        # Add a layer normalization layer if specified
        if layer_norm:
            network = nn.Sequential(network, nn.LayerNorm(normalized_shape=output_sizes[-1]))

        return network

    # forward GAT structure
    def forward(self, graph, image_feature):
        """
        Define the forward pass through the GATModel.

        Parameters:
            graph (GraphSet): The input graph containing node and edge features.
            image_feature (torch.Tensor): The image features associated with the graph.

        Returns:
            output (torch.Tensor): The final output of the GATModel (e.g., predicted node positions).
        """
        # Template Graph Encoding-->Encode the input graph and image features into a latent graph representation
        latent_graph = self.encoder(graph, image_feature)

        # Graph Attention Updating. -->Update the latent graph using attention-based message passing
        latent_graph = self.updater(latent_graph)

        # Cloth Mesh Decoding--> Decode the updated latent graph to produce the final output (e.g., 3D positions)
        output = self.decoder(latent_graph)

        return output

"""Template Graph Encoding.""" 
class Encoder(nn.Module):
    """
    Encoder Module

    Encodes the input graph node and edge features along with image features into latent representations.

    Methods:
        __init__: Initializes the Encoder.
        forward: Defines the forward pass for encoding features.
    """

    def __init__(self, make_mlp):
        """
        Initialize the Encoder.

        Parameters:
            make_mlp (callable): A function to create MLP networks.
        """
        super().__init__()

        # MLP to encode node features; output size is set to 128
        self.node_encoder = make_mlp([256, 128])

        # MLP to encode edge features; output size is set to 128
        self.edge_encoder = make_mlp([256, 128])

    def forward(self, graph, image_feature):
        """
        Forward pass to encode node and edge features.

        Parameters:
            graph (GraphSet): The input graph with node and edge features.
            image_feature (torch.Tensor): The image features.

        Returns:
            latent_graph (GraphSet): The graph with encoded latent node and edge features.
        """
        # Get batch size
        # pdb.set_trace()
        batch_size = image_feature.shape[0]

        # Encode node features
        node_num, node_dim = graph.node_features.shape
        # Expand node features to match batch size
        #Since the graph structure is the same across the batch (assuming identical graphs for each sample), 
        #the node features are replicated across the batch.
        node_feature = graph.node_features.view(1, node_num, node_dim).expand(batch_size, -1, -1)
        # Expand image features to match number of nodes
        node_image_feature = image_feature.view(batch_size, 1, image_feature.shape[1]).expand(-1, node_num, -1)
        # Concatenate node features with image features and encode
        node_latents = self.node_encoder(torch.cat([node_feature, node_image_feature], -1))

        # Encode edge features
        edge_num, edge_dim = graph.edge_set.features.shape
        # Expand edge features to match batch size
        edge_features = graph.edge_set.features.view(1, edge_num, edge_dim).expand(batch_size, -1, -1)
        # Encode edge features (edge_encoder is a mlp)
        edge_latents = self.edge_encoder(edge_features)

        # Return the graph with encoded node and edge features
        graphset = GraphSet(node_latents, graph.edge_set._replace(features=edge_latents))
        
        return graphset 
    
"""Graph Attention Updating."""    
class Updater(nn.Module):
    """
    Updater Module

    Performs attention-based message passing on the latent graph for a specified number of steps.

    Methods:
        __init__: Initializes the Updater.
        forward: Defines the forward pass for updating the graph.
    """

    def __init__(self, make_mlp, output_size, message_passing_steps):
        """
        Initialize the Updater.

        Parameters:
            make_mlp (callable): A function to create MLP networks.
            output_size (int): The size of the output features for the MLPs.
            message_passing_steps (int): The number of message passing iterations to perform.
        """
        super().__init__()

        # Use an OrderedDict to store the sequence of GraphAttentionBlocks
        self._submodules_ordered_dict = collections.OrderedDict()

        # Create multiple GraphAttentionBlocks for message passing
        for index in range(message_passing_steps):
            self._submodules_ordered_dict[str(index)] = GraphAttentionBlock(
                make_mlp=make_mlp,
                output_size=output_size
            )

        # Combine the blocks into a Sequential module
        self.submodules = nn.Sequential(self._submodules_ordered_dict)

    def forward(self, graph):
        """
        Forward pass to update the graph through attention-based message passing.

        Parameters:
            graph (GraphSet): The input latent graph.

        Returns:
            updated_graph (GraphSet): The updated latent graph after message passing.
        """
        # Pass the graph through the sequence of GraphAttentionBlocks
        # pdb.set_trace()
        #This calls the GraphAttentionBlock forward part which calls the  _update_edge_features, _update_node_features etc
        updated_graph = self.submodules(graph) 
        return updated_graph

"""Cloth Mesh Decoding."""
class Decoder(nn.Module):
    """
    Decoder Module

    Decodes the updated latent graph to produce the final output (e.g., node positions).

    Methods:
        __init__: Initializes the Decoder.
        forward: Defines the forward pass for decoding.
    """

    def __init__(self, make_mlp, output_size):
        """
        Initialize the Decoder.

        Parameters:
            make_mlp (callable): A function to create MLP networks.
            output_size (int): The size of the output features (e.g., 3 for 3D positions).
        """
        super().__init__()

        # Create an MLP to decode node features to the desired output size
        self.node_decoder = make_mlp(output_size)

    def forward(self, graph):
        """
        Forward pass to decode the node features into final outputs.

        Parameters:
            graph (GraphSet): The updated latent graph.

        Returns:
            output (torch.Tensor): The decoded output from the node features.
        """
        # Apply the decoder MLP to the node features
        # pdb.set_trace()
        output = self.node_decoder(graph.node_features)
        return output

# ------------------- LazyMLP, GraphNet, Attention ------------------- #

class LazyMLPBlock(nn.Module):
    """
    Basic MLP (Multi-Layer Perceptron) Block

    Constructs a sequence of linear layers with optional activation functions.

    Methods:
        __init__: Initializes the MLP block with specified output sizes.
        forward: Defines the forward pass through the MLP.
    """

    def __init__(self, output_sizes):
        """
        Initialize the LazyMLPBlock.

        Parameters:
            output_sizes (list): A list of integers specifying the output size of each layer.
        """
        super(LazyMLPBlock, self).__init__()

        # Get the number of layers based on the output sizes
        num_layers = len(output_sizes)

        # Use an OrderedDict to store layers for consistent ordering
        self._layers_ordered_dict = collections.OrderedDict()

        # Construct the MLP layers
        for index, output_size in enumerate(output_sizes):
            # Add a linear layer; LazyLinear infers the input size at runtime
            self._layers_ordered_dict[f"linear_{index}"] = nn.LazyLinear(output_size)
            # Add a ReLU activation after each layer except the last
            if index < (num_layers - 1):
                self._layers_ordered_dict[f"relu_{index}"] = nn.ReLU()

        # Combine the layers into a Sequential module
        self.layers = nn.Sequential(self._layers_ordered_dict)

    def forward(self, x):
        """
        Forward pass through the MLP.

        Parameters:
            x (torch.Tensor): The input tensor.

        Returns:
            y (torch.Tensor): The output tensor after passing through the MLP.
        """
        y = self.layers(x)
        return y

#Attention model for generating weightd
class AttentionBlock(nn.Module):
    """
        Basic attention weight function.
        w_{ij}^{(t+1)} = \frac{exp(\phi_{A}^(t)e_{ij}^{(t+1)})}{\sum{exp(\phi_{A}^(t)e_{ik}^{{(t+1)})}}
        
    """

    def __init__(self):
        super().__init__()
        self.linear = nn.LazyLinear(1)
        self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input, idx):
        input = self.linear(input)
        input = self.activation(input)
        # update attention_weights with softmax(MLP(edge_message_features))
        attention_weight = torch_scatter.composite.scatter_softmax(input, idx, dim=1)
        return attention_weight


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
            
            ē_ij^(t+1) = φ_E^(t)( [ ē_ij^(t), v̄_i^(t), v̄_j^(t) ] ), where 
                1. ē_ij^(t+1): The updated edge feature between nodes i and j at time step t+1.
                2. φ_E^(t): The edge update function at time step t.
                3. [ ē_ij^(t), v̄_i^(t), v̄_j^(t) ]: The concatenation of the current edge feature 
                and the features of the sender node i and receiver node j at time step t.
        """
        # pdb.set_trace()
        # get node_sender_features, node_receiver_features, edge_set.features
        node_sender_features   = torch.index_select(input=node_features, dim=1, index=edge_set.senders)
        node_receiver_features = torch.index_select(input=node_features, dim=1, index=edge_set.receivers)
        
        features = [node_sender_features, node_receiver_features, edge_set.features]
        # update edge_message_features = edge_MLP([sender_node_features, receiver_node_features, edge_features])
        return self.edge_model(torch.cat(features, -1))

    def _update_node_features(self, node_features, edge_set):
        """
            Aggregate edge features, apply node function.
            
            v̄_i^(t+1) = φ_V^(t)( [ v̄_i^(t), Σ_{k∈Neighbor(i)} w̄_{ik}^{(t+1)} ē_{ik}^{(t+1)} ] )
                1. v̄_i^(t+1): The updated node feature for node i at time step t+1.
                2. φ_V^(t): The node update function at time step t.
                3. v̄_i^(t): The current node feature of node i at time step t.
                4. Σ_{k∈Neighbor(i)} w̄_{ik}^{(t+1)} ē_{ik}^{(t+1)}: The sum over all neighbors k of node i 
                of the product of the updated edge features ē_{ik}^{(t+1)} and the corresponding attention weights w̄_{ik}^{(t+1)}.
                5. Neighbor(i): The set of nodes that are connected to node i (i.e., the neighbors of node i).
                6. w̄_{ik}^{(t+1)}: The attention weight between nodes i and k at time step t+1.
                7. ē_{ik}^{(t+1)}: The updated edge feature between nodes i and k at time step t+1.

        """
        features = [node_features]
        # get learnable attention_weights
        attention_weights = self.attention_model(edge_set.features, edge_set.receivers)
        # get attention_message_features: attention_weights * edge_message_features
        features.append(torch_scatter.scatter_add(torch.mul(edge_set.features, attention_weights), 
                                                  edge_set.receivers, dim=1))
        # update node_message_features: node_MLP([receiver_node_features, attention_message_features])
        return self.node_model(torch.cat(features, -1))

    def forward(self, graph, residual=True):
        """
            Update Latent Graph with Attention Message Passing.
            edge:          ē_ij^(t+1)
            vertex (node): v̄_i^(t+1)
        """
        # pdb.set_trace()
        # apply edge functions: update edge_features (graph.node_features, graph.edge_set are both in latent space)
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

    #######MLP MODEL TEST#######
    # Define the MLP architecture via output sizes.
    output_sizes = [100, 200, 300]
    
    # Instantiate the LazyMLPBlock model.
    model = LazyMLPBlock(output_sizes)

    print("== Model Architecture ==")
    print(model)
    
    # Create a dummy input tensor.
    # For instance, assume a batch size of 4 and an input feature dimension of 50.
    x = torch.randn(4, 50)
    
    # Print the input tensor shape.
    print("Input shape:", x.shape)
    
    # Pass the input through the MLP.
    output = model(x)
    
    # Print the output tensor shape.
    # The final output should have dimension 300 as specified in the output_sizes.
    print("Output shape:", output.shape)  

    #######ATTENTION MODEL TEST#######
    batch_size = 2
    num_neighbors = 6
    feature_dim = 10

    input_tensor = torch.randn(batch_size, num_neighbors, feature_dim)
    idx = torch.tensor([[0, 0, 0, 1, 1, 0],
                        [0, 0, 1, 1, 1, 0]], dtype=torch.long)
    
    print("== Dummy Input ==")
    print("Input tensor (features):")
    print(input_tensor)
    print("\nIndex tensor (group assignments):")
    print(idx)

    attn_block = AttentionBlock()
    attention_weights = attn_block(input_tensor, idx)
    
    print("\n== Output ==")
    print("Attention weights (after group-wise softmax):")
    print(attention_weights)
    
    print("\nSum of attention weights per group (should be close to 1):")
    for b in range(batch_size):
        unique_groups = torch.unique(idx[b])
        for group in unique_groups:
            group_mask = (idx[b] == group).unsqueeze(-1)
            group_sum = attention_weights[b][group_mask].sum()
            print(f"Batch {b}, Group {group.item()} sum: {group_sum.item()}")

    #######GRAPH ATTENTION  TEST#######
    # Hyperparametersand dimensions.
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

    #######GAT MODEL TEST#######
    # Define hyperparameters for the GAT model
    output_size = 3  # For example, 3D coordinates
    latent_size = 128
    num_layers = 2  # Number of layers in the MLP
    message_passing_steps = 10 # Number of message passing iterations in the Updater

    # Instantiate the GATModel
    gat_model = GATModel(output_size, latent_size, num_layers, message_passing_steps)

    print("\n== GAT Model Architecture ==")
    print(gat_model)

    # Create dummy input graph and image features
    batch_size = 2
    num_nodes = 5
    node_feature_dim = 6
    num_edges = 8
    edge_feature_dim = 6
    image_feature_dim = 10

    # Dummy node features
    node_features = torch.randn(num_nodes, node_feature_dim)

    # Dummy edge features
    edge_features = torch.randn(num_edges, edge_feature_dim)

    # Dummy senders and receivers
    senders = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2], dtype=torch.long)
    receivers = torch.tensor([1, 2, 3, 4, 0, 2, 3, 4], dtype=torch.long)

    # Create the EdgeSet and GraphSet
    edge_set = EdgeSet(name="edge_set_1", features=edge_features, senders=senders, receivers=receivers)
    graph = GraphSet(node_features=node_features, edge_set=edge_set)

    # Dummy image features
    image_features = torch.randn(batch_size, image_feature_dim)

    # Perform a forward pass through the GAT model
    output = gat_model(graph, image_features)

    # Print the output shape
    print("\n== GAT Model Output ==")
    print("Output shape:", output.shape)