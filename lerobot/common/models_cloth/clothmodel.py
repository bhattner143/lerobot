import abc
import logging
import os

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type, TypeVar
import torch
import torch_scatter
import functools
import collections
import torch.nn as nn
import matplotlib.pyplot as plt
import pdb
from clothmodel_configs import PreTrainedClothModelConfig, MeshGATConfig

import torch.nn as nn
import torchvision.models as models

import draccus
from termcolor import colored
from omegaconf import OmegaConf


T = TypeVar("T", bound="PreTrainedClothModel")

class PreTrainedClothModel(nn.Module, abc.ABC):
    """
    Base class for cloth models.
    """

    config_class: None
    name: None

    def __init__(self, config: PreTrainedClothModelConfig, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, PreTrainedClothModelConfig):
            raise ValueError(
                f"Parameter config in `{self.__class__.__name__}(config)` should be an instance of class "
                "`PreTrainedClothModelConfig`. To create a model from a pretrained model use "
                f"`model = {self.__class__.__name__}.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        self.config = config

    @classmethod
    def from_pretrained_cloth_model(
        cls: Type[T],
        pretrained_path: str | Path,
        cli_overrides: dict[str, str] | None = None,
        config: PreTrainedClothModelConfig| None = None,
    ) -> T:
        if config is None:
            config = PreTrainedClothModelConfig.from_pretrained_cloth_model(pretrained_path, 
                                                                            cli_overrides=cli_overrides)
            

class ClothModel(PreTrainedClothModel):
    """
        Template-based Mass-spring Cloth GNN
    """
    config_class = MeshGATConfig
    name         = "mesh_gat"

    def __init__(self, 
                 config: MeshGATConfig,
                 template_info: dict = None,
                 ):
        super().__init__(config)

        # init template_info
        if template_info is None:
            template_info = pickle.load(open(config.template_dir, mode='rb'))
        
        self.template_info     = template_info
        self.template_mesh_pos = self.template_info['mesh_pos']
        self.template_edge_idx = self.template_info['edge_idx'].astype(int)

        # init backbone ResNet model
        self.backbone_model = get_model('resnet34')

        # init learned GAT model
        self.learned_model = GATModel(
            output_size=3,
            latent_size=128,
            num_layers=2,
            message_passing_steps=template_info.get('message_passing_steps', 3)
        )

    # build template_graph with template_nodes, edge_senders, and edge_receivers
    def _build_template_graph(self):
        # init node features as template mesh positions: N (num of nodes) x 3 (positions)
        node_features = torch.from_numpy(self.template_mesh_pos).float().cuda()
      
        # create two-way connectivity of edge senders and receivers
        senders = self.template_edge_idx[:, 0]
        receivers = self.template_edge_idx[:, 1]
        sender = torch.from_numpy(np.concatenate([senders, receivers], 0)).to(torch.int64).cuda()
        receiver = torch.from_numpy(np.concatenate([receivers, senders], 0)).to(torch.int64).cuda()
        
        # assign edge features as edge vertices' relative coordinate and norm: E (num of edges) x 4 (vecter + norm)
        relative_edge_vector = (torch.index_select(node_features, 0, sender) - torch.index_select(node_features, 0, receiver))
        relative_edge_norm = torch.norm(relative_edge_vector, dim=-1, keepdim=True)
        edge_features = torch.cat((relative_edge_vector, relative_edge_norm), -1)

        # return gat model with node features and mesh edges
        mesh_edges = EdgeSet(
            name='mesh_edges',
            features=edge_features,
            receivers=receiver,
            senders=sender
        )

        return GraphSet(node_features=node_features, edge_set=mesh_edges)

    def forward(self, batch):
        
        # resnet encode image feature
        image_feature = self.backbone_model(batch)
        
        # build template graph from template_info
        template_graph = self._build_template_graph()
        
        # forward gat with message_passing_steps
        pred_mesh = self.learned_model(template_graph, image_feature)

        # batch_size, vertex_num, vertex_dim(3: xyz position)
        b, n_v, n_p = pred_mesh.shape
        
        return pred_mesh



# resnet model in the torchvision library
def resnet_list():
    return ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]

# ------------------------------------------------------ #
# ------------------- ResNet Encoder ------------------- #
# ------------------------------------------------------ #

class ResNet(nn.Module):
    """
    This is a resnet wrapper class which takes existing resnet architectures and
    adds a final linear layer at the end, ensuring proper output dimensionality
    """
    def __init__(self, model_name, pretrained):
        super().__init__()
        
        if pretrained:
            if model_name == "resnet18":
                model_func = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            elif model_name == "resnet34":
                model_func = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            elif model_name == "resnet50":
                model_func = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            elif model_name == "resnet101":
                model_func = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
            elif model_name == "resnet152":
                model_func = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
            else:
                raise Exception(f"Unknown backend model type: {model_name}")
        else:
            if model_name == "resnet18":
                model_func = models.resnet18
            elif model_name == "resnet34":
                model_func = models.resnet34
            elif model_name == "resnet50":
                model_func = models.resnet50
            elif model_name == "resnet101":
                model_func = models.resnet101
            elif model_name == "resnet152":
                model_func = models.resnet152
            else:
                raise Exception(f"Unknown backend model type: {model_name}")

        # construct the resnet encoder
        if pretrained:
            b_model = model_func
        else:
            b_model = model_func()
        resnet_encoder = nn.Sequential(
            b_model.conv1,
            b_model.bn1,
            b_model.relu,
            b_model.maxpool,
            b_model.layer1,
            b_model.layer2,
            b_model.layer3,
            b_model.layer4,
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )
        self.encoder = resnet_encoder

    def forward(self, x):
        # batch size, channels, height, width
        b, c, h, w = x.shape
        
        # get feature output
        f = self.encoder(x)
        
        # get final output
        f = f.flatten(start_dim=1)

        # batch_size, feature_dim (512)
        b, l = f.shape

        return f

# ------------------------------------------------------ #
# ---------------- Access ResNet Model ----------------- #
# ------------------------------------------------------ #

def get_model(model_name, pretrained=False):
    model = ResNet(model_name=model_name, pretrained=pretrained)
    return model


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

# Example usage
if __name__ == "__main__":
    import json
    # set current task
    mode = "train" # select mode from 'train', 'test', 'test_real'
    name_cloth = 't_shirt_l3'
    checkpoint_file = '2025-03-28/14-46-38/finalbestmodel_0299_0.01162.pt'

    # get address
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

    DATASET_DIR     = Path(f'/home/dips/Documents/datasets_lerobot/so100_test/mesh_gat/{name_cloth}')
    # ✅ Create a dummy "config.json" file for testing
    config_dict = {
        "type": "mesh_gat",
        "mode": "eval",
        "batch_size": 32,
        "epoch_size": 2000,
        "image_size": 720,
        "lr": 1e-4,
        "schedule_step": 150,
        "momentum": 0.9,
        "sample_ratio": 1.0,
        "save_step": 30,
        "num_threads": 16,
        "shuffle": True,
        "drop_last": False,
        "store_pred": True,
        "loss_weights": {
            "vertex_loss": 1.0,
            "keypoint_loss": 1.0,
            "chamfer_loss": 0.5
        },
        "use_chamfer": True,
        "chamfer_active_epoch": 0,
        "use_pixel": False,
        "use_normalize": False,
        "input_features": {
            "depth_image": {"type": "VISUAL", "shape": [3, 224, 224]}
        },
        "output_features": {
            "mesh": {"type": "STATE", "shape": [3]}
        }
    }

    config_dir = DATASET_DIR / "dummy_model"
    config_dir.mkdir(exist_ok=True)
    with open(config_dir / "config.json", "w") as f:
        json.dump(config_dict, f)
    
    print(config_dir.resolve())
    
    argtype = PreTrainedClothModelConfig

    config_path = config_dir / "config.json"
    
    config = draccus.parse(
                    PreTrainedClothModelConfig,
                    config_path=config_dir / "config.json",
                    # args=["--type", "mesh_gat"]  # 👈 passed as CLI override!
                )

    print(f"Config type: {config.type}")
    print(config)

    # Save the configuration as eval_config.json using json
    eval_config_path = config_dir / "eval_config.json"
    with open(eval_config_path, "w") as f:
        json.dump(OmegaConf.to_container(OmegaConf.structured(config), resolve=True), f, indent=4)
    print(colored(f"Configuration saved to {eval_config_path}", "green"))


    # Load the configuration using the from_pretrained_cloth_model method
    pretrained_path = config_dir
    cli_overrides = {"mode": "train", "batch_size": 64}
    config = MeshGATConfig.from_pretrained_cloth_model(pretrained_path, cli_overrides=cli_overrides)
    print(f"Loaded config: {config}")

    # Load the model
    import pickle
    template_info = pickle.load(open(config.template_dir, mode='rb'))
    model = ClothModel(config, template_info)
    