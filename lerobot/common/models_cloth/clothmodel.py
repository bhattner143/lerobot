import abc
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
import numpy as np
import pdb
import pickle
import logging
from termcolor import colored


from lerobot.common.models_cloth.clothmodel_configs import PreTrainedClothModelConfig, MeshGATConfig

import torch.nn as nn
import torchvision.models as models

import draccus
from termcolor import colored
from omegaconf import OmegaConf
import logging
from termcolor import colored


from lerobot.common.models_cloth.mesh_gat.model import gat_model, resnet_model


T = TypeVar("T", bound="PreTrainedClothModel")

class PreTrainedClothModel(nn.Module, abc.ABC):
    """
    Base abstract class for cloth models.
    """

    config_class: None
    name: None

    def __init__(self, 
                 config: PreTrainedClothModelConfig, 
                 *inputs, 
                 **kwargs):
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
        cls: Type[T], #Gets the class type for cloth model
        pretrained_path: str | Path,
        *, #ARguments passed after these must be passed as keyword arguments
        config: PreTrainedClothModelConfig| None = None,
        strict: bool = True,
        **kwargs,
        ) -> T:
        #if config is not passed then generate it from the pretrained_path
        if config is None:
            # TODO: Fix this to parse the model type from command line
            cli_overrides = {"type": "mesh_gat"}  # Default or replace with actual argument parsing if available
            config = PreTrainedClothModelConfig.from_pretrained_cloth_model(pretrained_path,
                                                                            cli_overrides=cli_overrides,
                                                                            **kwargs
                                                                            )
            
        cloth_model_id = str(pretrained_path)

        #Load the template mesh information and create the cloth model object
        template_info = pickle.load(open(config.path_config.template_dir, mode='rb'))
        kwargs['template_info'] = template_info
        model_obj= cls(config, **kwargs)


        # instance = cls(config) 
        if os.path.isdir(cloth_model_id):
            logging.info(colored("Loading pretrained cloth model weights from directory", "yellow"))
            #Get the model file
            model_checkpoint_file = config.path_config.checkpoint_file
            # Load the cloth model with the specified file
            if torch.cuda.is_available():
                model_checkpoint  = torch.load(model_checkpoint_file, weights_only=strict)
            else:
                logging.warning(colored("Loading model checkpoint on CPU", "red"))
                model_checkpoint  = torch.load(model_checkpoint_file, map_location=torch.device('cpu'), weights_only=strict)
            
            # Inspect the state dict loading
            cls._inspect_model_state_dict_with_model(model_obj, model_checkpoint['model_state_dict'])
            
            # Load the model state dictionary and restore the model to trained state
            model_obj.load_state_dict(model_checkpoint['model_state_dict'], strict=True)

            # Check if weights match
            if cls._check_weights_loaded(model_obj, model_checkpoint['model_state_dict']):
                logging.info(colored("Model restored to trained state.", "yellow"))
            else:
                logging.info(colored("Model not fully restored. Please check the checkpoint.", "yellow"))

        else:
            raise Exception(f"Path {cloth_model_id} is not a directory")
        if config.device is None:
            config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_obj.to(config.device)
        model_obj.eval()
        cls.model = model_obj
        return model_obj
    

    @classmethod
    def _inspect_model_state_dict_with_model(cls, model, checkpoint_dict):
        """
        Inspect the state dictionary of the model and compare it with the checkpoint dictionary.
        """
        # Get the set of keys from the model's state dictionary
        model_keys = set(model.state_dict().keys())
        # Get the set of keys from the checkpoint dictionary
        ckpt_keys = set(checkpoint_dict.keys())

        # Calculate missing keys (present in model but not in checkpoint)
        missing_keys = model_keys - ckpt_keys
        # Calculate unexpected keys (present in checkpoint but not in model)
        unexpected_keys = ckpt_keys - model_keys

        # Log the number of keys in the model and checkpoint
        print(f"\n✅ Model keys: {len(model_keys)}")
        print(f"📦 Checkpoint keys: {len(ckpt_keys)}")

        # Log missing keys if any
        if missing_keys:
            logging.info(colored(f"❌ Missing keys in checkpoint: {missing_keys}", "red"))
        # Log unexpected keys if any
        if unexpected_keys:
            logging.info(colored(f"⚠️ Unexpected keys in checkpoint: {unexpected_keys}", "yellow"))
        # Log success message if no missing or unexpected keys
        if not missing_keys and not unexpected_keys:
            logging.info(colored("✅ Model dict and model object keys match!", "green"))
    
    @classmethod
    def _check_weights_loaded(cls, model, state_dict):
        """
        Check if all weights from the model's state dictionary are present in the loaded state dictionary.
        """
        # Get the set of keys from the model's state dictionary
        model_keys = set(model.state_dict().keys())
        # Get the set of keys from the loaded state dictionary
        loaded_keys = set(state_dict.keys())

        # Calculate missing keys (present in model but not in the loaded state dictionary)
        missing_keys = model_keys - loaded_keys

        # Return True if no missing keys, otherwise False
        return not missing_keys



class ClothMeshGATModel(PreTrainedClothModel):
    """
        Template-based Mass-spring Cloth GNN
    """
    config_class = MeshGATConfig
    name         = "mesh_gat"

    def __init__(self, 
                 config: MeshGATConfig,
                 template_info: dict = None,
                 message_passing_steps: int = 15,
                 ):
        super().__init__(config)

        # init template_info
        if template_info is None:
            if not os.path.exists(config.path_config.template_dir):
                raise FileNotFoundError(f"Template file not found: {config.path_config.template_dir}")
            with open(config.path_config.template_dir, mode='rb') as f:
                template_info = pickle.load(f)
        
        self.template_info = template_info
        self.template_mesh_pos = self.template_info['mesh_pos']
        self.template_edge_idx = self.template_info['edge_idx'].astype(int)

        # init backbone ResNet model
        self.backbone_model = resnet_model.get_model('resnet34')

        # init learned GAT model
        self.learned_model = gat_model.GATModel(
            output_size=3,
            latent_size=128,
            num_layers=2,
            message_passing_steps=message_passing_steps
        )

    # build template_graph with template_nodes, edge_senders, and edge_receivers
    def _build_template_graph(self):
        # init node features as template mesh positions: N (num of nodes) x 3 (positions)
        if torch.cuda.is_available():
            node_features = torch.from_numpy(self.template_mesh_pos).float().cuda()
        else:
            logging.warning(colored("Loading node features on CPU", "red"))
            node_features = torch.from_numpy(self.template_mesh_pos).float()

      
        # create two-way connectivity of edge senders and receivers
        senders = self.template_edge_idx[:, 0]
        receivers = self.template_edge_idx[:, 1]

        if torch.cuda.is_available():
            sender = torch.from_numpy(np.concatenate([senders, receivers], 0)).to(torch.int64).cuda()
            receiver = torch.from_numpy(np.concatenate([receivers, senders], 0)).to(torch.int64).cuda()
        else:
            logging.warning(colored("Loading senders and receivers on CPU", "red"))
            sender = torch.from_numpy(np.concatenate([senders, receivers], 0)).to(torch.int64)
            receiver = torch.from_numpy(np.concatenate([receivers, senders], 0)).to(torch.int64)
        
        # assign edge features as edge vertices' relative coordinate and norm: E (num of edges) x 4 (vecter + norm)
        relative_edge_vector = (torch.index_select(node_features, 0, sender) - torch.index_select(node_features, 0, receiver))
        relative_edge_norm = torch.norm(relative_edge_vector, dim=-1, keepdim=True)
        edge_features = torch.cat((relative_edge_vector, relative_edge_norm), -1)

        # return gat model with node features and mesh edges
        mesh_edges = gat_model.EdgeSet(
            name='mesh_edges',
            features=edge_features,
            receivers=receiver,
            senders=sender
        )

        return gat_model.GraphSet(node_features=node_features, edge_set=mesh_edges)

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
    
    # @staticmethod
    # def predict(model: nn.Module, 
    #             input_data: np.ndarray) -> torch.Tensor:
    #     # Resize and normalize input data
    #     transform = ReshapeNormalizeImage()
    #     input_data = transform(input_data)
    #     input_data = torch.from_numpy(input_data).float().cuda()
    #     input_data = input_data.unsqueeze(0)

    #     print(f"input_data shape: {input_data.shape}")
    #     with torch.no_grad():
    #         pred_mesh = model(input_data)
    #     return pred_mesh
    def predict(self, input_data: np.ndarray) -> torch.Tensor:
        """
        Predicts the mesh for the given input image.
        
        Args:
            input_data (np.ndarray): Input image array (H x W x C or similar).
        
        Returns:
            torch.Tensor: Predicted mesh tensor of shape (1, vertex_count, 3).
        """

        # Resize and normalize input data
        transform = ReshapeNormalizeImage()
        input_data = transform(input_data)

        # Convert to torch tensor
        input_tensor = torch.from_numpy(input_data).float().to(self.config.device)
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

        print(f"Input tensor shape: {input_tensor.shape}")

        # Run inference
        self.eval()
        with torch.no_grad():
            pred_mesh = self.forward(input_tensor)

        return pred_mesh


import cv2
class ReshapeNormalizeImage:
    """
        Resize sample image to (224, 224), normalize to (0, 1)
    """
    def __init__(self, image_size=(224, 224)):
        self.image_size = tuple(image_size)

    def __call__(self, sample):
        sample = np.asarray(cv2.resize(sample, self.image_size).transpose(2, 0, 1) / 255.)
        c, h, w = sample.shape
        return sample
    




# Example usage
if __name__ == "__main__":
    import json
    import argparse
    # Create an instance of the MeshGATConfig class
    mesh_gat_config = MeshGATConfig(
        mode="train",
        batch_size=64,
        lr=1e-3,
        name_cloth="t_shirt_l3",
        distributed=True,
    )
    
    print(f"MeshGATConfig instance: {mesh_gat_config}")
    # #Read the "config.yaml" generated from training the cloth model and create a config_dict
    config_dir = mesh_gat_config.path_config.dataset_dir/ "configs"
    # # Parse the type argument from the command line
    parser = argparse.ArgumentParser(description="Load configuration for cloth model.")
    parser.add_argument("--cloth_model_type", type=str, default="mesh_gat", help="Type of the cloth model (e.g., mesh_gat).")
    args = parser.parse_args()

     #Example of slecting MeshGATConfig from the parent calss PreTrainedClothModelConfig
    # Load the saved configuration using the from_pretrained_cloth_model method
    pretrained_path = config_dir
    cli_overrides = {"type": args.cloth_model_type}
    config = PreTrainedClothModelConfig.from_pretrained_cloth_model(pretrained_path, cli_overrides=cli_overrides)
    print(f"Loaded config: {config}")


    # Load the model
    import pickle
    from mpl_toolkits.mplot3d import Axes3D
    template_info = pickle.load(open(config.path_config.template_dir, mode='rb'))
    model = ClothMeshGATModel(config)
    
    cli_overrides_model = {"type": args.cloth_model_type}
    kwargs = {}
    # kwargs["cli_overrides"] = '"type": args.type'
    model_obj = model.from_pretrained_cloth_model(pretrained_path=config_dir,
                                      config=config,                                     
                                    **kwargs)

    # Use kwargs to pass pretrained_path
    kwargs = {}
    kwargs["pretrained_path"]  = config_dir
    kwargs["config"]           = config

    model = ClothMeshGATModel.from_pretrained_cloth_model(**kwargs)

    # Load the input data
    TEST_DIR = '/home/dips/Documents/datasets_lerobot/so100_test/mesh_gat/t_shirt_l3/test/real'
    input_depth_image_path = Path(TEST_DIR) / '000020.depth.png'  # Example input data
    input_depth_image = cv2.imread(str(input_depth_image_path), cv2.IMREAD_COLOR)
    if input_depth_image is None:
        raise FileNotFoundError(f"Image not found at path: {input_depth_image_path}")
    pred_mesh = model.predict(input_depth_image)
    print(f"Predicted mesh shape: {pred_mesh.shape}")

    # Plot the predicted mesh and input depth image

    def plot_mesh_and_image(pred_mesh, input_image):
        fig = plt.figure(figsize=(12, 6))

        # Plot the input depth image
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(input_image)
        ax1.set_title("Input Depth Image")
        ax1.axis("off")

        # Plot the predicted mesh (top view)
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.scatter(pred_mesh[0, :, 0].cpu().numpy(),
                pred_mesh[0, :, 1].cpu().numpy(), s=1)
        ax2.set_title("Predicted Mesh (Top View)")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")

        plt.tight_layout()
        plt.show()

    # Call the function to plot
    plot_mesh_and_image(pred_mesh, input_depth_image)



    

