import abc
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Type, TypeVar

try:
    from utils.utils import auto_select_torch_device, is_torch_device_available #Change this to general one which is inside comon folder
except ImportError:
    from lerobot.common.utils.utils import auto_select_torch_device, is_torch_device_available

from enum import Enum

import draccus
from termcolor import colored
from omegaconf import OmegaConf
import argparse

# set current task
mode = "train" # select mode from 'train', 'test', 'test_real'
name_cloth = 't_shirt_l3'
checkpoint_file = 'finalbestmodel_0299_0.01162.pt'

# get address
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR     = Path(f'/home/dips/Documents/datasets_lerobot/so100_test/mesh_gat/{name_cloth}')
PREDICT_DIR     = DATASET_DIR / 'predict'
CHECKPOINT_DIR  = DATASET_DIR / 'checkpoints'
CHECKPOINT_FILE = CHECKPOINT_DIR / checkpoint_file
TEMPLATE_DIR    = DATASET_DIR / f'configs/template_{name_cloth}.pickle'

@dataclass
class PathConfig:
    project_dir: Path
    dataset_dir: Path
    predict_dir: Path
    checkpoint_dir: Path
    checkpoint_file: Path
    template_dir: Path

    @classmethod
    def from_dataset(cls, root: Path, name_cloth: str, checkpoint_file_name: str) -> "PathConfig":
        dataset_dir = root / name_cloth
        return cls(
            project_dir=Path(__file__).resolve().parent,
            dataset_dir=dataset_dir,
            predict_dir=dataset_dir / "predict",
            checkpoint_dir=dataset_dir / "checkpoints",
            checkpoint_file=dataset_dir / "checkpoints" / checkpoint_file_name,
            template_dir=dataset_dir / f"configs/template_{name_cloth}.pickle",
        )

class FeatureType(str, Enum):
    STATE  = "STATE"
    VISUAL = "VISUAL"


@dataclass
class ClothMeshGATModelFeature:
    type: FeatureType
    shape: tuple

# Generic variable that is either PreTrainedConfig or a subclass thereof
T = TypeVar("T", bound="PreTrainedClothModelConfig")

###########ABSTRACT CLASS FOR PRETRAINED CLOTH MODEL CONFIG###########
@dataclass
class PreTrainedClothModelConfig(draccus.ChoiceRegistry, abc.ABC):
    
    """Base Abstract class for pretrained cloth model configs."""
    type: str = "base"  # Add the type field with a default value
    input_features: dict[str, ClothMeshGATModelFeature]  = field(default_factory=dict)
    output_features: dict[str, ClothMeshGATModelFeature] = field(default_factory=dict)
    
    device: str | None = None  # cuda | cpu | mp

    project_dir: dict = field(default_factory=dict)
    dataset_dir: dict = field(default_factory=dict)
    predict_dir: dict = field(default_factory=dict)
    checkpoint_dir: dict = field(default_factory=dict)
    checkpoint_file: dict = field(default_factory=dict)

    def __post_init__(self):
        # Set the device to the appropriate one based on the environment
        # if not self.device or not is_torch_device_available(self.device):
        #     self.device = auto_select_torch_device()
        #     logging.info(f"Device set to {self.device}")
        pass

   
    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)
    
    @classmethod
    def from_pretrained_cloth_model(
        cls: Type[T],
        pretrained_path: str | Path,
        cli_overrides: dict[str, str] | None = None,
        **kwargs,
    ) -> T:
        
        model_id = str(pretrained_path)
        config_file: str | None = None
        
        if Path(model_id).is_dir():
            # Check if the yaml configuration file exists in the directory
            if "config.yaml" in os.listdir(model_id):
                # Read the YAML configuration file and save it as a JSON file
                cls._read_config_yaml_to_save_config_json(Path(model_id))

                #Read the recently saved config.json
                config_path = Path(pretrained_path/ "config.json")
                
                cls.config = draccus.parse(
                                cls, #Configuration class of the selected model config (E.g. MeshGATConfig)
                                config_path=config_path,
                                args = [f"--{key}={value}" for key, value in cli_overrides.items()] # Pass the command
                                )
                print(f"Config type: {cls.config.type}")
                print(cls.config)
                
                # Save all the configuration, which includes mesh gan info as cloth_model_config.json using json
                cls._save_pretrained(pretrained_path)
                
            else:
                logging.warning(f"config.yaml not found in {Path(model_id).resolve()}")
        else:
            raise Exception(f"Path {model_id} is not a directory")

        return cls.config

    @classmethod
    def _read_config_yaml_to_save_config_json(cls, config_dir: str | Path) -> None:
        """Read the YAML configuration file and return it as a dictionary."""
        config_file_path = config_dir / "config.yaml"
        if config_file_path.exists():
            with open(config_file_path, "r") as f:
                config_dict = OmegaConf.to_container(OmegaConf.load(f), resolve=True)
            #Create the directory if it doesn't exist and save the config_dict as "config.json from yaml"
            config_dir.mkdir(exist_ok=True)
            with open(config_dir / "config.json", "w") as f:
                json.dump(config_dict, f)

            print(config_dir.resolve())
        else:
            raise FileNotFoundError(f"{config_file_path} does not exist.")
        
    @classmethod
    def _save_pretrained(cls, save_directory: Path) -> None:
        """Save the configuration to a file."""
        cloth_model_config_path = save_directory / "cloth_model_config.json"
        with open(cloth_model_config_path, "w") as f:
                json.dump(OmegaConf.to_container(OmegaConf.structured(cls.config), resolve=True), f, indent=4)
        print(colored(f"Configuration saved to {cloth_model_config_path}", "green"))

@PreTrainedClothModelConfig.register_subclass("mesh_gat")
@dataclass
class MeshGATConfig(PreTrainedClothModelConfig):
    """Configuration for MeshGAT model."""
    type: str = "mesh_gat"
    mode: str = "eval"
    name_cloth: str  = name_cloth
    # # device: str | None = None  # cuda | cpu 
    # project_dir: str = field(default_factory=lambda: Path(PROJECT_DIR))
    # dataset_dir: str = field(default_factory=lambda: Path(DATASET_DIR))
    # predict_dir: str = field(default_factory=lambda: Path(PREDICT_DIR))
    # checkpoint_dir: str = field(default_factory=lambda: Path(CHECKPOINT_DIR))
    # checkpoint_file: str = field(default_factory=lambda: Path(CHECKPOINT_FILE))
    # template_dir: str = field(default_factory=lambda: Path(TEMPLATE_DIR))

    distributed: bool = False
    # SYSTEM
    main_seed: int = 0

    # TRAIN
    batch_size: int = 32
    epoch_size: int = 2000
    image_size: int = 720
    lr: float = 1e-4
    schedule_step: int = 150
    momentum: float = 0.9
    sample_ratio: float = 1.0
    save_step: int = 30

    # DATALOADER
    num_threads: int = 16
    shuffle: bool = True
    drop_last: bool = False

    # PREDICT
    store_pred: bool = True

    message_passing_steps: int = 15

    # LOSS
    #declares a field called loss_weights of type dict, and assigns it a default value using a lambda function.
    # Directly defining with dict will make the value shared across all instances of the class, which is dangerous 
    # for mutable types like dict or list.
    loss_weights: dict = field(default_factory=lambda: {
        "vertex_loss": 1.0,
        "keypoint_loss": 1.0,
        "chamfer_loss": 0.5,
    })
    use_chamfer: bool = True
    chamfer_active_epoch: int = 0
    use_pixel: bool = False
    use_normalize: bool = False

    def __post_init__(self):
        super().__post_init__()

        # Initialize all paths using the utility class
        root_dataset_dir = Path("/home/dips/Documents/datasets_lerobot/so100_test/mesh_gat")
        self.path_config = PathConfig.from_dataset(
            root=root_dataset_dir,
            name_cloth=self.name_cloth,
            checkpoint_file=self.checkpoint_file_name,
        )

        self.project_dir = self.path_config.project_dir
        self.dataset_dir = self.path_config.dataset_dir
        self.predict_dir = self.path_config.predict_dir
        self.checkpoint_dir = self.path_config.checkpoint_dir
        self.checkpoint_file = self.path_config.checkpoint_file


# Example usage
if __name__ == "__main__":
    
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
    config_dir = DATASET_DIR / "configs"
    config_file_path = config_dir / "config.yaml"

    if config_file_path.exists():
        with open(config_file_path, "r") as f:
            config_dict = OmegaConf.to_container(OmegaConf.load(f), resolve=True)
    else:
        raise FileNotFoundError(f"{config_file_path} does not exist.")

    #Create the directory if it doesn't exist and save the config_dict as "config.json"
    config_dir.mkdir(exist_ok=True)
    with open(config_dir / "config.json", "w") as f:
        json.dump(config_dict, f)
    
    print(config_dir.resolve())
    
    
    # # Parse the type argument from the command line
    parser = argparse.ArgumentParser(description="Load configuration for cloth model.")
    parser.add_argument("--cloth_model_type", type=str, default="mesh_gat", help="Type of the cloth model (e.g., mesh_gat).")
    args = parser.parse_args()
    
    # Example of directly calling MeshGATConfig
    # Load the saved configuration using the from_pretrained_cloth_model method
    pretrained_path = config_dir
    cli_overrides = {"mode": "train", "batch_size": 64, "type": args.cloth_model_type}
    config = MeshGATConfig.from_pretrained_cloth_model(pretrained_path, cli_overrides=cli_overrides)
    print(f"Loaded config: {config}")

    #Example of slecting MeshGATConfig from the parent calss PreTrainedClothModelConfig
    # Load the saved configuration using the from_pretrained_cloth_model method
    pretrained_path = config_dir
    cli_overrides = {"type": args.cloth_model_type}
    config = PreTrainedClothModelConfig.from_pretrained_cloth_model(pretrained_path, cli_overrides=cli_overrides)
    print(f"Loaded config: {config}")


    # Example of using the factory method to create a specific cloth model config
    def cloth_model_factory(model_type: str, **kwargs) -> PreTrainedClothModelConfig:
        if model_type == "mesh_gat":
            return MeshGATConfig(**kwargs)
        elif model_type == "dummy":
            return DummyClothMeshGATModelConfig(**kwargs)
        else:
            raise ValueError(f"Unknown cloth model type: {model_type}")
        
    cloth_model_config = cloth_model_factory(
                        args.cloth_model_type,
                        )
