import abc
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Type, TypeVar
from utils.utils import auto_select_torch_device, is_torch_device_available #Change this to general one which is inside comon folder
from enum import Enum

import draccus
from termcolor import colored
from omegaconf import OmegaConf

# set current task
mode = "train" # select mode from 'train', 'test', 'test_real'
name_cloth = 't_shirt_l3'
checkpoint_file = '2025-03-28/14-46-38/finalbestmodel_0299_0.01162.pt'

# get address
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR     = Path(f'/home/dips/Documents/datasets_lerobot/so100_test/mesh_gat/{name_cloth}')
PREDICT_DIR     = DATASET_DIR / 'predict'
CHECKPOINT_DIR  = DATASET_DIR / 'checkpoints'
CHECKPOINT_FILE = CHECKPOINT_DIR / checkpoint_file
TEMPLATE_DIR    = DATASET_DIR / f'configs/template_{name_cloth}.pickle'


class FeatureType(str, Enum):
    STATE  = "STATE"
    VISUAL = "VISUAL"


@dataclass
class ClothModelFeature:
    type: FeatureType
    shape: tuple

# Generic variable that is either PreTrainedConfig or a subclass thereof
T = TypeVar("T", bound="PreTrainedClothModelConfig")

###########ABSTRACT CLASS FOR PRETRAINED CLOTH MODEL CONFIG###########
@dataclass
class PreTrainedClothModelConfig(draccus.ChoiceRegistry, abc.ABC):
    
    """Base Abstract class for pretrained cloth model configs."""
    input_features: dict[str, ClothModelFeature]  = field(default_factory=dict)
    output_features: dict[str, ClothModelFeature] = field(default_factory=dict)
    
    device: str | None = None  # cuda | cpu | mp

    project_dir: dict = field(default_factory=dict)
    dataset_dir: dict = field(default_factory=dict)
    predict_dir: dict = field(default_factory=dict)
    checkpoint_dir: dict = field(default_factory=dict)
    checkpoint_file: dict = field(default_factory=dict)

    def __post_init__(self):
        # Set the device to the appropriate one based on the environment
        if not self.device or not is_torch_device_available(self.device):
            self.device = auto_select_torch_device()
            logging.info(f"Device set to {self.device}")

   
    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)
    
    @classmethod
    def from_pretrained_cloth_model(
        cls: Type[T],
        pretrained_path: str | Path,
        cli_overrides: dict[str, str] | None = None,
    ) -> T:
        model_id = str(pretrained_path)
        config_file: str | None = None
        if Path(model_id).is_dir():
            # Check if the configuration file exists in the directory
            if "config.json" in os.listdir(model_id):
                config_model_file = os.path.join(model_id, "config.json")
            else:
                logging.warning(f"config.json not found in {Path(model_id).resolve()}")
        else:
            raise Exception(f"Path {model_id} is not a directory")

        # Load the configuration using draccus
        config = draccus.parse(cls, config_path=config_model_file)

        # Apply any command-line overrides using draccus
        if cli_overrides:
            config = draccus.parse(
            cls,
            config_path=config_model_file,
            args=[f"--{key}={value}" for key, value in cli_overrides.items()]
            )

        return config


@PreTrainedClothModelConfig.register_subclass("mesh_gat")
@dataclass
class MeshGATConfig(PreTrainedClothModelConfig):
    """Configuration for MeshGAT model."""
    type: str = "mesh_gat"
    mode: str = "eval"
    name_cloth: str  = name_cloth
    project_dir: str = field(default_factory=lambda: Path(PROJECT_DIR))
    dataset_dir: str = field(default_factory=lambda: Path(DATASET_DIR))
    predict_dir: str = field(default_factory=lambda: Path(PREDICT_DIR))
    checkpoint_dir: str = field(default_factory=lambda: Path(CHECKPOINT_DIR))
    checkpoint_file: str = field(default_factory=lambda: Path(CHECKPOINT_FILE))
    template_dir: str = field(default_factory=lambda: Path(TEMPLATE_DIR))

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





# Example usage
if __name__ == "__main__":
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