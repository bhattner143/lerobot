import os
import argparse
from omegaconf import OmegaConf

def load_yaml(yaml_path):
    """Load configuration from a YAML file using OmegaConf."""
    return OmegaConf.load(yaml_path)

def get_args():
    # Load YAML configuration using OmegaConf
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    address_yaml = f"{project_dir}/configs/configs_steam_mesh.yaml"
    config = load_yaml(address_yaml)

    return config