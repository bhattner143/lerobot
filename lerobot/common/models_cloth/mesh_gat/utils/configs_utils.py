import argparse
from omegaconf import OmegaConf

def load_yaml(yaml_path):
    """Load configuration from a YAML file using OmegaConf."""
    return OmegaConf.load(yaml_path)

def get_args(PROJECT_DIR, name_cloth):
    # Load YAML configuration using OmegaConf
    address_yaml = f"{PROJECT_DIR}/configs/configs_{name_cloth}.yaml"
    config = load_yaml(address_yaml)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a CAT model")
    parser.add_argument('--batch_size', type=int, help="Batch size for training")
    args = parser.parse_args()

    # Convert argparse Namespace to a dictionary, filtering out None values
    cli_args = {k: v for k, v in vars(args).items() if v is not None}

    # Merge CLI arguments only if they exist
    if cli_args:
        cli_config = OmegaConf.create(cli_args)
        config = OmegaConf.merge(config, cli_config)

    return config