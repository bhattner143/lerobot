from dataclasses import dataclass, field
import logging
from termcolor import colored
from pathlib import Path

# Assume these are defined elsewhere
class NormalizationMode:
    MEAN_STD = "mean_std"

class PreTrainedClothModelConfig:
    @classmethod
    def from_pretrained_cloth_model(cls, path, cli_overrides=None):
        # Dummy loader (replace with real logic)
        logging.info(f"Loading config from {path} with overrides {cli_overrides}")
        return cls()

# Assume this is your base class
class PreTrainedConfig:
    _registry = {}

    @classmethod
    def register_subclass(cls, name):
        def decorator(subclass):
            cls._registry[name] = subclass
            return subclass
        return decorator

    @classmethod
    def from_pretrained_cloth_model(cls, path, cli_overrides=None):
        model_type = cli_overrides.get("type")
        if model_type not in cls._registry:
            raise ValueError(f"Unknown config type: {model_type}")
        return cls._registry[model_type]()  # Optionally pass config path

    def __post_init__(self):
        pass

# Your actual config class
@PreTrainedConfig.register_subclass("act_mesh_gat")
@dataclass
class ACTMeshGATConfig(PreTrainedConfig):
    """Configuration class for the Action Chunking Transformers policy."""

    n_obs_steps: int = 1
    chunk_size: int = 100
    n_action_steps: int = 100

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    cloth_model_config: str | PreTrainedClothModelConfig | None = None

    def __post_init__(self):
        super().__post_init__()

        if isinstance(self.cloth_model_config, str):
            config_dir = Path("DATASET_DIR/configs")  # Replace with actual DATASET_DIR
            cli_overrides = {"type": self.cloth_model_config}

            self.cloth_model_config = PreTrainedClothModelConfig.from_pretrained_cloth_model(
                config_dir,
                cli_overrides=cli_overrides
            )

            logging.info(colored(f"Loaded cloth model config: {self.cloth_model_config}", "yellow", attrs=["bold"]))


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Example usage
    config = ACTMeshGATConfig(cloth_model_config="example_model")
    print(config)
    print(config.cloth_model_config)
    print(config.normalization_mapping)