#!/usr/bin/env python

from dataclasses import dataclass, field

from lerobot.common.optim.optimizers import AdamWConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode

from lerobot.common.models_cloth.clothmodel_configs import*

import logging
from termcolor import colored  # For colored terminal output.

@PreTrainedConfig.register_subclass("act_mesh_gat")
@dataclass
class ACTMeshGATConfig(PreTrainedConfig):
    """Configuration class for the Action Chunking Transformers policy.

    Defaults are configured for training on bimanual Aloha tasks like "insertion" or "transfer".
    """

    # Input / output structure.
    
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

    # Architecture.
    # Pre-trained cloth model.
    cloth_model_config: str | PreTrainedClothModelConfig | None = None
    name_cloth:str| None = "t_shirt_l3"
    # cloth_model_config: PreTrainedClothModelConfig | None = None#= field(default_factory=PreTrainedClothModelConfig)
    # cloth_model_config_dummy: str
    # Vision backbone.
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: int = False
    # Transformer layers.
    pre_norm: bool = False
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    feedforward_activation: str = "relu"
    n_encoder_layers: int = 4
    # Note: Although the original ACT implementation has 7 for `n_decoder_layers`, there is a bug in the code
    # that means only the first layer is used. Here we match the original implementation by setting this to 1.
    # See this issue https://github.com/tonyzhaozh/act/issues/25#issue-2258740521.
    n_decoder_layers: int = 1
    # VAE.
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4

    # Inference.
    # Note: the value used in ACT when temporal ensembling is enabled is 0.01.
    temporal_ensemble_coeff: float | None = None

    # Training and loss computation.
    dropout: float = 0.1
    kl_weight: float = 10.0

    # Training preset
    optimizer_lr: float = 1e-5
    optimizer_weight_decay: float = 1e-4
    optimizer_lr_backbone: float = 1e-5

    @property
    def type(self) -> str:
        return "act_mesh_gat"

    def __post_init__(self):
        super().__post_init__()
        # Load pre-trained cloth model config.
        if self.cloth_model_config:
            config_dir = DATASET_ROOT /  self.name_cloth/ "configs" 
            pretrained_path = config_dir
            cli_overrides = {"type": self.cloth_model_config}
            self.cloth_model_config = PreTrainedClothModelConfig.from_pretrained_cloth_model(pretrained_path, 
                                                                                             cli_overrides=cli_overrides)
            logging.info(colored(f"Loaded cloth model config: {self.cloth_model_config}", "yellow", attrs=["bold"]))

        """Input validation (not exhaustive)."""
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )
        if self.temporal_ensemble_coeff is not None and self.n_action_steps > 1:
            raise NotImplementedError(
                "`n_action_steps` must be 1 when using temporal ensembling. This is "
                "because the policy needs to be queried every step to compute the ensembled action."
            )
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not handled yet. Got `nobs_steps={self.n_obs_steps}`"
            )
        logging.info(colored("Initialized --> ACTConfigAdvanced", "yellow", attrs=["bold"]))

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if not self.image_features and not self.env_state_feature:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

##########################
if __name__ == "__main__":
    # Example usage
    # Create an instance of the MeshGATConfig class
    mesh_gat_config = MeshGATConfig(
        mode="train",
        batch_size=64,
        lr=1e-3,
        name_cloth="t_shirt_l3",
        distributed=True,
    )
    config = ACTMeshGATConfig(cloth_model_config=mesh_gat_config)
    print(config)

    # # Parse the type argument from the command line
    parser = argparse.ArgumentParser(description="Load configuration for cloth model.")
    parser.add_argument("--cloth_model_type", type=str, default="mesh_gat", help="Type of the cloth model (e.g., mesh_gat).")
    args = parser.parse_args()

    #Example of slecting MeshGATConfig from the parent calss PreTrainedClothModelConfig
    # Load the saved configuration using the from_pretrained_cloth_model method
    config_dir = DATASET_ROOT / "configs"
    pretrained_path = config_dir
    cli_overrides = {"type": args.cloth_model_type}
    config = PreTrainedClothModelConfig.from_pretrained_cloth_model(pretrained_path, cli_overrides=cli_overrides)
    print(f"Loaded config: {config}")