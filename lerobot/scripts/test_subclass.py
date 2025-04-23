# config_example.py

from dataclasses import dataclass, field
from typing import Optional, Type
import draccus
import torch
# ---- Base + Subclass Registry ----

@dataclass
class DatasetConfig:
    """
    Configuration for datasets used in training or evaluation.
    """
    repo_id: str
    root: str | None = None
    episodes: list[int] | None = None
    revision: str | None = None
    use_imagenet_stats: bool = True

@dataclass
# draccus.ChoiceRegistry allows dynamic registration and instantiation of subclasses based on a string key. 
# It is particularly useful for managing configurations where different types of objects 
# (e.g., policies, models, optimizers) need to be dynamically selected and instantiated 
# at runtime based on user input, such as command-line arguments or configuration files.
class PreTrainedConfig(draccus.ChoiceRegistry):  # Ensure it inherits from dataclass
    n_obs_steps: int = 1
    normalization_mapping: dict[str, str] = field(default_factory=dict)
    pass

@dataclass
class OptimizerConfig(draccus.ChoiceRegistry):
    lr: float
    weight_decay: float
    grad_clip_norm: float

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)
    
# The register_subclass decorator adds ACTConfig to an internal 
# registry in PreTrainedConfig with the key "act".
@PreTrainedConfig.register_subclass("act")
@dataclass
class ACTConfig(PreTrainedConfig):
    chunk_size: int = 100

@OptimizerConfig.register_subclass("adam")
@dataclass
class AdamConfig(OptimizerConfig):
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip_norm: float = 10.0

    # def build(self, params: dict) -> torch.optim.Optimizer:
    #     kwargs = asdict(self)
    #     kwargs.pop("grad_clip_norm")
    #     return torch.optim.Adam(params, **kwargs)


# ---- Top-level Config ----
@dataclass
class TrainPipelineConfig:
    # Dataset configuration
    dataset: DatasetConfig
    job_name: str = "default_training_job"
    policy: Optional[PreTrainedConfig] = None  # This will be ACTConfig if type=act
    # Optimizer configuration (optional)
    optimizer: OptimizerConfig | None = None

# ---- Simulate CLI args and parse ----
def main():
    cli_args = [
        "--dataset.repo_id=example_repo",
        "--job_name=training_with_act",
        "--policy.type=act",
        "--optimizer.type=adam",
    ]

    config = draccus.parse(
        config_class=TrainPipelineConfig,
        args=cli_args
    )

    print("Parsed Configuration:")
    print(f"Job Name: {config.job_name}")
    print(f"Dataset Repo ID: {config.dataset.repo_id}")
    print(f"Policy Type: {type(config.policy).__name__}")
    print(f"Optimizer Type: {type(config.optimizer).__name__}")
    if config.policy:
        print(f"Policy Details: {config.policy}")

if __name__ == "__main__":
    main()