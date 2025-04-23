import logging
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any

import torch
from termcolor import colored
from torch.amp import GradScaler
from torch.optim import Optimizer

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.sampler import EpisodeAwareSampler
from lerobot.common.datasets.utils import cycle
from lerobot.common.envs.factory import make_env
from lerobot.common.optim.factory import make_optimizer_and_scheduler
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
from lerobot.common.utils.wandb_utils import WandBLogger
from lerobot.configs import parser_dips
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.eval import eval_policy
import argparse

from pathlib import Path  # For handling file system paths

from dataclasses import dataclass, field  # For creating data classes
# from lerobot.configs.default import DatasetConfig, EvalConfig, WandBConfig  # Default configurations
# from lerobot.configs.policies import PreTrainedConfig  # Pre-trained policy configuration

@dataclass
class DatasetConfig:
    repo_id: str
    episodes: list[int] | None = None
    video_backend: str = "pyav"

@dataclass
class PreTrainedConfig:
    type: str = "act"
    checkpoint_path: str | None = None
    device: str = "cuda"

@dataclass
class TrainPipelineConfig:
    # Dataset configuration
    dataset: DatasetConfig
    # Environment configuration (optional)
    # env: envs.EnvConfig | None = None
    # Pre-trained policy configuration (optional)
    policy: PreTrainedConfig | None = None
    # Directory to save all outputs of the training run
    output_dir: Path | None = None
    # Name of the training job
    job_name: str | None = None
    # Whether to resume a previous training run
    resume: bool = False
    # Random seed for reproducibility
    seed: int | None = 1000
    # Number of workers for the data loader
    num_workers: int = 4
    # Batch size for training
    batch_size: int = 8
    # Total number of training steps
    steps: int = 100_000
    # Frequency of evaluation during training
    eval_freq: int = 20_000
    # Frequency of logging during training
    log_freq: int = 200
    # Whether to save checkpoints during training
    save_checkpoint: bool = True
    # Frequency of saving checkpoints
    save_freq: int = 20_000
    # Whether to use policy training presets
    use_policy_training_preset: bool = True
    # Optimizer configuration (optional)
    # optimizer: OptimizerConfig | None = None
    # Scheduler configuration (optional)
    # scheduler: LRSchedulerConfig | None = None
    # Evaluation configuration
    # eval: EvalConfig = field(default_factory=EvalConfig)
    # Weights and Biases (WandB) configuration
    # wandb: WandBConfig = field(default_factory=WandBConfig)


# def main():
#     # Create a sample dataset configuration
#     dataset_config = DatasetConfig(
#         repo_id="lerobot/so_100_tele_op_cloth_flatening",
#         episodes=[1, 2, 3],
#         video_backend="pyav"
#     )

#     # Create a sample pre-trained policy configuration
#     policy_config = PreTrainedConfig(
#         type="act",
#         checkpoint_path="/path/to/checkpoint",
#         device="cuda"
#     )

#     # Combine into a training pipeline configuration
#     train_pipeline_config = TrainPipelineConfig(
#         dataset=dataset_config,
#         policy=policy_config
#     )

#     # Print the configuration to verify
#     print(train_pipeline_config)


#In the training script, the main function train expects a TrainPipelineConfig object
@parser_dips.wrap()
def train(cfg: TrainPipelineConfig):
    cfg.validate()
    logging.info(pformat(cfg.to_dict()))

    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed)

#In the training script, the main function train expects a TrainPipelineConfig object
@parser_dips.wrap()
def train(cfg: TrainPipelineConfig):
    cfg.validate()
    print(cfg.to_dict())  # or actual training logic


if __name__ == "__main__":
    # main()

    
    config_args = [
        "--dataset_repo_id=lerobot/so_100_tele_op_cloth_flatening",
        "--policy_type=act",
        "--output_dir=outputs/train/act_so_100_test",
        "--job_name=act_so_100_test",
        "--device=cuda",
        "--wandb_enable=true",
    ]

    # Call parse_arg to extract specific arguments
    dataset_repo_id = parser_dips.parse_arg("dataset_repo_id", config_args)
    policy_type = parser_dips.parse_arg("policy_type", config_args)
    output_dir = parser_dips.parse_arg("output_dir", config_args)
    job_name = parser_dips.parse_arg("job_name", config_args)
    device = parser_dips.parse_arg("device", config_args)
    wandb_enable = parser_dips.parse_arg("wandb_enable", config_args)

    # Print the extracted arguments
    print(f"Dataset Repo ID: {dataset_repo_id}")
    print(f"Policy Type: {policy_type}")
    print(f"Output Directory: {output_dir}")
    print(f"Job Name: {job_name}")
    print(f"Device: {device}")
    print(f"Weights & Biases Enabled: {wandb_enable}")

    # Pass the simulated CLI args to the wrapped function
    # train(cli_args=config_args)
     #Use parser_dips to parse the configuration and pass it to the train function
    cfg = parser_dips.parse_arg(config_args)

    train()