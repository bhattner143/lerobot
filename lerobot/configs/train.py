# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Import necessary modules
import datetime as dt  # For handling date and time
import os  # For interacting with the operating system
from dataclasses import dataclass, field  # For creating data classes
from pathlib import Path  # For handling file system paths
from typing import Type  # For type annotations

import draccus  # For configuration parsing and serialization
from huggingface_hub import hf_hub_download  # For downloading from HuggingFace Hub
from huggingface_hub.errors import HfHubHTTPError  # For handling HuggingFace Hub errors

# Importing custom modules from the lerobot package
# from lerobot.common import envs  # Environment configurations
from lerobot.common.optim import OptimizerConfig  # Optimizer configuration
from lerobot.common.optim.schedulers import LRSchedulerConfig  # Scheduler configuration
from lerobot.common.utils.hub import HubMixin  # Mixin for HuggingFace Hub integration
from lerobot.configs import parser  # Command-line argument parser
from lerobot.configs.default import DatasetConfig, EvalConfig, WandBConfig  # Default configurations
from lerobot.configs.policies import PreTrainedConfig  # Pre-trained policy configuration

# Name of the training configuration file
TRAIN_CONFIG_NAME = "train_config.json"

# Define the main configuration class for the training pipeline
@dataclass
class TrainPipelineConfig(HubMixin):
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
    optimizer: OptimizerConfig | None = None
    # Scheduler configuration (optional)
    scheduler: LRSchedulerConfig | None = None
    # Evaluation configuration
    eval: EvalConfig = field(default_factory=EvalConfig)
    # Weights and Biases (WandB) configuration
    wandb: WandBConfig = field(default_factory=WandBConfig)

    # Post-initialization logic
    def __post_init__(self):
        # Initialize the checkpoint path to None
        self.checkpoint_path = None

    ########## Validate the configuration#############
    def validate(self):
        """
        Validate the training configuration.
        1. Check if the dataset is set.
        """
        # Parse command-line arguments for the policy path
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            # Load the policy configuration from the specified path
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        elif self.resume:
            # Handle resuming a previous training run
            config_path = parser.parse_arg("config_path")
            if not config_path:
                raise ValueError(
                    f"A config_path is expected when resuming a run. Please specify path to {TRAIN_CONFIG_NAME}"
                )
            if not Path(config_path).resolve().exists():
                raise NotADirectoryError(
                    f"{config_path=} is expected to be a local path. "
                    "Resuming from the hub is not supported for now."
                )
            policy_path = Path(config_path).parent
            self.policy.pretrained_path = policy_path
            self.checkpoint_path = policy_path.parent

        # Generate a job name if not provided
        if not self.job_name:
            self.job_name = f"{self.policy.type}"


        # Handle output directory conflicts
        if not self.resume and isinstance(self.output_dir, Path) and self.output_dir.is_dir():
            raise FileExistsError(
                f"Output directory {self.output_dir} already exists and resume is {self.resume}. "
                f"Please change your output directory so that {self.output_dir} is not overwritten."
            )
        elif not self.output_dir:
            # Generate a default output directory based on the current date and time
            now = dt.datetime.now()
            train_dir = f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}"
            self.output_dir = Path("outputs/train") / train_dir

        # Check for unsupported multi-dataset configurations
        if isinstance(self.dataset.repo_id, list):
            raise NotImplementedError("LeRobotMultiDataset is not currently implemented.")

        # Validate optimizer and scheduler configurations
        if not self.use_policy_training_preset and (self.optimizer is None or self.scheduler is None):
            raise ValueError("Optimizer and Scheduler must be set when the policy presets are not used.")
        elif self.use_policy_training_preset and not self.resume:
            # Use presets from the policy configuration
            self.optimizer = self.policy.get_optimizer_preset()
            self.scheduler = self.policy.get_scheduler_preset()

    # Define path fields for the parser
    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]

    # Convert the configuration to a dictionary
    def to_dict(self) -> dict:
        return draccus.encode(self)

    # Save the configuration to a file
    def _save_pretrained(self, save_directory: Path) -> None:
        with open(save_directory / TRAIN_CONFIG_NAME, "w") as f, draccus.config_type("json"):
            draccus.dump(self, f, indent=4)

    # Load a configuration from a pre-trained model or file
    @classmethod
    def from_pretrained(
        cls: Type["TrainPipelineConfig"],
        pretrained_name_or_path: str | Path,
        *,
        force_download: bool = False,
        resume_download: bool = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **kwargs,
    ) -> "TrainPipelineConfig":
        # Handle different input types for the pre-trained configuration
        model_id = str(pretrained_name_or_path)
        config_file: str | None = None
        if Path(model_id).is_dir():
            # Check if the configuration file exists in the directory
            if TRAIN_CONFIG_NAME in os.listdir(model_id):
                config_file = os.path.join(model_id, TRAIN_CONFIG_NAME)
            else:
                print(f"{TRAIN_CONFIG_NAME} not found in {Path(model_id).resolve()}")
        elif Path(model_id).is_file():
            # Use the provided file path
            config_file = model_id
        else:
            # Attempt to download the configuration from the HuggingFace Hub
            try:
                config_file = hf_hub_download(
                    repo_id=model_id,
                    filename=TRAIN_CONFIG_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except HfHubHTTPError as e:
                raise FileNotFoundError(
                    f"{TRAIN_CONFIG_NAME} not found on the HuggingFace Hub in {model_id}"
                ) from e

        # Parse the configuration file with optional command-line arguments
        cli_args = kwargs.pop("cli_args", [])
        cfg      = draccus.parse(cls, config_file, args=cli_args)

        return cfg
    
if __name__ == "__main__":

    # Create a sample DatasetConfig
    dataset_config = DatasetConfig(
        repo_id="datasets_lerobot/so100_test",
        root=Path("/path/to/dataset")
    )

    # Load the PreTrainedConfig using the `from_pretrained` method
    policy_config = PreTrainedConfig.from_pretrained(
        pretrained_name_or_path="/path/to/pretrained/policy",
        cli_overrides=["--n_obs_steps=5", "--use_amp=true"]
    )

    # Create a sample OptimizerConfig
    optimizer_config = OptimizerConfig(
        learning_rate=0.001,
        weight_decay=0.01
    )

    # Create a sample SchedulerConfig
    scheduler_config = LRSchedulerConfig(
        scheduler_type="linear",
        warmup_steps=1000
    )

    # Create a TrainPipelineConfig object
    train_config = TrainPipelineConfig(
        dataset=dataset_config,
        policy=policy_config,
        output_dir=Path("outputs/train/act_so100_test"),
        job_name="act_advanced_training",
        resume=False,
        seed=42,
        num_workers=8,
        batch_size=16,
        steps=50000,
        eval_freq=5000,
        log_freq=100,
        save_checkpoint=True,
        save_freq=10000,
        use_policy_training_preset=False,
        optimizer=optimizer_config,
        scheduler=scheduler_config,
        eval=EvalConfig(),
        wandb=WandBConfig(enable=False)
    )

    # Print the configuration
    print(train_config)
