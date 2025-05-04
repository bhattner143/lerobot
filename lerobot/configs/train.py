
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
from lerobot.configs import parser_dips  # Command-line argument parser
from lerobot.configs.default import DatasetConfig, EvalConfig, WandBConfig  # Default configurations
from lerobot.configs.policies import PreTrainedConfig  # Pre-trained policy configuration
# from lerobot.common.models_cloth.clothmodel_configs import PreTrainedModelConfig  # Pre-trained model configuration

import logging
from termcolor import colored

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
    #
    # cloth_model: PreTrainedModelConfig | None = None
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
        logging.info(colored("Initialized --> TrainPipelineConfig", "yellow", attrs=["bold"]))
        # Initialize the checkpoint path to None
        self.checkpoint_path = None

    ########## Validate the configuration#############
    def validate(self):
        """
        Validate the training configuration.
        1. Check if the dataset is set.
        """
        # Parse command-line arguments for the policy path
        policy_path = parser_dips.get_path_arg("policy")
        if policy_path:
            # Load the policy configuration from the specified path
            cli_overrides = parser_dips.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        elif self.resume:
            # Handle resuming a previous training run
            config_path = parser_dips.parse_arg("config_path")
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

    # # Load a configuration from a pre-trained model or file
    # @classmethod
    # def from_pretrained(
    #     cls: Type["TrainPipelineConfig"],
    #     pretrained_name_or_path: str | Path,
    #     *,
    #     force_download: bool = False,
    #     resume_download: bool = None,
    #     proxies: dict | None = None,
    #     token: str | bool | None = None,
    #     cache_dir: str | Path | None = None,
    #     local_files_only: bool = False,
    #     revision: str | None = None,
    #     **kwargs,
    # ) -> "TrainPipelineConfig":
    #     # Handle different input types for the pre-trained configuration
    #     model_id = str(pretrained_name_or_path)
    #     config_file: str | None = None
    #     if Path(model_id).is_dir():
    #         # Check if the configuration file exists in the directory
    #         if TRAIN_CONFIG_NAME in os.listdir(model_id):
    #             config_file = os.path.join(model_id, TRAIN_CONFIG_NAME)
    #         else:
    #             print(f"{TRAIN_CONFIG_NAME} not found in {Path(model_id).resolve()}")
    #     elif Path(model_id).is_file():
    #         # Use the provided file path
    #         config_file = model_id
    #     else:
    #         Exception(f"File {config_file} not found")

    #     # Parse the configuration file with optional command-line arguments
    #     cli_args = kwargs.pop("cli_args", [])
    #     cfg      = draccus.parse(cls, config_file, args=cli_args)

    #     return cfg
    
