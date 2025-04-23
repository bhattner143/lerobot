#!/usr/bin/env python

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
import logging  # For logging messages and warnings
import os  # For interacting with the operating system
import re  # For regular expression operations
from glob import glob  # For file pattern matching
from pathlib import Path  # For working with filesystem paths

from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE  # Constant for safe tensor file
from termcolor import colored  # For colored terminal output

from lerobot.common.constants import PRETRAINED_MODEL_DIR  # Directory for pretrained models
from lerobot.configs.train import TrainPipelineConfig  # Configuration class for training pipeline


def cfg_to_group(cfg: TrainPipelineConfig, return_list: bool = False) -> list[str] | str:
    """Return a group name for logging. Optionally returns group name as list."""
    lst = [
        f"policy:{cfg.policy.type}",  # Policy type from the configuration
        f"dataset:{cfg.dataset.repo_id}",  # Dataset repository ID
        f"seed:{cfg.seed}",  # Random seed value
    ]
    return lst if return_list else "-".join(lst)  # Return as list or joined string


def get_wandb_run_id_from_filesystem(log_dir: Path) -> str:
    # Get the WandB run ID from the filesystem.
    paths = glob(str(log_dir / "wandb/latest-run/run-*"))  # Find matching run files
    if len(paths) != 1:  # Ensure exactly one matching file is found
        raise RuntimeError("Couldn't get the previous WandB run ID for run resumption.")
    match = re.search(r"run-([^\.]+).wandb", paths[0].split("/")[-1])  # Extract run ID using regex
    if match is None:  # Ensure a match is found
        raise RuntimeError("Couldn't get the previous WandB run ID for run resumption.")
    wandb_run_id = match.groups(0)[0]  # Extract the first group (run ID)
    return wandb_run_id


def get_safe_wandb_artifact_name(name: str):
    """WandB artifacts don't accept ":" or "/" in their name."""
    return name.replace(":", "_").replace("/", "_")  # Replace invalid characters with underscores


class WandBLogger:
    """A helper class to log objects using wandb."""

    def __init__(self, cfg: TrainPipelineConfig):
        self.cfg = cfg.wandb  # WandB-specific configuration
        self.log_dir = cfg.output_dir  # Directory for logs
        self.job_name = cfg.job_name  # Name of the job
        self.env_fps = None  # Frames per second for video logging
        self._group = cfg_to_group(cfg)  # Group name for logging

        # Set up WandB.
        os.environ["WANDB_SILENT"] = "True"  # Suppress WandB output
        import wandb  # Import WandB library

        wandb_run_id = (
            cfg.wandb.run_id  # Use provided run ID if available
            if cfg.wandb.run_id
            else get_wandb_run_id_from_filesystem(self.log_dir)  # Retrieve from filesystem if resuming
            if cfg.resume
            else None  # No run ID if not resuming
        )
        wandb.init(
            id=wandb_run_id,  # Initialize WandB with the run ID
            project=self.cfg.project,  # Project name
            entity=self.cfg.entity,  # Entity (team or user)
            name=self.job_name,  # Job name
            notes=self.cfg.notes,  # Additional notes
            tags=cfg_to_group(cfg, return_list=True),  # Tags for grouping
            dir=self.log_dir,  # Directory for WandB logs
            config=cfg.to_dict(),  # Configuration dictionary
            save_code=False,  # Disable saving code (can be enabled later)
            job_type="train_eval",  # Job type (e.g., training and evaluation)
            resume="must" if cfg.resume else None,  # Resume mode if applicable
            mode=self.cfg.mode if self.cfg.mode in ["online", "offline", "disabled"] else "online",  # WandB mode
        )
        print(colored("Logs will be synced with wandb.", "blue", attrs=["bold"]))  # Inform user about logging
        logging.info(f"Track this run --> {colored(wandb.run.get_url(), 'yellow', attrs=['bold'])}")  # Log WandB URL
        self._wandb = wandb  # Store WandB instance

    def log_policy(self, checkpoint_dir: Path):
        """Checkpoints the policy to wandb."""
        if self.cfg.disable_artifact:  # Skip if artifact logging is disabled
            return

        step_id = checkpoint_dir.name  # Use checkpoint directory name as step ID
        artifact_name = f"{self._group}-{step_id}"  # Create artifact name
        artifact_name = get_safe_wandb_artifact_name(artifact_name)  # Sanitize artifact name
        artifact = self._wandb.Artifact(artifact_name, type="model")  # Create WandB artifact
        artifact.add_file(checkpoint_dir / PRETRAINED_MODEL_DIR / SAFETENSORS_SINGLE_FILE)  # Add model file
        self._wandb.log_artifact(artifact)  # Log artifact to WandB

    def log_dict(self, d: dict, step: int, mode: str = "train"):
        if mode not in {"train", "eval"}:  # Ensure mode is valid
            raise ValueError(mode)

        for k, v in d.items():  # Iterate over dictionary items
            if not isinstance(v, (int, float, str)):  # Skip unsupported types
                logging.warning(
                    f'WandB logging of key "{k}" was ignored as its type is not handled by this wrapper.'
                )
                continue
            self._wandb.log({f"{mode}/{k}": v}, step=step)  # Log key-value pair with step

    def log_video(self, video_path: str, step: int, mode: str = "train"):
        if mode not in {"train", "eval"}:  # Ensure mode is valid
            raise ValueError(mode)

        wandb_video = self._wandb.Video(video_path, fps=self.env_fps, format="mp4")  # Create WandB video object
        self._wandb.log({f"{mode}/video": wandb_video}, step=step)  # Log video to WandB
