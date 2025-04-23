from pprint import pprint

import torch
from huggingface_hub import HfApi

import lerobot
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

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
from pathlib import Path

from lerobot.common.datasets.transforms import ImageTransforms
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig

def resolve_delta_timestamps(
    cfg: PreTrainedConfig, ds_meta: LeRobotDatasetMetadata
) -> dict[str, list] | None:
    """Resolves delta_timestamps by reading from the 'delta_indices' properties of the PreTrainedConfig.

    Args:
        cfg (PreTrainedConfig): The PreTrainedConfig to read delta_indices from.
        ds_meta (LeRobotDatasetMetadata): The dataset from which features and fps are used to build
            delta_timestamps against.

    Returns:
        dict[str, list] | None: A dictionary of delta_timestamps, e.g.:
            {
                "observation.state": [-0.04, -0.02, 0]
                "observation.action": [-0.02, 0, 0.02]
            }
            returns `None` if the the resulting dict is empty.
    """
    delta_timestamps = {}
    for key in ds_meta.features:
        if key == "next.reward" and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
        if key == "action" and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        if key.startswith("observation.") and cfg.observation_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]

    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps


#In the training script, the main function train expects a TrainPipelineConfig object
@parser_dips.wrap(config_path=Path("/home/dips/Documents/datasets_lerobot/so100_test/train_config/train_config.json"))
def generate_dataset(cfg: TrainPipelineConfig):

    ###############CREATE DATASET(IF NOT AVAILABLE) FROM CFG######################
    logging.info("Creating dataset")
    ######dataset = make_dataset(cfg) part

    # Check if the dataset repository ID is a single string (indicating a single dataset)
    if isinstance(cfg.dataset.repo_id, str):
        # Create metadata for the dataset using the repository ID, root directory, and revision
        ds_meta = LeRobotDatasetMetadata(
            cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
        )
        
        # Resolve delta timestamps based on the policy configuration and dataset metadata
        delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
        
        # Create a single LeRobotDataset instance with the resolved parameters
        dataset_default = LeRobotDataset(
            cfg.dataset.repo_id,  # Repository ID of the dataset
            root=cfg.dataset.root,  # Root directory where the dataset is stored
            episodes=cfg.dataset.episodes,  # Number of episodes to load
            delta_timestamps=delta_timestamps,  # Delta timestamps for temporal features
            image_transforms=None,  # Image transformations to apply
            revision=cfg.dataset.revision,  # Dataset revision to use
            video_backend=cfg.dataset.video_backend,  # Backend for video processing
        )
    #     repo_id: str,  # Repository ID for the dataset.
    #     fps: int,  # Frames per second used during data collection.
    #     root: str | Path | None = None,  # Root directory for the dataset.
    #     robot: Robot | None = None,  # Robot instance to extract features from (optional).
    #     robot_type: str | None = None,  # Type of robot used for data collection (optional).
    #     features: dict | None = None,  # Dictionary of dataset features (optional).
    #     use_videos: bool = True,  # Whether to use videos for visual modalities.
    #     tolerance_s: float = 1e-4,  # Tolerance in seconds for timestamp synchronization.
    #     image_writer_processes: int = 0,  # Number of processes for the image writer (optional).
    #     image_writer_threads: int = 0,  # Number of threads for the image writer (optional).
    #     video_backend: str | None = None,  # Video backend to use for decoding videos (optional).
        dataset_custom = LeRobotDataset.create(
            repo_id = 'so100_test_2',  # Repository ID of the dataset
            fps = 50,  # Frames per second used during data collection
            root = '/home/dips/Documents/datasets_lerobot/so100_test_2',  # Root directory where the dataset is stored
            robot_type = 'Denso', # Type of robot used for data collection (optional),
            features = {} # Dictionary of dataset features (optional)
        )
        logging.info("Dataset created")


    

if __name__ == "__main__":
  
    config_args = [
        "--dataset_repo_id=lerobot/so_100_tele_op_cloth_flatening",
        "--policy_type=act",
        "--output_dir=outputs/train/act_so_100_test",
        "--job_name=act_so_100_test",
        "--device=cuda",
        "--wandb_enable=true",
    ]
    
    # Initialize logging and run the training
    init_logging()
    # parser.parse_args(config_args)
    generate_dataset()
    # train()



    

