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
from pprint import pformat

import torch

from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata
)
from lerobot.common.datasets.transforms import ImageTransforms
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig

IMAGENET_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],  # (c,1,1)
    "std": [[[0.229]], [[0.224]], [[0.225]]],  # (c,1,1)
}


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


def make_dataset(cfg: TrainPipelineConfig) -> LeRobotDataset:
    """Handles the logic of setting up delta timestamps and image transforms before creating a dataset.

    Args:
        cfg (TrainPipelineConfig): A TrainPipelineConfig config which contains a DatasetConfig and a PreTrainedConfig.

    Raises:
        NotImplementedError: The MultiLeRobotDataset is currently deactivated.

    Returns:
        LeRobotDataset | MultiLeRobotDataset
    """
    # Initialize image transforms if enabled in the dataset configuration
    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )
    # Create metadata for the dataset using the repository ID, root directory, and revision
    # It loads the info, stats(v2.0)/episodes_stats(v2.1), episodes, and tasks files from the meta directory
    ds_meta = LeRobotDatasetMetadata(
        cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
    )
    
    # Resolve delta timestamps based on the policy configuration and dataset metadata
    delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
    
    # Create a single LeRobotDataset instance with the resolved parameters
    dataset = LeRobotDataset(
        cfg.dataset.repo_id,  # Repository ID of the dataset
        root=cfg.dataset.root,  # Root directory where the dataset is stored
        episodes=cfg.dataset.episodes,  # Number of episodes to load
        delta_timestamps=delta_timestamps,  # Delta timestamps for temporal features
        image_transforms=image_transforms,  # Image transformations to apply
        revision=cfg.dataset.revision,  # Dataset revision to use
        video_backend=cfg.dataset.video_backend,  # Backend for video processing
    )

    # If ImageNet statistics are enabled in the configuration, apply them to the dataset
    if cfg.dataset.use_imagenet_stats:
        for key in dataset.meta.camera_keys:  # Iterate over all camera keys in the dataset metadata
            for stats_type, stats in IMAGENET_STATS.items():  # Iterate over mean and std statistics
                # Update the dataset metadata with ImageNet statistics for each camera key
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    # Return the constructed dataset (either single or multi-dataset)
    return dataset
