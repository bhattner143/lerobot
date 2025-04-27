#!/usr/bin/env python

from dataclasses import dataclass, field

from lerobot.common import (
    policies,  # noqa: F401 - Imported for potential future use, even if unused currently.
)
from lerobot.common.datasets.transforms import ImageTransformsConfig  # Configuration for image transformations.
from lerobot.common.datasets.video_utils import get_safe_default_codec  # Utility to get a safe default video codec.

import logging
from termcolor import colored  # For colored terminal output.

@dataclass
class DatasetConfig:
    """
    Configuration for datasets used in training or evaluation.

    Attributes:
        repo_id (str): Identifier for the dataset repository (e.g., HuggingFace Hub repo ID).
        root (str | None): Root directory where the dataset will be stored. If None, defaults to a system-defined path.
        episodes (list[int] | None): List of episode indices to use. If None, all episodes are used.
        image_transforms (ImageTransformsConfig): Configuration for image transformations applied to the dataset.
        revision (str | None): Specific revision of the dataset to use (e.g., a Git commit hash or branch name).
        use_imagenet_stats (bool): Whether to normalize images using ImageNet statistics.
        video_backend (str): Backend codec to use for video processing. Defaults to a safe codec.
    """
    repo_id: str
    root: str | None = None
    episodes: list[int] | None = None
    image_transforms: ImageTransformsConfig = field(default_factory=ImageTransformsConfig)
    revision: str | None = None
    use_imagenet_stats: bool = True
    video_backend: str = field(default_factory=get_safe_default_codec)

    def __post_init__(self):
        """
        Post-initialization logic for DatasetConfig.
        """
        logging.info(colored("Initialized --> DatasetConfig", "yellow", attrs=["bold"]))


@dataclass
class WandBConfig:
    """
    Configuration for Weights & Biases (WandB) integration.

    Attributes:
        enable (bool): Whether to enable WandB logging.
        disable_artifact (bool): If True, disables saving artifacts even if training.save_checkpoint=True.
        project (str): Name of the WandB project.
        entity (str | None): Name of the WandB entity (team or user). If None, defaults to the current user.
        notes (str | None): Notes or description for the WandB run.
        run_id (str | None): Unique identifier for the WandB run. If None, a new run ID is generated.
        mode (str | None): Mode for WandB logging. Allowed values: 'online', 'offline', 'disabled'. Defaults to 'online'.
    """
    enable: bool = False
    disable_artifact: bool = False
    project: str = "lerobot"
    entity: str | None = None
    notes: str | None = None
    run_id: str | None = None
    mode: str | None = None  # Allowed values: 'online', 'offline', 'disabled'. Defaults to 'online'.

    # Post-initialization logic
    def __post_init__(self):
        logging.info(colored("Initialized --> WandBConfig", "yellow", attrs=["bold"]))

@dataclass
class EvalConfig:
    """
    Configuration for evaluation settings.

    Attributes:
        n_episodes (int): Number of episodes to evaluate.
        batch_size (int): Number of environments to use in a gym.vector.VectorEnv during evaluation.
        use_async_envs (bool): Whether to use asynchronous environments (multiprocessing) for evaluation.

    Methods:
        __post_init__: Validates the configuration after initialization. Ensures batch_size does not exceed n_episodes.
    """
    n_episodes: int = 50
    batch_size: int = 50
    use_async_envs: bool = False

    def __post_init__(self):
        """
        Validates the evaluation configuration after initialization.

        Raises:
            ValueError: If the batch size exceeds the number of episodes, as this would result in inefficiencies.
        """
        if self.batch_size > self.n_episodes:
            raise ValueError(
                "The eval batch size is greater than the number of eval episodes "
                f"({self.batch_size} > {self.n_episodes}). As a result, {self.batch_size} "
                f"eval environments will be instantiated, but only {self.n_episodes} will be used. "
                "This might significantly slow down evaluation. To fix this, you should update your command "
                f"to increase the number of episodes to match the batch size (e.g. `eval.n_episodes={self.batch_size}`), "
                f"or lower the batch size (e.g. `eval.batch_size={self.n_episodes}`)."
            )
        
        logging.info(colored("Initialized --> EvalConfig", "yellow", attrs=["bold"]))
