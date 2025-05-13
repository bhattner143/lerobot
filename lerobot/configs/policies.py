# -*- coding: utf-8 -*-
import abc
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Type, TypeVar

import draccus

from lerobot.common.optim.optimizers import OptimizerConfig
from lerobot.common.optim.schedulers import LRSchedulerConfig

from lerobot.common.utils.utils import auto_select_torch_device, is_amp_available, is_torch_device_available
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.common.models_cloth.clothmodel_configs import PreTrainedClothModelConfig

import logging
from termcolor import colored  # For colored terminal output.

# Generic variable that is either PreTrainedConfig or a subclass thereof
T = TypeVar("T", bound="PreTrainedConfig")

# draccus.ChoiceRegistry allows dynamic registration and instantiation of subclasses based on a string key. 
# It is particularly useful for managing configurations where different types of objects 
# (e.g., policies, models, optimizers) need to be dynamically selected and instantiated 
# at runtime based on user input, such as command-line arguments or configuration files.
@dataclass
class PreTrainedConfig(draccus.ChoiceRegistry, abc.ABC):
    """
    This is the base class for all policy configurations. 
    It defines the common structure and behavior for policy configurations, 
    such as handling input/output features, normalization, and device settings.

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy (takes the
            current step and additional steps going back).
        input_shapes: A dictionary defining the shapes of the input data for the policy.
        output_shapes: A dictionary defining the shapes of the output data for the policy.
        input_normalization_modes: A dictionary with key representing the modality and the value specifies the
            normalization mode to apply.
        output_normalization_modes: Similar dictionary as `input_normalization_modes`, but to unnormalize to
            the original scale.
    """
    n_obs_steps: int = 1
    normalization_mapping: dict[str, NormalizationMode] = field(default_factory=dict)

    input_features: dict[str, PolicyFeature] = field(default_factory=dict)
    output_features: dict[str, PolicyFeature] = field(default_factory=dict)

    device: str | None = None  # cuda | cpu | mp
    # `use_amp` determines whether to use Automatic Mixed Precision (AMP) for training and evaluation. With AMP,
    # automatic gradient scaling is used.
    use_amp: bool = False

    # Added for the cloth model config
    cloth_model_config: str | PreTrainedClothModelConfig | None = None

    def __post_init__(self):
        self.pretrained_path = None
        if not self.device or not is_torch_device_available(self.device):
            auto_device = auto_select_torch_device()
            logging.warning(f"Device '{self.device}' is not available. Switching to '{auto_device}'.")
            self.device = auto_device.type

        # Automatically deactivate AMP if necessary
        if self.use_amp and not is_amp_available(self.device):
            logging.warning(
                f"Automatic Mixed Precision (amp) is not available on device '{self.device}'. Deactivating AMP."
            )
            self.use_amp = False

        logging.info(colored("Initialized --> PreTrainedConfig", "yellow", attrs=["bold"]))

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)


    @property
    def robot_state_feature(self) -> PolicyFeature | None:
        for _, ft in self.input_features.items():
            if ft.type is FeatureType.STATE:
                return ft
        return None

    @property
    def env_state_feature(self) -> PolicyFeature | None:
        for _, ft in self.input_features.items():
            if ft.type is FeatureType.ENV:
                return ft
        return None

    @property
    def image_features(self) -> dict[str, PolicyFeature]:
        return {key: ft for key, ft in self.input_features.items() if ft.type is FeatureType.VISUAL}

    @property
    def action_feature(self) -> PolicyFeature | None:
        for _, ft in self.output_features.items():
            if ft.type is FeatureType.ACTION:
                return ft
        return None

    def _save_pretrained(self, save_directory: Path) -> None:
        with open(save_directory / CONFIG_NAME, "w") as f, draccus.config_type("json"):
            draccus.dump(self, f, indent=4)

    @classmethod
    def from_pretrained(
        cls: Type[T],
        pretrained_name_or_path: str | Path,
        # *,
        # force_download: bool = False,
        # resume_download: bool = None,
        # proxies: dict | None = None,
        # token: str | bool | None = None,
        # cache_dir: str | Path | None = None,
        # local_files_only: bool = False,
        # revision: str | None = None,
        **policy_kwargs,
    ) -> T:
        model_id = str(pretrained_name_or_path)
        config_file: str | None = None
        if Path(model_id).is_dir():
            if CONFIG_NAME in os.listdir(model_id):
                config_file = os.path.join(model_id, CONFIG_NAME)
            else:
                print(f"{CONFIG_NAME} not found in {Path(model_id).resolve()}")
        else:
            Exception(f"Path {model_id} is not a directory")

        # HACK: this is very ugly, ideally we'd like to be able to do that natively with draccus
        # something like --policy.path (in addition to --policy.type)
        cli_overrides = policy_kwargs.pop("cli_overrides", [])
        cfg = draccus.parse(cls, config_file, args=cli_overrides)
        return cfg
