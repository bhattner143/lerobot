#!/usr/bin/env python

import logging

from torch import nn

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
# from lerobot.common.envs.configs import EnvConfig
# from lerobot.common.envs.utils import env_to_policy_features
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
from lerobot.common.policies.pi0fast.configuration_pi0fast import PI0FASTConfig
from lerobot.common.policies.act_advanced.configuration_act_advanced import ACTConfigAdvanced #My addition
from lerobot.common.policies.act_mesh_gat.configuration_act_mesh_gat import ACTMeshGATConfig#My addition
from lerobot.common.policies.pretrained import PreTrainedPolicy
# from lerobot.common.policies.tdmpc.configuration_tdmpc import TDMPCConfig
# from lerobot.common.policies.vqbet.configuration_vqbet import VQBeTConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType


def get_policy_class(name: str) -> PreTrainedPolicy:
    """Get the policy's class and config class given a name (matching the policy class' `name` attribute)."""
    if name == "tdmpc":
        from lerobot.common.policies.tdmpc.modeling_tdmpc import TDMPCPolicy

        return TDMPCPolicy
    elif name == "diffusion":
        from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

        return DiffusionPolicy
    elif name == "act":
        from lerobot.common.policies.act.modeling_act import ACTPolicy

        return ACTPolicy
    elif name == "vqbet":
        from lerobot.common.policies.vqbet.modeling_vqbet import VQBeTPolicy

        return VQBeTPolicy
    elif name == "pi0":
        from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy

        return PI0Policy
    elif name == "pi0fast":
        from lerobot.common.policies.pi0fast.modeling_pi0fast import PI0FASTPolicy

        return PI0FASTPolicy
    
    elif name == "act_advanced":
        from lerobot.common.policies.act_advanced.modeling_act_advanced import ACTPolicyAdvanced

        return ACTPolicyAdvanced
    
    elif name == "act_mesh_gat":
        from lerobot.common.policies.act_mesh_gat.modeling_act_mesh_gat import ACTMeshGATPolicy

        return ACTMeshGATPolicy
    
    else:
        raise NotImplementedError(f"Policy with name {name} is not implemented.")


def make_policy_config(policy_type: str, **kwargs) -> PreTrainedConfig:
    if policy_type == "tdmpc":
        return TDMPCConfig(**kwargs)
    elif policy_type == "diffusion":
        return DiffusionConfig(**kwargs)
    elif policy_type == "act":
        return ACTConfig(**kwargs)
    elif policy_type == "vqbet":
        return VQBeTConfig(**kwargs)
    elif policy_type == "pi0":
        return PI0Config(**kwargs)
    elif policy_type == "pi0fast":
        return PI0FASTConfig(**kwargs)
    # elif policy_type == "act_advanced":
    #     return ACTConfigAdvanced(**kwargs)
    elif policy_type == "act_mesh_gat":
        return ACTMeshGATConfig(**kwargs)
    else:
        raise ValueError(f"Policy type '{policy_type}' is not available.")


def make_policy(
    cfg: PreTrainedConfig,  # Configuration object for the policy
    ds_meta: LeRobotDatasetMetadata | None = None,  # Optional dataset metadata
    # env_cfg: EnvConfig | None = None,  # Optional environment configuration
) -> PreTrainedPolicy:
    """Make an instance of a policy class.

    This function exists because (for now) we need to parse features from either a dataset or an environment
    in order to properly dimension and instantiate a policy for that dataset or environment.

    Args:
        cfg (PreTrainedConfig): The config of the policy to make. If `pretrained_path` is set, the policy will
            be loaded with the weights from that path.
        ds_meta (LeRobotDatasetMetadata | None, optional): Dataset metadata to take input/output shapes and
            statistics to use for (un)normalization of inputs/outputs in the policy. Defaults to None.
    Raises:
        ValueError: Either ds_meta or env and env_cfg must be provided.
        NotImplementedError: if the policy.type is 'vqbet' and the policy device 'mps' (due to an incompatibility)

    Returns:
        PreTrainedPolicy: _description_
    """
    # if bool(ds_meta) == bool(env_cfg):
    #     raise ValueError("Either one of a dataset metadata or a sim env must be provided.")

    # Retrieve the policy class based on the type specified in the configuration
    policy_cls = get_policy_class(cfg.type)

    # Initialize a dictionary to hold additional arguments for policy instantiation
    kwargs = {}

    if ds_meta is not None:
        # If dataset metadata is provided, extract features from the dataset
        features = dataset_to_policy_features(ds_meta.features)
        # Add dataset statistics to the arguments for policy instantiation
        kwargs["dataset_stats"] = ds_meta.stats
    else:
        # If no dataset metadata is provided, ensure a pretrained path is not required
        if not cfg.pretrained_path:
            logging.warning(
                "You are instantiating a policy from scratch and its features are parsed from an environment "
                "rather than a dataset. Normalization modules inside the policy will have infinite values "
                "by default without stats from a dataset."
            )

    # Separate features into input and output features based on their type
    cfg.output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    cfg.input_features = {key: ft for key, ft in features.items() if key not in cfg.output_features}
    # Add the configuration to the arguments for policy instantiation
    
    kwargs["config"] = cfg
    if cfg.cloth_model_config:
        # If a cloth model configuration is provided, add it to the arguments
        kwargs["cloth_model_config"] = cfg.cloth_model_config

    if cfg.pretrained_path:
        # If a pretrained path is specified, load the pretrained policy
        # and override the configuration if necessary (e.g., for inference-time hyperparameters)
        kwargs["pretrained_name_or_path"] = cfg.pretrained_path
        policy = policy_cls.from_pretrained(**kwargs)
    else:
        # If no pretrained path is specified, create a new policy instance
        policy = policy_cls(**kwargs)

    # Move the policy to the specified device (e.g., CPU, GPU)
    policy.to(cfg.device)
    # Ensure the policy is an instance of PyTorch's nn.Module
    assert isinstance(policy, nn.Module)

    # Optionally, compile the policy for performance optimization (commented out for now)
    # policy = torch.compile(policy, mode="reduce-overhead")

    # Return the instantiated policy
    return policy
