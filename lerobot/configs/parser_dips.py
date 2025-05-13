# Copyright 2024 The HuggingFace Inc. team. All rights reserved.

import inspect
import sys
from functools import wraps
from pathlib import Path

import draccus

from lerobot.common.utils.utils import has_method
from lerobot.common.utils.parser_utils import *
from omegaconf import OmegaConf
import json

# Set the default configuration type for draccus to JSON
draccus.set_config_type("json")


def wrap(config_path: Path | None = None): ## <- accepts arguments
    """
    A decorator that wraps a function to handle configuration and plugin loading.

    Args:
        config_path (Path | None): The path to the configuration file. Defaults to None.

    Returns:
        Callable: The wrapped function.

    Notes:
        - Removes '.path' arguments from the CLI to process them later.
        - Initializes the main config class from a pretrained model if `config_path` is provided.
        - Loads plugins specified in the CLI arguments.
    """

    def wrapper_outer(fn):#<- takes the original function
        @wraps(fn) # <- preserves the original function's
        def wrapper_inner(*args, **kwargs):#<- the actual replacement function
            # Get the function's argument specification
            argspec = inspect.getfullargspec(fn)
            argtype = argspec.annotations[argspec.args[0]]  # Get the type of the first argument, which is the default cfg from TrainPipelineConfig

            if len(args) > 0 and type(args[0]) is argtype:
                # If the first argument matches the expected type, use it as the config
                cfg = args[0]
                args = args[1:]  # Remove the first argument from args as it's already used
            else:
                # Parse CLI arguments
                cli_args = sys.argv[1:]  # Get all command-line arguments except the script name
                plugin_args = parse_plugin_args(PLUGIN_DISCOVERY_SUFFIX, cli_args)

                # Load plugins specified in the CLI arguments
                for plugin_cli_arg, plugin_path in plugin_args.items():
                    try:
                        load_plugin(plugin_path)  # Attempt to load the plugin
                    except PluginLoadError as e:
                        # Raise a detailed error if plugin loading fails
                        raise PluginLoadError(f"{e}\nFailed plugin CLI Arg: {plugin_cli_arg}") from e
                    # Remove the plugin argument from CLI arguments after processing
                    cli_args = filter_arg(plugin_cli_arg, cli_args)

                # Parse the config path from CLI arguments
                config_path_cli = parse_arg("config_path", cli_args)

                # Handle path fields if the config class supports it
                if has_method(argtype, "__get_path_fields__"):
                    path_fields = argtype.__get_path_fields__()  # Get all path-related fields
                    cli_args = filter_path_args(path_fields, cli_args)  # Remove path-related arguments
                
                # Check if save_config_train and overwrite_config_train arguments exist, then filter them out (Dips)
                save_config_train_arg = parse_arg("save_config_train", cli_args)
                if save_config_train_arg is not None:
                    save_config_train_arg = True if save_config_train_arg.lower() == "true" else False
                    cli_args = filter_arg("save_config_train", cli_args)

                # Check if dataset.root argument exists, then filter it out and save it as a separate argument
                dataset_root_arg = parse_arg("dataset.root", cli_args)
                if dataset_root_arg:
                    cli_args = filter_arg("dataset.root", cli_args)
                
                # Initialize the config class from a pretrained model if applicable
                if has_method(argtype, "from_pretrained") and config_path_cli:
                    # If a config path is provided, use it to initialize the config
                    cli_args = filter_arg("config_path", cli_args)  # Remove the config path argument
                    cfg = argtype.from_pretrained(config_path_cli, cli_args=cli_args)
                else:
                    # Parse the config using draccus if no pretrained model is used
                    cfg = draccus.parse(config_class=argtype,     # Classthat defines the structure of the config
                                        config_path=config_path,  # Optional path to a config file (e.g. YAML/JSON)
                                        args=cli_args             #List of CLI arguments passed to override or extend
                                        )
                    # Add dataset_root_arg to cfg.dataset.root if it exists
                    if dataset_root_arg:
                        if hasattr(cfg.dataset, "root") and cfg.dataset.root:
                            raise ValueError("The 'cfg.dataset.root' already exists and cannot be overwritten.")
                        cfg.dataset.root = dataset_root_arg

                    # Convert the parsed config to a dictionary
                    cfg_dict = cfg.to_dict()
            
                    
                    if save_config_train_arg:
                        # Convert all Path objects in the configuration to strings
                        cfg_dict_without_path_obj = convert_paths_to_str(cfg_dict)
                        # Save the configuration as a JSON file
                        save_path_json = Path(f"/home/dips/Documents/{cfg.dataset.repo_id}/meta/config_train_{cfg.policy.type}_all.json")
                        # if save_path_json.exists():
                        #     raise FileExistsError(f"The file {save_path_json} already exists. Cannot overwrite.")
                        with open(save_path_json, "w") as f:
                            json.dump(cfg_dict_without_path_obj, f, indent=4)

                        # # Convert the parsed config to an OmegaConf object
                        # cfg_omegaconf = OmegaConf.create(cfg_dict)

                        # # Save the OmegaConf object to a YAML file
                        # save_path_yaml = Path(f"/home/dips/Documents/{cfg.dataset.repo_id}/meta/config_train.yaml")
                        # if save_path_yaml.exists():
                        #     raise FileExistsError(f"The file {save_path_yaml} already exists. Cannot overwrite.")
                        # with open(save_path_yaml, "w") as f:
                        #     OmegaConf.save(cfg_omegaconf, f)

                # Call the wrapped function with the parsed config and remaining arguments
                response = fn(cfg, *args, **kwargs)
                return response

        return wrapper_inner

    return wrapper_outer
