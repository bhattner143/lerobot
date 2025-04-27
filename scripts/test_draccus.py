import draccus
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# Define a configuration class
@dataclass
class TrainConfig:
    dataset_repo_id: str = "default_repo_id"  # Default dataset repository ID
    policy_type: str = "default_policy"       # Default policy type
    output_dir: str = "outputs/default"       # Default output directory
    job_name: str = "default_job"             # Default job name
    device: str = "cpu"                       # Default device
    wandb_enable: bool = False                # Default Weights & Biases logging
    steps: int = 1000                         # Default number of training steps
    batch_size: int = 32                      # Default batch size
    extra_args: List[str] = field(default_factory=list)  # Additional arguments
    optimizer: str = "adam"                   # Default optimizer
    scheduler: str = "linear"                 # Default learning rate scheduler

import importlib
import inspect
import pkgutil
import sys
from argparse import ArgumentError
from functools import wraps
from pathlib import Path

# Define a decorator to wrap a function for configuration and plugin handling
def wrap(config_path: Path | None = None):## <- accepts arguments
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
    # Outer wrapper function
    def wrapper_outer(fn): #<- takes the original function
        print(f"Function name: {fn.__name__}")  # Print the name of the function being wrapped

        @wraps(fn) # <- preserves the original function's
        def wrapper_inner(*args, **kwargs):#<- the actual replacement function
            # Get the function's argument specification
            argspec = inspect.getfullargspec(fn)
            argtype = argspec.annotations[argspec.args[0]]  # Get the type of the first argument
            cli_args = sys.argv[1:]  # Get all command-line arguments except the script name
            
            # Parse the config using draccus if no pretrained model is used
            cfg = draccus.parse(
                config_class=argtype,  # Parse the config using draccus
                config_path=config_path,  # Path to the config file (used for pretrained model)
                args=cli_args  # Remaining CLI arguments
            )
                    
            print(f"Parsed config: {cfg}")  # Print the parsed config
            # Call the wrapped function with the parsed config and remaining arguments
            response = fn(cfg, *args, **kwargs) #<- call the original function with the parsed config
            return response

        return wrapper_inner

    return wrapper_outer

# Define the training function and wrap it with the `wrap` decorator
@wrap(config_path=Path("test_config_draccus.json"))  # Specify the path to the config file
def train(cfg: TrainConfig):
    """
    Training function that uses the parsed configuration.

    Args:
        cfg (TrainConfig): The parsed training configuration.
    """
    print(f"Training with config: {cfg}")

# Main function to simulate command-line arguments and invoke the training function
def main():
    # Simulate command-line arguments
    cli_args = [
        "--optimizer=adam_dips",  # Example CLI argument to override the optimizer
        "--scheduler=linear_dips",  # Example CLI argument to override the scheduler
    ]

    train()  # Call the training function

# Entry point of the script
if __name__ == "__main__":
    main()