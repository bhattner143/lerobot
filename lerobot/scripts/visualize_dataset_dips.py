#!/usr/bin/env python

"""
Visualize data of **all** frames of any episode of a dataset of type LeRobotDataset.
"""

import argparse
import gc
import logging
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import rerun as rr
import torch
import torch.utils.data
import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.utils.utils import has_method
from lerobot.common.utils.parser_utils import *

import inspect
import sys
from functools import wraps
from pathlib import Path
from typing import Sequence

import draccus


def wrap(config_path: Path | None = None):
    """
    A decorator to handle configuration and plugin loading for functions.
    """
    def wrapper_outer(fn):
        @wraps(fn)
        def wrapper_inner(*args, **kwargs):
            # Get the function's argument specification
            argspec = inspect.getfullargspec(fn)
            argtype = argspec.annotations[argspec.args[0]]  # Type of the first argument (config)

            if len(args) > 0 and type(args[0]) is argtype:
                # If the first argument matches the expected type, use it as the config
                cfg = args[0]
                args = args[1:]  # Remove the first argument from args
            else:
                # Parse CLI arguments
                cli_args = sys.argv[1:]
                cli_args_copy = cli_args.copy()

                # Handle path fields if the config class supports it
                if has_method(argtype, "__get_path_fields__"):
                    path_fields = argtype.__get_path_fields__()
                    cli_args = filter_path_args(path_fields, cli_args)

                # Filter out unused arguments
                unused_args = ["output_dir", "episode_index", "web_port", "ws_port", 
                               "save", "mode", "batch_size", "num_workers"]
                for unused_arg in unused_args:
                    cli_args = filter_arg(unused_arg, cli_args)

                # Parse the configuration using draccus
                cfg = draccus.parse(config_class=argtype, config_path=config_path, args=cli_args)

            # Parse additional arguments like episode_index
            ep_idx = parse_int_arg("episode_index", cli_args_copy)
            args = (ep_idx,) if ep_idx is not None else ()

            # Create kwargs for unused arguments
            unused_kwargs = {}
            for unused_arg in unused_args:
                arg_value = parse_arg(unused_arg, cli_args_copy)
                if arg_value is not None:
                    unused_kwargs[unused_arg] = arg_value
            
            kwargs.update(unused_kwargs)

            # Remove episode_index from kwargs
            kwargs.pop("episode_index", None)

            # Convert numeric strings to integers
            for k, v in kwargs.items():
                if isinstance(v, str) and v.isdigit():
                    kwargs[k] = int(v)

            # Call the wrapped function with the parsed config and arguments
            response = fn(cfg, *args, **kwargs)
            return response
        return wrapper_inner

    return wrapper_outer


class EpisodeSampler(torch.utils.data.Sampler):
    """
    Custom sampler to iterate over frames of a specific episode in the dataset.
    """
    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        # Get the range of frame indices for the specified episode
        from_idx = dataset.episode_data_index["from"][episode_index].item()
        to_idx = dataset.episode_data_index["to"][episode_index].item()
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self) -> Iterator:
        # Return an iterator over the frame indices
        return iter(self.frame_ids)

    def __len__(self) -> int:
        # Return the number of frames in the episode
        return len(self.frame_ids)


def to_hwc_uint8_numpy(chw_float32_torch: torch.Tensor) -> np.ndarray:
    """
    Convert a PyTorch tensor (CHW format, float32) to a NumPy array (HWC format, uint8).
    """
    assert chw_float32_torch.dtype == torch.float32
    assert chw_float32_torch.ndim == 3
    c, h, w = chw_float32_torch.shape
    assert c < h and c < w, f"Expect channel-first images, but got {chw_float32_torch.shape}"
    hwc_uint8_numpy = (chw_float32_torch * 255).type(torch.uint8).permute(1, 2, 0).numpy()
    return hwc_uint8_numpy



@wrap()
def visualize_dataset(
    dataset: LeRobotDataset,
    episode_index: int,
    batch_size: int = 32,
    num_workers: int = 0,
    mode: str = "local",
    web_port: int = 9090,
    ws_port: int = 9087,
    save: bool = False,
    output_dir: Path | None = None,
) -> Path | None:
    """
    Visualize the dataset for a specific episode.
    """
    if save:
        # Ensure output directory is provided when saving
        assert output_dir is not None, (
            "Set an output directory where to write .rrd files with `--output-dir path/to/directory`."
        )

    repo_id = dataset.repo_id

    logging.info("Loading dataloader")
    # Create a DataLoader for the specified episode
    episode_sampler = EpisodeSampler(dataset, episode_index)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        sampler=episode_sampler,
    )

    logging.info("Starting Rerun")

    # Validate the mode
    if mode not in ["local", "distant"]:
        raise ValueError(mode)

    # Determine whether to spawn a local viewer
    spawn_local_viewer = mode == "local" and not save
    rr.init(f"{repo_id}/episode_{episode_index}", spawn=spawn_local_viewer)

    # Manually call garbage collector to avoid hanging issues
    gc.collect()

    if mode == "distant":
        # Start a server for distant mode
        rr.serve(open_browser=False, web_port=web_port, ws_port=ws_port)

    logging.info("Logging to Rerun")

    # Iterate over the DataLoader and log data to Rerun
    for batch in tqdm.tqdm(dataloader, total=len(dataloader)):
        for i in range(len(batch["index"])):
            rr.set_time_sequence("frame_index", batch["frame_index"][i].item())
            rr.set_time_seconds("timestamp", batch["timestamp"][i].item())

            # Log camera images
            for key in dataset.meta.camera_keys:
                rr.log(key, rr.Image(to_hwc_uint8_numpy(batch[key][i])))

            # Log action dimensions
            if "action" in batch:
                for dim_idx, val in enumerate(batch["action"][i]):
                    rr.log(f"action/{dim_idx}", rr.Scalar(val.item()))

            # Log observed state dimensions
            if "observation.state" in batch:
                for dim_idx, val in enumerate(batch["observation.state"][i]):
                    rr.log(f"state/{dim_idx}", rr.Scalar(val.item()))

            # Log additional data like done, reward, and success
            if "next.done" in batch:
                rr.log("next.done", rr.Scalar(batch["next.done"][i].item()))
            if "next.reward" in batch:
                rr.log("next.reward", rr.Scalar(batch["next.reward"][i].item()))
            if "next.success" in batch:
                rr.log("next.success", rr.Scalar(batch["next.success"][i].item()))

    if mode == "local" and save:
        # Save the .rrd file locally
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        repo_id_str = repo_id.replace("/", "_")
        rrd_path = output_dir / f"{repo_id_str}_episode_{episode_index}.rrd"
        rr.save(rrd_path)
        return rrd_path

    elif mode == "distant":
        # Keep the process running for distant mode
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Ctrl-C received. Exiting.")


if __name__ == "__main__":
    # Entry point for the script
    visualize_dataset()
