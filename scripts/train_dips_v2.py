#!/usr/bin/env python
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
# from scripts.eval import eval_policy

import numpy as np

def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    grad_scaler: GradScaler,
    lr_scheduler=None,
    use_amp: bool = False,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    policy.train()
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        loss, output_dict = policy.forward(batch)
        # TODO(rcadene): policy.unnormalize_outputs(out_dict)
    grad_scaler.scale(loss).backward()

    # Unscale the gradient of the optimizer's assigned params in-place **prior to gradient clipping**.
    grad_scaler.unscale_(optimizer)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        grad_clip_norm,
        error_if_nonfinite=False,
    )

    # Optimizer's gradients are already unscaled, so scaler.step does not unscale them,
    # although it still skips optimizer.step() if the gradients contain infs or NaNs.
    with lock if lock is not None else nullcontext():
        grad_scaler.step(optimizer)
    # Updates the scale for next iteration.
    grad_scaler.update()

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    if has_method(policy, "update"):
        # To possibly update an internal buffer (for instance an Exponential Moving Average like in TDMPC).
        policy.update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict

# The main training function that orchestrates the entire training pipeline
# The `@parser_dips.wrap()` decorator is used to parse the configuration from various sources (e.g., CLI, config file)
@parser_dips.wrap()
def train(cfg: TrainPipelineConfig):
    """
    #################GET CONFIGURATION (dataset, env, policy, ...,optimizer,scheduler,wandb)######################
    """
    # Validate the configuration to ensure all required fields are set correctly
    cfg.validate()
    logging.info(pformat(cfg.to_dict()))  # Log the configuration for debugging and reproducibility

    # Initialize Weights & Biases (WandB) logger if enabled in the configuration
    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    # Set the random seed for reproducibility if specified in the configuration
    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Ensure the specified device (e.g., CPU or GPU) is available and log its details
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark        = True  # Enable cuDNN auto-tuner for better performance
    torch.backends.cuda.matmul.allow_tf32 = True  # Allow TensorFloat32 for faster computation on supported GPUs
    """
    #################GET THE DATASET######################
    """
    # Create the dataset based on the configuration
    logging.info("Creating dataset")
    dataset = make_dataset(cfg)

    """
    #################CREATE AN INSTANCE OF THE POLICY######################
    """
    # Create the policy (model) based on the configuration and dataset metadata
    logging.info("Creating policy")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
    )

    """
     #################CREATE OPTIMIZER AND SCHEDULER######################
    """
    # Create the optimizer and learning rate scheduler for training
    logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)  # Gradient scaler for mixed precision training

    # Initialize the training step counter
    step = 0

    """
    #################FOR RESUME TRAINING######################
    """
    # Resume training from a checkpoint if specified in the configuration
    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    """
    #################LOGGING######################
    """
    # Log the number of learnable and total parameters in the policy
    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())
    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")

    logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
    logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
    logging.info(f"{dataset.num_episodes=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    """
    #################DATA LOADER######################
    """
    # Configure the data loader for training
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.episode_data_index,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=1, #cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )
    dl_iter = cycle(dataloader)  # Create an infinite iterator for the data loader

    # Set the policy to training mode
    policy.train()

    # Initialize metrics for tracking training progress
    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    # Initialize a metrics tracker for logging and monitoring
    train_tracker = MetricsTracker(
        cfg.batch_size, dataset.num_frames, dataset.num_episodes, train_metrics, initial_step=step
    )

    """
    #################TRAINING LOOP######################
    """
    # Start the main training loop
    logging.info("Start offline training on a fixed dataset")
    for _ in range(step, cfg.steps):
        # Measure the time taken to load a batch of data
        start_time = time.perf_counter()
        batch = next(dl_iter)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        # Move the batch data to the specified device (e.g., GPU) for training
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)

        # Update the policy using the current batch of data
        train_tracker, output_dict = update_policy(
            train_tracker,  # Metrics tracker to log training progress
            policy,         # The policy (model) being trained
            batch,          # Current batch of data
            optimizer,      # Optimizer for updating model parameters
            cfg.optimizer.grad_clip_norm,  # Gradient clipping norm from config
            grad_scaler=grad_scaler,       # Gradient scaler for mixed precision training
            lr_scheduler=lr_scheduler,    # Learning rate scheduler
            use_amp=cfg.policy.use_amp,   # Whether to use Automatic Mixed Precision (AMP)
        )
        visualize_first_image_from_presnt_batch = False
        # Visualize the first image from the current batch if specified
        if visualize_first_image_from_presnt_batch:
            import matplotlib.pyplot as plt

            if cfg.dataset.repo_id == 'lerobot/aloha_mobile_cabinet':
                # Extract the first image from 'observation.images.cam_high' for visualization
                image = batch['observation.images.cam_high'][0].cpu().numpy().transpose(1, 2, 0)
            elif cfg.dataset.repo_id == 'datasets_lerobot/so100_test':
                # Extract the first image from 'observation.images.cam_high' for visualization
                image = batch['observation.images.rgb_intel_real_sense'][0].cpu().numpy().transpose(1, 2, 0)
            else:
                # Extract the first image from 'observation.images.top' for visualization
                image = batch['observation.images.top'][0].cpu().numpy().transpose(1, 2, 0)

            # Normalize the image to the range [0, 255] if necessary
            image = (image - image.min()) / (image.max() - image.min()) * 255
            image = image.astype(np.uint8)

            # Display the image using Matplotlib
            plt.imshow(image)
            plt.axis("off")  # Turn off axis labels
            plt.title("Top Observation Image")
            plt.show()

        # Increment the training step counter
        step += 1
        train_tracker.step()

        # Determine whether to log, save a checkpoint, or evaluate the policy
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        # Log training metrics at regular intervals
        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        # Save a checkpoint of the policy at regular intervals
        if cfg.save_checkpoint and is_saving_step:
            logging.info(f"Checkpoint policy after step {step}")
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            save_checkpoint(checkpoint_dir, step, cfg, policy, optimizer, lr_scheduler)
            update_last_checkpoint(checkpoint_dir)
            if wandb_logger:
                wandb_logger.log_policy(checkpoint_dir)

        

    # Log the end of training
    logging.info("End of training")


if __name__ == "__main__":
  
    """
    Check test_subclass program to undersatnd how all the objects are dynamically created
    """
    # Initialize logging and run the training
    init_logging()
    # parser.parse_args(config_args)
    train()
    # train()



    