#!/bin/bash

# Automatically detect the number of GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# Get the number of physical CPU cores
NUM_CPUS=$(lscpu | awk '/^CPU\(s\):/ {print $2}')

# Get the number of physical CPU cores
NUM_USED=6

# Set OMP_NUM_THREADS (adjust based on your system)
export OMP_NUM_THREADS=16

# Training script parameters
SCRIPT="main.py"  # Change this to your training script

# Print system info
echo "Detected $NUM_GPUS GPUs and $NUM_CPUS CPU cores."
echo "Setting OMP_NUM_THREADS=$OMP_NUM_THREADS per process."
echo "Used $NUM_USED GPUs for training."
echo "Starting distributed training..."

# Run PyTorch Distributed Data Parallel (DDP) with torchrun
torchrun --nproc_per_node=$NUM_USED $SCRIPT