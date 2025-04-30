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

"""
Utilities to control a robot.

Useful to record a dataset, replay a recorded episode, run the policy on your robot
and record an evaluation dataset, and to recalibrate your robot if needed.

Examples of usage:
... (examples omitted for brevity)
"""

import logging  # For logging messages to the console or files
import os  # For interacting with the operating system
import time  # For time-related operations
from dataclasses import asdict  # For converting dataclass instances to dictionaries
from pprint import pformat  # For pretty-printing data structures

import rerun as rr  # For visualizing robot control data

# Importing necessary modules and classes from the project
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # Dataset management
from lerobot.common.policies.factory import make_policy  # Policy creation
from lerobot.common.robot_devices.control_configs import (  # Configuration classes for control modes
    CalibrateControlConfig,
    ControlConfig,
    ControlPipelineConfig,
    RecordControlConfig,
    RemoteRobotConfig,
    ReplayControlConfig,
    TeleoperateControlConfig,
)
from lerobot.common.robot_devices.control_utils import (  # Utility functions for robot control
    control_loop,
    init_keyboard_listener,
    is_headless,
    log_control_info,
    record_episode,
    reset_environment,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
    stop_recording,
    warmup_record,
)
from lerobot.common.robot_devices.robots.utils import Robot, make_robot_from_config  # Robot utilities
from lerobot.common.robot_devices.utils import busy_wait, safe_disconnect  # Utility functions for robot devices
from lerobot.common.utils.utils import has_method, init_logging, log_say  # General utility functions
from lerobot.configs import parser_dips  # Configuration parser

########################################################################################
# Control modes
########################################################################################

@safe_disconnect  # Ensures the robot disconnects safely in case of an error
def calibrate(robot: Robot, cfg: CalibrateControlConfig):

    arms = robot.available_arms if cfg.arms is None else cfg.arms  # Determine which arms to calibrate
    unknown_arms = [arm_id for arm_id in arms if arm_id not in robot.available_arms]  # Check for invalid arms
    available_arms_str = " ".join(robot.available_arms)  # Format available arms as a string
    unknown_arms_str = " ".join(unknown_arms)  # Format unknown arms as a string

    if arms is None or len(arms) == 0:  # Raise an error if no arms are provided
        raise ValueError(
            "No arm provided. Use `--arms` as argument with one or more available arms.\n"
            f"For instance, to recalibrate all arms add: `--arms {available_arms_str}`"
        )

    if len(unknown_arms) > 0:  # Raise an error if unknown arms are provided
        raise ValueError(
            f"Unknown arms provided ('{unknown_arms_str}'). Available arms are `{available_arms_str}`."
        )

    for arm_id in arms:  # Iterate over the arms to calibrate
        arm_calib_path = robot.calibration_dir / f"{arm_id}.json"  # Path to the calibration file
        if arm_calib_path.exists():  # Remove existing calibration files
            print(f"Removing '{arm_calib_path}'")
            arm_calib_path.unlink()
        else:
            print(f"Calibration file not found '{arm_calib_path}'")

    if robot.is_connected:  # Disconnect the robot if it's connected
        robot.disconnect()

    if robot.robot_type.startswith("lekiwi") and "main_follower" in arms:  # Special handling for "lekiwi" follower arm
        print("Calibrating only the lekiwi follower arm 'main_follower'...")
        robot.calibrate_follower()
        return

    if robot.robot_type.startswith("lekiwi") and "main_leader" in arms:  # Special handling for "lekiwi" leader arm
        print("Calibrating only the lekiwi leader arm 'main_leader'...")
        robot.calibrate_leader()
        return

    # Calling `connect` automatically runs calibration when the calibration file is missing
    robot.connect()
    robot.disconnect()
    print("Calibration is done! You can now teleoperate and record datasets!")  # Inform the user that calibration is complete

@safe_disconnect  # Ensures safe disconnection in case of errors
def teleoperate(robot: Robot, cfg: TeleoperateControlConfig):
    # Start the control loop for teleoperation
    control_loop(
        robot,
        control_time_s=cfg.teleop_time_s,  # Duration of teleoperation
        fps=cfg.fps,  # Frames per second for control updates
        teleoperate=True,  # Enable teleoperation mode
        display_data=cfg.display_data,  # Whether to display data during teleoperation
    )

@safe_disconnect  # Ensures safe disconnection in case of errors
def record(
    robot: Robot,
    cfg: RecordControlConfig,
) -> LeRobotDataset:
    # TODO: Add option to record logs for debugging or analysis
    if cfg.resume:  # Resume recording from an existing dataset
        dataset = LeRobotDataset(
            cfg.repo_id,  # Repository ID for the dataset
            root=cfg.root,  # Root directory for the dataset
        )
        if len(robot.cameras) > 0:  # Start image writer processes if cameras are available
            dataset.start_image_writer(
                num_processes=cfg.num_image_writer_processes,
                num_threads=cfg.num_image_writer_threads_per_camera * len(robot.cameras),
            )
        sanity_check_dataset_robot_compatibility(dataset, robot, cfg.fps, cfg.video)  # Ensure dataset compatibility
    else:  # Create a new dataset
        sanity_check_dataset_name(cfg.repo_id, cfg.policy)  # Validate dataset name
        dataset = LeRobotDataset.create(
            cfg.repo_id,
            cfg.fps,
            root=cfg.root,
            robot=robot,
            use_videos=cfg.video,
            image_writer_processes=cfg.num_image_writer_processes,
            image_writer_threads=cfg.num_image_writer_threads_per_camera * len(robot.cameras),
        )

    policy = None if cfg.policy is None else make_policy(cfg.policy, ds_meta=dataset.meta)  # Load pretrained policy if provided

    if not robot.is_connected:  # Connect the robot if not already connected
        robot.connect()

    listener, events = init_keyboard_listener()  # Initialize keyboard listener for user input

    # Warmup phase to prepare the robot and environment
    enable_teleoperation = policy is None  # Enable teleoperation if no policy is provided
    log_say("Warmup record", cfg.play_sounds)  # Log the start of the warmup phase
    warmup_record(robot, events, enable_teleoperation, cfg.warmup_time_s, cfg.display_data, cfg.fps)

    if has_method(robot, "teleop_safety_stop"):  # Stop the robot safely if the method is available
        robot.teleop_safety_stop()

    recorded_episodes = 0  # Counter for recorded episodes
    while True:  # Loop to record multiple episodes
        if recorded_episodes >= cfg.num_episodes:  # Stop if the required number of episodes is recorded
            break

        log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)  # Log the start of a new episode
        record_episode(
            robot=robot,
            dataset=dataset,
            events=events,
            episode_time_s=cfg.episode_time_s,
            display_data=cfg.display_data,
            policy=policy,
            fps=cfg.fps,
            single_task=cfg.single_task,
        )

        # Reset the environment between episodes
        if not events["stop_recording"] and (
            (recorded_episodes < cfg.num_episodes - 1) or events["rerecord_episode"]
        ):
            log_say("Reset the environment", cfg.play_sounds)
            reset_environment(robot, events, cfg.reset_time_s, cfg.fps)

        if events["rerecord_episode"]:  # Handle re-recording of the current episode
            log_say("Re-record episode", cfg.play_sounds)
            events["rerecord_episode"] = False
            events["exit_early"] = False
            dataset.clear_episode_buffer()
            continue

        dataset.save_episode()  # Save the recorded episode
        recorded_episodes += 1  # Increment the counter

        if events["stop_recording"]:  # Stop recording if the user requests it
            break

    log_say("Stop recording", cfg.play_sounds, blocking=True)  # Log the end of recording
    stop_recording(robot, listener, cfg.display_data)  # Stop the recording process

    if cfg.push_to_hub:  # Push the dataset to the hub if required
        dataset.push_to_hub(tags=cfg.tags, private=cfg.private)

    log_say("Exiting", cfg.play_sounds)  # Log the exit
    return dataset  # Return the recorded dataset

# Additional functions omitted for brevity...
