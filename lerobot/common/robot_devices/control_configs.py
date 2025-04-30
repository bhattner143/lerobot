
from dataclasses import dataclass
from pathlib import Path

import draccus

from lerobot.common.robot_devices.robots.configs import RobotConfig
from lerobot.configs import parser_dips
from lerobot.configs.policies import PreTrainedConfig
import logging
from lerobot.common.utils.utils import (
    init_logging,
)


@dataclass
class ControlConfig(draccus.ChoiceRegistry):
    pass


@ControlConfig.register_subclass("calibrate")
@dataclass
class CalibrateControlConfig(ControlConfig):
    # List of arms to calibrate (e.g. `--arms='["left_follower","right_follower"]' left_leader`)
    arms: list[str] | None = None

    def __post_init__(self):
        # Initialize logging
        init_logging()
        # Set the logging level to INFO
        logging.getLogger().setLevel(logging.INFO)
        # Log the configuration
        logging.info(f"CalibrateControlConfig initialized with arms: {self.arms}")


@ControlConfig.register_subclass("teleoperate")
@dataclass
class TeleoperateControlConfig(ControlConfig):
    # Limit the maximum frames per second. By default, no limit.
    fps: int | None = None
    teleop_time_s: float | None = None
    # Display all cameras on screen
    display_data: bool = False


@ControlConfig.register_subclass("record")
@dataclass
class RecordControlConfig(ControlConfig):
    # Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).
    repo_id: str
    # A short but accurate description of the task performed during the recording (e.g. "Pick the Lego block and drop it in the box on the right.")
    single_task: str
    # Root directory where the dataset will be stored (e.g. 'dataset/path').
    root: str | Path | None = None
    policy: PreTrainedConfig | None = None
    # Limit the frames per second. By default, uses the policy fps.
    fps: int | None = None
    # Number of seconds before starting data collection. It allows the robot devices to warmup and synchronize.
    warmup_time_s: int | float = 10
    # Number of seconds for data recording for each episode.
    episode_time_s: int | float = 60
    # Number of seconds for resetting the environment after each episode.
    reset_time_s: int | float = 60
    # Number of episodes to record.
    num_episodes: int = 50
    # Encode frames in the dataset into video
    video: bool = True
    # Upload dataset to Hugging Face hub.
    push_to_hub: bool = True
    # Upload on private repository on the Hugging Face hub.
    private: bool = False
    # Add tags to your dataset on the hub.
    tags: list[str] | None = None
    # Number of subprocesses handling the saving of frames as PNG. Set to 0 to use threads only;
    # set to ≥1 to use subprocesses, each using threads to write images. The best number of processes
    # and threads depends on your system. We recommend 4 threads per camera with 0 processes.
    # If fps is unstable, adjust the thread count. If still unstable, try using 1 or more subprocesses.
    num_image_writer_processes: int = 0
    # Number of threads writing the frames as png images on disk, per camera.
    # Too many threads might cause unstable teleoperation fps due to main thread being blocked.
    # Not enough threads might cause low camera fps.
    num_image_writer_threads_per_camera: int = 4
    # Display all cameras on screen
    display_data: bool = False
    # Use vocal synthesis to read events.
    play_sounds: bool = True
    # Resume recording on an existing dataset.
    resume: bool = False
    # Use local files only. By default, this script will fetch from local files
    local_files_only: bool = True

    def __post_init__(self):
        print(f"Initialized RecordControlConfig with repo_id={self.repo_id}, single_task={self.single_task}")
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser_dips.get_path_arg("control.policy")
        if policy_path:
            cli_overrides = parser_dips.get_cli_overrides("control.policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path


@ControlConfig.register_subclass("replay")
@dataclass
class ReplayControlConfig(ControlConfig):
    # Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).
    repo_id: str
    # Index of the episode to replay.
    episode: int
    # Root directory where the dataset will be stored (e.g. 'dataset/path').
    root: str | Path | None = None
    # Limit the frames per second. By default, uses the dataset fps.
    fps: int | None = None
    # Use vocal synthesis to read events.
    play_sounds: bool = True


@dataclass
class ControlPipelineConfig:
    robot: RobotConfig
    control: ControlConfig

    def __post_init__(self):
        # Set the logging level to INFO
        logging.getLogger().setLevel(logging.INFO)
        # Log the configuration
        logging.info(f"ControlPipelineConfig initialized with robot: {self.robot}, control: {self.control}")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["control.policy"]
    
    # Convert the configuration to a dictionary
    def to_dict(self) -> dict:
        return draccus.encode(self)
