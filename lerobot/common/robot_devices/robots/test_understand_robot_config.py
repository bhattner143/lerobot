# -*- coding: utf-8 -*-
import abc
import logging
from dataclasses import dataclass, field
from typing import Sequence

from termcolor import colored


# ----------------------------
# Dummy ChoiceRegistry system
# ----------------------------
class ChoiceRegistry:
    _registry = {}

    @classmethod
    def register_subclass(cls, name):
        def decorator(subclass):
            cls._registry[name] = subclass
            return subclass
        return decorator

    @classmethod
    def get_choice_name(cls, subclass):
        for key, val in cls._registry.items():
            if val == subclass:
                return key
        return "unknown"


# ----------------------------
# Dummy Motor and Camera Configs
# ----------------------------
@dataclass
class MotorsBusConfig:
    port: str
    motors: dict[str, list]
    mock: bool = False


@dataclass
class FeetechMotorsBusConfig(MotorsBusConfig):
    pass


@dataclass
class CameraConfig:
    camera_index: int
    fps: int
    width: int
    height: int
    mock: bool = False


@dataclass
class OpenCVCameraConfig(CameraConfig):
    pass


# ----------------------------
# Abstract Base Config
# ----------------------------
@dataclass
class RobotConfig(ChoiceRegistry, abc.ABC):
    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

    def __post_init__(self):
        logging.info(colored(f"RobotConfig initialized with type: {self.type}", "green"))


# ----------------------------
# Intermediate Manipulator Config
# ----------------------------
@dataclass
class ManipulatorRobotConfig(RobotConfig):
    leader_arms: dict[str, MotorsBusConfig] = field(default_factory=dict)
    follower_arms: dict[str, MotorsBusConfig] = field(default_factory=dict)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    max_relative_target: list[float] | float | None = None
    gripper_open_degree: float | None = None
    mock: bool = False

    def __post_init__(self):
        logging.info(colored(f"ManipulatorRobotConfig initialized with type: {self.type}", "yellow"))

        if self.mock:
            for arm in self.leader_arms.values():
                arm.mock = True
            for arm in self.follower_arms.values():
                arm.mock = True
            for cam in self.cameras.values():
                cam.mock = True

        if self.max_relative_target is not None and isinstance(self.max_relative_target, Sequence):
            for name in self.follower_arms:
                if len(self.follower_arms[name].motors) != len(self.max_relative_target):
                    raise ValueError(
                        f"len(max_relative_target)={len(self.max_relative_target)} but follower arm '{name}' has "
                        f"{len(self.follower_arms[name].motors)} motors. Ensure lengths match."
                    )


# ----------------------------
# Concrete So100 Config
# ----------------------------
@RobotConfig.register_subclass("so100")
@dataclass
class So100RobotConfig(ManipulatorRobotConfig):
    calibration_dir: str = "/path/to/calibration/so100"
    max_relative_target: int | None = None

    leader_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": FeetechMotorsBusConfig(
                port="/dev/ttyACM1",
                motors={
                    "shoulder_pan": [1, "sts3215"],
                    "shoulder_lift": [2, "sts3215"],
                    "elbow_flex": [3, "sts3215"],
                    "wrist_flex": [4, "sts3215"],
                    "wrist_roll": [5, "sts3215"],
                    "gripper": [6, "sts3215"],
                },
            )
        }
    )

    follower_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": FeetechMotorsBusConfig(
                port="/dev/ttyACM0",
                motors={
                    "shoulder_pan": [1, "sts3215"],
                    "shoulder_lift": [2, "sts3215"],
                    "elbow_flex": [3, "sts3215"],
                    "wrist_flex": [4, "sts3215"],
                    "wrist_roll": [5, "sts3215"],
                    "gripper": [6, "sts3215"],
                },
            )
        }
    )

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "pcd_intel_real_sense": OpenCVCameraConfig(
                camera_index=4,
                fps=30,
                width=640,
                height=480,
            ),
            "rgb_intel_real_sense": OpenCVCameraConfig(
                camera_index=6,
                fps=30,
                width=640,
                height=480,
            ),
        }
    )

    mock: bool = False

    def __post_init__(self):
        super().__post_init__()
        logging.info(colored(f"So100RobotConfig initialized with type: {self.type}", "cyan"))


# ----------------------------
# Example Usage
# ----------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = So100RobotConfig(mock=True)
    print(config)