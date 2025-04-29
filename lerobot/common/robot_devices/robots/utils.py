# Copyright 2024 The HuggingFace Inc. team. All rights reserved.

from typing import Protocol

from lerobot.common.robot_devices.robots.configs import (
    ManipulatorRobotConfig,
    RobotConfig,
    So100RobotConfig,
    DensoRobotConfig
)


def get_arm_id(name, arm_type):
    """Returns the string identifier of a robot arm. For instance, for a bimanual manipulator
    like Aloha, it could be left_follower, right_follower, left_leader, or right_leader.
    """
    return f"{name}_{arm_type}"


class Robot(Protocol):
    # TODO(rcadene, aliberts): Add unit test checking the protocol is implemented in the corresponding classes
    robot_type: str
    features: dict

    def connect(self): ...
    def run_calibration(self): ...
    def teleop_step(self, record_data=False): ...
    def capture_observation(self): ...
    def send_action(self, action): ...
    def disconnect(self): ...


def make_robot_config(robot_type: str, **kwargs) -> RobotConfig:
    if robot_type == "denso":
        return DensoRobotConfig(**kwargs)
    elif robot_type == "so100":
        return So100RobotConfig(**kwargs)
    else:
        raise ValueError(f"Robot type '{robot_type}' is not available.")


def make_robot_from_config(config: RobotConfig):
    if isinstance(config, ManipulatorRobotConfig):
        from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot

        return ManipulatorRobot(config)
    elif isinstance(config, DensoRobotConfig):
        from lerobot.common.robot_devices.robots.remote_manipulator import RemoteMobileManipulator

        return RemoteMobileManipulator(config)
    else:
        raise ValueError(
            f"Robot type '{config.type}' is not available. Please check the robot type in the configuration."
        )


def make_robot(robot_type: str, **kwargs) -> Robot:
    config = make_robot_config(robot_type, **kwargs)
    return make_robot_from_config(config)
