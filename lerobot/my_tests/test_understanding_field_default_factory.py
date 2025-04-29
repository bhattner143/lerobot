from dataclasses import dataclass, field

# Step 1: Define a simple base class for motor bus configuration
@dataclass
class MotorsBusConfig:
    port: str
    motors: dict[str, list]  # motor name → [index, model]

# Step 2: Define a specific type of bus (inherits from base)
@dataclass
class FeetechMotorsBusConfig(MotorsBusConfig):
    pass

# Step 3: Define a robot config class that includes leader arms
@dataclass
class MyRobotConfig:
    leader_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": FeetechMotorsBusConfig(
                port="/dev/ttyACM1",
                motors={
                    "joint1": [1, "motor_model"],
                    "joint2": [2, "motor_model"]
                }
            )
        }
    )

# Step 4: Create an instance and print it
robot = MyRobotConfig()

# Access and print details
print("Leader arm name:", list(robot.leader_arms.keys())[0])
print("Port used:", robot.leader_arms["main"].port)
print("Motors:", robot.leader_arms["main"].motors)