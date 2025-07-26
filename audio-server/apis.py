"""
APIs for the shitbro robot.
"""
import logging
import torch
import os
import time
from dataclasses import asdict
from pprint import pformat

import rerun as rr
import numpy as np

# from safetensors.torch import load_file, save_file
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.factory import make_policy
from lerobot.common.robot_devices.control_configs import (
    CalibrateControlConfig,
    ControlConfig,
    ControlPipelineConfig,
    RecordControlConfig,
    RemoteRobotConfig,
    ReplayControlConfig,
    TeleoperateControlConfig,
)
from lerobot.common.robot_devices.control_utils import (
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
    go_to_rest_pose,
)
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.robot_devices.robots.utils import Robot, make_robot_from_config
from lerobot.common.robot_devices.utils import busy_wait, safe_disconnect
from lerobot.common.utils.utils import has_method, log_say
from lerobot.common.datasets.image_writer import safe_stop_image_writer
from lerobot.configs import parser
from lerobot.scripts.control_robot import _init_rerun, init_logging



##############################################
# Config robot
##############################################
from lerobot.common.robot_devices.robots.configs import (
    RobotConfig,
    ManipulatorRobotConfig,
)
from lerobot.common.robot_devices.cameras.configs import (
    CameraConfig,
    OpenCVCameraConfig,
)
from lerobot.common.robot_devices.motors.configs import (
    StaraiMotorsBusConfig,
    MotorsBusConfig,
)
from dataclasses import dataclass, field


@RobotConfig.register_subclass("starai")
@dataclass
class StaraiRobotConfig(ManipulatorRobotConfig):
    calibration_dir: str = ".cache/calibration/starai"
    
    max_relative_target: int | None = None

    follower_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": StaraiMotorsBusConfig(
                port="/dev/ttyUSB0",
                interval = 500,
                motors={
                    # name: (index, model)
                    "joint1": [0, "rx8-u50"],
                    "joint2": [1, "rx8-u50"],
                    "joint3": [2, "rx8-u50"],
                    "joint4": [3, "rx8-u50"],
                    "joint5": [4, "rx8-u50"],
                    "joint6": [5, "rx8-u50"],
                    "gripper": [6, "rx8-u50"],
                },
            ),
        }
    )

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            # "wrist": OpenCVCameraConfig(
            #     camera_index=0,
            #     fps=30,
            #     width=640,
            #     height=480,
            # ),
            # "agentview": OpenCVCameraConfig(
            #     camera_index=2,
            #     fps=30,
            #     width=640,
            #     height=480,
            # ),
        }
    )

    mock: bool = False



@safe_disconnect
def replay(
    robot: Robot,
    cfg: ReplayControlConfig,
    path: str = "data/hold.pt",
):
    
    go_to_rest_pose(
        robot=robot,
    )

    actions = torch.load(path)

    if not robot.is_connected:
        robot.connect()

    log_say("Replaying episode", cfg.play_sounds, blocking=True)
    for idx in range(len(actions)):
        start_episode_t = time.perf_counter()

        action = actions[idx]
        robot.send_action(action)

        dt_s = time.perf_counter() - start_episode_t
        busy_wait(1 / cfg.fps - dt_s)

        dt_s = time.perf_counter() - start_episode_t
        log_control_info(robot, dt_s, fps=cfg.fps)
        
    go_to_rest_pose(
        robot=robot,
    )
    
def hold_phone(
    robot: Robot,
    cfg: ControlConfig,
    signal: bool = False,
):
    go_to_rest_pose(
        robot=robot,
    )

    actions = torch.load("data/hold.pt")

    if not robot.is_connected:
        robot.connect()

    log_say("Replaying episode", cfg.play_sounds, blocking=True)
    last_action = actions[-1]
    for idx in range(len(actions)):
        start_episode_t = time.perf_counter()

        action = actions[idx]
        robot.send_action(action)

        dt_s = time.perf_counter() - start_episode_t
        busy_wait(1 / cfg.fps - dt_s)

        dt_s = time.perf_counter() - start_episode_t
        log_control_info(robot, dt_s, fps=cfg.fps)


@parser.wrap()
def control_robot(cfg: ControlPipelineConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    robot = make_robot_from_config(cfg.robot)
    
    return robot, cfg
    # import IPython; IPython.embed()

    # if isinstance(cfg.control, ReplayControlConfig):
    #     replay(robot, cfg.control)
        
    #     cfg.control.repo_id = "Felix-Zhenghao/agreeBot"
    #     # replay(robot, cfg.control)

    # if robot.is_connected:
    #     # Disconnect manually to avoid a "Core dump" during process
    #     # termination due to camera threads not properly exiting.
    #     robot.disconnect()


def get_napkins():
    pass

def hold_phone():
    pass

def go_to_rest_pos():
    pass

def be_happy():
    pass

def be_angry():
    pass

def be_sad():
    pass

def be_cute():
    pass

def agree():
    pass

def disagree():
    pass


"""
python audio-server/apis.py   --robot.type=starai  --robot.cameras='{}' --control.type=replay   --control.fps=30   --control.repo_id=Felix-Zhenghao/agreeBot   --control.episode=0
"""
if __name__ == "__main__":
    control_robot()

"""
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import torch
dataset = LeRobotDataset("Felix-Zhenghao/napkinBot")
actions = dataset.hf_dataset.select_columns("action")
all_actions = []
for idx in range(593):
    if idx <=300 or idx >=439:
        all_actions.append(actions[idx]['action'])

import torch
torch.save(all_actions, "data/napkin.pt")
"""

