import logging
import torch
import os
import time
from dataclasses import asdict
from pprint import pformat

import rerun as rr

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

    leader_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": StaraiMotorsBusConfig(
                port="/dev/ttyUSB0",
                interval = 300,
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
            "wrist": OpenCVCameraConfig(
                camera_index=0,
                fps=30,
                width=640,
                height=480,
            ),
            "agentview": OpenCVCameraConfig(
                camera_index=2,
                fps=30,
                width=640,
                height=480,
            ),
        }
    )

    mock: bool = False



def teleop_step(
    robot, record_data=False
):
    if not robot.is_connected:
        robot.connect()

    # Prepare to assign the position of the leader to the follower
    leader_pos = {}
    for name in robot.leader_arms:
        # NOTE: only one name: "main"
        before_lread_t = time.perf_counter()
        leader_pos[name] = robot.leader_arms[name].read("Present_Position")
        leader_pos[name] = torch.from_numpy(leader_pos[name])
        # NOTE: In [6]: leader_pos
        # Out[6]: {'main': tensor([ -2.5000, -88.0000,  87.8000,   0.9000,   6.7000, 178.7000,   5.0000])}
        robot.logs[f"read_leader_{name}_pos_dt_s"] = time.perf_counter() - before_lread_t


    # Early exit when recording data is not requested
    if not record_data:
        return

    state = leader_pos["main"]

    action = leader_pos["main"] # NOTE: action here is the goal pos. It is the abs degree (not delta) of each DoF.

    # Capture images from cameras
    images = {}
    for name in robot.cameras:
        before_camread_t = time.perf_counter()
        images[name] = robot.cameras[name].async_read()
        images[name] = torch.from_numpy(images[name])
        robot.logs[f"read_camera_{name}_dt_s"] = robot.cameras[name].logs["delta_timestamp_s"]
        robot.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

    # Populate output dictionaries
    obs_dict, action_dict = {}, {}
    obs_dict["observation.state"] = state
    action_dict["action"] = action
    for name in robot.cameras:
        obs_dict[f"observation.images.{name}"] = images[name]

    return obs_dict, action_dict

@safe_stop_image_writer
def control_loop(
    robot,
    control_time_s=None,
    teleoperate=False,
    display_data=False,
    dataset: LeRobotDataset | None = None,
    events=None,
    policy: PreTrainedPolicy = None,
    fps: int | None = None,
    single_task: str | None = None,
):
    # TODO(rcadene): Add option to record logs
    if not robot.is_connected:
        robot.connect()

    if events is None:
        events = {"exit_early": False}

    if control_time_s is None:
        control_time_s = float("inf")

    if teleoperate and policy is not None:
        raise ValueError("When `teleoperate` is True, `policy` should be None.")

    if dataset is not None and single_task is None:
        raise ValueError("You need to provide a task as argument in `single_task`.")

    if dataset is not None and fps is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset['fps']} != {fps}).")

    timestamp = 0
    start_episode_t = time.perf_counter()

    # Controls starts, if policy is given it needs cleaning up
    if policy is not None:
        policy.reset()

    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if teleoperate:
            # NOTE: action here is the goal pos. It is the abs degree (not delta) of each DoF.
            observation, action = teleop_step(robot=robot, record_data=True)
            print(action)

        if dataset is not None:
            observation = {k: v for k, v in observation.items() if k not in ["task", "robot_type"]}
            frame = {**observation, **action, "task": single_task}
            dataset.add_frame(frame)

        # TODO(Steven): This should be more general (for RemoteRobot instead of checking the name, but anyways it will change soon)
        if (display_data and not is_headless()) or (display_data and robot.robot_type.startswith("lekiwi")):
            if action is not None:
                for k, v in action.items():
                    for i, vv in enumerate(v):
                        rr.log(f"sent_{k}_{i}", rr.Scalar(vv.numpy()))

            image_keys = [key for key in observation if "image" in key]
            for key in image_keys:
                rr.log(key, rr.Image(observation[key].numpy()), static=True)

        if fps is not None:
            dt_s = time.perf_counter() - start_loop_t
            busy_wait(1 / fps - dt_s)

        dt_s = time.perf_counter() - start_loop_t
        log_control_info(robot, dt_s, fps=fps)

        timestamp = time.perf_counter() - start_episode_t
        if events["exit_early"]:
            events["exit_early"] = False
            break

@safe_disconnect
def record(
    robot: Robot,
    cfg: RecordControlConfig,
) -> LeRobotDataset:
    
    # TODO(rcadene): Add option to record logs
    if cfg.resume:
        dataset = LeRobotDataset(
            cfg.repo_id,
            root=cfg.root,
        )
        if len(robot.cameras) > 0:
            dataset.start_image_writer(
                num_processes=cfg.num_image_writer_processes,
                num_threads=cfg.num_image_writer_threads_per_camera * len(robot.cameras),
            )
        sanity_check_dataset_robot_compatibility(dataset, robot, cfg.fps, cfg.video)
    else:
        # Create empty dataset or load existing saved episodes
        sanity_check_dataset_name(cfg.repo_id, cfg.policy)
        dataset = LeRobotDataset.create(
            cfg.repo_id,
            cfg.fps,
            root=cfg.root,
            robot=robot,
            use_videos=cfg.video,
            image_writer_processes=cfg.num_image_writer_processes,
            image_writer_threads=cfg.num_image_writer_threads_per_camera * len(robot.cameras),
        )

    # Load pretrained policy
    policy = None if cfg.policy is None else make_policy(cfg.policy, ds_meta=dataset.meta)

    if not robot.is_connected:
        robot.connect()

    """
    # Definition of events:
    
    if key == keyboard.Key.right:
        print("Right arrow key pressed. Exiting loop...")
        events["exit_early"] = True
    elif key == keyboard.Key.left:
        print("Left arrow key pressed. Exiting loop and rerecord the last episode...")
        events["rerecord_episode"] = True
        events["exit_early"] = True
    elif key == keyboard.Key.esc:
        print("Escape key pressed. Stopping data recording...")
        events["stop_recording"] = True
        events["exit_early"] = True
    """
    listener, events = init_keyboard_listener()


    recorded_episodes = 0
    while True:
        # NOTE: the only way to exit the loop is to exceed the cfg.num_episodes
        if recorded_episodes >= cfg.num_episodes:
            break

        log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)
        control_loop(
            robot=robot,
            control_time_s=cfg.episode_time_s,
            display_data=False,
            dataset=dataset,
            events=events,
            policy=policy,
            fps=cfg.fps,
            teleoperate=policy is None,
            single_task=cfg.single_task,
        )

        # NOTE: manual reset of the environment, just teleop the robot to the initial position
        if not events["stop_recording"] and (
            (recorded_episodes < cfg.num_episodes - 1) or events["rerecord_episode"]
        ):
            log_say("Reset the environment", cfg.play_sounds)

        if events["rerecord_episode"]:
            log_say("Re-record episode", cfg.play_sounds)
            events["rerecord_episode"] = False
            events["exit_early"] = False
            dataset.clear_episode_buffer()
            continue

        dataset.save_episode()
        recorded_episodes += 1

        if events["stop_recording"]:
            break

    log_say("Stop recording", cfg.play_sounds, blocking=True)
    stop_recording(robot, listener, cfg.display_data)

    if cfg.push_to_hub:
        dataset.push_to_hub(tags=cfg.tags, private=cfg.private)

    log_say("Exiting", cfg.play_sounds)
    return dataset


@parser.wrap()
def control_robot(cfg: ControlPipelineConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    robot = make_robot_from_config(cfg.robot)

    # TODO(Steven): Blueprint for fixed window size

    if isinstance(cfg.control, RecordControlConfig):
        _init_rerun(control_config=cfg.control, session_name="lerobot_control_loop_record")
        record(robot, cfg.control)

    if robot.is_connected:
        # Disconnect manually to avoid a "Core dump" during process
        # termination due to camera threads not properly exiting.
        robot.disconnect()
        
        
"""
python impedence.py \
    --robot.type=starai \
    --robot.cameras='{}' \
    --control.type=record \
    --control.fps=30 \
    --control.single_task="Agree." \
    --control.repo_id="Felix-Zhenghao/agreeBot" \
    --control.tags='["advx"]' \
    --control.warmup_time_s=0 \
    --control.episode_time_s=7 \
    --control.reset_time_s=5 \
    --control.num_episodes=1 \
    --control.display_data=false \
    --control.push_to_hub=false
"""

"""
python impedence.py \
    --robot.type=starai \
    --robot.cameras='{}' \
    --control.type=record \
    --control.fps=30 \
    --control.single_task="Disagree." \
    --control.repo_id="Felix-Zhenghao/disagreeBot" \
    --control.tags='["advx"]' \
    --control.warmup_time_s=0 \
    --control.episode_time_s=7 \
    --control.reset_time_s=5 \
    --control.num_episodes=1 \
    --control.display_data=false \
    --control.push_to_hub=false
"""


"""
python impedence.py \
    --robot.type=starai \
    --robot.cameras='{}' \
    --control.type=record \
    --control.fps=30 \
    --control.single_task="Get napkin." \
    --control.repo_id="Felix-Zhenghao/napkinBot" \
    --control.tags='["advx"]' \
    --control.warmup_time_s=0 \
    --control.episode_time_s=20 \
    --control.reset_time_s=5 \
    --control.num_episodes=1 \
    --control.display_data=false \
    --control.push_to_hub=false
"""

"""

python impedence.py \
    --robot.type=starai \
    --robot.cameras='{}' \
    --control.type=record \
    --control.fps=30 \
    --control.single_task="Dance." \
    --control.repo_id="Felix-Zhenghao/danceBot" \
    --control.tags='["advx"]' \
    --control.warmup_time_s=0 \
    --control.episode_time_s=20 \
    --control.reset_time_s=5 \
    --control.num_episodes=1 \
    --control.display_data=false \
    --control.push_to_hub=false
"""

"""
python impedence.py \
    --robot.type=starai \
    --robot.cameras='{}' \
    --control.type=record \
    --control.fps=30 \
    --control.single_task="Hold phone." \
    --control.repo_id="Felix-Zhenghao/holdBot" \
    --control.tags='["advx"]' \
    --control.warmup_time_s=0 \
    --control.episode_time_s=20 \
    --control.reset_time_s=5 \
    --control.num_episodes=1 \
    --control.display_data=false \
    --control.push_to_hub=false
"""

if __name__ == "__main__":
    control_robot()
