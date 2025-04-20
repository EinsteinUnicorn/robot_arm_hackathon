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

- Recalibrate your robot:
```bash
python lerobot/scripts/control_robot.py \
    --robot.type=so100 \
    --control.type=calibrate
```

- Unlimited teleoperation at highest frequency (~200 Hz is expected), to exit with CTRL+C:
```bash
python lerobot/scripts/control_robot.py \
    --robot.type=so100 \
    --robot.cameras='{}' \
    --control.type=teleoperate

# Add the cameras from the robot definition to visualize them:
python lerobot/scripts/control_robot.py \
    --robot.type=so100 \
    --control.type=teleoperate
```

- Unlimited teleoperation at a limited frequency of 30 Hz, to simulate data recording frequency:
```bash
python lerobot/scripts/control_robot.py \
    --robot.type=so100 \
    --control.type=teleoperate \
    --control.fps=30
```

- Record one episode in order to test replay:
```bash
python lerobot/scripts/control_robot.py \
    --robot.type=so100 \
    --control.type=record \
    --control.fps=30 \
    --control.single_task="Grasp a lego block and put it in the bin." \
    --control.repo_id=$USER/koch_test \
    --control.num_episodes=1 \
    --control.push_to_hub=True
```

- Visualize dataset:
```bash
python lerobot/scripts/visualize_dataset.py \
    --repo-id $USER/koch_test \
    --episode-index 0
```

- Replay this test episode:
```bash
python lerobot/scripts/control_robot.py replay \
    --robot.type=so100 \
    --control.type=replay \
    --control.fps=30 \
    --control.repo_id=$USER/koch_test \
    --control.episode=0
```

- Record a full dataset in order to train a policy, with 2 seconds of warmup,
30 seconds of recording for each episode, and 10 seconds to reset the environment in between episodes:
```bash
python lerobot/scripts/control_robot.py record \
    --robot.type=so100 \
    --control.type=record \
    --control.fps 30 \
    --control.repo_id=$USER/koch_pick_place_lego \
    --control.num_episodes=50 \
    --control.warmup_time_s=2 \
    --control.episode_time_s=30 \
    --control.reset_time_s=10
```

- For remote controlled robots like LeKiwi, run this script on the robot edge device (e.g. RaspBerryPi):
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=lekiwi \
  --control.type=remote_robot
```

**NOTE**: You can use your keyboard to control data recording flow.
- Tap right arrow key '->' to early exit while recording an episode and go to resseting the environment.
- Tap right arrow key '->' to early exit while resetting the environment and got to recording the next episode.
- Tap left arrow key '<-' to early exit and re-record the current episode.
- Tap escape key 'esc' to stop the data recording.
This might require a sudo permission to allow your terminal to monitor keyboard events.

**NOTE**: You can resume/continue data recording by running the same data recording command and adding `--control.resume=true`.

- Train on this dataset with the ACT policy:
```bash
python lerobot/scripts/train.py \
  --dataset.repo_id=${HF_USER}/koch_pick_place_lego \
  --policy.type=act \
  --output_dir=outputs/train/act_koch_pick_place_lego \
  --job_name=act_koch_pick_place_lego \
  --device=cuda \
  --wandb.enable=true
```

- Run the pretrained policy on the robot:
```bash
python lerobot/scripts/control_robot.py \
    --robot.type=so100 \
    --control.type=record \
    --control.fps=30 \
    --control.single_task="Grasp a lego block and put it in the bin." \
    --control.repo_id=$USER/eval_act_koch_pick_place_lego \
    --control.num_episodes=10 \
    --control.warmup_time_s=2 \
    --control.episode_time_s=30 \
    --control.reset_time_s=10 \
    --control.push_to_hub=true \
    --control.policy.path=outputs/train/act_koch_pick_place_lego/checkpoints/080000/pretrained_model
```
"""

import logging
import os
import time
import torch
from dataclasses import asdict
from pprint import pformat

import rerun as rr

# from safetensors.torch import load_file, save_file
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.factory import make_policy
from lerobot.common.robot_devices.control_configs import (
    ActionStreamControlConfig,
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
)
from lerobot.common.robot_devices.robots.utils import Robot, make_robot_from_config
from lerobot.common.robot_devices.utils import busy_wait, safe_disconnect
from lerobot.common.utils.utils import has_method, init_logging, log_say
from lerobot.configs import parser

########################################################################################
# Control modes
########################################################################################


@safe_disconnect
def calibrate(robot: Robot, cfg: CalibrateControlConfig):
    # TODO(aliberts): move this code in robots' classes
    if robot.robot_type.startswith("stretch"):
        if not robot.is_connected:
            robot.connect()
        if not robot.is_homed():
            robot.home()
        return

    arms = robot.available_arms if cfg.arms is None else cfg.arms
    unknown_arms = [arm_id for arm_id in arms if arm_id not in robot.available_arms]
    available_arms_str = " ".join(robot.available_arms)
    unknown_arms_str = " ".join(unknown_arms)

    if arms is None or len(arms) == 0:
        raise ValueError(
            "No arm provided. Use `--arms` as argument with one or more available arms.\n"
            f"For instance, to recalibrate all arms add: `--arms {available_arms_str}`"
        )

    if len(unknown_arms) > 0:
        raise ValueError(
            f"Unknown arms provided ('{unknown_arms_str}'). Available arms are `{available_arms_str}`."
        )

    for arm_id in arms:
        arm_calib_path = robot.calibration_dir / f"{arm_id}.json"
        if arm_calib_path.exists():
            print(f"Removing '{arm_calib_path}'")
            arm_calib_path.unlink()
        else:
            print(f"Calibration file not found '{arm_calib_path}'")

    if robot.is_connected:
        robot.disconnect()

    if robot.robot_type.startswith("lekiwi") and "main_follower" in arms:
        print("Calibrating only the lekiwi follower arm 'main_follower'...")
        robot.calibrate_follower()
        return

    if robot.robot_type.startswith("lekiwi") and "main_leader" in arms:
        print("Calibrating only the lekiwi leader arm 'main_leader'...")
        robot.calibrate_leader()
        return

    # Calling `connect` automatically runs calibration
    # when the calibration file is missing
    robot.connect()
    robot.disconnect()
    print("Calibration is done! You can now teleoperate and record datasets!")


@safe_disconnect
def teleoperate(robot: Robot, cfg: TeleoperateControlConfig):
    control_loop(
        robot,
        control_time_s=cfg.teleop_time_s,
        fps=cfg.fps,
        teleoperate=True,
        display_data=cfg.display_data,
    )

@safe_disconnect
def action_stream(robot: Robot, cfg: ActionStreamControlConfig):
    if not robot.is_connected:
        print("Connecting to robot")
        robot.connect()
        print("Connected to robot!!!!")
        
    actions = [
        {"action": torch.tensor([  4.88, 157.95, -80.92, -82.46, -23.02, -64.34])},
        {"action": torch.tensor([ 15.11, 157.11, -88.98, -81.45, -28.71, -61.51])},
        {"action": torch.tensor([ 24.67, 155.10, -96.06, -79.66, -34.30, -58.25])},
        {"action": torch.tensor([ 34.22, 151.92, -99.36, -77.69, -40.38, -55.00])},
        {"action": torch.tensor([ 42.66, 148.93, -99.47, -75.56, -47.15, -51.48])},
        {"action": torch.tensor([ 48.47, 145.86, -97.99, -73.86, -54.42, -47.95])},
        {"action": torch.tensor([ 53.88, 143.10, -94.54, -72.13, -61.35, -43.97])},
        {"action": torch.tensor([ 59.58, 140.11, -89.78, -70.21, -67.67, -40.30])},
        {"action": torch.tensor([ 65.14, 137.47, -84.53, -67.99, -72.53, -36.34])},
        {"action": torch.tensor([ 70.79, 135.26, -78.26, -65.79, -76.45, -32.46])},
        {"action": torch.tensor([ 76.33, 133.32, -71.69, -63.33, -79.66, -28.76])},
        {"action": torch.tensor([ 81.71, 131.18, -64.80, -60.84, -82.27, -24.74])},
        {"action": torch.tensor([ 86.43, 128.89, -58.05, -58.47, -84.26, -20.71])},
        {"action": torch.tensor([ 90.50, 126.50, -50.79, -56.06, -85.74, -16.79])},
        {"action": torch.tensor([ 94.21, 123.78, -43.36, -53.46, -87.16, -12.73])},
        {"action": torch.tensor([ 97.41, 120.67, -35.99, -50.71, -88.59,  -8.58])},
        {"action": torch.tensor([100.48, 117.19, -28.71, -47.92, -89.53,  -4.15])},
        {"action": torch.tensor([103.58, 113.40, -21.62, -44.82, -89.93,   0.38])},
        {"action": torch.tensor([106.81, 109.69, -14.55, -41.28, -89.57,   4.85])},
        {"action": torch.tensor([109.77, 105.88,  -7.34, -37.64, -88.36,   9.17])},
        {"action": torch.tensor([112.43, 101.71,   0.05, -33.99, -86.56,  13.28])},
        {"action": torch.tensor([115.13,  97.33,   7.39, -30.08, -84.01,  17.40])},
        {"action": torch.tensor([117.69,  93.13,  14.36, -25.81, -80.82,  21.41])},
        {"action": torch.tensor([120.32,  89.35,  20.87, -21.24, -77.41,  25.39])},
        {"action": torch.tensor([123.32,  85.79,  26.96, -16.70, -73.98,  29.41])},
        {"action": torch.tensor([126.32,  82.53,  32.99, -12.11, -70.48,  33.56])},
        {"action": torch.tensor([129.32,  79.49,  38.92,  -7.51, -66.91,  37.84])},
        {"action": torch.tensor([132.45,  76.53,  44.93,  -2.91, -63.35,  42.19])},
        {"action": torch.tensor([135.55,  73.39,  51.02,   1.87, -59.82,  46.56])},
        {"action": torch.tensor([138.80,  70.37,  56.82,   6.87, -56.07,  50.82])},
        {"action": torch.tensor([141.99,  67.32,  62.19,  12.23, -51.83,  54.94])},
        {"action": torch.tensor([144.77,  64.13,  66.87,  17.85, -47.42,  58.79])},
        {"action": torch.tensor([147.23,  60.91,  70.88,  23.65, -42.90,  62.40])},
        {"action": torch.tensor([149.53,  57.87,  74.41,  29.25, -38.20,  65.89])},
        {"action": torch.tensor([151.68,  54.89,  77.67,  34.65, -33.31,  69.26])},
        {"action": torch.tensor([153.86,  52.04,  80.80,  39.98, -28.40,  72.43])},
        {"action": torch.tensor([156.10,  49.12,  83.63,  45.28, -23.42,  75.49])},
        {"action": torch.tensor([158.49,  46.03,  86.09,  50.59, -18.43,  78.44])},
        {"action": torch.tensor([160.95,  42.79,  88.38,  55.76, -13.41,  81.36])},
        {"action": torch.tensor([163.43,  39.53,  90.75,  60.63,  -8.35,  84.29])},
        {"action": torch.tensor([165.70,  36.51,  93.08,  65.31,  -3.35,  87.29])},
        {"action": torch.tensor([167.79,  33.73,  95.12,  69.97,   1.55,  90.42])},
        {"action": torch.tensor([169.93,  30.99,  96.82,  74.73,   6.32,  93.56])},
        {"action": torch.tensor([172.19,  28.15,  98.31,  79.53,  10.95,  96.66])},
        {"action": torch.tensor([174.57,  25.18,  99.66,  84.15,  15.44,  99.68])},
        {"action": torch.tensor([176.95,  22.19, 100.90,  88.47,  19.75, 102.65])},
        {"action": torch.tensor([179.26,  19.38, 101.99,  92.53,  23.92, 105.59])},
        {"action": torch.tensor([181.48,  16.83, 102.78,  96.33,  27.97, 108.48])},
        {"action": torch.tensor([183.58,  14.46, 103.32,  99.96,  31.87, 111.33])},
        {"action": torch.tensor([185.65,  12.20, 103.69, 103.57,  35.74, 114.16])},
        {"action": torch.tensor([187.74,  10.08, 104.03, 107.31,  39.73, 117.06])},
        {"action": torch.tensor([189.91,   8.10, 104.41, 111.24,  43.93, 120.10])},
        {"action": torch.tensor([192.08,   6.14, 104.71, 115.15,  48.25, 123.26])},
        {"action": torch.tensor([194.04,   4.18, 104.65, 118.71,  52.57, 126.39])},
        {"action": torch.tensor([195.91,   2.16, 104.26, 121.96,  56.83, 129.44])},
        {"action": torch.tensor([197.83,   0.17, 103.78, 125.19,  61.15, 132.50])},
        {"action": torch.tensor([199.81,  -1.77, 103.39, 128.61,  65.62, 135.63])},
        {"action": torch.tensor([201.69,  -3.83, 103.00, 132.13,  70.18, 138.79])},
        {"action": torch.tensor([203.34,  -5.97, 102.52, 135.50,  74.67, 141.90])},
        {"action": torch.tensor([204.89,  -8.14, 101.93, 138.61,  79.08, 144.95])},
        {"action": torch.tensor([206.46, -10.29, 101.22, 141.54,  83.44, 147.93])},
        {"action": torch.tensor([208.09, -12.38, 100.41, 144.42,  87.80, 150.86])},
        {"action": torch.tensor([209.73, -14.39,  99.58, 147.38,  92.21, 153.76])},
        {"action": torch.tensor([211.33, -16.32,  98.80, 150.45,  96.66, 156.63])},
        {"action": torch.tensor([212.88, -18.20,  98.13, 153.61, 101.14, 159.47])},
        {"action": torch.tensor([214.43, -20.07,  97.53, 156.75, 105.62, 162.32])},
        {"action": torch.tensor([215.99, -21.97,  96.90, 159.84, 110.05, 165.18])},
        {"action": torch.tensor([217.59, -23.87,  96.19, 162.91, 114.45, 168.02])},
        {"action": torch.tensor([219.15, -25.72,  95.39, 166.05, 118.86, 170.84])},
        {"action": torch.tensor([220.60, -27.48,  94.53, 169.29, 123.33, 173.61])},
        {"action": torch.tensor([221.97, -29.17,  93.67, 172.60, 127.83, 176.36])},
        {"action": torch.tensor([223.37, -30.85,  92.82, 175.91, 132.35, 179.11])}
    ]

    
    for frame in actions:
        print(f"Moving to position: {frame['action']}")
        action = frame["action"]
        print(f"Sending action: {action}")
        robot.send_action(action)
        print("Sent action")

        time.sleep(.1)  # Wait 2 seconds at each position

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

    listener, events = init_keyboard_listener()

    # Execute a few seconds without recording to:
    # 1. teleoperate the robot to move it in starting position if no policy provided,
    # 2. give times to the robot devices to connect and start synchronizing,
    # 3. place the cameras windows on screen
    enable_teleoperation = policy is None
    log_say("Warmup record", cfg.play_sounds)
    warmup_record(robot, events, enable_teleoperation, cfg.warmup_time_s, cfg.display_data, cfg.fps)

    if has_method(robot, "teleop_safety_stop"):
        robot.teleop_safety_stop()

    recorded_episodes = 0
    while True:
        if recorded_episodes >= cfg.num_episodes:
            break

        log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)
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

        # Execute a few seconds without recording to give time to manually reset the environment
        # Current code logic doesn't allow to teleoperate during this time.
        # TODO(rcadene): add an option to enable teleoperation during reset
        # Skip reset for the last episode to be recorded
        if not events["stop_recording"] and (
            (recorded_episodes < cfg.num_episodes - 1) or events["rerecord_episode"]
        ):
            log_say("Reset the environment", cfg.play_sounds)
            reset_environment(robot, events, cfg.reset_time_s, cfg.fps)

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


@safe_disconnect
def replay(
    robot: Robot,
    cfg: ReplayControlConfig,
):
    # TODO(rcadene, aliberts): refactor with control_loop, once `dataset` is an instance of LeRobotDataset
    # TODO(rcadene): Add option to record logs

    dataset = LeRobotDataset(cfg.repo_id, root=cfg.root, episodes=[cfg.episode])
    actions = dataset.hf_dataset.select_columns("action")

    if not robot.is_connected:
        robot.connect()

    log_say("Replaying episode", cfg.play_sounds, blocking=True)
    for idx in range(dataset.num_frames):
        start_episode_t = time.perf_counter()

        action = actions[idx]["action"]
        robot.send_action(action)

        dt_s = time.perf_counter() - start_episode_t
        busy_wait(1 / cfg.fps - dt_s)

        dt_s = time.perf_counter() - start_episode_t
        log_control_info(robot, dt_s, fps=cfg.fps)


def _init_rerun(control_config: ControlConfig, session_name: str = "lerobot_control_loop") -> None:
    """Initializes the Rerun SDK for visualizing the control loop.

    Args:
        control_config: Configuration determining data display and robot type.
        session_name: Rerun session name. Defaults to "lerobot_control_loop".

    Raises:
        ValueError: If viewer IP is missing for non-remote configurations with display enabled.
    """
    if (control_config.display_data and not is_headless()) or (
        control_config.display_data and isinstance(control_config, RemoteRobotConfig)
    ):
        # Configure Rerun flush batch size default to 8KB if not set
        batch_size = os.getenv("RERUN_FLUSH_NUM_BYTES", "8000")
        os.environ["RERUN_FLUSH_NUM_BYTES"] = batch_size

        # Initialize Rerun based on configuration
        rr.init(session_name)
        if isinstance(control_config, RemoteRobotConfig):
            viewer_ip = control_config.viewer_ip
            viewer_port = control_config.viewer_port
            if not viewer_ip or not viewer_port:
                raise ValueError(
                    "Viewer IP & Port are required for remote config. Set via config file/CLI or disable control_config.display_data."
                )
            logging.info(f"Connecting to viewer at {viewer_ip}:{viewer_port}")
            rr.connect_tcp(f"{viewer_ip}:{viewer_port}")
        else:
            # Get memory limit for rerun viewer parameters
            memory_limit = os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "10%")
            rr.spawn(memory_limit=memory_limit)


@parser.wrap()
def control_robot(cfg: ControlPipelineConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    robot = make_robot_from_config(cfg.robot)

    # TODO(Steven): Blueprint for fixed window size
    print(cfg.control)
    print(type(cfg.control))

    if isinstance(cfg.control, CalibrateControlConfig):
        calibrate(robot, cfg.control)
    elif isinstance(cfg.control, TeleoperateControlConfig):
        _init_rerun(control_config=cfg.control, session_name="lerobot_control_loop_teleop")
        teleoperate(robot, cfg.control)
    elif isinstance(cfg.control, RecordControlConfig):
        _init_rerun(control_config=cfg.control, session_name="lerobot_control_loop_record")
        record(robot, cfg.control)
    elif isinstance(cfg.control, ReplayControlConfig):
        replay(robot, cfg.control)
    elif isinstance(cfg.control, RemoteRobotConfig):
        from lerobot.common.robot_devices.robots.lekiwi_remote import run_lekiwi

        _init_rerun(control_config=cfg.control, session_name="lerobot_control_loop_remote")
        run_lekiwi(cfg.robot)
    elif isinstance(cfg.control, ActionStreamControlConfig): 
        _init_rerun(control_config=cfg.control, session_name="lerobot_control_loop_action_stream")
        action_stream(robot, cfg.control)

    if robot.is_connected:
        # Disconnect manually to avoid a "Core dump" during process
        # termination due to camera threads not properly exiting.
        robot.disconnect()


if __name__ == "__main__":
    control_robot()
