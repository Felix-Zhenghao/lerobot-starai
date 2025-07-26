# Record and replay test

- First, record one episode (no cameras)

```
python lerobot/scripts/control_robot.py \
  --robot.type=starai \
  --robot.cameras='{}' \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Replay waving in the air." \
  --control.repo_id="Felix-Zhenghao/testReplay" \
  --control.tags='["starai","tutorial"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=20 \
  --control.reset_time_s=15 \
  --control.num_episodes=2 \
  --control.display_data=false \
  --control.push_to_hub=false
```

- Second, replay the episode

```
python lerobot/scripts/control_robot.py \
  --robot.type=starai \
  --robot.cameras='{}' \
  --control.type=replay \
  --control.fps=30 \
  --control.repo_id=Felix-Zhenghao/testReplay \
  --control.episode=0
```


### Happy robot (拿餐巾纸)

- Collect
```
python lerobot/scripts/control_robot.py   --robot.type=starai   --control.type=record   --control.fps=30   --control.single_task="Happy."   --control.repo_id="Felix-Zhenghao/happyRobot"   --control.tags='["advx"]'   --control.warmup_time_s=2   --control.episode_time_s=30   --control.reset_time_s=5   --control.num_episodes=1   --control.display_data=false  --control.push_to_hub=false
```

- Replay
```
python lerobot/scripts/control_robot.py \
  --robot.type=starai \
  --control.type=replay \
  --control.fps=30 \
  --control.repo_id=Felix-Zhenghao/happyRobot \
  --control.episode=0
```

### Open robot

- Collect
```
python lerobot/scripts/control_robot.py   --robot.type=starai   --control.type=record   --control.fps=30   --control.single_task="Open the shitbro."   --control.repo_id="Felix-Zhenghao/happyRobot"   --control.tags='["advx"]'   --control.warmup_time_s=2   --control.episode_time_s=30   --control.reset_time_s=3   --control.num_episodes=1   --control.display_data=false  --control.push_to_hub=false
```

# Angry robot

- Replay
```
python lerobot/scripts/control_robot.py \
  --robot.type=starai \
  --control.type=replay \
  --control.fps=30 \
  --control.repo_id=Felix-Zhenghao/angryBot \
  --control.episode=0
```

# Changes

- 1. line 507-510 commented.

```
lerobot/common/datasets/lerobot_dataset.py
```


# Jetson

### Python package

- torchvision
```
https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
```

- cusparse
```
https://developer.nvidia.com/cusparselt-downloads?target_os=Linux&target_arch=aarch64-jetson&Compilation=Native&Distribution=Ubuntu&target_version=22.04&target_type=deb_network
```

- torch
```
https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html
```

### Others

- Give USB autho
```
sudo chmod a+rw /dev/ttyUSB0
```
