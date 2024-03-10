# fly-craft

An efficient goal-conditioned reinforcement learning environment for fixed-wing UAV attitude control.

[![PyPI version](https://img.shields.io/pypi/v/flycraft.svg?logo=pypi&logoColor=FFE873)](https://pypi.org/project/flycraft/)
[![Downloads](https://static.pepy.tech/badge/flycraft)](https://pepy.tech/project/flycraft)
[![GitHub](https://img.shields.io/github/license/gongxudong/fly-craft.svg)](LICENSE.txt)

## Installation

### Using PyPI

```bash
pip install flycraft
```

### From source

```bash
git clone https://github.com/GongXudong/fly-craft.git
pip install -e fly-craft
```

## Usage

```python
import gymnasium as gym
import flycraft

env = gym.make('FlyCraft-v0')

observation, info = env.reset()

for _ in range(500):
    action = env.action_space.sample() # random action
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

## Documentation

TODO:

## Citation

Cite as

```bib
@article{gong2024flycraft,
  title        = {fly-craft: An Efficient Goal-Conditioned Environment for Fixed-Wing UAV Attitude Control},
  author       = {Xudong, Gong and Hao, Wang and Dawei, Feng and Weijia, Wang},
  year         = 2024,
  journal      = {},
}
```
