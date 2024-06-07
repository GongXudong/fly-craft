# fly-craft

An efficient goal-conditioned reinforcement learning environment for fixed-wing UAV velocity vector control.

[![PyPI version](https://img.shields.io/pypi/v/flycraft.svg?logo=pypi&logoColor=FFE873)](https://pypi.org/project/flycraft/)
[![Downloads](https://static.pepy.tech/badge/flycraft)](https://pepy.tech/project/flycraft)
[![GitHub](https://img.shields.io/github/license/gongxudong/fly-craft.svg)](LICENSE.txt)

## Demos

The policies are trained by "Iterative Regularized Policy Optimization with Imperfect Demonstrations (ICML2024)". [Code](https://github.com/GongXudong/IRPO)

### Target velocity vector (v, $\mu$, $\chi$) from (200, 0, 0) to (140, -40, -165)
![target velocity vector (v, $\mu$, $\chi$) from (200, 0, 0) to (140, -40, -165)](assets/traj_140_-40_-165.gif)

### Target velocity vector (v, $\mu$, $\chi$) from (200, 0, 0) to (120, 50, 170)
![target velocity vector (v, $\mu$, $\chi$) from (200, 0, 0) to (120, 50, 170)](assets/traj_120_50_170.gif)


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

## Citation

Cite as

```bib
@article{gong2024flycraft,
  title        = {FlyCraft: An Efficient Goal-Conditioned Reinforcement Learning Environment for Fixed-Wing UAV Velocity Vector Control},
  author       = {Xudong, Gong and Hao, Wang and Dawei, Feng and Weijia, Wang},
  year         = 2024,
  journal      = {},
}
```
