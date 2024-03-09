# fly-craft

## Documentation

TODO:

## Installation

### Using PyPI

```bash
pip install fly-craft
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
  title        = {fly-craft: Open-Source Goal-Conditioned Environments for Fixed-Wing UAV Attitude Control},
  author       = {Xudong, Gong and Hao, Wang and Dawei, Feng and Weijia, Wang},
  year         = 2024,
  journal      = {},
}
```
