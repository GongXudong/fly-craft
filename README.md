# fly-craft

An efficient goal-conditioned reinforcement learning environment for fixed-wing UAV velocity vector control based on **Gymnasium**.

[![PyPI version](https://img.shields.io/pypi/v/flycraft.svg?logo=pypi&logoColor=FFE873)](https://pypi.org/project/flycraft/)
[![Downloads](https://static.pepy.tech/badge/flycraft)](https://pepy.tech/project/flycraft)
[![GitHub](https://img.shields.io/github/license/gongxudong/fly-craft.svg)](LICENSE.txt)

## Demos

The policies are trained by "Iterative Regularized Policy Optimization with Imperfect Demonstrations (ICML2024)". [Code](https://github.com/GongXudong/IRPO)

### Target velocity vector (v, $\mu$, $\chi$) from (200, 0, 0) to (140, -40, -165)
![target velocity vector (v, $\mu$, $\chi$) from (200, 0, 0) to (140, -40, -165)](https://github.com/GongXudong/fly-craft/blob/main/assets/traj_140_-40_-165.gif)

### Target velocity vector (v, $\mu$, $\chi$) from (200, 0, 0) to (120, 50, 170)
![target velocity vector (v, $\mu$, $\chi$) from (200, 0, 0) to (120, 50, 170)](https://github.com/GongXudong/fly-craft/blob/main/assets/traj_120_50_170.gif)

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

[Here](https://github.com/GongXudong/fly-craft/tree/main/flycraft/configs/MR_for_HER.json) is an example of the configuration, which consists of 4 blocks:

### task

The configurations about task and simulator, including：

* **control_mode** Str: the model to be trained, _guidance_law_mode_ for guidance law model, _end_to_end_mode_ for end-to-end model
* **step_frequence** Int (Hz): simulation frequency.
* **max_simulate_time** Int (s): maximum simulation time, max_simulate_time * step_frequence equals maximum length of an episode.
* **h0** Int (m): initial altitude of the aircraft.
* **v0** Int (m/s): initial true air speed of the aircraft.

### goal

The configurations about the definition and sampling method of the desired goal, including:

* **use_fixed_goal** Boolean: whether to use a fixed desired goal.
* **goal_v** Float: the true air speed of the fixed desired goal.
* **goal_mu** Float: the flight path elevator angle of the fixed desired goal.
* **goal_chi** Float: the flight path azimuth angle of the fixed desired goal.
* **sample_random** Boolean: if don't use fixed desired goal, whether sample desired goal randomly from ([v_min, v_max], [mu_min, mu_max], [chi_min, chi_max])
* **v_min** Float: the min value of true air speed of desired goal.
* **v_max** Float: the max value of true air speed of desired goal.
* **mu_min** Float: the min value of flight path elevator angle of desired goal.
* **mu_max** Float: the max value of flight path elevator angle of desired goal.
* **chi_min** Float: the min value of flight path azimuth angle of desired goal.
* **chi_max** Float: the max value of flight path azimuth angle of desired goal.
* **available_goals_file** Str: path of the file of available desired goals. If don't use fixed desired goal and don't sample desired goal randomly, then sample desired goal from the file of available desired goals. The file is a .csv file that has at least four columns: v, mu, chi, length. The column 'length' is used to indicate whether the desired goal represented by the row can be achieved by an expert. If it can be completed, it represents the number of steps required to achieved the desired goal. If it cannot be completed, the value is 0.
* **sample_reachable_goal** Boolean: when sampling desired goals from available_goals_file, should only those desired goals with length>0 be sampled.
* **sample_goal_noise_std** Tuple[Float]: a tuple with three float. The standard deviation used to add Gaussian noise to the true air speed, flight path elevation angle, and flight path azimuth angle of the sampled desired goal.

### rewards

The configurations about rewards, including:

* **dense** Dict: The configurations of the dense reward: _use_, a boolean indicating whether use this reward; _b_, a float indicating the exponent used for each reward component; _angle_weight_: a float about the coefficient of the angle error component of reward; _angle_scale_, a float used to scale the error in direction of velocity vector; _velocity_scale_, a float used to scale the error in true air speed of velocity vector.
* **sparse** Dict: The configurations of the sparse reward: _use_, a boolean indicating whether use this reward; _reward_constant_, a float indicating the reward when achieving the desired goal.

### terminations

The configurations about termination conditions, including:

* **RT** Dict: The configurations of the Reach Target Termination (used by non-Markovian reward), _use_, a boolean indicating whether use this termination; _integral_time_length_, a integer that represents the number of consecutive seconds required to achieve the accuracy of determining achievement; _v_threshold_, a float that represents the error band required to determine whether true air speed meets the requirements; _angle_threshold_, a float that represents the error band required to determine whether the direction of velocity vector meets the requirements; _termination_reward_, a float representing the reward that the agent can receive after triggering RT.
* **RT_SINGLE_STEP** Dict: The configurations of the Reach Target Termination (used by Markovian reward), _use_, a boolean indicating whether use this termination; _v_threshold_, a float that represents the error band required to determine whether true air speed meets the requirements; _angle_threshold_, a float that represents the error band required to determine whether the direction of velocity vector meets the requirements; _termination_reward_, a float representing the reward that the agent can receive after triggering RT_SINGLE_STEP.
* **C** Dict: The configurations of Crash Termination, _use_, a boolean indicating whether use this termination; _h0_, a float, when the altitude of aircraft is below this value, this termination triggers; _is_termination_reward_based_on_steps_left_, a boolean indicating whether calculate the reward (penalty) based on the max_episode_step and the current steps; _termination_reward_, a float indicating the reward when triggers this termination if 'is_termination_reward_based_on_steps_left' is set to False.
* **ES** Dict: The configurations of Extreme State Termination, _use_, a boolean indicating whether use this termination; _v_max_, a float, when the true air spped exceeding this value, this termination triggers; _p_max_, a float, when the roll angular spped exceeding this value, this termination triggers; _is_termination_reward_based_on_steps_left_, a boolean indicating whether calculate the reward (penalty) based on the max_episode_step and the current steps; _termination_reward_, a float indicating the reward when triggers this termination if 'is_termination_reward_based_on_steps_left' is set to False.
* **T** Dict: The configurations of Timeout Termination, _use_, a boolean indicating whether use this termination; _termination_reward_, a float indicating the reward when triggers this termination if 'is_termination_reward_based_on_steps_left' is set to False.
* **CMA** Dict: The configurations of Continuously Move Away Termination, _use_, a boolean indicating whether use this termination; _time_window_, an integer, the time window used to detect whether this termination condition will be triggered; _ignore_mu_error_, a float, when the error of flight path elevator angle is less than this value, the termination condition will no longer be considered; _ignore_chi_error_, a float, when the error of flight path azimuth angle is less than this value, the termination condition will no longer be considered; _is_termination_reward_based_on_steps_left_, a boolean indicating whether calculate the reward (penalty) based on the max_episode_step and the current steps; _termination_reward_, a float indicating the reward when triggers this termination if 'is_termination_reward_based_on_steps_left' is set to False.
* **CR** Dict: The configurations of Continuously Roll Termination, _use_, a boolean indicating whether use this termination; _continuousely_roll_threshold_, a float, when the angle of continuous roll exceeds this value, this termination condition is triggered; _is_termination_reward_based_on_steps_left_, a boolean indicating whether calculate the reward (penalty) based on the max_episode_step and the current steps; _termination_reward_, a float indicating the reward when triggers this termination if 'is_termination_reward_based_on_steps_left' is set to False.
* **NOBR** Dict: The configurations of Negative Overload and Big Roll Termination, _use_, a boolean indicating whether use this termination; _time_window_, an integer, the time window used to detect whether this termination condition will be triggered; _negative_overload_threshold_, a float, when the overloat exceeds this value for at least 'time_window' seconds, this termination condition is triggered; _big_phi_threshold_, a float, when the roll angle exceeds this value for at least 'time_window' seconds, this termination condition is triggered; _is_termination_reward_based_on_steps_left_, a boolean indicating whether calculate the reward (penalty) based on the max_episode_step and the current steps; _termination_reward_, a float indicating the reward when triggers this termination if 'is_termination_reward_based_on_steps_left' is set to False.

## Applications

### Examples

1. Examples based on [StableBaselines3](https://github.com/DLR-RM/stable-baselines3) and [Imitation](https://github.com/HumanCompatibleAI/imitation): https://github.com/GongXudong/fly-craft-examples

### Researches on FlyCraft

1. Xudong, Gong, et al. "Iterative Regularized Policy Optimization with Imperfect Demonstrations." Forty-first International Conference on Machine Learning. 2024.

## Citation

Cite as

```bib
@article{gong2024flycraft,
  title        = {FlyCraft: An Efficient Goal-Conditioned Reinforcement Learning Environment for Fixed-Wing UAV Velocity Vector Control},
  author       = {Gong, Xudong and Wang, Hao and Feng, Dawei and Wang, Weijia},
  year         = 2024,
  journal      = {},
}
```
