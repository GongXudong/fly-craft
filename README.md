# Fly-Craft <img src="assets/logo.png" align="left" width="5%"/>

An efficient goal-conditioned reinforcement learning environment for fixed-wing UAV velocity vector control based on **Gymnasium**.

[![PyPI version](https://img.shields.io/pypi/v/flycraft.svg?logo=pypi&logoColor=FFE873)](https://pypi.org/project/flycraft/)
[![Downloads](https://static.pepy.tech/badge/flycraft)](https://pepy.tech/project/flycraft)
[![GitHub](https://img.shields.io/github/license/gongxudong/fly-craft.svg)](LICENSE.txt)
[![Static Badge](https://img.shields.io/badge/Paper-ICLR2025-green?link=https%3A%2F%2Fopenreview.net%2Fforum%3Fid%3D5xSRg3eYZz)](https://openreview.net/forum?id=5xSRg3eYZz)

<!-- TODO: README 目录 -->

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

### Basic usage

```python
import gymnasium as gym
import flycraft

env = gym.make('FlyCraft-v0')  # use default configurations
observation, info = env.reset()

for _ in range(500):
    action = env.action_space.sample() # random action
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

### The four methods to initialize environment

```python
# 1.Initialize environment with default configurations
env = gym.make('FlyCraft-v0')

# 2.Initialize environment by passing configurations through config_file (Path or str)
env = gym.make('FlyCraft-v0', config_file=PROJECT_ROOT_DIR / "configs" / "NMR.json")

# 3.Initialize environment by passing configurations through custom_config (dict), this method will load default configurations from default path, then update the default config with custom_config
env = gym.make(
    'FlyCraft-v0', 
    custom_config={
        "task": {
            "control_mode": "end_to_end_mode",
        }
    }
)

# 4.Initialize environment by passing configurations through both config_file and custom_config. FlyCraft load config from config_file firstly, then update the loaded config with custom_config
env = gym.make(
    'FlyCraft-v0',
    config_file=PROJECT_ROOT_DIR / "configs" / "NMR.json",
    custom_config={
        "task": {
            "control_mode": "end_to_end_mode",
        }
    }
)
```

### Visualization

We provide a visualization method based on [Tacview](https://www.tacview.net/). For more details, please refer to [fly-craft-examples](https://github.com/GongXudong/fly-craft-examples).

## Applications

### Application examples

1. We provide a sister repository, [fly-craft-examples](https://github.com/GongXudong/fly-craft-examples), for flycraft, which offers a variety of training scripts based on [StableBaselines3](https://github.com/DLR-RM/stable-baselines3) and [Imitation](https://github.com/HumanCompatibleAI/imitation).

### Researches on FlyCraft

1. Xudong, Gong, et al. "**_Improving the Continuity of Goal-Achievement Ability via Policy Self-Regularization for Goal-Conditioned Reinforcement Learning_**." Forty-second International Conference on Machine Learning. **ICML, 2025**. [[Paper]]()

2. Xudong, Gong, et al. "**_VVC-Gym: A Fixed-Wing UAV Reinforcement Learning Environment for Multi-Goal Long-Horizon Problems_**." International Conference on Learning Representations. **ICLR, 2025**. [[Paper]](https://openreview.net/forum?id=5xSRg3eYZz)

3. Xudong, Gong, et al. "**_Iterative Regularized Policy Optimization with Imperfect Demonstrations_**." Forty-first International Conference on Machine Learning. **ICML, 2024**. [[Paper]](https://openreview.net/pdf?id=Gp5F6qzwGK)

4. Xudong, Gong, et al. "**_Goal-Conditioned On-Policy Reinforcement Learning_**." Advances in Neural Information Processing Systems. **NeurIPS, 2024**. [[Paper]](https://openreview.net/pdf?id=KP7EUORJYI)

5. Xudong, Gong, et al. "**_V-Pilot: A Velocity Vector Control Agent for Fixed-Wing UAVs from Imperfect Demonstrations_**." IEEE International Conference on Robotics and Automation. **ICRA, 2025**.

6. Dawei, Feng, et al. "**_Think Before Acting: The Necessity of Endowing Robot Terminals With the Ability to Fine-Tune Reinforcement Learning Policies_**." IEEE International Symposium on Parallel and Distributed Processing with Applications. **ISPA, 2024**.

## Architecture

![Architecture](assets/framework.png)

## Configuration Details

[Here](https://github.com/GongXudong/fly-craft/tree/main/flycraft/configs/MR_for_HER.json) is an example of the configuration, which consists of 4 blocks:

### Task

The configurations about task and simulator, including：

* **control_mode** Str: the model to be trained, _guidance_law_mode_ for guidance law model, _end_to_end_mode_ for end-to-end model
* **step_frequence** Int (Hz): simulation frequency.
* **max_simulate_time** Int (s): maximum simulation time, max_simulate_time * step_frequence equals maximum length of an episode.
* **h0** Int (m): initial altitude of the aircraft.
* **v0** Int (m/s): initial true air speed of the aircraft.

### Desired Goal

The configurations about the definition and sampling method of the desired goal, including:

* **use_fixed_goal** Boolean: whether to use a fixed desired goal.
* **goal_v** Float (m/s): the true air speed of the fixed desired goal.
* **goal_mu** Float (deg): the flight path elevator angle of the fixed desired goal.
* **goal_chi** Float (deg): the flight path azimuth angle of the fixed desired goal.
* **sample_random** Boolean: if don't use fixed desired goal, whether sample desired goal randomly from ([v_min, v_max], [mu_min, mu_max], [chi_min, chi_max])
* **v_min** Float (m/s): the min value of true air speed of desired goal.
* **v_max** Float (m/s): the max value of true air speed of desired goal.
* **mu_min** Float (deg): the min value of flight path elevator angle of desired goal.
* **mu_max** Float (deg): the max value of flight path elevator angle of desired goal.
* **chi_min** Float (deg): the min value of flight path azimuth angle of desired goal.
* **chi_max** Float (deg): the max value of flight path azimuth angle of desired goal.
* **available_goals_file** Str: path of the file of available desired goals. If don't use fixed desired goal and don't sample desired goal randomly, then sample desired goal from the file of available desired goals. The file is a .csv file that has at least four columns: v, mu, chi, length. The column 'length' is used to indicate whether the desired goal represented by the row can be achieved by an expert. If it can be completed, it represents the number of steps required to achieved the desired goal. If it cannot be completed, the value is 0.
* **sample_reachable_goal** Boolean: when sampling desired goals from _available_goals_file_, should only those desired goals with length>0 be sampled.
* **sample_goal_noise_std** Tuple[Float]: a tuple with three float. The standard deviation used to add Gaussian noise to the true air speed, flight path elevation angle, and flight path azimuth angle of the sampled desired goal.

### Rewards

The configurations about rewards, including:

* **dense** Dict: The configurations of the dense reward that calculated by the error on angle and on the true air speed
  * _use_ Boolean: whether use this reward;
  * _b_ Float: indicates the exponent used for each reward component;
  * _angle_weight_ Float [0.0, 1.0]: the coefficient of the angle error component of reward;
  * _angle_scale_ Float (deg): the scalar used to scale the error in direction of velocity vector;
  * _velocity_scale_ Float (m/s): the scalar used to scale the error in true air speed of velocity vector.
* **dense_angle_only** Dict: The configurations of the dense reward that calculated by the error on angle only
  * _use_ Boolean: whether use this reward;
  * _b_ Float: indicates the exponent used for each reward component;
  * _angle_scale_ Float (deg): the scalar used to scale the error in direction of velocity vector.
* **sparse** Dict: The configurations of the sparse reward
  * _use_ Boolean: whether use this reward;
  * _reward_constant_ Float: the reward when achieving the desired goal.

### Terminations

The configurations about termination conditions, including:

* **RT** Dict: The configurations of the Reach Target Termination (used by non-Markovian reward)
  * _use_ Boolean: whether use this termination;
  * _integral_time_length_ Integer (s): the number of consecutive seconds required to achieve the accuracy of determining achievement;
  * _v_threshold_ Float (m/s): the error band used to determine whether true air speed meets the requirements;
  * _angle_threshold_ Float (deg): the error band used to determine whether the direction of velocity vector meets the requirements;
  * _termination_reward_ Float: the reward the agent receives when triggering RT.
* **RT_SINGLE_STEP** Dict: The configurations of the Reach Target Termination (used by Markovian reward, judge achievement by the error of true airspeed and the error of angle of velocity)
  * _use_ Boolean: whether use this termination;
  * _v_threshold_ Float (m/s): the error band used to determine whether true air speed meets the requirements;
  * _angle_threshold_ Float (deg): the error band used to determine whether the direction of velocity vector meets the requirements;
  * _termination_reward_ Float: the reward the agent receives when triggering RT_SINGLE_STEP.
* **RT_V_MU_CHI_SINGLE_STEP** Dict: The configurations of the Reach Target Termination (used by Markovian reward, judge achievement by the error of true airspeed, the error of flight path elevator angle, and the error of flight path azimuth angle)
  * _use_ Boolean: whether use this termination;
  * _v_threshold_ Float (m/s): the error band used to determine whether true air speed meets the requirements;
  * _angle_threshold_ Float (deg): the error band used to determine whether the direction of velocity vector meets the requirements;
  * _termination_reward_ Float: the reward the agent receives when triggering RT_V_MU_CHI_SINGLE_STEP.
* **C** Dict: The configurations of Crash Termination
  * _use_ Boolean: whether use this termination;
  * _h0_ Float (m): the altitude threshold below which this termination triggers;
  * _is_termination_reward_based_on_steps_left_ Boolean: whether calculate the reward (penalty) based on the max_episode_step and the current steps;
  * _termination_reward_ Float: the reward when triggers this termination under the condition of 'is_termination_reward_based_on_steps_left == False'.
* **ES** Dict: The configurations of Extreme State Termination
  * _use_ Boolean: whether use this termination;
  * _v_max_ Float (m/s): the maximum value of true air speed. when the true air speed exceeding this value, this termination triggers;
  * _p_max_ Float (deg/s): the maximum value of roll angular speed. when the roll angular speed exceeding this value, this termination triggers;
  * _is_termination_reward_based_on_steps_left_ Boolean: whether calculate the reward (penalty) based on the max_episode_step and the current steps;
  * _termination_reward_ Float: the reward when triggers this termination under the condition of 'is_termination_reward_based_on_steps_left == False'.
* **T** Dict: The configurations of Timeout Termination
  * _use_ Boolean: whether use this termination;
  * _termination_reward_ Float: the reward when triggers this termination.
* **CMA** Dict: The configurations of Continuously Move Away Termination
  * _use_ Boolean: whether use this termination;
  * _time_window_ Integer (s): the time window used to detect whether this termination condition will be triggered;
  * _ignore_mu_error_ Float (deg): when the error of flight path elevator angle is less than this value, the termination condition will no longer be considered;
  * _ignore_chi_error_ Float (deg): when the error of flight path azimuth angle is less than this value, the termination condition will no longer be considered;
  * _is_termination_reward_based_on_steps_left_ Boolean: whether calculate the reward (penalty) based on the max_episode_step and the current steps;
  * _termination_reward_ Float: the reward when triggers this termination under the condition of 'is_termination_reward_based_on_steps_left == False'.
* **CR** Dict: The configurations of Continuously Roll Termination
  * _use_ Boolean: whether use this termination;
  * _continuousely_roll_threshold_ Float (deg): when the angle of continuous roll exceeds this value, this termination condition is triggered;
  * _is_termination_reward_based_on_steps_left_ Boolean: whether calculate the reward (penalty) based on the max_episode_step and the current steps;
  * _termination_reward_ Float: the reward when triggers this termination under the condition of 'is_termination_reward_based_on_steps_left == False'.
* **NOBR** Dict: The configurations of Negative Overload and Big Roll Termination
  * _use_ Boolean: whether use this termination;
  * _time_window_ Integer (s): the time window used to detect whether this termination condition will be triggered;
  * _negative_overload_threshold_ Float: when the overloat exceeds this value for at least 'time_window' seconds, this termination condition is triggered;
  * _big_phi_threshold_ Float (deg): when the roll angle exceeds this value for at least 'time_window' seconds, this termination condition is triggered;
  * _is_termination_reward_based_on_steps_left_ Boolean: whether calculate the reward (penalty) based on the max_episode_step and the current steps;
  * _termination_reward_ Float: the reward when triggers this termination under the condition of 'is_termination_reward_based_on_steps_left == False'.

### Cases

1. Using fixed desired goal of $(u, \mu, \chi) = (100, -25， 75)$, [link](https://github.com/GongXudong/fly-craft-examples/blob/main/configs/env/fixed_target/env_config_for_ppo_100_-25_75.json).

2. Sampling desired goal $(u, \mu, \chi)$ randomly from $[150, 250] \times [-30， 30] \times [-60, 60]$, [link](https://github.com/GongXudong/fly-craft-examples/blob/main/configs/env/D2D/env_config_for_ppo_medium_b_05.json).

3. Sampling desired goal $(u, \mu, \chi)$ randomly from a pre-defined set (specified by config["goal"]["available_goals_file"]), [link](https://github.com/GongXudong/fly-craft-examples/blob/main/configs/env/IRPO/env_hard_guidance_MR_config_for_ppo_with_dg_from_demo1.json).

## Research Areas supported by FlyCraft

### 1. Goal-Conditioned Reinforcement Learning

FlyCraft is a typical multi-goal problem. Its interface adheres to the design principles of [Gymnasium-Robotics](https://github.com/Farama-Foundation/Gymnasium-Robotics). The observation is implemented using a dictionary type, which includes three keys: "_observation_", "_desired\_goal_", and "_achieved\_goal_". Additionally, it provides a _compute\_reward()_ function. This design makes Flycraft compatible with many open-source Goal-Centric Reinforcement Learning (GCRL) algorithms, facilitating the direct reuse of these algorithms for GCRL research.

### 2. Demonstration-based Reinforcement learning (Imitation Learning, Offline Reinforcement Learning, Offline-to-Online Reinforcement Learning)

We provide scripts for generating demonstration data using PID controllers and for updating demonstrations using trained RL policies. Users can utilize these scripts to generate their own demonstrations. Additionally, we have open-sourced eight datasets of varying quality and quantity, as listed below:

|Demonstration|Trajectory Number|Average Trajectory Length|Transition Number|Link|Collect Method|
|:-:|:-:|:-:|:-:|:-:|:-:|
|$D_E^0$|10,184|282.01±149.98|2,872,051|[link](https://www.openml.org/d/46000)|from PID|
|$\overline{D_E^0}$|10,264|281.83±149.48|2,892,731|[link](https://www.openml.org/d/46011)|Augment $D_E^0$|
|$D_E^1$|24,924|124.64±53.07|3,106,516|[link](https://www.openml.org/d/46012)|optimized $\overline{D_E^0}$ with RL trained policies|
|$\overline{D_E^1}$|27,021|119.64±47.55|3,232,896|[link](https://www.openml.org/d/46013)|Augment $D_E^1$|
|$D_E^2$|33,114|117.65±46.24|3,895,791|[link](https://www.openml.org/d/46014)|optimized $\overline{D_E^1}$ with RL trained policies|
|$\overline{D_E^2}$|34,952|115.76±45.65|4,045,887|[link](https://www.openml.org/d/46015)|Augment $D_E^2$|
|$D_E^3$|38,654|116.59±46.81|4,506,827|[link](https://www.openml.org/d/46016)|optimized $\overline{D_E^2}$ with RL trained policies|
|$\overline{D_E^3}$|39,835|116.56±47.62|4,643,048|[link](https://www.openml.org/d/46017)|Augment $D_E^3$|

For the specific details on how the datasets were generated, please refer to our ICLR 2025 paper: [VVCGym](https://openreview.net/pdf?id=5xSRg3eYZz).

### 3. Exploration

FlyCraft represents a typical exploration challenge problem, primarily due to the following complexities:

**Spatial Exploration Complexity (Multi-Goal)**: For non-recurrent policies, the policy search space becomes the Cartesian product of the state space, action space, and desired goal space. This significantly increases the challenge of finding effective policies.

**Temporal Exploration Complexity (Long-Horizon)**: Demonstrations typically exceed an average length of 100 steps, and some challenging tasks may require over 300 steps to complete successfully.

Furthermore, directly applying common RL algorithms like PPO or SAC on FlyCraft, even with dense rewards, yields policies with virtually no beneficial effect. Detailed experimental results are available in our ICLR 2025 paper: [VVCGym](https://openreview.net/pdf?id=5xSRg3eYZz).

### 4. Curriculum Learning

Since common RL algorithms struggle to learn from scratch on FlyCraft, it serves as an suitable testbed for researching Curriculum Learning.

We recommend adjusting task difficulty through two primary methods:

(1) **Adjusting the Desired Goal Space**

A smaller desired goal space, closer to the initial state, makes the task easier. The aircraft's initial velocity is defined by env_config["task"]["v0"]. The initial values for the flight path elevator angle and flight path azimuth angle are both 0. The desired goal space is defined by the following configuration:

* Minimum desired true airspeed: env_config["goal"]["v_min"]
* Maximum desired true airspeed: env_config["goal"]["v_max"]
* Minimum desired flight path elevator angle: env_config["goal"]["mu_min"]
* Maximum desired flight path elevator angle: env_config["goal"]["mu_max"]
* Minimum desired flight path azimuth angle: env_config["goal"]["chi_min"]
* Maximum desired flight path azimuth angle: env_config["goal"]["chi_max"]

Additionally, users can define their own desired goal sampling strategy by referring to _tasks.goal_samplers.goal_sampler_for_velocity_vector_control.py_.

(2) **Adjusting the Action Space**

Different action spaces can be employed by setting env_config["task"]["control_mode"]:

* **end_to_end_mode**: The action space consists of final control commands: the deflections of the aileron actuator, elevator actuator, rudder actuator, and power level actuator. This mode exhibits significant oscillation in actions during training. Combined with the long-horizon nature of the Velocity Vector Control task, training suffers greatly from the temporal complexity of exploration.
* **guidance_law_mode**: The action space consists of intermediate control commands: roll rate command, overload command, and the position of the power level actuator. These three commands are then converted by a PD controller-based control law model into the deflections of the aileron actuator, elevator actuator, rudder actuator, and power level actuator. The PD controller-based control law model acts to smooth the actions, serving a purpose similar to the temporal abstraction provided by frame-skipping, thereby alleviating the temporal complexity of exploration.

A related Curriculum Learning study can be found in the NeurIPS 2024 paper: [GCPO](https://openreview.net/pdf?id=KP7EUORJYI)。

### 5. Hierarchical Reinforcement Learning

The long-horizon nature of fixed-wing UAV tasks makes FlyCraft a suitable testbed for Hierarchical RL research. The task horizon can be adjusted by setting the simulation frequency env_config["task"]["step_frequence"]. We recommend setting the simulation frequency within the range of [10, 100], where a higher value corresponds to a longer task horizon.

For researchers wishing to avoid the long-horizon challenge, we suggest: (1) setting env_config["task"]["step_frequence"] to 10; (2) employing FrameSkip techniques.

### 6. Continual Learning

Users can refer to the methods for modifying task difficulty described in the Curriculum Learning section to create different MDPs, thereby facilitating research in Continual Learning.

### 7. Applications of Fixed-Wing UAVs

FlyCraft is fundamentally a simulation engine for fixed-wing UAVs. It allows researchers to study various fixed-wing UAV control problems by defining specific tasks. Currently, we have implemented:

**Velocity Vector Control**: This involves manipulating the UAV's velocity vector to match a desired velocity vector.
**Attitude Control**: This involves manipulating the UAV's attitude to match a desired attitude.

Potential future tasks include Basic Flight Maneuvers (BFMs) such as Level Turn, Slow Roll, and Knife Edge, among others.

## Citation

Cite as

```bib
@inproceedings{gong2025vvcgym,
  title        = {VVC-Gym: A Fixed-Wing UAV Reinforcement Learning Environment for Multi-Goal Long-Horizon Problems},
  author       = {Gong, Xudong and Feng, Dawei and Xu, kele and Wang, Weijia and Sun, Zhangjun and Zhou, Xing and Ding, Bo and Wang, Huaimin},
  booktitle    = {International Conference on Learning Representations},
  year         = {2025}
}
```
