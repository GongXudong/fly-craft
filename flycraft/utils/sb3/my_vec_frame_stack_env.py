from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
from gymnasium import spaces

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper
from stable_baselines3.common.vec_env.stacked_observations import StackedObservations


class MyVecFrameStack(VecEnvWrapper):
    """
    因为观测中包含目标信息，所以原始的FrameStack中存在大量的冗余目标信息。
    前n_stack-1个观测去掉目标信息，只有当前观测包含目标信息。
    
    例：假设原始观测为obs=(sensors_obs, target)，n_stack=3时，
    framestack后的观测为(sensors_obs_0, sensors_obs_1, sensors_obs_2, target)

    此外，reset时，stack中的所有观测置为初始观测！！！

    :param venv: Vectorized environment to wrap
    :param n_stack: Number of frames to stack
    :param channels_order: If "first", stack on first image dimension. If "last", stack on last dimension.
        If None, automatically detect channel to stack over in case of image observation or default to "last" (default).
        Alternatively channels_order can be a dictionary which can be used with environments with Dict observation spaces
    """

    def __init__(self, 
        venv: VecEnv, 
        n_stack: int, 
        channels_order: Optional[Union[str, Mapping[str, str]]] = None,
        obs_cnt: int = 8, 
        target_cnt: int = 3
    ) -> None:
        assert isinstance(
            venv.observation_space, (spaces.Box, spaces.Dict)
        ), "VecFrameStack only works with gym.spaces.Box and gym.spaces.Dict observation spaces"

        self.obs_cnt = obs_cnt
        self.target_cnt = target_cnt

        self.stacked_obs = StackedObservations(
            venv.num_envs, 
            n_stack, 
            # venv.observation_space, 
            spaces.Box(
                low=venv.observation_space.low[:obs_cnt], 
                high=venv.observation_space.high[:obs_cnt]
            ),
            channels_order
        )
        # observation_space = self.stacked_obs.stacked_observation_space
        observation_space = spaces.Box(
            low=np.concatenate((self.stacked_obs.stacked_observation_space.low, venv.observation_space.low[obs_cnt:obs_cnt+target_cnt]), axis=0),
            high=np.concatenate((self.stacked_obs.stacked_observation_space.high, venv.observation_space.high[obs_cnt:obs_cnt+target_cnt]), axis=0)
        )
        super().__init__(venv, observation_space=observation_space)

    def step_wait(
        self,
    ) -> Tuple[Union[np.ndarray, Dict[str, np.ndarray]], np.ndarray, np.ndarray, List[Dict[str, Any]],]:
        observations, rewards, dones, infos = self.venv.step_wait()
        tmp_observations, infos = self.stacked_obs.update(observations[:, :self.obs_cnt], dones, infos)
        observations = np.concatenate((tmp_observations, observations[:, self.obs_cnt:(self.obs_cnt+self.target_cnt)]), axis=1)
        return observations, rewards, dones, infos

    def reset(self) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Reset all environments
        """
        observation = self.venv.reset()  # pytype:disable=annotation-type-mismatch
        tmp_observation = self.stacked_obs.reset(observation[:, :self.obs_cnt])
        for i in range(self.stacked_obs.n_stack-1):
            tmp_observation, infos = self.stacked_obs.update(observation[:, :self.obs_cnt], np.array([False for i in range(self.venv.num_envs)]), [{} for i in range(self.venv.num_envs)])
        observation = np.concatenate((tmp_observation, observation[:, self.obs_cnt:(self.obs_cnt+self.target_cnt)]), axis=1)
        return observation
