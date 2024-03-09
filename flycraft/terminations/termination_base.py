from abc import ABC, abstractmethod
from typing import Tuple
from collections import namedtuple
import logging
import numpy as np


class TerminationBase(ABC):

    def __init__(self, 
        termination_reward: float=-1.,
        is_termination_reward_based_on_steps_left: bool=False,
        env_config: dict=None,
        my_logger: logging.Logger=None
    ) -> None:
        """
        Args:
            termination_reward (float, optional): 结束奖励. Defaults to -1..
            is_termination_reward_based_on_steps_left (bool, optional): 计算结束奖励时，是否基于剩余的仿真步数。不是的话，结束奖励就是termination_reward；是的话，结束奖励按剩余每一步的奖励是-1计算，因此累计起来是(-1) * (1 - gamma^{step_left})/(1-gamma). Defaults to False.
            max_episode_steps (int, optional): 环境的最大仿真步长. Defaults to 400.
            my_logger (logging.Logger, optional): _description_. Defaults to None.
        """
        self.termination_reward = termination_reward
        self.is_termination_reward_based_on_steps_left = is_termination_reward_based_on_steps_left
        self.env_config = env_config
        self.logger = my_logger

    @abstractmethod
    def get_termination(self, state: namedtuple, **kwargs) -> Tuple[bool, bool]:
        """返回episode是否结束，是否被截断

        Returns:
            Tuple[bool, bool]: terminated, truncated
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_termination_and_reward(self, state: namedtuple, **kwargs) -> Tuple[bool, bool, float]:
        """返回episode是否结束，是否被截断，以及结束的对应的奖励

        Args:
            state (_type_): _description_

        Returns:
            Tuple[float, bool, bool]: _description_
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    def get_penalty_base_on_steps_left(self, steps_cnt: int=1):
        """用于环境期望仿真长度固定的情况。
        
        当触发了提前终止条件后，假设后续每一步的奖励是-1，

        累加起来就是 - (1 - gamma^steps_left) / (1 - gamma).

        Args:
            steps_cnt (int, optional): 环境已经仿真的步数. Defaults to 1.

        Returns:
            _type_: _description_
        """
        return - (1 -  np.power(self.rl_gamma, self.max_episode_steps - steps_cnt)) / (1 - self.rl_gamma)
    
    def get_termination_penalty(self, terminated: bool=False, steps_cnt: int=1):
        """计算仿真结束对应的惩罚值。

        Args:
            terminated (bool, optional): 仿真是否结束. Defaults to False.
            steps_cnt (int, optional): 环境已经仿真的步数. Defaults to 1.

        Returns:
            _type_: _description_
        """
        if terminated:
            if self.is_termination_reward_based_on_steps_left:
                reward = self.get_penalty_base_on_steps_left(steps_cnt)
            else:
                reward = self.termination_reward
        else:
            reward = 0.
        
        return reward
    
    @property
    def rl_gamma(self):
        return self.env_config["task"].get("gamma")

    @property
    def step_frequence(self):
        return self.env_config["task"].get("step_frequence")

    @property
    def max_simulate_time(self):
        return self.env_config["task"].get("max_simulate_time")

    @property
    def max_episode_steps(self):
        return self.max_simulate_time * self.step_frequence