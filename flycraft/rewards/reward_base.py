from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Union
import numpy as np
from typing import Any
import logging


class RewardBase(ABC):

    def __init__(
        self, 
        is_potential: bool=False, 
        log_history_reward: bool=False, 
        my_logger: logging.Logger=None
    ) -> None:
        self.is_potential = is_potential  # 奖励是否基于势
        self.pre_reward = 0.  # 上一回合的奖励值，如果奖励基于势能，该值记录势
        self.reward_trajectory = []
        self.log_history_reward = log_history_reward
        self.logger = my_logger
    
    @abstractmethod
    def reset(self):
        self.pre_reward = 0.
        self.reward_trajectory = []

    @abstractmethod
    def get_reward(self, state: Union[namedtuple, np.ndarray], **kwargs) -> float:
        raise NotImplementedError

    def _process(self, new_reward, log: Any=None):
        """根据是否基于势能做额外处理，并记录
        """
        reward = new_reward
        
        if self.is_potential:
            reward, self.pre_reward = new_reward - self.pre_reward, new_reward
        
        if self.log_history_reward:
            if log is None:
                self.reward_trajectory.append(reward)
            else: 
                self.reward_trajectory.append(log)
        
        return reward


