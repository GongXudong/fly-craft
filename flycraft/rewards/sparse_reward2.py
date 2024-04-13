from typing import Union
import numpy as np
from collections import namedtuple
from pathlib import Path

import sys

PROJECT_ROOT_DIR = Path(__file__).parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))
    
from rewards.reward_base import RewardBase


class SparseReward2(RewardBase):

    def __init__(self, 
        is_potential: bool = False, log_history_reward: bool=False,
        step_frequence: int=100, integral_time_length: float=1.,
        v_threshold: float=10., mu_threshold: float=1., chi_threshold: float=1.,
        reach_target_reward: float=0., else_reward: float=0. 
    ) -> None:
        super().__init__(is_potential=is_potential, log_history_reward=log_history_reward)

        self.is_potential = is_potential
        
        self.step_frequence = step_frequence
        self.integral_time_length = integral_time_length
        
        self.v_threshold = v_threshold
        self.mu_threshold = mu_threshold
        self.chi_threshold = chi_threshold

        self.reach_target_reward = reach_target_reward
        self.else_reward = else_reward

    def get_reward(self, state: Union[namedtuple, np.ndarray], **kwargs) -> float:
        assert "goal_v" in kwargs, "参数中需要包括goal_v，再调用get_termination方法"
        assert "goal_mu" in kwargs, "参数中需要包括goal_mu，再调用get_termination方法"
        assert "goal_chi" in kwargs, "参数中需要包括goal_chi，再调用get_termination方法"
        assert "state_list" in kwargs, "参数中需要包括state_list（list[namedtuple]类型，所有历史观测），再调用get_reward方法"
        assert type(kwargs["state_list"]) is list, "state_list参数的类型应该是list[namedtuple]"

        state_list = kwargs["state_list"]  # list[namedtuple], GuideEnv.get_state_vars()返回的namedtuple

        if len(state_list) < self.integral_window_length:
            return self.else_reward
        else:
            v_flag, mu_flag, chi_flag = False, False, False
            if sum([abs(kwargs["goal_v"] - item.v) for item in state_list[-self.integral_window_length:]]) < self.v_integral_threshold:
                v_flag = True
            if sum([abs(kwargs["goal_mu"] - item.mu) for item in state_list[-self.integral_window_length:]]) < self.mu_integral_threshold:
                mu_flag = True
            if sum([abs(kwargs["goal_chi"] - item.chi) for item in state_list[-self.integral_window_length:]]) < self.chi_integral_threshold:
                chi_flag = True
            if v_flag and mu_flag and chi_flag:
                return self.reach_target_reward
            else:
                return self.else_reward
    
    def reset(self):
        super().reset()

    @property
    def integral_window_length(self):
        """v, mu, chi的积分窗口长度

        Returns:
            _type_: _description_
        """
        return self.integral_time_length * self.step_frequence

    @property
    def v_integral_threshold(self):
        """v的误差积分阈值，当v在最后self.integral_window_length上的积分小于该值时，认为v达到目标值

        Returns:
            _type_: _description_
        """
        return self.v_threshold * self.integral_window_length
    
    @property
    def mu_integral_threshold(self):
        """mu的误差积分阈值，当mu在最后self.integral_window_length上的积分小于该值时，认为mu达到目标值

        Returns:
            _type_: _description_
        """
        return self.mu_threshold * self.integral_window_length
    
    @property
    def chi_integral_threshold(self):
        """chi的误差积分阈值，当chi在最后self.integral_window_length上的积分小于该值时，认为chi达到目标值

        Returns:
            _type_: _description_
        """
        return self.chi_threshold * self.integral_window_length