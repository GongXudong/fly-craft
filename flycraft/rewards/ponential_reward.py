from abc import ABC, abstractmethod
import numpy as np
from collections import namedtuple
from typing import Union, Callable
import logging
from pathlib import Path
import sys

PROJECT_ROOT_DIR = Path(__file__).parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from rewards.reward_base import RewardBase


class PonentialRewardBase(RewardBase, ABC):

    def __init__(self, gamma: float=0.99, my_logger: logging.Logger=None) -> None:
        super().__init__(is_potential=False, log_history_reward=True, my_logger=my_logger)  # 直接计算基于势的奖励，不使用基类的默认逻辑
        self.gamma = gamma
    
    def get_reward(self, state: namedtuple, **kwargs) -> float:
        """根据（state, next_state, done）计算ponential reward。
        
        参考论文：“Reward Shaping in Episodic Reinforcement Learning”
        
        计算公式为：\gamma * \Phi(next_state) - \Phi(state), 
        
        其中需要特殊处理的是，当done=True时，\Phi(next_state)取值为0！！！
        """
        assert "goal_v" in kwargs, "参数中需要包括goal_v，再调用get_termination方法"
        assert "goal_mu" in kwargs, "参数中需要包括goal_mu，再调用get_termination方法"
        assert "goal_chi" in kwargs, "参数中需要包括goal_chi，再调用get_termination方法"
        assert "next_state" in kwargs, "调用前，需要把namedtuple类型的next_state放入additional_info"
        assert "done" in kwargs, ""

        next_state = kwargs["next_state"]
        done = kwargs["done"]

        reward_v = self.gamma * (0. if done else self.phi_v(next_state, kwargs["goal_v"])) - self.phi_v(state, kwargs["goal_v"])
        reward_mu = self.gamma * (0. if done else self.phi_mu(next_state, kwargs["goal_mu"])) - self.phi_mu(state, kwargs["goal_mu"])
        reward_chi = self.gamma * (0. if done else self.phi_chi(next_state, kwargs["goal_chi"])) - self.phi_chi(state, kwargs["goal_chi"])
        reward = reward_v * reward_mu * reward_chi

        reward_log = {
            "reward_v": reward_v,
            "reward_mu": reward_mu,
            "reward_chi": reward_chi,
            "reward": reward
        }

        return self._process(new_reward=reward, log=reward_log)

    def reset(self):
        super().reset()

    @abstractmethod
    def phi_v(self, state: namedtuple, goal_v: float):
        raise NotImplementedError

    @abstractmethod
    def phi_mu(self, state: namedtuple, goal_mu: float):
        raise NotImplementedError
    
    @abstractmethod
    def phi_chi(self, state: namedtuple, goal_chi: float):
        raise NotImplementedError


class PonentialReward1(PonentialRewardBase):
    """计算势的方法：

    - abs(target - state) / (state.max - state.min)

    即：目标状态处势为0,其他状态的势为负数，值根据误差和归一化因子计算。

    Args:
        PonentialRewardBase (_type_): _description_
    """

    def __init__(self, gamma: float = 0.99, 
        v_min: float=0., v_max: float=1000., 
        mu_min: float=-90., mu_max: float=90., 
        chi_min: float=-180., chi_max: float=180.,
    ) -> None:
        super().__init__(gamma=gamma)

        self.v_min, self.v_max = v_min, v_max
        self.mu_min, self.mu_max = mu_min, mu_max
        self.chi_min, self.chi_max = chi_min, chi_max        

    def phi_v(self, state: namedtuple, goal_v: float):
        return -abs(goal_v - state.v) / (self.v_max - self.v_min)

    def phi_mu(self, state: namedtuple, goal_mu: float):
        return -abs(goal_mu - state.mu) / (self.mu_max - self.mu_min)

    def phi_chi(self, state: namedtuple, goal_chi: float):
        return -abs(goal_chi - state.chi) / (self.chi_max - self.chi_min)
    
    def __str__(self) -> str:
        return "ponential_reward_1"


class PonentialReward2(PonentialRewardBase):
    """计算势的方法，参考了论文“Autonomous Control of Simulated Fixed Wing Aircraft using Deep Reinforcement Learning”的3.6.3节
    """

    def __init__(self, 
        gamma: float = 0.99,
        coef_k_for_v: float=10., coef_k_for_mu: float=5., coef_k_for_chi: float=5.
    ) -> None:
        """_summary_

        Args:
            gamma (float, optional): 同MDP中的gamma. Defaults to 0.99.
            coef_k_for_v (float, optional): 系数越小，对误差越敏感. Defaults to 10..
            coef_k_for_mu (float, optional): 系数越小，对误差越敏感. Defaults to 5..
            coef_k_for_chi (float, optional): 系数越小，对误差越敏感. Defaults to 5..
        """
        super().__init__(gamma=gamma)

        self.coef_k_for_v, self.coef_k_for_mu, self.coef_k_for_chi = coef_k_for_v, coef_k_for_mu, coef_k_for_chi

    def phi_v(self, state: namedtuple, goal_v: float):
        err = abs(goal_v - state.v) / self.coef_k_for_v  # 计算误差
        normalized_err =  err / (1 + err)  # 将误差缩放到[0, 1]
        return 1. - normalized_err

    def phi_mu(self, state: namedtuple, goal_mu: float):
        err = abs(goal_mu - state.mu) / self.coef_k_for_mu  # 计算误差
        normalized_err =  err / (1 + err)  # 将误差缩放到[0, 1]
        return 1. - normalized_err
    
    def phi_chi(self, state: namedtuple, goal_chi: float):
        err = abs(goal_chi - state.chi) / self.coef_k_for_chi  # 计算误差
        normalized_err =  err / (1 + err)  # 将误差缩放到[0, 1]
        return 1. - normalized_err

    def __str__(self) -> str:
        return "ponential_reward_2"
    

class ScaledPonentialReward2(PonentialReward2):
    """将PonentialReward2的势函数计算的值放大scale_coef倍，尝试解决PonentialReward2计算的奖励数值过小的问题

    Args:
        PonentialReward2 (_type_): _description_
    """
    def __init__(self, scale_coef: float=100., gamma: float = 0.99, coef_k_for_v: float = 10, coef_k_for_mu: float = 5, coef_k_for_chi: float = 5) -> None:
        super().__init__(gamma, coef_k_for_v, coef_k_for_mu, coef_k_for_chi)
        self.scale_coef = scale_coef
    
    def phi_v(self, state: namedtuple, goal_v: float):
        return self.scale_coef * super().phi_v(state, goal_v)

    def phi_mu(self, state: namedtuple, goal_mu: float):
        return self.scale_coef * super().phi_mu(state, goal_mu)
    
    def phi_chi(self, state: namedtuple, goal_chi: float):
        return self.scale_coef * super().phi_chi(state, goal_chi)
    
    def __str__(self) -> str:
        return "scaled_ponential_reward_2"
    
