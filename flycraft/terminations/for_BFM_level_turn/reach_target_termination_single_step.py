from typing import Tuple, List
from collections import namedtuple
import logging
import numpy as np

from flycraft.terminations.termination_base import TerminationBase


class ReachTargetTerminationSingleStep(TerminationBase):

    def __init__(self,
        mu_tolerance: float=1.,
        v_threshold: float=10., 
        chi_threshold: float=1.,
        termination_reward: float=1.,
        env_config: dict=None,
        my_logger: logging.Logger=None
    ) -> None:
        super().__init__(
            termination_reward=termination_reward, 
            is_termination_reward_based_on_steps_left=False,
            env_config=env_config, 
            my_logger=my_logger
        )
        
        self.mu_tolerance = mu_tolerance
        self.v_threshold = v_threshold
        self.chi_threshold = chi_threshold

    def _get_termination(
        self, 
        v: float, mu: float, chi: float, 
        goal_v: float, goal_chi: float
    ) -> Tuple[bool, bool]:
        
        if np.abs(mu) < self.mu_tolerance:
            v_flag, chi_flag = False, False
            if abs(goal_v - v) < self.v_threshold:
                v_flag = True
            if abs(goal_chi - chi) < self.chi_threshold:
                chi_flag = True
            
            if v_flag and chi_flag:
                return True, False
            else:
                return False, False
        else:
            return False, False

    def get_termination(self, state, **kwargs) -> Tuple[bool, bool]:
        assert "goal_v" in kwargs, "参数中需要包括goal_v，再调用get_termination方法"
        assert "goal_chi" in kwargs, "参数中需要包括goal_chi，再调用get_termination方法"

        return self._get_termination(
            v=state.v,
            mu=state.mu,
            chi=state.chi,
            goal_v=kwargs["goal_v"], 
            goal_chi=kwargs["goal_chi"],
        )

    def get_termination_and_reward(self, state, **kwargs) -> Tuple[bool, bool, float]:
        assert "goal_v" in kwargs, "参数中需要包括goal_v，再调用get_termination方法"
        assert "goal_chi" in kwargs, "参数中需要包括goal_chi，再调用get_termination方法"

        terminated, truncated = self._get_termination(
            v=state.v,
            mu=state.mu,
            chi=state.chi,
            goal_v=kwargs["goal_v"], 
            goal_chi=kwargs["goal_chi"],
        )
        reward = self.termination_reward if terminated else 0.
        
        return terminated, truncated, reward

    def reset(self):
        pass

    def __str__(self) -> str:
        return "reach_target_termination_single_step_based_on_chi_and_velocity"