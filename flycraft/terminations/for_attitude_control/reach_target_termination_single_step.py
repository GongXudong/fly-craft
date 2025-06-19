from typing import Tuple, List
from collections import namedtuple
import logging

from flycraft.terminations.termination_base import TerminationBase


class ReachTargetTerminationSingleStep(TerminationBase):

    def __init__(self,
        v_threshold: float=10., 
        phi_threshold: float=1.,
        theta_threshold: float=1.,
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
        
        self.v_threshold = v_threshold
        self.phi_threshold = phi_threshold
        self.theta_threshold = theta_threshold

    def _get_termination(
        self, 
        v: float, phi: float, theta: float, 
        goal_v: float, goal_phi: float, goal_theta: float
    ) -> Tuple[bool, bool]:
        
        v_flag, phi_flag, theta_flag = False, False, False
        if abs(goal_v - v) < self.v_threshold:
            v_flag = True
        if abs(goal_phi - phi) < self.phi_threshold:
            phi_flag = True
        if abs(goal_theta - theta) < self.theta_threshold:
            theta_flag = True
        
        if v_flag and phi_flag and theta_flag:
            return True, False
        else:
            return False, False

    def get_termination(self, state, **kwargs) -> Tuple[bool, bool]:
        assert "goal_v" in kwargs, "参数中需要包括goal_v，再调用get_termination方法"
        assert "goal_phi" in kwargs, "参数中需要包括goal_phi，再调用get_termination方法"
        assert "goal_theta" in kwargs, "参数中需要包括goal_theta，再调用get_termination方法"

        return self._get_termination(
            v=state.v,
            phi=state.phi,
            theta=state.theta,
            goal_v=kwargs["goal_v"], 
            goal_phi=kwargs["goal_phi"], 
            goal_theta=kwargs["goal_theta"],
        )

    def get_termination_and_reward(self, state, **kwargs) -> Tuple[bool, bool, float]:
        assert "goal_v" in kwargs, "参数中需要包括goal_v，再调用get_termination方法"
        assert "goal_phi" in kwargs, "参数中需要包括goal_phi，再调用get_termination方法"
        assert "goal_theta" in kwargs, "参数中需要包括goal_theta，再调用get_termination方法"

        terminated, truncated = self._get_termination(
            v=state.v,
            phi=state.phi,
            theta=state.theta,
            goal_v=kwargs["goal_v"], 
            goal_phi=kwargs["goal_phi"], 
            goal_theta=kwargs["goal_theta"],
        )
        reward = self.termination_reward if terminated else 0.
        
        return terminated, truncated, reward

    def reset(self):
        pass

    def __str__(self) -> str:
        return "reach_target_termination_single_step_based_on_v_phi_theta"