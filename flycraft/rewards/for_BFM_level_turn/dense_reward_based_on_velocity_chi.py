from typing import Union
import numpy as np
from collections import namedtuple
import logging

from flycraft.rewards.reward_base import RewardBase


class DenseRewardBasedOnAngleAndVelocity(RewardBase):
    """The dense reward calculated by the error of chi and velocity.
    """
    def __init__(
            self, 
            b: float = 1., 
            mu_tolerance: float = 1.0,
            chi_scale: float = 180., 
            velocity_scale: float = 100.,
            chi_weight: float = 0.5, 
            log_history_reward: bool = True, 
            my_logger: logging.Logger = None
        ) -> None:
        assert 0. <= chi_weight <= 1., "chi_weight must be in [0,1]!"

        self.b = b
        self.mu_tolerance = mu_tolerance
        self.chi_scale = chi_scale
        self.velocity_scale = velocity_scale
        self.chi_weight = chi_weight
        super().__init__(is_potential=False, log_history_reward=log_history_reward, my_logger=my_logger)
    
    def get_reward(self, state: Union[namedtuple, np.ndarray], **kwargs) -> float:
        assert "goal_v" in kwargs, "The arguments needs to include goal_v, and then call the get_termination method!"
        assert "goal_chi" in kwargs, "The arguments needs to include goal_chi, and then call the get_termination method!"
        assert "next_state" in kwargs, "kwargs must contains next_state (type: namedtuple)"

        next_state = kwargs["next_state"]

        if np.abs(next_state.mu) < self.mu_tolerance:
            # refer to section 4.3 of "Reinforcement learning for UAV attitude control"
            chi_error = np.abs(kwargs["goal_chi"] - next_state.chi)
            cliped_chi_error = np.clip(chi_error / self.chi_scale, a_min=0., a_max=1.)
            chi_base_reward = - np.power(cliped_chi_error, self.b)

            velocity_error = np.abs(kwargs["goal_v"] - next_state.v)
            cliped_velocity_error = np.clip(velocity_error / self.velocity_scale, a_min=0., a_max=1.)
            velocity_base_reward = - np.power(cliped_velocity_error, self.b)

            return self.chi_weight *  chi_base_reward + (1. - self.chi_weight) * velocity_base_reward
        else:
            return 0.

    def reset(self):
        super().reset()