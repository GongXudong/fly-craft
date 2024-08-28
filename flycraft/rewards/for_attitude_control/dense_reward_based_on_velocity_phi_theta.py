from typing import Union, List
import numpy as np
from collections import namedtuple
from pathlib import Path
import logging
import sys

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from rewards.reward_base import RewardBase


class DenseRewardBasedOnAngleAndVelocity(RewardBase):
    """The dense reward calculated by the error of phi, theta, and velocity.
    """

    def __init__(
            self, 
            b: float = 1., 
            phi_scale: float = 180., 
            theta_scale: float = 90.,
            velocity_scale: float = 100.,
            weights: List[float] = [1./3., 1./3., 1./3.], 
            log_history_reward: bool = True, 
            my_logger: logging.Logger = None
        ) -> None:
        assert 0. <= weights[0] <= 1., "phi_weight must be in [0,1]!"
        assert 0. <= weights[1] <= 1., "theta_weight must be [0,1]!"
        assert 0. <= weights[2] <= 1., "v_weight must be in [0,1]!"
        assert np.allclose(np.array(weights).sum(), 1.), "The sum of weights must be 1!"

        self.b = b
        self.phi_scale = phi_scale
        self.theta_scale = theta_scale
        self.velocity_scale = velocity_scale
        self.weights = weights
        super().__init__(is_potential=False, log_history_reward=log_history_reward, my_logger=my_logger)
    
    def get_reward(self, state: Union[namedtuple, np.ndarray], **kwargs) -> float:
        assert "goal_v" in kwargs, "The arguments needs to include goal_v, and then call the get_termination method!"
        assert "goal_phi" in kwargs, "The arguments needs to include goal_phi, and then call the get_termination method!"
        assert "goal_theta" in kwargs, "The arguments needs to include goal_theta, and then call the get_termination method!"
        assert "next_state" in kwargs, "kwargs must contains next_state (type: namedtuple)"

        next_state = kwargs["next_state"]

        # refer to section 4.3 of "Reinforcement learning for UAV attitude control"
        phi_error = np.abs(kwargs["goal_phi"] - next_state.phi)
        cliped_phi_error = np.clip(phi_error / self.phi_scale, a_min=0., a_max=1.)
        phi_base_reward = - np.power(cliped_phi_error, self.b)

        theta_error = np.abs(kwargs["goal_theta"] - next_state.theta)
        cliped_theta_error = np.clip(theta_error / self.theta_scale, a_min=0., a_max=1.)
        theta_base_reward = - np.power(cliped_theta_error, self.b)

        velocity_error = np.abs(kwargs["goal_v"] - next_state.v)
        cliped_velocity_error = np.clip(velocity_error / self.velocity_scale, a_min=0., a_max=1.)
        velocity_base_reward = - np.power(cliped_velocity_error, self.b)

        return self.weights[0] *  phi_base_reward + self.weights[1] * theta_base_reward + self.weights[2] * velocity_base_reward

    def reset(self):
        super().reset()