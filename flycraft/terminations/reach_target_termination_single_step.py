from typing import Tuple, List
from collections import namedtuple
import logging
from pathlib import Path
import sys

PROJECT_ROOT_DIR = Path(__file__).parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from terminations.termination_base import TerminationBase
from utils.geometry_utils import angle_of_2_velocity

class ReachTargetTerminationSingleStep(TerminationBase):

    def __init__(self,
        v_threshold: float=10., 
        angle_threshold: float=1.,
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
        self.angle_threshold = angle_threshold

    def _get_termination(
        self, 
        v: float, mu: float, chi: float, 
        goal_v: float, goal_mu: float, goal_chi: float
    ) -> Tuple[bool, bool]:
        
        v_flag, angle_flag = False, False
        if abs(goal_v - v) < self.v_threshold:
            v_flag = True
        if angle_of_2_velocity(v, mu, chi, goal_v, goal_mu, goal_chi) < self.angle_threshold:
            angle_flag = True
        
        if v_flag and angle_flag:
            return True, False
        else:
            return False, False

    def get_termination(self, state, **kwargs) -> Tuple[bool, bool]:
        assert "goal_v" in kwargs, "参数中需要包括goal_v，再调用get_termination方法"
        assert "goal_mu" in kwargs, "参数中需要包括goal_mu，再调用get_termination方法"
        assert "goal_chi" in kwargs, "参数中需要包括goal_chi，再调用get_termination方法"

        return self._get_termination(
            v=state.v,
            mu=state.mu,
            chi=state.chi,
            goal_v=kwargs["goal_v"], 
            goal_mu=kwargs["goal_mu"], 
            goal_chi=kwargs["goal_chi"],
        )

    def get_termination_and_reward(self, state, **kwargs) -> Tuple[bool, bool, float]:
        assert "goal_v" in kwargs, "参数中需要包括goal_v，再调用get_termination方法"
        assert "goal_mu" in kwargs, "参数中需要包括goal_mu，再调用get_termination方法"
        assert "goal_chi" in kwargs, "参数中需要包括goal_chi，再调用get_termination方法"

        terminated, truncated = self._get_termination(
            v=state.v,
            mu=state.mu,
            chi=state.chi,
            goal_v=kwargs["goal_v"], 
            goal_mu=kwargs["goal_mu"], 
            goal_chi=kwargs["goal_chi"],
        )
        reward = self.termination_reward if terminated else 0.
        
        return terminated, truncated, reward

    def reset(self):
        pass

    def __str__(self) -> str:
        return "reach_target_termination_single_step_based_on_angle_of_velocity_vector"