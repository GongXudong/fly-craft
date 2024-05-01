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

class ReachTargetTermination2(TerminationBase):

    def __init__(self,
        integral_time_length: float=1.,
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

        self.integral_time_length = integral_time_length
        
        self.v_threshold = v_threshold
        self.angle_threshold = angle_threshold

    def _get_termination(self, goal_v: float, goal_mu: float, goal_chi: float, state_list: List[namedtuple]):
        
        if len(state_list) < self.integral_window_length:
            return False, False
        else:
            v_flag, angle_flag = False, False
            if sum([abs(goal_v - item.v) for item in state_list[-self.integral_window_length:]]) < self.v_integral_threshold:
                v_flag = True
            if sum([angle_of_2_velocity(item.v, item.mu, item.chi, goal_v, goal_mu, goal_chi) for item in state_list[-self.integral_window_length:]]) < self.angle_integral_threshold:
                angle_flag = True
            if v_flag and angle_flag:
                return True, False
            else:
                return False, False

    def get_termination(self, state, **kwargs) -> Tuple[bool, bool]:
        assert "goal_v" in kwargs, "参数中需要包括goal_v，再调用get_termination方法"
        assert "goal_mu" in kwargs, "参数中需要包括goal_mu，再调用get_termination方法"
        assert "goal_chi" in kwargs, "参数中需要包括goal_chi，再调用get_termination方法"
        assert "state_list" in kwargs, "参数中需要包括state_list（list[namedtuple]类型，所有历史观测），再调用get_termination方法"
        assert type(kwargs["state_list"]) is list, "state_list参数的类型应该是list[namedtuple]"

        return self._get_termination(
            goal_v=kwargs["goal_v"], 
            goal_mu=kwargs["goal_mu"], 
            goal_chi=kwargs["goal_chi"],
            state_list=kwargs["state_list"]
        )

    def get_termination_and_reward(self, state, **kwargs) -> Tuple[bool, bool, float]:
        assert "goal_v" in kwargs, "参数中需要包括goal_v，再调用get_termination方法"
        assert "goal_mu" in kwargs, "参数中需要包括goal_mu，再调用get_termination方法"
        assert "goal_chi" in kwargs, "参数中需要包括goal_chi，再调用get_termination方法"
        assert "state_list" in kwargs, "参数中需要包括state_list（list[namedtuple]类型，所有历史观测），再调用get_termination方法"
        assert type(kwargs["state_list"]) is list, "state_list参数的类型应该是list[namedtuple]"

        terminated, truncated = self._get_termination(
            goal_v=kwargs["goal_v"], 
            goal_mu=kwargs["goal_mu"], 
            goal_chi=kwargs["goal_chi"],
            state_list=kwargs["state_list"]
        )
        reward = self.termination_reward if terminated else 0.
        
        return terminated, truncated, reward

    def reset(self):
        pass
    
    @property
    def integral_window_length(self) -> int:
        """v, mu, chi的积分窗口长度

        Returns:
            _type_: _description_
        """
        return round(self.integral_time_length * self.step_frequence)

    @property
    def v_integral_threshold(self):
        """v的误差积分阈值，当v在最后self.integral_window_length上的积分小于该值时，认为v达到目标值

        Returns:
            _type_: _description_
        """
        return self.v_threshold * self.integral_window_length
    
    @property
    def angle_integral_threshold(self):
        """速度矢量方向的误差积分阈值，当速度矢量方向在最后self.integral_window_length上的积分小于该值时，认为速度矢量方向达到目标值

        Returns:
            _type_: _description_
        """
        return self.angle_threshold * self.integral_window_length

    def __str__(self) -> str:
        return "reach_target_termination_based_on_angle_of_velocity_vector"