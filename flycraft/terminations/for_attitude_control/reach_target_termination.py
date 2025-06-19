from typing import Tuple, List
from collections import namedtuple
import logging

from flycraft.terminations.termination_base import TerminationBase


class ReachTargetTermination(TerminationBase):

    def __init__(self,
        integral_time_length: float=1.,
        v_threshold=10., phi_threshold=1., theta_threshold=1., 
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
        self.phi_threshold = phi_threshold
        self.theta_threshold = theta_threshold

    def _get_termination(self, goal_v: float, goal_phi: float, goal_theta: float, state_list: List[namedtuple], ):
        
        if len(state_list) < self.integral_window_length:
            return False, False
        else:
            v_flag, mu_flag, chi_flag = False, False, False
            if sum([abs(goal_v - item.v) for item in state_list[-self.integral_window_length:]]) < self.v_integral_threshold:
                v_flag = True
            if sum([abs(goal_phi - item.phi) for item in state_list[-self.integral_window_length:]]) < self.phi_integral_threshold:
                mu_flag = True
            if sum([abs(goal_theta - item.theta) for item in state_list[-self.integral_window_length:]]) < self.theta_integral_threshold:
                chi_flag = True
            if v_flag and mu_flag and chi_flag:
                return True, False
            else:
                return False, False

    def get_termination(self, state, **kwargs) -> Tuple[bool, bool]:
        assert "goal_v" in kwargs, "参数中需要包括goal_v，再调用get_termination方法"
        assert "goal_phi" in kwargs, "参数中需要包括goal_phi，再调用get_termination方法"
        assert "goal_theta" in kwargs, "参数中需要包括goal_theta，再调用get_termination方法"
        assert "state_list" in kwargs, "参数中需要包括state_list（list[namedtuple]类型，所有历史观测），再调用get_termination方法"
        assert type(kwargs["state_list"]) is list, "state_list参数的类型应该是list[namedtuple]"

        return self._get_termination(
            goal_v=kwargs["goal_v"], 
            goal_phi=kwargs["goal_phi"], 
            goal_theta=kwargs["goal_theta"],
            state_list=kwargs["state_list"]
        )

    def get_termination_and_reward(self, state, **kwargs) -> Tuple[bool, bool, float]:
        assert "goal_v" in kwargs, "参数中需要包括goal_v，再调用get_termination方法"
        assert "goal_phi" in kwargs, "参数中需要包括goal_phi，再调用get_termination方法"
        assert "goal_theta" in kwargs, "参数中需要包括goal_theta，再调用get_termination方法"
        assert "state_list" in kwargs, "参数中需要包括state_list（list[namedtuple]类型，所有历史观测），再调用get_termination方法"
        assert type(kwargs["state_list"]) is list, "state_list参数的类型应该是list[namedtuple]"

        terminated, truncated = self._get_termination(
            goal_v=kwargs["goal_v"],
            goal_phi=kwargs["goal_phi"],
            goal_theta=kwargs["goal_theta"],
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
    def phi_integral_threshold(self):
        """phi的误差积分阈值，当phi在最后self.integral_window_length上的积分小于该值时，认为phi达到目标值

        Returns:
            _type_: _description_
        """
        return self.phi_threshold * self.integral_window_length
    
    @property
    def theta_integral_threshold(self):
        """theta的误差积分阈值，当theta在最后self.integral_window_length上的积分小于该值时，认为theta达到目标值

        Returns:
            _type_: _description_
        """
        return self.theta_threshold * self.integral_window_length

    def __str__(self) -> str:
        return "reach_target_termination"