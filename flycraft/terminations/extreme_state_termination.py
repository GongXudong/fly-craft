from typing import Tuple, List
from collections import namedtuple
import logging

from flycraft.terminations.termination_base import TerminationBase


class ExtremeStateTermination(TerminationBase):

    def __init__(self, 
        v_max: float=1000., 
        p_max: float=500., 
        termination_reward: float=-1.,
        is_termination_reward_based_on_steps_left: bool=False,
        env_config: dict=None,
        my_logger: logging.Logger=None
    ) -> None:
        """飞机速度大于v_max，或者角速度不在[-p_max, p_max]范围内，判定飞机处于极限状态.

        Args:
            v_max (float, optional): 速度最大临界值。 Defaults to 1000..
            p_max (float, optional): 角速度最大临界值。 Defaults to 500..
        """
        super().__init__(
            termination_reward=termination_reward, 
            is_termination_reward_based_on_steps_left=is_termination_reward_based_on_steps_left,
            env_config=env_config,
            my_logger=my_logger
        )
        self.v_max = v_max
        self.p_max = p_max
    
    def _get_termination(self, v: float, p: float) -> Tuple[bool, bool]:
        if v > self.v_max:
            return True, True
        if not (-self.p_max <= p <= self.p_max):
            return True, True
        return False, False

    def get_termination(self, state: namedtuple, **kwargs) -> Tuple[bool, bool]:
        assert "next_state" in kwargs, "参数中需要包括next_state，再调用get_termination方法"
        assert 'v' in kwargs["next_state"]._fields, "next_state中必须包含v"
        assert 'p' in kwargs["next_state"]._fields, "next_state中必须包含p"
        v = kwargs["next_state"].v
        p = kwargs["next_state"].p
        return self._get_termination(v=v, p=p)

    def get_termination_and_reward(self, state: namedtuple, **kwargs) -> Tuple[bool, bool, float]:
        assert "next_state" in kwargs, "参数中需要包括next_state，再调用get_termination方法"
        assert 'v' in kwargs["next_state"]._fields, "next_state中必须包含v"
        assert 'p' in kwargs["next_state"]._fields, "next_state中必须包含p"
        assert "step_cnt" in kwargs, "参数中需要包括step_cnt"

        v = kwargs["next_state"].v
        p = kwargs["next_state"].p
        terminated, truncated = self._get_termination(v=v, p=p)
        # reward = self.termination_reward if terminated else 0.

        return terminated, truncated, self.get_termination_penalty(terminated=terminated, steps_cnt=kwargs["step_cnt"])

    def reset(self):
        pass

    def __str__(self) -> str:
        return "extreme_state_termination"