from typing import Tuple, List
from math import fabs
from collections import namedtuple
import logging
from pathlib import Path
import sys

PROJECT_ROOT_DIR = Path(__file__).parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from utils.attitude_angle_calc_utils import RollDirection, get_roll_direction, get_roll_deg
from terminations.termination_base import TerminationBase

class ContinuouselyRollTermination(TerminationBase):
    """连续滚转：连续滚转2圈
    """
    def __init__(self,
        continuousely_roll_threshold: float=720.,
        termination_reward: float=-1.,
        is_termination_reward_based_on_steps_left: bool=False,
        env_config: dict=None,
        my_logger: logging.Logger = None) -> None:
        super().__init__(
            termination_reward=termination_reward, 
            is_termination_reward_based_on_steps_left=is_termination_reward_based_on_steps_left,
            env_config=env_config,
            my_logger=my_logger
        )

        self.continuousely_roll_threshold = continuousely_roll_threshold

        self.roll_flag: RollDirection = RollDirection.NOROLL
        self.accumulate_roll = 0.

    def _get_termination(self, state: namedtuple, next_state: namedtuple) -> Tuple[bool, bool]:
        roll_1, roll_2 = state.phi, next_state.phi
        cur_roll_direction = get_roll_direction(roll_1, roll_2)
        cur_roll_deg = get_roll_deg(roll_1=roll_1, roll_2=roll_2)
        # print(f"roll: {roll_1} -> {roll_2}, direction: {cur_roll_direction}, pre_roll: {self.roll_flag}, sum: {self.accumulate_roll}")
        if cur_roll_direction == RollDirection.NOROLL:
            self.reset()
            return False, False
        else:
            if cur_roll_direction == self.roll_flag:
                # TODO: 需要处理的边界情况：(-180) - 170，通过叉积求旋转角度
                self.accumulate_roll += cur_roll_deg
                if self.accumulate_roll > self.continuousely_roll_threshold:
                    if self.logger is not None:
                        self.logger.info(f"持续转弯超过{self.continuousely_roll_threshold}度！！！")
                    return True, False
                else:
                    return False, False
            else:
                self.accumulate_roll = cur_roll_deg
                self.roll_flag = cur_roll_direction
                return False, False

    def get_termination(self, state: namedtuple, **kwargs) -> Tuple[bool, bool]:
        assert "next_state" in kwargs, "参数中需要包括next_state，再调用get_termination方法"

        return self._get_termination(state=state, next_state=kwargs["next_state"])
    
    def get_termination_and_reward(self, state: namedtuple, **kwargs) -> Tuple[bool, bool, float]:
        assert "next_state" in kwargs, "参数中需要包括next_state，再调用get_termination方法"
        assert "step_cnt" in kwargs, "参数中需要包括step_cnt"

        terminated, truncated = self._get_termination(state=state, next_state=kwargs["next_state"])
        # reward = self.termination_reward if terminated else 0.

        return terminated, truncated, self.get_termination_penalty(terminated=terminated, steps_cnt=kwargs["step_cnt"])

    def reset(self):
        self.roll_flag = RollDirection.NOROLL
        self.accumulate_roll = 0.
    
    def __str__(self) -> str:
        return "continuousely_roll_termination"
