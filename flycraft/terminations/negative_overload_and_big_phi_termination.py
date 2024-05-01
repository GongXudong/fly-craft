from typing import Tuple, List
from collections import namedtuple
import logging
from pathlib import Path
import sys

PROJECT_ROOT_DIR = Path(__file__).parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from terminations.termination_base import TerminationBase


class NegativeOverloadAndBigPhiTermination(TerminationBase):
    """负过载且大幅度滚转：nz<0且phi>60(deg)连续超过2秒
    """
    def __init__(self,
        time_window: float=2.,
        negative_overload_threshold: float=0.,
        big_phi_threshold: float=60.,
        termination_reward: float=-1.,
        is_termination_reward_based_on_steps_left: bool=False,
        env_config: dict=None,
        my_logger: logging.Logger=None
    ) -> None:
        super().__init__(
            termination_reward=termination_reward, 
            is_termination_reward_based_on_steps_left=is_termination_reward_based_on_steps_left,
            env_config=env_config,
            my_logger=my_logger
        )

        self.time_window = time_window
        self.negative_overload_threshold = negative_overload_threshold
        self.big_phi_threshold = big_phi_threshold

        self.invalid_cnt = 0

    def _get_termination(self, state: namedtuple, nz: float) -> Tuple[bool, bool]:
        if abs(state.phi) > self.big_phi_threshold and nz < self.negative_overload_threshold:
            self.invalid_cnt += 1
        else:
            self.invalid_cnt = 0
        
        if self.invalid_cnt >= self.time_window_step_length:
            return True, False
        
        return False, False
    
    def get_termination(self, state: namedtuple, **kwargs) -> Tuple[bool, bool]:
        assert "nz" in kwargs, "参数中需要包括nz，再调用get_termination方法"

        return self._get_termination(state=state, nz=kwargs["nz"])

    def get_termination_and_reward(self, state: namedtuple, **kwargs) -> Tuple[bool, bool, float]:
        assert "nz" in kwargs, "参数中需要包括nz，再调用get_termination方法"
        assert "step_cnt" in kwargs, "参数中需要包括step_cnt"

        terminated, truncated = self._get_termination(state=state, nz=kwargs["nz"])
        # reward = self.termination_reward if terminated else 0.
        
        return terminated, truncated, self.get_termination_penalty(terminated=terminated, steps_cnt=kwargs["step_cnt"])

    
    def reset(self):
        self.invalid_cnt = 0

    @property
    def time_window_step_length(self) -> int:
        """判断使用的时间窗口step数

        Returns:
            _type_: _description_
        """
        return round(self.time_window * self.step_frequence)

    def __str__(self) -> str:
        return "negative_overload_and_big_phi_termination"