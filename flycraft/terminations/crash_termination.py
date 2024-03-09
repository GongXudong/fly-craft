from typing import Tuple, List
from collections import namedtuple
import logging
from pathlib import Path
import sys

PROJECT_ROOT_DIR = Path(__file__).parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from terminations.termination_base import TerminationBase


class CrashTermination(TerminationBase):

    def __init__(self, 
        h0: float=0., 
        termination_reward: float=-1.,
        is_termination_reward_based_on_steps_left: bool=False,
        env_config: dict=None,
        my_logger: logging.Logger=None
    ) -> None:
        """

        Args:
            h0 (float, optional): 飞机高度小于h0，判定坠机. Defaults to 0..
        """
        super().__init__(
            termination_reward=termination_reward, 
            is_termination_reward_based_on_steps_left=is_termination_reward_based_on_steps_left,
            env_config=env_config,
            my_logger=my_logger
        )
        self.h0 = h0
    
    def _get_termination(self, h: float) -> Tuple[bool, bool]:
        if h < self.h0:
            return True, True
        return False, False

    def get_termination(self, state: namedtuple, **kwargs) -> Tuple[bool, bool]:
        assert 'h' in state._fields, "state中必须包含h"
        h = state.h
        return self._get_termination(h=h)

    def get_termination_and_reward(self, state: namedtuple, **kwargs) -> Tuple[bool, bool, float]:
        assert 'h' in state._fields, "state中必须包含h"
        assert "step_cnt" in kwargs, "参数中需要包括step_cnt"

        h = state.h
        terminated, truncated = self._get_termination(h=h)
        # reward = self.termination_reward if terminated else 0.
        return terminated, truncated, self.get_termination_penalty(terminated=terminated, steps_cnt=kwargs["step_cnt"])

    def reset(self):
        pass

    def __str__(self) -> str:
        return "crash_termination"