from typing import Tuple
import logging
from pathlib import Path
import sys

PROJECT_ROOT_DIR = Path(__file__).parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from terminations.termination_base import TerminationBase

class TimeoutTermination(TerminationBase):
    """超时：40秒内未达到目标速度矢量
    """
    def __init__(self, 
        termination_reward: float=-1.,
        env_config: dict=None,
        my_logger: logging.Logger=None
    ) -> None:
        super().__init__(
            termination_reward=termination_reward, 
            is_termination_reward_based_on_steps_left=False, 
            env_config=env_config, 
            my_logger=my_logger
        )

    def _get_termination(self, step_cnt: int) -> Tuple[bool, bool]:
        if step_cnt >= self.max_episode_steps - 1:
            return True, True

        return False, False

    def get_termination(self, state, **kwargs) -> Tuple[bool, bool]:
        assert "step_cnt" in kwargs, "参数中需要包括step_cnt"
        assert type(kwargs["step_cnt"]) is int, "step_cnt必须为int类型"

        return self._get_termination(step_cnt=kwargs["step_cnt"])
    
    def get_termination_and_reward(self, state, **kwargs) -> Tuple[bool, bool, float]:
        assert "step_cnt" in kwargs, "参数中需要包括step_cnt"
        assert type(kwargs["step_cnt"]) is int, "step_cnt必须为int类型"

        terminated, truncated = self._get_termination(step_cnt=kwargs["step_cnt"])
        # reward = self.termination_reward if terminated else 0.
        
        return terminated, truncated, self.get_termination_penalty(terminated=terminated, steps_cnt=kwargs["step_cnt"])
        
    def reset(self):
        pass

    def __str__(self) -> str:
        return "timeout_termination"