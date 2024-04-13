from typing import Union
import numpy as np
from collections import namedtuple
from pathlib import Path

import sys

PROJECT_ROOT_DIR = Path(__file__).parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))
    
from rewards.reward_base import RewardBase


class SparseReward(RewardBase):

    def __init__(self, 
        is_potential: bool = False, 
        log_history_reward: bool=False,
        reward_constant: float=0.
    ) -> None:
        super().__init__(is_potential=is_potential, log_history_reward=log_history_reward)

        self.is_potential = is_potential
        
        self.reward_constant = reward_constant

    def get_reward(self, state: Union[namedtuple, np.ndarray], **kwargs) -> float:
        return self.reward_constant
    
    def reset(self):
        super().reset()
