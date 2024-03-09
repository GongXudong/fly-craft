from typing import Union
import numpy as np
from collections import namedtuple
from pathlib import Path
import logging
import sys

PROJECT_ROOT_DIR = Path(__file__).parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from rewards.reward_base import RewardBase
from utils.geometry_utils import angle_of_2_3d_vectors


class DenseReward(RewardBase):
    """过程奖励，根据目标速度矢量与飞机当前速度矢量的夹角给出的负奖励。例如：目标速度矢量与飞机当前速度矢量的夹角为a，奖励为：-(a / 180) ^ b，其中b为预设的常量。

    Args:
        RewardBase (_type_): _description_
    """
    def __init__(self, b: float = 1., log_history_reward: bool = True, my_logger: logging.Logger = None) -> None:
        self.b = b
        super().__init__(is_potential=False, log_history_reward=log_history_reward, my_logger=my_logger)
    
    def get_reward(self, state: Union[namedtuple, np.ndarray], **kwargs) -> float:
        assert "goal_v" in kwargs, "参数中需要包括goal_v，再调用get_termination方法"
        assert "goal_mu" in kwargs, "参数中需要包括goal_mu，再调用get_termination方法"
        assert "goal_chi" in kwargs, "参数中需要包括goal_chi，再调用get_termination方法"
        # assert "next_ve" in kwargs, "调用前，需要传入next_ve"
        # assert "next_vn" in kwargs, "调用前，需要传入next_vn"
        # assert "next_vh" in kwargs, "调用前，需要传入next_vh"
        assert "next_state" in kwargs, "调用前，需要把namedtuple类型的next_state放入函数的参数中"
        
        next_state = kwargs["next_state"]

        # plane_current_velocity_vector = [
        #     kwargs["next_ve"], 
        #     kwargs["next_vn"], 
        #     kwargs["next_vh"]
        # ]
        plane_current_velocity_vector = [
            next_state.v * np.cos(np.deg2rad(next_state.mu)) * np.sin(np.deg2rad(next_state.chi)), 
            next_state.v * np.cos(np.deg2rad(next_state.mu)) * np.cos(np.deg2rad(next_state.chi)), 
            next_state.v * np.sin(np.deg2rad(next_state.mu))
        ]
        target_velocity_vector = [
            kwargs["goal_v"] * np.cos(np.deg2rad(kwargs["goal_mu"])) * np.sin(np.deg2rad(kwargs["goal_chi"])), 
            kwargs["goal_v"] * np.cos(np.deg2rad(kwargs["goal_mu"])) * np.cos(np.deg2rad(kwargs["goal_chi"])),
            kwargs["goal_v"] * np.sin(np.deg2rad(kwargs["goal_mu"])),
        ]

        angle = angle_of_2_3d_vectors(plane_current_velocity_vector, target_velocity_vector)

        return -np.power(angle / 180., self.b)

    def reset(self):
        super().reset()