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


class DenseRewardBasedOnAngleAndVelocity(RewardBase):
    """基于飞机速度矢量的大小与方向的过程奖励。

    Args:
        RewardBase (_type_): _description_
    """
    def __init__(
            self, 
            b: float = 1., 
            angle_scale: float = 180., 
            velocity_scale: float = 100.,
            angle_weight: float = 0.5, 
            log_history_reward: bool = True, 
            my_logger: logging.Logger = None
        ) -> None:
        assert 0. <= angle_weight <= 1., "angle_weight不在[0,1]范围内"

        self.b = b
        self.angle_scale = angle_scale
        self.velocity_scale = velocity_scale
        self.angle_weight = angle_weight
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
        angle_base_reward = - np.power(angle / self.angle_scale, self.b)

        # 奖励公式参考论文“Reinforcement learning for UAV attitude control”的4.3节
        velocity_error = np.abs(kwargs["goal_v"] - next_state.v)
        cliped_velocity_error = np.clip(velocity_error / self.velocity_scale, a_min=0., a_max=1.)
        velocity_base_reward = - np.power(cliped_velocity_error, self.b)

        return self.angle_weight *  angle_base_reward + (1. - self.angle_weight) * velocity_base_reward

    def reset(self):
        super().reset()