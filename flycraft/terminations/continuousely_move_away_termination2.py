from typing import Tuple, List
from collections import namedtuple
import logging
from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT_DIR = Path(__file__).parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from terminations.termination_base import TerminationBase
from utils.geometry_utils import angle_of_2_3d_vectors


class ContinuouselyMoveAwayTermination2(TerminationBase):
    """持续远离目标：连续2秒，当前速度矢量和目标速度矢量之间的误差增大
    """
    def __init__(self,
        time_window: float=2.,
        ignore_velocity_vector_angle_error: float=1.,
        termination_reward: float=-1.,
        is_termination_reward_based_on_steps_left: bool=False,
        env_config: dict=None,
        my_logger: logging.Logger=None
    ) -> None:
        """_summary_

        Args:
            step_frequence (int, optional): _description_. Defaults to 100.
            time_window (float, optional): _description_. Defaults to 2..
            termination_reward (float, optional): _description_. Defaults to -1..
            ignore_velocity_vector_angle_error (float, optional): 速度矢量误差大于这个值的时候，才考虑持续远离条件. Defaults to 1..
            my_logger (logging.Logger, optional): _description_. Defaults to None.
        """
        super().__init__(
            termination_reward=termination_reward, 
            is_termination_reward_based_on_steps_left=is_termination_reward_based_on_steps_left,
            env_config=env_config,
            my_logger=my_logger
        )

        self.time_window = time_window
        self.ignore_velocity_vector_angle_error = ignore_velocity_vector_angle_error

        # 例子：
        # velocity_vector_errors:                          2, 3, 5, 3, 6, 7
        # velocity_vector_continuously_increasing_num 计数：0, 1, 2, 0, 1, 2
        self.prev_velocity_vector_errors = 180.  # 上一拍的速度矢量误差
        self.velocity_vector_continuously_increasing_num = -1  # 到当前step为止，连续增大的速度矢量误差的数量

    
    def _get_termination(self, state: namedtuple, goal_v: float, goal_mu: float, goal_chi: float, ve: float, vn: float, vh: float):
        
        plane_current_velocity_vector = [
            ve, 
            vn, 
            vh
        ]

        target_velocity_vector = [
            goal_v * np.cos(np.deg2rad(goal_mu)) * np.sin(np.deg2rad(goal_chi)), 
            goal_v * np.cos(np.deg2rad(goal_mu)) * np.cos(np.deg2rad(goal_chi)),
            goal_v * np.sin(np.deg2rad(goal_mu)),
        ]

        cur_velocity_vector_error = angle_of_2_3d_vectors(plane_current_velocity_vector, target_velocity_vector)
        # print('velocity angle: ', cur_velocity_vector_error)
        
        if cur_velocity_vector_error <= self.ignore_velocity_vector_angle_error:
            self.velocity_vector_continuously_increasing_num = 0
        else:
            self.velocity_vector_continuously_increasing_num = 0 if cur_velocity_vector_error <= self.prev_velocity_vector_errors else self.velocity_vector_continuously_increasing_num + 1

        self.prev_velocity_vector_errors = cur_velocity_vector_error

        if self.velocity_vector_continuously_increasing_num >= self.time_window_step_length:

            if self.logger is not None:
                self.logger.info(f"速度矢量误差时持续增大{self.time_window_step_length}拍！！！")
            
            terminated, truncated = True, False
        else:
            terminated, truncated = False, False

        return terminated, truncated
    
    def get_termination(self, state: namedtuple, **kwargs) -> Tuple[bool, bool]:
        assert "goal_v" in kwargs, "参数中需要包括goal_v，再调用get_termination方法"
        assert "goal_mu" in kwargs, "参数中需要包括goal_mu，再调用get_termination方法"
        assert "goal_chi" in kwargs, "参数中需要包括goal_chi，再调用get_termination方法"
        # assert "ve" in kwargs, "参数中需要包括ve，再调用get_termination方法"
        # assert "vn" in kwargs, "参数中需要包括vn，再调用get_termination方法"
        # assert "vh" in kwargs, "参数中需要包括vh，再调用get_termination方法"

        return self._get_termination(
            state=state, 
            goal_v=kwargs["goal_v"],
            goal_mu=kwargs["goal_mu"],
            goal_chi=kwargs["goal_chi"],
            ve=state.v * np.cos(np.deg2rad(state.mu)) * np.sin(np.deg2rad(state.chi)), 
            vn=state.v * np.cos(np.deg2rad(state.mu)) * np.cos(np.deg2rad(state.chi)), 
            vh=state.v * np.sin(np.deg2rad(state.mu))
        )
    
    def get_termination_and_reward(self, state: namedtuple, **kwargs) -> Tuple[bool, bool, float]:
        assert "goal_v" in kwargs, "参数中需要包括goal_v，再调用get_termination方法"
        assert "goal_mu" in kwargs, "参数中需要包括goal_mu，再调用get_termination方法"
        assert "goal_chi" in kwargs, "参数中需要包括goal_chi，再调用get_termination方法"
        # assert "ve" in kwargs, "参数中需要包括ve，再调用get_termination方法"
        # assert "vn" in kwargs, "参数中需要包括vn，再调用get_termination方法"
        # assert "vh" in kwargs, "参数中需要包括vh，再调用get_termination方法"
        assert "step_cnt" in kwargs, "参数中需要包括step_cnt"

        terminated, truncated = self._get_termination(
            state=state, 
            goal_v=kwargs["goal_v"],
            goal_mu=kwargs["goal_mu"],
            goal_chi=kwargs["goal_chi"],
            ve=state.v * np.cos(np.deg2rad(state.mu)) * np.sin(np.deg2rad(state.chi)), 
            vn=state.v * np.cos(np.deg2rad(state.mu)) * np.cos(np.deg2rad(state.chi)), 
            vh=state.v * np.sin(np.deg2rad(state.mu))
        )
        # reward = self.termination_reward if terminated else 0.

        return terminated, truncated, self.get_termination_penalty(terminated=terminated, steps_cnt=kwargs["step_cnt"])

    def reset(self):
        self.prev_velocity_vector_errors = 180.
        self.velocity_vector_continuously_increasing_num = -1

    @property
    def time_window_step_length(self) -> int:
        """判断使用的时间窗口step数

        Returns:
            _type_: _description_
        """
        return round(self.time_window * self.step_frequence)

    def __str__(self) -> str:
        return "continuousely_move_away_termination_based_on_velocity_vector_error"