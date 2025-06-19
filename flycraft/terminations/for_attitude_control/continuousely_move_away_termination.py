from typing import Tuple, List
from collections import namedtuple
import logging

from flycraft.terminations.termination_base import TerminationBase


class ContinuouselyMoveAwayTermination(TerminationBase):
    """持续远离目标：连续2秒，当前状态和目标状态之间的角度差增大。（当phi的误差小于ignore_phi_error时，不再考虑这个终止条件；theta同理）
    """
    def __init__(self,
        time_window: float=2.,
        ignore_phi_error: float=1.,
        ignore_theta_error: float=1.,
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
        self.ignore_phi_error = ignore_phi_error
        self.ignore_theta_error = ignore_theta_error

        # 例子：
        # phi errors:                          2, 3, 5, 3, 6, 7
        # phi_continuously_increasing_num 计数：0, 1, 2, 0, 1, 2
        self.prev_phi_errors = -360.  # 上一拍phi误差
        self.phi_continuously_increasing_num = -1  # 到当前step为止，连续增大的phi误差的数量
        self.prev_theta_errors = -360.  # 上一拍的theta误差
        self.theta_continuously_increasing_num = -1  # 到当前step为止，连续增大的theta误差的数量

    
    def _get_termination(self, state: namedtuple, goal_v: float, goal_phi: float, goal_theta: float):
        # phi的范围是[-180, 180]，存在-180与180的跨越问题
        # 例如，从target_phi=-179, phi=179, 此时tmp_phi_error=358, 二者间的误差实际应该是360 - 358 = 2
        tmp_phi_error = abs(goal_phi - state.phi)
        cur_phi_error = min(tmp_phi_error, 360. - tmp_phi_error)

        # theta的范围是[-90, 90]，不存在-90与90的跨越问题
        cur_theta_error = abs(goal_theta - state.theta)

        # 当phi和theta的error很小时，不再考虑这个终止条件
        if cur_phi_error <= self.ignore_phi_error:
            self.phi_continuously_increasing_num = 0
        else:
            self.phi_continuously_increasing_num = 0 if cur_phi_error <= self.prev_phi_errors else self.phi_continuously_increasing_num + 1
        
        if cur_theta_error <= self.ignore_theta_error:
            self.theta_continuously_increasing_num = 0
        else:
            self.theta_continuously_increasing_num = 0 if cur_theta_error <= self.prev_theta_errors else self.theta_continuously_increasing_num + 1
        
        self.prev_phi_errors, self.prev_theta_errors = cur_phi_error, cur_theta_error

        if self.phi_continuously_increasing_num >= self.time_window_step_length and \
            self.theta_continuously_increasing_num >= self.time_window_step_length:

            if self.logger is not None:
                self.logger.info(f"phi和theta同时持续增大{self.time_window_step_length}拍！！！")
            
            terminated, truncated = True, False
        # 注意：phi和theta的误差，只有一个持续增大时，并不足以说明飞机持续远离目标！！！
        # 一个能够说明的例子：当目标v=130，phi=10，theta=85时，专家轨迹phi先减小到-16在逐渐增大到10，theta持续增大到85.
        # phi虽然持续远离了目标phi（原因是转弯掉高），但theta误差却减小了，飞机速度矢量与目标速度矢量靠近了！！！

        # elif self.phi_continuously_increasing_num >= self.time_window_step_length and \
        #     self.theta_continuously_increasing_num < self.time_window_step_length:

        #     if self.logger is not None:
        #         self.logger.info(f"phi持续增大{self.time_window_step_length}拍！！！")
            
        #     terminated, truncated = True, False
        # elif self.phi_continuously_increasing_num < self.time_window_step_length and \
        #     self.theta_continuously_increasing_num >= self.time_window_step_length:

        #     if self.logger is not None:
        #         self.logger.info(f"theta持续增大{self.time_window_step_length}拍！！！")
            
        #     terminated, truncated = True, False
        else:
            terminated, truncated = False, False

        return terminated, truncated
    
    def get_termination(self, state: namedtuple, **kwargs) -> Tuple[bool, bool]:
        assert "goal_v" in kwargs, "参数中需要包括goal_v，再调用get_termination方法"
        assert "goal_phi" in kwargs, "参数中需要包括goal_phi，再调用get_termination方法"
        assert "goal_theta" in kwargs, "参数中需要包括goal_theta，再调用get_termination方法"

        return self._get_termination(
            state=state, 
            goal_v=kwargs["goal_v"], 
            goal_phi=kwargs["goal_phi"], 
            goal_theta=kwargs["goal_theta"]
        )
    
    def get_termination_and_reward(self, state: namedtuple, **kwargs) -> Tuple[bool, bool, float]:
        assert "goal_v" in kwargs, "参数中需要包括goal_v，再调用get_termination方法"
        assert "goal_phi" in kwargs, "参数中需要包括goal_phi，再调用get_termination方法"
        assert "goal_theta" in kwargs, "参数中需要包括goal_theta，再调用get_termination方法"
        assert "step_cnt" in kwargs, "参数中需要包括step_cnt"

        terminated, truncated = self._get_termination(
            state=state,
            goal_v=kwargs["goal_v"], 
            goal_phi=kwargs["goal_phi"], 
            goal_theta=kwargs["goal_theta"]
        )
        # reward = self.termination_reward if terminated else 0.

        return terminated, truncated, self.get_termination_penalty(terminated=terminated, steps_cnt=kwargs["step_cnt"])

    def reset(self):
        self.prev_phi_errors = -360.
        self.phi_continuously_increasing_num = -1
        self.prev_theta_errors = -360.
        self.theta_continuously_increasing_num = -1

    @property
    def time_window_step_length(self) -> int:
        """判断使用的时间窗口step数

        Returns:
            _type_: _description_
        """
        return round(self.time_window * self.step_frequence)

    def __str__(self) -> str:
        return "continuousely_move_away_termination_based_on_phi_error_and_theta_error"