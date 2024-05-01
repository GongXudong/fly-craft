from typing import Tuple, List
from collections import namedtuple
import logging
from pathlib import Path
import sys

PROJECT_ROOT_DIR = Path(__file__).parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from terminations.termination_base import TerminationBase


class ContinuouselyMoveAwayTermination(TerminationBase):
    """持续远离目标：连续2秒，当前状态和目标状态之间的角度差增大。（当mu的误差小于ignore_mu_error时，不再考虑这个终止条件；chi同理）
    """
    def __init__(self,
        time_window: float=2.,
        ignore_mu_error: float=1.,
        ignore_chi_error: float=1.,
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
        self.ignore_mu_error = ignore_mu_error
        self.ignore_chi_error = ignore_chi_error

        # 例子：
        # mu errors:                          2, 3, 5, 3, 6, 7
        # mu_continuously_increasing_num 计数：0, 1, 2, 0, 1, 2
        self.prev_mu_errors = -360.  # 上一拍的mu误差
        self.mu_continuously_increasing_num = -1  # 到当前step为止，连续增大的mu误差的数量
        self.prev_chi_errors = -360.  # 上一拍的chi误差
        self.chi_continuously_increasing_num = -1  # 到当前step为止，连续增大的chi误差的数量

    
    def _get_termination(self, state: namedtuple, goal_v: float, goal_mu: float, goal_chi: float):
        # mu的范围是[-90, 90]，不存在-90与90的跨越问题
        cur_mu_error = abs(goal_mu - state.mu)

        # chi的范围是[-180, 180]，存在-180与180的跨越问题
        # 例如，从target_chi=-179, chi=179, 此时tmp_chi_error=358, 二者间的误差实际应该是360 - 358 = 2
        tmp_chi_error = abs(goal_chi - state.chi)
        cur_chi_error = min(tmp_chi_error, 360. - tmp_chi_error)

        # 当mu和chi的error很小时，不再考虑这个终止条件
        if cur_mu_error <= self.ignore_mu_error:
            self.mu_continuously_increasing_num = 0
        else:
            self.mu_continuously_increasing_num = 0 if cur_mu_error <= self.prev_mu_errors else self.mu_continuously_increasing_num + 1
        
        if cur_chi_error <= self.ignore_chi_error:
            self.chi_continuously_increasing_num = 0
        else:
            self.chi_continuously_increasing_num = 0 if cur_chi_error <= self.prev_chi_errors else self.chi_continuously_increasing_num + 1
        
        self.prev_mu_errors, self.prev_chi_errors = cur_mu_error, cur_chi_error

        if self.mu_continuously_increasing_num >= self.time_window_step_length and \
            self.chi_continuously_increasing_num >= self.time_window_step_length:

            if self.logger is not None:
                self.logger.info(f"mu和chi同时持续增大{self.time_window_step_length}拍！！！")
            
            terminated, truncated = True, False
        # 注意：mu和chi的误差，只有一个持续增大时，并不足以说明飞机持续远离目标！！！
        # 一个能够说明的例子：当目标v=130，mu=10，chi=145时，专家轨迹mu先减小到-16在逐渐增大到10，chi持续增大到145.
        # mu虽然持续远离了目标mu（原因是转弯掉高），但chi误差却减小了，飞机速度矢量与目标速度矢量靠近了！！！

        # elif self.mu_continuously_increasing_num >= self.time_window_step_length and \
        #     self.chi_continuously_increasing_num < self.time_window_step_length:

        #     if self.logger is not None:
        #         self.logger.info(f"mu持续增大{self.time_window_step_length}拍！！！")
            
        #     terminated, truncated = True, False
        # elif self.mu_continuously_increasing_num < self.time_window_step_length and \
        #     self.chi_continuously_increasing_num >= self.time_window_step_length:

        #     if self.logger is not None:
        #         self.logger.info(f"chi持续增大{self.time_window_step_length}拍！！！")
            
        #     terminated, truncated = True, False
        else:
            terminated, truncated = False, False

        return terminated, truncated
    
    def get_termination(self, state: namedtuple, **kwargs) -> Tuple[bool, bool]:
        assert "goal_v" in kwargs, "参数中需要包括goal_v，再调用get_termination方法"
        assert "goal_mu" in kwargs, "参数中需要包括goal_mu，再调用get_termination方法"
        assert "goal_chi" in kwargs, "参数中需要包括goal_chi，再调用get_termination方法"

        return self._get_termination(
            state=state, 
            goal_v=kwargs["goal_v"], 
            goal_mu=kwargs["goal_mu"], 
            goal_chi=kwargs["goal_chi"]
        )
    
    def get_termination_and_reward(self, state: namedtuple, **kwargs) -> Tuple[bool, bool, float]:
        assert "goal_v" in kwargs, "参数中需要包括goal_v，再调用get_termination方法"
        assert "goal_mu" in kwargs, "参数中需要包括goal_mu，再调用get_termination方法"
        assert "goal_chi" in kwargs, "参数中需要包括goal_chi，再调用get_termination方法"
        assert "step_cnt" in kwargs, "参数中需要包括step_cnt"

        terminated, truncated = self._get_termination(
            state=state,
            goal_v=kwargs["goal_v"], 
            goal_mu=kwargs["goal_mu"], 
            goal_chi=kwargs["goal_chi"]
        )
        # reward = self.termination_reward if terminated else 0.

        return terminated, truncated, self.get_termination_penalty(terminated=terminated, steps_cnt=kwargs["step_cnt"])

    def reset(self):
        self.prev_mu_errors = -360.
        self.mu_continuously_increasing_num = -1
        self.prev_chi_errors = -360.
        self.chi_continuously_increasing_num = -1

    @property
    def time_window_step_length(self) -> int:
        """判断使用的时间窗口step数

        Returns:
            _type_: _description_
        """
        return round(self.time_window * self.step_frequence)

    def __str__(self) -> str:
        return "continuousely_move_away_termination_based_on_mu_error_and_chi_error"