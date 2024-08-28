from typing import Any, Dict, List, Union
import numpy as np
from collections import namedtuple
import logging
from gymnasium.utils import seeding

from planes.f16_plane import F16Plane
from tasks.task_base import Task
from flycraft.tasks.goal_samplers.goal_sampler_for_attitude_control import GoalSampler

from rewards.reward_base import RewardBase
from rewards.for_attitude_control.dense_reward_based_on_velocity_phi_theta import DenseRewardBasedOnAngleAndVelocity
from rewards.sparse_reward import SparseReward

from terminations.termination_base import TerminationBase
from terminations.for_attitude_control.reach_target_termination import ReachTargetTermination
from terminations.for_attitude_control.reach_target_termination_single_step import ReachTargetTerminationSingleStep
from terminations.crash_termination import CrashTermination
from terminations.extreme_state_termination import ExtremeStateTermination
from terminations.timeout_termination import TimeoutTermination
from terminations.for_attitude_control.continuousely_move_away_termination import ContinuouselyMoveAwayTermination
from terminations.continuousely_roll_termination import ContinuouselyRollTermination
from terminations.negative_overload_and_big_phi_termination import NegativeOverloadAndBigPhiTermination

class AttitudeControlTask(Task):

    def __init__(
        self, 
        plane: F16Plane,
        env_config: dict,
        np_random: np.random.Generator=None,
        my_logger: logging.Logger=None
    ) -> None:
        super().__init__(plane)
        self.config = {
            "flag_str": "Train",
            "debug_mode": True,
            "task": {
                "h0": 5000,
                "v0": 200,
                "step_frequence": 10,
                "max_simulate_time": 40,
                "gamma": 0.995
            },
            "goal": {
                "use_fixed_goal": False,
                "goal_v": 0, 
                "goal_phi": 0, 
                "goal_theta": 0,
                "sample_random": True,
                "v_min": 100., 
                "v_max": 300.,
                "phi_min": -180.,
                "phi_max": 180.,
                "theta_min": -90.,
                "theta_max": 90.,
                "available_goals_file": "res.csv",
                "sample_reachable_goal": False,
                "sample_goal_noise_std": [5, 0.5, 0.5],  # noise std for [v, mu, chi]
            },
            "rewards": {
                "dense": {
                    "use": True,
                    "b": 0.5
                },
            },
            "terminations": {
                "RT": {
                    "use": True, 
                    "integral_time_length": 1,  # 在连续多长的时间窗口内满足条件
                    "v_threshold": 10,  # 触发RT的最大v误差
                    "phi_threshold": 1,  # 触发RT的最大角度误差
                    "theta_threshold": 1,  # 触发RT的最大角度误差
                    "termination_reward": 0.0,  # 触发RT的奖励
                },
                "RT_SINGLE_STEP": {
                    "use": True,
                    "v_threshold": 10,
                    "phi_threshold": 1,
                    "theta_threshold": 1,
                    "termination_reward": 0.0
                },
                "C": {
                    "use": True,
                    "h0": 0,
                    "is_termination_reward_based_on_steps_left": True,  # 是否根据剩余步长计算惩罚
                    "termination_reward": -1,  # 不跟据剩余步长计算惩罚时使用的立即惩罚大小
                },
                "ES": {
                    "use": True,
                    "v_max": 400,
                    "p_max": 300,
                    "is_termination_reward_based_on_steps_left": True,  # 是否根据剩余步长计算惩罚
                    "termination_reward": -1,  # 不跟据剩余步长计算惩罚时使用的立即惩罚大小
                },
                "T": {
                    "use": True,
                    "termination_reward": -1,
                },
                "CMA": {
                    "use": True,
                    "time_window": 2,  # 在这个时间窗口内持续远离目标，就触发该终止条件
                    "ignore_phi_error": 1,  # 当phi的误差小于ignore_phi_error时，不再考虑这个终止条件
                    "ignore_theta_error": 1,  # 当theta的误差小于ignore_theta_error时，不再考虑这个终止条件
                    "is_termination_reward_based_on_steps_left": True,  # 是否根据剩余步长计算惩罚
                    "termination_reward": -1,  # 不跟据剩余步长计算惩罚时使用的立即惩罚大小
                },
                "CR": {
                    "use": True,
                    "continuousely_roll_threshold": 720,
                    "is_termination_reward_based_on_steps_left": True,  # 是否根据剩余步长计算惩罚
                    "termination_reward": -1,  # 不跟据剩余步长计算惩罚时使用的立即惩罚大小
                },
                "NOBR": {
                    "use": True,
                    "time_window": 2,  # 负过载且大幅度滚转：nz<0且phi>60(deg)连续超过time_window秒，就触发该终止条件
                    "negative_overload_threshold": 0,
                    "big_phi_threshold": 60,
                    "is_termination_reward_based_on_steps_left": True,  # 是否根据剩余步长计算惩罚
                    "termination_reward": -1,  # 不跟据剩余步长计算惩罚时使用的立即惩罚大小
                }
            }
        }
        self.config.update(env_config)

        self.reward_funcs: List[RewardBase] = []
        self._prep_reward_funcs()

        self.termination_funcs: List[TerminationBase] = []
        self._prep_termination_funcs()

        if np_random != None:
            self.np_random: np.random.Generator = np_random
        else:
            self.np_random, _ = seeding.np_random(np.random.randint(0, 1_000_000))
        
        self.goal_sampler: GoalSampler = GoalSampler(
            np_random=self.np_random, 
            env_config=self.config
        )

        self.logger: logging.Logger = my_logger

    def _prep_reward_funcs(self):
        self.reward_funcs = []
        for rwd in self.config["rewards"]:
            tmp_cfg = self.config["rewards"][rwd]
            if tmp_cfg["use"]:
                if rwd == "dense":
                    self.reward_funcs.append(
                        DenseRewardBasedOnAngleAndVelocity(
                            b=tmp_cfg.get("b", 0.5),
                            phi_scale=tmp_cfg.get("phi_scale", 180.),
                            theta_scale=tmp_cfg.get("theta_scale", 90.),
                            velocity_scale=tmp_cfg.get("velocity_scale", 100.),
                            weights=tmp_cfg.get("weights", [1./3., 1./3., 1./3.]),
                        )
                    )
                elif rwd == "sparse":
                    self.reward_funcs.append(
                        SparseReward(
                            reward_constant=tmp_cfg.get("reward_constant", 0.0),
                        )
                    )
                # TODO: other rewards

    def _prep_termination_funcs(self):
        # TODO: 如何设置优先级？？？
        self.termination_funcs = []

        for tmnt in self.config["terminations"]:
            tmp_cfg = self.config["terminations"][tmnt]
            if tmp_cfg["use"]:
                if tmnt == "RT":
                    self.termination_funcs.append(
                        ReachTargetTermination(
                            integral_time_length=tmp_cfg["integral_time_length"],
                            v_threshold=tmp_cfg["v_threshold"],
                            phi_threshold=tmp_cfg["phi_threshold"],
                            theta_threshold=tmp_cfg["theta_threshold"],
                            termination_reward=tmp_cfg["termination_reward"],
                            env_config=self.config,
                            # my_logger=self.logger
                        )
                    )
                elif tmnt == "RT_SINGLE_STEP":
                    self.termination_funcs.append(
                        ReachTargetTerminationSingleStep(
                            v_threshold=tmp_cfg["v_threshold"],
                            phi_threshold=tmp_cfg["phi_threshold"],
                            theta_threshold=tmp_cfg["theta_threshold"],
                            termination_reward=tmp_cfg["termination_reward"],
                            env_config=self.config,
                        )
                    )
                elif tmnt == "C":
                    self.termination_funcs.append(
                        CrashTermination(
                            h0=tmp_cfg["h0"],
                            is_termination_reward_based_on_steps_left=tmp_cfg["is_termination_reward_based_on_steps_left"],
                            termination_reward=tmp_cfg["termination_reward"],
                            env_config=self.config,
                            # my_logger=self.logger
                        )
                    )
                elif tmnt == "ES":
                    self.termination_funcs.append(
                        ExtremeStateTermination(
                            v_max=tmp_cfg["v_max"],
                            p_max=tmp_cfg["p_max"],
                            is_termination_reward_based_on_steps_left=tmp_cfg["is_termination_reward_based_on_steps_left"],
                            termination_reward=tmp_cfg["termination_reward"],
                            env_config=self.config,
                            # my_logger=self.logger
                        )
                    )
                elif tmnt == "T":
                    self.termination_funcs.append(
                        TimeoutTermination(
                            termination_reward=tmp_cfg["termination_reward"],
                            env_config=self.config,
                            # my_logger=self.logger
                        )
                    )
                elif tmnt == "CMA":
                    self.termination_funcs.append(
                        ContinuouselyMoveAwayTermination(
                            time_window=tmp_cfg["time_window"],
                            ignore_phi_error=tmp_cfg["ignore_phi_error"],
                            ignore_theta_error=tmp_cfg["ignore_theta_error"],
                            is_termination_reward_based_on_steps_left=tmp_cfg["is_termination_reward_based_on_steps_left"],
                            termination_reward=tmp_cfg["termination_reward"],
                            env_config=self.config,
                            # my_logger=self.logger
                        )
                    )
                elif tmnt == "CR":
                    self.termination_funcs.append(
                        ContinuouselyRollTermination(
                            continuousely_roll_threshold=tmp_cfg["continuousely_roll_threshold"],
                            is_termination_reward_based_on_steps_left=tmp_cfg["is_termination_reward_based_on_steps_left"],
                            termination_reward=tmp_cfg["termination_reward"],
                            env_config=self.config,
                            # my_logger=self.logger
                        )
                    )
                elif tmnt == "NOBR":
                    self.termination_funcs.append(
                        NegativeOverloadAndBigPhiTermination(
                            time_window=tmp_cfg["time_window"],
                            negative_overload_threshold=tmp_cfg["negative_overload_threshold"],
                            big_phi_threshold=tmp_cfg["big_phi_threshold"],
                            is_termination_reward_based_on_steps_left=tmp_cfg["is_termination_reward_based_on_steps_left"],
                            termination_reward=tmp_cfg["termination_reward"],
                            env_config=self.config,
                            # my_logger=self.logger
                        )
                    )

    def get_obs(self) -> np.ndarray:
        current_state_dict = self.plane.current_obs
        return np.array([
            current_state_dict['phi'],
            current_state_dict['theta'],
            current_state_dict['psi'],
            current_state_dict['v'],
            current_state_dict['mu'],
            current_state_dict['chi'],
            current_state_dict['p'],
            current_state_dict['h']
        ])

    def get_achieved_goal(self) -> np.ndarray:
        current_state_dict = self.plane.current_obs
        return np.array([
            current_state_dict['v'],
            current_state_dict['phi'],
            current_state_dict['theta'],
        ])

    def reset(self) -> None:
        goal_dict = self.goal_sampler.sample_goal()
        self.goal = np.array([goal_dict["v"], goal_dict["phi"], goal_dict["theta"]])
        
        for t_func in self.termination_funcs:
            t_func.reset()
        
        for r_func in self.reward_funcs:
            r_func.reset()
    
    def is_success(
        self, 
        achieved_goal: np.ndarray, 
        desired_goal: np.ndarray, 
        info: Dict[str, Any] = {}
    ) -> np.ndarray:
        """use only ReachTargetTerminationSingleStep to judge whether achieving desired goal for off-policy RL algorithms.

        Args:
            achieved_goal (np.ndarray): _description_
            desired_goal (np.ndarray): _description_
            info (Dict[str, Any], optional): _description_. Defaults to {}.

        Returns:
            np.ndarray: _description_
        """

        # pick out the reach target termination
        reach_target_termination_func = None
        for t in self.termination_funcs:
            if isinstance(t, ReachTargetTerminationSingleStep):
                reach_target_termination_func = t
                break
        
        if reach_target_termination_func == None:
            raise ValueError("AttitudeControlTask: when using off-policy algorithms, must use the termination condition: ReachTargetTerminationSingleStep!!!")

        # make tmp_achieved_goal and tmp_desired_goal be of shape (batch, goal_dim)
        if len(achieved_goal.shape) == 1:
            tmp_achieved_goal = achieved_goal.reshape((1, -1))
            tmp_desired_goal = desired_goal.reshape((1, -1))
        elif len(achieved_goal.shape) == 2:
            tmp_achieved_goal = achieved_goal
            tmp_desired_goal = desired_goal
        else:
            raise ValueError("AttitudeControlTask: the shape of achieved goal mush be 1-D or 2-D!")

        terminated_arr = []
        for tmp_a, tmp_d in zip(tmp_achieved_goal, tmp_desired_goal):
            state_var = AttitudeControlTask.get_state_vars()
            cur_state_namedtuple = state_var(phi=tmp_a[1], theta=tmp_a[2], psi=0, v=tmp_a[0], mu=0, chi=0, p=0, h=0)

            ternimated, truncated = reach_target_termination_func.get_termination(
                state=cur_state_namedtuple,
                goal_v=tmp_d[0],
                goal_phi=tmp_d[1],
                goal_theta=tmp_d[2]
            )

            terminated_arr.append(ternimated)
        
        if len(achieved_goal.shape) == 1:
            return terminated_arr[0]
        else:
            return np.array(terminated_arr, dtype=bool)

    def compute_reward(
        self, 
        achieved_goal: np.ndarray, 
        desired_goal: np.ndarray, 
        info: Union[Dict[str, Any], List[Dict]] = {}
    ) -> np.ndarray:
        
        if len(achieved_goal.shape) == 1:
            tmp_achieved_goals = achieved_goal.reshape((1, -1))
            tmp_desired_goals = desired_goal.reshape((1, -1))
            tmp_infos = [info]
        elif len(achieved_goal.shape) == 2:
            tmp_achieved_goals = achieved_goal
            tmp_desired_goals = desired_goal
            tmp_infos = info
        else:
            raise ValueError("AttitudeControlTask: the shape of achieved goal mush be 1-D or 2-D!")
        
        # compute reward: base on self.reward_funcs
        reward_arr = []
        state_var = AttitudeControlTask.get_state_vars()

        for tmp_a, tmp_d, tmp_info in zip(tmp_achieved_goals, tmp_desired_goals, tmp_infos):
            # 使用self.reward_funcs中的所有奖励函数计算reward
            reward = 0.
            for r_func in self.reward_funcs:
                tmp_reward = r_func.get_reward(
                    state=state_var(0., 0., 0., 0., 0., 0., 0., 0.),
                    goal_v=tmp_d[0],
                    goal_phi=tmp_d[1],
                    goal_theta=tmp_d[2],
                    next_state=state_var(tmp_a[1], tmp_a[2], 0., tmp_a[0], 0., 0., 0., 0.)
                )
                reward += tmp_reward / len(self.reward_funcs)
            reward_arr.append(reward)
        
        if len(achieved_goal.shape) == 1:
            return reward_arr[0]
        else:
            return np.array(reward_arr, dtype=np.float32)
        
    @staticmethod
    def get_state_vars():
        """学习器使用的观测

        Returns:
            _type_: _description_
        """
        return namedtuple("state_vars", ["phi", "theta", "psi", "v", "mu", "chi", "p", "h"])
    
    @staticmethod
    def get_goal_vars():

        return namedtuple("goal_vars", ["v", "phi", "theta"])
    
    @staticmethod
    def convert_dict_to_state_vars(state_dict:dict) -> namedtuple:
        """将仿真器返回的字典类型观测转换为环境定义的观测(AttitudeControlTask.get_state_vars()定义的namedtuple)

        Args:
            state_dict (dict): 键包括：lef, npos, epos, h, alpha, beta, phi, theta, psi, p, q, r, v, vn, ve, vh, nx, ny, nz, ele, ail, rud, thrust, lon, lat, mu, chi

        Returns:
            namedtuple: _description_
        """

        state_vars_type = AttitudeControlTask.get_state_vars()
        return state_vars_type(
            phi=state_dict['phi'], theta=state_dict['theta'], psi=state_dict['psi'], 
            v=state_dict['v'], mu=state_dict['mu'], chi=state_dict['chi'],
            p=state_dict['p'], h=state_dict['h']
        )

    @staticmethod
    def get_state_lower_bounds():
        """返回观测的下限

        Returns:
            _type_: _description_
        """
        state_vars_type = AttitudeControlTask.get_state_vars()
        return state_vars_type(phi=-180., theta=-90., psi=-180., v=0., mu=-90., chi=-180., p=-300., h=0.)

    @staticmethod
    def get_state_higher_bounds():
        """返回观测的上限

        Returns:
            _type_: _description_
        """
        state_vars_type = AttitudeControlTask.get_state_vars()
        return state_vars_type(phi=180., theta=90., psi=180., v=1000., mu=90., chi=180., p=300., h=20000.)

    @staticmethod
    def get_goal_lower_bounds():
        goal_vars_type = AttitudeControlTask.get_goal_vars()
        return goal_vars_type(v=0., phi=-180., theta=-90.)
    
    @staticmethod
    def get_goal_higher_bounds():
        goal_vars_type = AttitudeControlTask.get_goal_vars()
        return goal_vars_type(v=1000., phi=180., theta=90.)
    