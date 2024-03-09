import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
import pandas as pd
from collections import namedtuple
from typing import List
import logging
from pathlib import Path
import sys
from copy import deepcopy
import gc
import time
from typing import Union, Dict

PROJECT_ROOT_DIR = Path(__file__).parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from planes.f16_plane import F16Plane
from tasks.attitude_control_task import AttitudeControlTask
from utils.load_config import load_config
from utils.dict_utils import update_nested_dict
from terminations.reach_target_termination import ReachTargetTermination
from terminations.reach_target_termination2 import ReachTargetTermination2
from terminations.reach_target_termination_single_step import ReachTargetTerminationSingleStep

from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.common.envs import BitFlippingEnv

class FlyCraftEnv(gym.Env):

    def __init__(self, 
        config_file: Union[Path, str],
        custom_config: dict={}
    ) -> None:
        # define spaces
        state_mins = AttitudeControlTask.get_state_lower_bounds()
        state_maxs = AttitudeControlTask.get_state_higher_bounds()
        goal_mins = AttitudeControlTask.get_goal_lower_bounds()
        goal_maxs = AttitudeControlTask.get_goal_higher_bounds()
        self.observation_space = spaces.Dict(
            dict(
                observation = spaces.Box(low=np.array(state_mins, dtype=np.float32), high=np.array(state_maxs, dtype=np.float32)),  # phi, theta, psi, v, mu, chi, p, h
                desired_goal = spaces.Box(low=np.array(goal_mins, dtype=np.float32), high=np.array(goal_maxs, dtype=np.float32)),
                achieved_goal = spaces.Box(low=np.array(goal_mins, dtype=np.float32), high=np.array(goal_maxs, dtype=np.float32)),
            )
        )

        action_mins = F16Plane.get_action_lower_bounds()
        action_maxs = F16Plane.get_action_higher_bounds()
        self.action_space = spaces.Box(low=np.array(action_mins, dtype=np.float32), high=np.array(action_maxs, dtype=np.float32))  # p, nz, pla

        self.env_config: dict = load_config(config_file)
        update_nested_dict(self.env_config, custom_config)
        
        self.plane: F16Plane = F16Plane(env_config=self.env_config)
        self.task: AttitudeControlTask = AttitudeControlTask(
            plane=self.plane,
            np_random=self.np_random,
            env_config=self.env_config
        )
        
        # log data
        self.step_cnt = 0
        self.plane_state_dict_list: List[Dict] = []
        self.plane_state_namedtuple_list: List[namedtuple] = []
        self.action_list = []
        
        # use for debug
        self.debug_mode: bool = self.env_config.get("debug_mode")
        self.flag_str: str = self.env_config.get("flag_str")

        # used by HerReplayBuffer
        self.compute_reward = self.task.compute_reward
    
    def _get_obs(self) -> Dict[str, np.ndarray]:
        return {
            "observation": np.array(AttitudeControlTask.convert_dict_to_state_vars(self.plane_state_dict_list[-1]), dtype=np.float32),
            "achieved_goal": self.task.get_achieved_goal().astype(np.float32),
            "desired_goal": self.task.get_goal().astype(np.float32),
        }

    def reset(self, *, seed = None, options = None):
        """

        Args:
            seed (_type_, optional): _description_. Defaults to None.
            options (_type_, optional): _description_. Defaults to None.
        
        Returns:
            tuple[ObsType, dict[str, Any]]: obs, info
        """
        super().reset(seed=seed, options=options)
        if seed != None:
            self.task.np_random = self.np_random
            self.task.goal_sampler.np_random = self.np_random

        plane_state_dict = self.plane.reset()
        self.task.reset()

        self.step_cnt = 0
        self.plane_state_dict_list = [plane_state_dict]
        self.plane_state_namedtuple_list = [AttitudeControlTask.convert_dict_to_state_vars(plane_state_dict)]
        self.action_list = []

        info = {}
        info["plane_state"] = plane_state_dict

        return self._get_obs(), info

    def close(self):
        self.plane.close()

    def step(self, action):
        """_summary_

        Args:
            action (_type_): action是制导律输出的[p, nz, pla]

        Returns:
            tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]: next_obs, reward, terminated, truncated, info

            info = {
                "plane_state": dict,
                "plane_next_state": dict,
                "rewards": float,
                "termination": bool,
                "step": int,
                "is_success": bool,
            }
        """
        self.step_cnt += 1
        self.action_list.append(action)

        plane_state_dict = self.plane.step(action)
        self.plane_state_dict_list.append(deepcopy(plane_state_dict))
        plane_state_namedtuple = AttitudeControlTask.convert_dict_to_state_vars(plane_state_dict)
        self.plane_state_namedtuple_list.append(plane_state_namedtuple)
        
        # check obs for NaN!!!!!!!!!
        if np.any(np.isnan(plane_state_namedtuple)):
            tmp = []
            for item_obs, item_act in zip(self.plane_state_namedtuple_list, self.action_list):
                if self.debug_mode:
                    print(item_obs, item_act)
                tmp.append([*item_obs, *item_act])
            tmp_pd = pd.DataFrame(data=tmp, columns=['s_phi', 's_theta', 's_psi', 's_v', 's_mu', 's_chi', 's_p', 's_h', 'target_v', 'target_mu', 'target_chi','a_p', 'a_nz', 'a_pla'])
            tmp_pd.to_csv(str((PROJECT_ROOT_DIR / "my_logs" / 'nan_states.csv').absolute()), index=False)

        terminated, truncated, reward, info = False, False, 0., {}
        info["step"] = self.step_cnt
        # tmp_goal = self.task.get_goal()
        # info["target_v"] = tmp_goal[0]
        # info["target_mu"] = tmp_goal[1]
        # info["target_chi"] = tmp_goal[2]
        info['is_success'] = False  # https://stable-baselines3.readthedocs.io/en/feat-gymnasium-support/common/logger.html#eval 评估中success_rate要求这个字段

        desired_goal = self.task.get_goal()

        # judge termination
        for t_func in self.task.termination_funcs:
            terminated, truncated, reward = t_func.get_termination_and_reward(
                state=self.plane_state_namedtuple_list[-2], 
                next_state=self.plane_state_namedtuple_list[-1],  # ContinuouselyRollTermination,
                step_cnt=self.step_cnt,  # TimeoutTermination,
                state_list=self.plane_state_namedtuple_list,  # ReachTargetTermination, 
                nz=self.plane_state_dict_list[-1]["nz"],  # NegativeOverloadAndBigPhiTermination,
                goal_v=desired_goal[0],
                goal_mu=desired_goal[1],
                goal_chi=desired_goal[2]
            )
            if terminated:
                info["termination"] = str(t_func)
                
                if isinstance(t_func, ReachTargetTermination) or isinstance(t_func, ReachTargetTermination2) or isinstance(t_func, ReachTargetTerminationSingleStep):
                    info['is_success'] = True

                if self.debug_mode:
                    if isinstance(t_func, ReachTargetTermination) or isinstance(t_func, ReachTargetTermination2):
                        # reach target, 绿色打印
                        print(f"print, {self.flag_str}, ", f"\033[32m{str(t_func)}。\033[0m", f"steps: {self.step_cnt}。target: ({desired_goal[0]}, {desired_goal[1]}, {desired_goal[2]})。expert steps: {self.task.goal_sampler.goal_expert_length}。")
                    else:
                        # 其它终止条件，红色打印
                        print(f"print, {self.flag_str}, ", f'\033[31m{str(t_func)}。\033[0m', f"steps: {self.step_cnt}。target: ({desired_goal[0]}, {desired_goal[1]}, {desired_goal[2]})。expert steps: {self.task.goal_sampler.goal_expert_length}。")

                if "rewards" not in info:
                    info["rewards"] = {}
                info["rewards"][str(t_func)] = reward
                break
        
        # compute reward
        for r_func in self.task.reward_funcs:
            tmp_reward = r_func.get_reward(
                state=self.plane_state_namedtuple_list[-2], 
                next_state=self.plane_state_namedtuple_list[-1],
                done=terminated,
                goal_v=desired_goal[0],
                goal_mu=desired_goal[1],
                goal_chi=desired_goal[2],
            )
            if "rewards" not in info:
                info["rewards"] = {}
            info["rewards"][str(r_func)] = tmp_reward
            reward += tmp_reward / len(self.task.reward_funcs)
        
        info["action"] = {
            "p": action[0],
            "nz": action[1],
            "pla": action[2],
            "rud": 0.
        }
        info["desired_goal"] = [*desired_goal]
        info["plane_state"] = deepcopy(self.plane_state_dict_list[-2])
        info["plane_next_state"] = deepcopy(self.plane_state_dict_list[-1])

        return self._get_obs(), reward, terminated, truncated, info
