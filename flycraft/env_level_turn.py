import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
import pandas as pd
from collections import namedtuple
from typing import List
from pathlib import Path
from copy import deepcopy
from typing import Union, Dict

from flycraft.planes.f16_plane import F16Plane
from flycraft.tasks.BFM_level_turn_task import BFMLevelTurnTask
from flycraft.utils_common.dict_utils import update_nested_dict
from flycraft.utils_common.load_config import load_config

PROJECT_ROOT_DIR = Path(__file__).parent


class FlyCraftLevelTurnEnv(gym.Env):
    """Velocity Control Task, controlling aircraft's velocity vector to achieve desired velocity vector $(v, \mu, \chi)$.
    """

    def __init__(
        self,
        custom_config: dict={},
        config_file: Union[Path, str]=Path(PROJECT_ROOT_DIR / "configs" / "NMR.json"),
    ) -> None:
        # define spaces
        state_mins = BFMLevelTurnTask.get_state_lower_bounds()
        state_maxs = BFMLevelTurnTask.get_state_higher_bounds()
        goal_mins = BFMLevelTurnTask.get_goal_lower_bounds()
        goal_maxs = BFMLevelTurnTask.get_goal_higher_bounds()
        self.observation_space = spaces.Dict(
            dict(
                observation = spaces.Box(low=np.array(state_mins, dtype=np.float32), high=np.array(state_maxs, dtype=np.float32)),  # phi, theta, psi, v, mu, chi, p, h
                desired_goal = spaces.Box(low=np.array(goal_mins, dtype=np.float32), high=np.array(goal_maxs, dtype=np.float32)),
                achieved_goal = spaces.Box(low=np.array(goal_mins, dtype=np.float32), high=np.array(goal_maxs, dtype=np.float32)),
            )
        )

        print(f"load config from: {config_file}")

        self.env_config: dict = load_config(config_file)
        update_nested_dict(self.env_config, custom_config)

        self.plane: F16Plane = F16Plane(env_config=self.env_config)
        self.task: BFMLevelTurnTask = BFMLevelTurnTask(
            plane=self.plane,
            env_config=self.env_config,
            np_random=self.np_random,
        )

        action_mins = F16Plane.get_action_lower_bounds(self.plane.control_mode)
        action_maxs = F16Plane.get_action_higher_bounds(self.plane.control_mode)
        self.action_space = spaces.Box(
            low=np.array(action_mins, dtype=np.float32),
            high=np.array(action_maxs, dtype=np.float32),
        )  # (p, nz, pla) if control_mode == "guidance_law_mode" else (ail, ele, rud, pla)

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
            "observation": np.array(BFMLevelTurnTask.convert_dict_to_state_vars(self.plane_state_dict_list[-1]), dtype=np.float32),
            "achieved_goal": self.task.get_achieved_goal().astype(np.float32),
            "desired_goal": self.task.get_goal().astype(np.float32),
        }

    def reset(self, *, seed = None, options = None):
        """ 重置simulator和task

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
        self.plane_state_namedtuple_list = [BFMLevelTurnTask.convert_dict_to_state_vars(plane_state_dict)]
        self.action_list = []

        info = {}
        info["plane_state"] = plane_state_dict

        return self._get_obs(), info

    def close(self):
        self.plane.close()

    def step(self, action):
        """仿真器仿真一步，基于task的定义获得下一步观测，奖励，episode是否终止，episode是否被truncated，辅助信息

        Args:
            action (_type_): guidance_law_mode模式下action是制导律输出的[p, nz, pla]，end_to_end_mode模式下action是[ail, ele, rud, pla]

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
        plane_state_namedtuple = BFMLevelTurnTask.convert_dict_to_state_vars(plane_state_dict)
        self.plane_state_namedtuple_list.append(plane_state_namedtuple)

        # check obs for NaN!!!!!!!!!
        if np.any(np.isnan(plane_state_namedtuple)):

            tmp = []

            for item_obs, item_act in zip(self.plane_state_namedtuple_list, self.action_list):
                if self.debug_mode:
                    print(item_obs, item_act)
                tmp.append([*item_obs, *item_act])

            if self.plane.control_mode_guidance_law:
                tmp_pd = pd.DataFrame(
                    data=tmp,
                    columns=['s_phi', 's_theta', 's_psi', 's_v', 's_mu', 's_chi', 's_p', 's_h', 'target_v', 'target_mu', 'target_chi','a_p', 'a_nz', 'a_pla'],
                )
            else:
                tmp_pd = pd.DataFrame(
                    data=tmp,
                    columns=['s_phi', 's_theta', 's_psi', 's_v', 's_mu', 's_chi', 's_p', 's_h', 'target_v', 'target_mu', 'target_chi','a_ail', 'a_ele', 'a_rud', 'a_pla'],
                )

            tmp_pd.to_csv(str((PROJECT_ROOT_DIR / "my_logs" / 'nan_states.csv').absolute()), index=False)

        terminated, truncated, reward, info = False, False, 0., {}
        info["step"] = self.step_cnt
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
                goal_chi=desired_goal[2],
            )
            if terminated:
                info["termination"] = str(t_func)

                # if isinstance(t_func, ReachTargetTermination) or isinstance(t_func, ReachTargetTermination2) or isinstance(t_func, ReachTargetTerminationSingleStep) or isinstance(t_func, ReachTargetTerminationVMuChiSingleStep):
                if self.task.is_reach_target_termination(t_func):
                    info['is_success'] = True

                if self.debug_mode:
                    # if isinstance(t_func, ReachTargetTermination) or isinstance(t_func, ReachTargetTermination2) or isinstance(t_func, ReachTargetTerminationSingleStep) or isinstance(t_func, ReachTargetTerminationVMuChiSingleStep):
                    if self.task.is_reach_target_termination(t_func):
                        # reach target, print with green
                        print(f"print, {self.flag_str}, \033[32m{str(t_func)}。\033[0m steps: {self.step_cnt}。target: ({desired_goal[0]:.2f}, {desired_goal[1]:.2f}, {desired_goal[2]:.2f})。achieved target: ({plane_state_dict['v']:.2f}, {plane_state_dict['mu']:.2f}, {plane_state_dict['chi']:.2f})。expert steps: {self.task.goal_sampler.goal_expert_length}。")
                    else:
                        # the other terminations, print with red
                        print(f"print, {self.flag_str}, \033[31m{str(t_func)}。\033[0m steps: {self.step_cnt}。target: ({desired_goal[0]:.2f}, {desired_goal[1]:.2f}, {desired_goal[2]:.2f})。achieved target: ({plane_state_dict['v']:.2f}, {plane_state_dict['mu']:.2f}, {plane_state_dict['chi']:.2f})。expert steps: {self.task.goal_sampler.goal_expert_length}。")

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

        if self.plane.control_mode_guidance_law:
            info["action"] = {
                "p": action[0],
                "nz": action[1],
                "pla": action[2],
                "rud": 0.,
            }
        else:
            info["action"] = {
                "ail": action[0],
                "ele": action[1],
                "rud": action[2],
                "pla": action[3],
            }
        info["desired_goal"] = [*desired_goal]
        info["plane_state"] = deepcopy(self.plane_state_dict_list[-2])
        info["plane_next_state"] = deepcopy(self.plane_state_dict_list[-1])

        return self._get_obs(), reward, terminated, truncated, info
