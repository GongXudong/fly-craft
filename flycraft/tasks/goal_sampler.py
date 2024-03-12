import sys
from pathlib import Path
import numpy as np
import pandas as pd
from gymnasium.utils import seeding

PROJECT_ROOT_DIR = Path(__file__).parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from utils.load_config import load_config
from utils.dict_utils import update_nested_dict

class GoalSampler(object):

    def __init__(
        self, 
        np_random: np.random.Generator, 
        env_config: dict
    ) -> None:
        self.env_config: dict = env_config

        self.flag_str = self.env_config.get("flag_str", "Train")
        self.debug_mode = self.env_config.get("debug_mode", True)

        self.np_random: np.random.Generator = np_random

        self.use_fixed_goal = self.env_config["goal"].get("use_fixed_goal")
        self.goal_v = self.env_config["goal"].get("goal_v")
        self.goal_mu = self.env_config["goal"].get("goal_mu")
        self.goal_chi = self.env_config["goal"].get("goal_chi")
        self.goal_expert_length = 0

        self.sample_random = self.env_config["goal"].get("sample_random")  # does sample randomly
        if not self.use_fixed_goal and not self.sample_random:
            self._load_available_goals()
            
            # does only sample reachable goals (length > 0) from available goals
            self.sample_reachable_goal = self.env_config["goal"].get("sample_reachable_goal")
            if self.sample_reachable_goal:
                self.available_goals = self.available_goals[self.available_goals.length > 0]
            
            self.sample_goal_noise_std = self.env_config["goal"].get("sample_goal_noise_std")

    def _load_available_goals(self):
        self.available_goals_file: str = self.env_config["goal"].get("available_goals_file")
        tmp_file_path = Path(self.available_goals_file)
        
        # self.available_goals are pd.DataFrame with columns: goal_v, goal_mu, goal_chi, length
        # length represents the trajectory length of reaching (goal_v, goal_mu, goal_chi)
        self.available_goals = pd.read_csv(tmp_file_path.absolute())
        self.available_goals = self.available_goals[["v", "mu", "chi", "length"]]

    def sample_goal(self) -> dict:
        if not self.use_fixed_goal:
            if self.sample_random:
                self.goal_v = self.np_random.uniform(low=self.env_config["goal"]["v_min"], high=self.env_config["goal"]["v_max"])
                self.goal_mu = self.np_random.uniform(low=self.env_config["goal"]["mu_min"], high=self.env_config["goal"]["mu_max"])
                self.goal_chi = self.np_random.uniform(low=self.env_config["goal"]["chi_min"], high=self.env_config["goal"]["chi_max"])

                # if self.debug_mode:
                #     print(f"In sampler, sample randomly: {self.goal_v}, {self.goal_mu}, {self.goal_chi}")
            else:
                if self.sample_reachable_goal:
                    while True:
                        tmp_goal = self.np_random.choice(self.available_goals, 1).squeeze()
                        if tmp_goal[3] == 0:
                            continue
                        else:
                            sampled_noise = self.sample_noise()

                            self.goal_v = tmp_goal[0] + sampled_noise[0]
                            self.goal_mu = tmp_goal[1] + sampled_noise[1]
                            self.goal_chi = tmp_goal[2] + sampled_noise[2]
                            self.goal_expert_length = tmp_goal[3]
                            break

                    # if self.debug_mode:
                    #     print(f"In sampler: sample reachable goal: {self.goal_v}, {self.goal_mu}, {self.goal_chi}, {self.goal_expert_length}")
                else:
                    tmp_goal = self.np_random.choice(self.available_goals, 1).squeeze()
                    sampled_noise = self.sample_noise()

                    self.goal_v = tmp_goal[0] + sampled_noise[0]
                    self.goal_mu = tmp_goal[1] + sampled_noise[1]
                    self.goal_chi = tmp_goal[2] + sampled_noise[2]
                    self.goal_expert_length = tmp_goal[3]

                    # if self.debug_mode:
                    #     print(f"In sampler, sample available goal: {self.goal_v}, {self.goal_mu}, {self.goal_chi}, {self.goal_expert_length}")

        return {
            "v": self.goal_v,
            "mu": self.goal_mu,
            "chi": self.goal_chi
        }
    
    def sample_noise(self):
        """采样噪声
        """
        return (
            (2 * self.np_random.random() -1.) * self.sample_goal_noise_std[0], 
            (2 * self.np_random.random() -1.) * self.sample_goal_noise_std[1], 
            (2 * self.np_random.random() -1.) * self.sample_goal_noise_std[2]
        )