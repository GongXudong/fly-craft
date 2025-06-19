from pathlib import Path
import numpy as np
import pandas as pd


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
        self.goal_phi = self.env_config["goal"].get("goal_phi")
        self.goal_theta = self.env_config["goal"].get("goal_theta")
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
        
        # self.available_goals are pd.DataFrame with columns: goal_v, goal_phi, goal_theta, length
        # length represents the trajectory length of reaching (goal_v, goal_phi, goal_theta)
        self.available_goals = pd.read_csv(tmp_file_path.absolute())
        self.available_goals = self.available_goals[["v", "phi", "theta", "length"]]

    def sample_goal(self) -> dict:
        if not self.use_fixed_goal:
            if self.sample_random:
                self.goal_v = self.np_random.uniform(low=self.env_config["goal"]["v_min"], high=self.env_config["goal"]["v_max"])
                self.goal_phi = self.np_random.uniform(low=self.env_config["goal"]["phi_min"], high=self.env_config["goal"]["phi_max"])
                self.goal_theta = self.np_random.uniform(low=self.env_config["goal"]["theta_min"], high=self.env_config["goal"]["theta_max"])

                # if self.debug_mode:
                #     print(f"In sampler, sample randomly: {self.goal_v}, {self.goal_phi}, {self.goal_theta}")
            else:
                if self.sample_reachable_goal:
                    while True:
                        tmp_goal = self.np_random.choice(self.available_goals, 1).squeeze()
                        if tmp_goal[3] == 0:
                            continue
                        else:
                            sampled_noise = self.sample_noise()

                            self.goal_v = tmp_goal[0] + sampled_noise[0]
                            self.goal_phi = tmp_goal[1] + sampled_noise[1]
                            self.goal_theta = tmp_goal[2] + sampled_noise[2]
                            self.goal_expert_length = tmp_goal[3]
                            break

                    # if self.debug_mode:
                    #     print(f"In sampler: sample reachable goal: {self.goal_v}, {self.goal_phi}, {self.goal_theta}, {self.goal_expert_length}")
                else:
                    tmp_goal = self.np_random.choice(self.available_goals, 1).squeeze()
                    sampled_noise = self.sample_noise()

                    self.goal_v = tmp_goal[0] + sampled_noise[0]
                    self.goal_phi = tmp_goal[1] + sampled_noise[1]
                    self.goal_theta = tmp_goal[2] + sampled_noise[2]
                    self.goal_expert_length = tmp_goal[3]

                    # if self.debug_mode:
                    #     print(f"In sampler, sample available goal: {self.goal_v}, {self.goal_phi}, {self.goal_theta}, {self.goal_expert_length}")

        return {
            "v": self.goal_v,
            "phi": self.goal_phi,
            "theta": self.goal_theta
        }
    
    def sample_noise(self):
        """采样噪声
        """
        return (
            (2 * self.np_random.random() -1.) * self.sample_goal_noise_std[0], 
            (2 * self.np_random.random() -1.) * self.sample_goal_noise_std[1], 
            (2 * self.np_random.random() -1.) * self.sample_goal_noise_std[2]
        )