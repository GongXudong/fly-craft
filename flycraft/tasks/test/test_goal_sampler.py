import unittest
import numpy as np
from pathlib import Path
import sys
import math
from gymnasium.utils import seeding

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from flycraft.tasks.goal_samplers.goal_sampler_for_velocity_vector_control import GoalSampler
from utils.load_config import load_config
from utils.dict_utils import update_nested_dict


class GoalSamplerTest(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.config: dict = load_config(PROJECT_ROOT_DIR / "configs" / "NMR.json")
        self._np_random, _ = seeding.np_random(np.random.randint(0, 1000))
        self.goal_sampler = GoalSampler(np_random=self._np_random, env_config=self.config)
        
    def test_1(self):
        """fixed goal
        """
        tmp_config = {
            "debug_mode": False,
            "goal":{
                "use_fixed_goal": True,
                "goal_v": 200., 
                "goal_mu": 0., 
                "goal_chi": 90.,
            }
        }
        update_nested_dict(self.config, tmp_config)

        self.goal_sampler = GoalSampler(self._np_random, self.config)
        for i in range(10):
            goal = self.goal_sampler.sample_goal()
            goal_arr = [goal["v"], goal["mu"], goal["chi"]]
            self.assertAlmostEqual(goal_arr, [200., 0., 90.])

    def test_2(self):
        """sample random goal
        """
        tmp_config = {
            "debug_mode": True,
            "goal": {
                "use_fixed_goal": False,
                "sample_random": True,
            }
        }
        update_nested_dict(self.config, tmp_config)

        self.goal_sampler = GoalSampler(self._np_random, self.config)

        for i in range(10):
            goal = self.goal_sampler.sample_goal()
            self.assertGreaterEqual(goal["v"], self.goal_sampler.env_config["goal"]["v_min"])
            self.assertLessEqual(goal["v"], self.goal_sampler.env_config["goal"]["v_max"])
            self.assertGreaterEqual(goal["mu"], self.goal_sampler.env_config["goal"]["mu_min"])
            self.assertLessEqual(goal["mu"], self.goal_sampler.env_config["goal"]["mu_max"])
            self.assertGreaterEqual(goal["chi"], self.goal_sampler.env_config["goal"]["chi_min"])
            self.assertLessEqual(goal["chi"], self.goal_sampler.env_config["goal"]["chi_max"])

    def test_3(self):
        """sample available goals, no noise
        """
        tmp_config = {
            "debug_mode": True,
            "goal": {
                "use_fixed_goal": False,
                "sample_random": False,
                "sample_reachable_goal": False,
                "available_goals_file": (Path(__file__).parent / "available_goals_for_test.csv").absolute().__str__(),
                "sample_goal_noise_std": [0., 0., 0.],
            }
        }
        update_nested_dict(self.config, tmp_config)

        self.goal_sampler = GoalSampler(self._np_random, self.config)
        for i in range(10):
            goal = self.goal_sampler.sample_goal()
    
    def test_4(self):
        """sample available goals, with noise
        """
        tmp_config = {
            "debug_mode": True,
            "goal": {
                "use_fixed_goal": False,
                "sample_random": False,
                "sample_reachable_goal": False,
                "available_goals_file": (Path(__file__).parent / "available_goals_for_test.csv").absolute().__str__(),
                "sample_goal_noise_std": [5., 0.5, 0.5],
            }
        }
        update_nested_dict(self.config, tmp_config)

        self.goal_sampler = GoalSampler(self._np_random, self.config)
        for i in range(10):
            goal = self.goal_sampler.sample_goal()

    def test_5(self):
        """sample achievable goals, no noise
        """
        tmp_config = {
            "debug_mode": True,
            "goal": {
                "use_fixed_goal": False,
                "sample_random": False,
                "sample_reachable_goal": True,
                "available_goals_file": (Path(__file__).parent / "available_goals_for_test.csv").absolute().__str__(),
                "sample_goal_noise_std": [0., 0., 0.],
            }
        }
        update_nested_dict(self.config, tmp_config)

        self.goal_sampler = GoalSampler(self._np_random, self.config)
        for i in range(10):
            goal = self.goal_sampler.sample_goal()
    
    def test_6(self):
        """sample achievable goals, with noise
        """
        tmp_config = {
            "debug_mode": True,
            "goal": {
                "use_fixed_goal": False,
                "sample_random": False,
                "sample_reachable_goal": True,
                "available_goals_file": (Path(__file__).parent / "available_goals_for_test.csv").absolute().__str__(),
                "sample_goal_noise_std": [5., 0.5, 0.5],
            }
        }
        update_nested_dict(self.config, tmp_config)

        self.goal_sampler = GoalSampler(self._np_random, self.config)
        for i in range(10):
            goal = self.goal_sampler.sample_goal()

if __name__ == "__main__":
    unittest.main()