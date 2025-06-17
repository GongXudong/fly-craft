import unittest
from pathlib import Path
import sys
import numpy as np
from tqdm import tqdm

PROJECT_ROOT_DIR = Path(__file__).parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from env import FlyCraftEnv
from utils_common.load_config import load_config


class FlyCraftEnvRandomSampleTest(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()

    def test_init_from_config_file1(self):
        self.env = FlyCraftEnv(
            config_file=PROJECT_ROOT_DIR / "configs" / "MR_for_HER_end2end.json",
            custom_config={
                "debug_mode": True
            }
        )

        # 测试10个episodes
        traj_length = []
        traj_n = 10
        for i in tqdm(range(traj_n)):
            obs, info = self.env.reset()
            terminated = False
            steps = 0
            while not terminated:
                action = self.env.action_space.sample()
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                steps += 1
            traj_length.append(steps)
        
        print(np.array(traj_length).mean())

    def test_init_from_custom_config1(self):
        self.env = FlyCraftEnv(
            custom_config=load_config(PROJECT_ROOT_DIR / "configs" / "MR_for_HER_end2end.json")
        )

        # 测试10个episodes
        traj_length = []
        traj_n = 10
        for i in tqdm(range(traj_n)):
            obs, info = self.env.reset()
            terminated = False
            steps = 0
            while not terminated:
                action = self.env.action_space.sample()
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                steps += 1
            traj_length.append(steps)
        
        print(np.array(traj_length).mean())


if __name__ == "__main__":
    unittest.main()