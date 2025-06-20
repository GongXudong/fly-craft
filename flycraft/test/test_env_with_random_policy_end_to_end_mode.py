import unittest
from pathlib import Path
import numpy as np
from tqdm import tqdm

from flycraft.env import FlyCraftEnv

PROJECT_ROOT_DIR = Path(__file__).parent.parent


class FlyCraftEnvRandomSampleTest(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.env = FlyCraftEnv(
            config_file=PROJECT_ROOT_DIR / "configs" / "MR_for_HER_end2end.json",
            custom_config={
                "debug_mode": True
            }
        )

    def test_step(self):
        """task不使用CMA终止条件，随机策略采样的轨迹平均长367
        """
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