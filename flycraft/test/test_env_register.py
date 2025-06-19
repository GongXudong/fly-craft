import unittest
from pathlib import Path
import gymnasium as gym
from gymnasium.envs.registration import register

from flycraft.env import FlyCraftEnv

PROJECT_ROOT_DIR = Path(__file__).parent.parent

register(
    id="FlyCraft-v0",
    entry_point=FlyCraftEnv,
)


class FlyCraftEnvRegisterTest(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.env = gym.make(
            id="FlyCraft-v0", 
            config_file=str(PROJECT_ROOT_DIR / "configs" / "MR_for_HER.json")
        )

    def test_reset(self):
        print("In test reset: ")
        obs, info = self.env.reset()
        print(obs, info)

    def test_step_1(self):
        obs, info = self.env.reset()
        terminated = False
        while not terminated:
            action = self.env.action_space.sample()
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            print(action)
            print(next_obs["observation"])
            # print(f"Step {i}:\n", next_obs, reward, terminated, truncated, info)
        print(info)


if __name__ == "__main__":
    unittest.main()