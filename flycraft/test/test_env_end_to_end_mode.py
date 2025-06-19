import unittest
from pathlib import Path

from flycraft.env import FlyCraftEnv

PROJECT_ROOT_DIR = Path(__file__).parent.parent


class FlyCraftEnvTest(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.env = FlyCraftEnv(
            config_file=PROJECT_ROOT_DIR / "configs" / "MR_for_HER_end2end.json",
            custom_config={}
        )

    def test_reset(self):
        print("In test reset: ")
        obs, info = self.env.reset(seed=1)
        self.assertEqual(self.env.np_random, self.env.task.goal_sampler.np_random)
        # print(obs, info)
        obs, info = self.env.reset()
        self.assertEqual(self.env.np_random, self.env.task.goal_sampler.np_random)
        # print(obs, info)
        obs, info = self.env.reset()
        self.assertEqual(self.env.np_random, self.env.task.goal_sampler.np_random)

    def test_step_1(self):
        obs, info = self.env.reset()
        terminated = False
        while not terminated:
            action = self.env.action_space.sample()
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            # print(action)
            # print(next_obs["observation"])
            # print(f"Step {i}:\n", next_obs, reward, terminated, truncated, info)
        # print(info)


if __name__ == "__main__":
    unittest.main()