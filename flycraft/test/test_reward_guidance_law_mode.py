import unittest
from pathlib import Path

from flycraft.env import FlyCraftEnv

PROJECT_ROOT_DIR = Path(__file__).parent.parent


class FlyCraftEnvRewardTest(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.env = FlyCraftEnv(
            config_file=PROJECT_ROOT_DIR / "configs" / "MR_for_HER.json"
        )

    def test_reward_1(self):
        obs, info = self.env.reset()
        terminated = False
        step = 0
        while not terminated:
            action = self.env.action_space.sample()
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            # print(f"step: {step}, {info}")
            step += 1
            if not terminated:
                self.assertAlmostEqual(
                    reward,
                    self.env.compute_reward(
                        achieved_goal=self.env.task.get_achieved_goal(),
                        desired_goal=self.env.task.get_goal(),
                        info=info
                    )
                )


if __name__ == "__main__":
    unittest.main()