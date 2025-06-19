import unittest
from pathlib import Path
import gymnasium as gym

import flycraft

PROJECT_ROOT_DIR = Path(__file__).parent.parent


class FlyCraftEnvRandomSampleTest(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()

    def test_with_random_actions_and_init_env_from_default(self):
        
        env = gym.make('FlyCraft-v0')  # use default configurations

        observation, info = env.reset()

        for _ in range(500):
            action = env.action_space.sample() # random action
            observation, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                observation, info = env.reset()

        env.close()

    def test_with_random_actions_and_init_env_from_config_file(self):
        
        # env = gym.make('FlyCraft-v0')  # use default configurations
        env = gym.make('FlyCraft-v0', config_file=PROJECT_ROOT_DIR / "configs" / "NMR.json")  # pass configurations through config_file

        observation, info = env.reset()

        for _ in range(500):
            action = env.action_space.sample() # random action
            observation, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                observation, info = env.reset()

        env.close()

    def test_with_random_actions_and_init_env_from_custom_config(self):
        
        # env = gym.make('FlyCraft-v0')  # use default configurations
        # env = gym.make('FlyCraft-v0', config_file=PROJECT_ROOT_DIR / "configs" / "NMR.json")  # pass configurations through config_file (Path or str)
        env = gym.make('FlyCraft-v0', custom_config={
            "task": {
                "control_mode": "end_to_end_mode",
            }
        })  # pass configurations through custom_config (dict)

        observation, info = env.reset()

        for _ in range(500):
            action = env.action_space.sample() # random action
            observation, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                observation, info = env.reset()

        env.close()

    def test_with_random_actions_and_init_env_from_both_config_file_and_custom_config(self):
        
        # env = gym.make('FlyCraft-v0')  # use default configurations
        # env = gym.make('FlyCraft-v0', config_file=PROJECT_ROOT_DIR / "configs" / "NMR.json")  # pass configurations through config_file (Path or str)
        env = gym.make(
            'FlyCraft-v0',
            config_file=PROJECT_ROOT_DIR / "configs" / "NMR.json",
            custom_config={
                "task": {
                    "control_mode": "end_to_end_mode",
                }
            }
        )  # pass configurations through both config_file and custom_config

        observation, info = env.reset()

        for _ in range(500):
            action = env.action_space.sample() # random action
            observation, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                observation, info = env.reset()

        env.close()

if __name__ == "__main__":
    unittest.main()