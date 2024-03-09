import unittest
from pathlib import Path
import sys

PROJECT_ROOT_DIR = Path(__file__).parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from env import FlyCraftEnv
from utils.sb3.my_wrappers import ScaledActionWrapper, ScaledObservationWrapper


class FlyCraftEnvWrapperTest(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.env = FlyCraftEnv(
            config_file=PROJECT_ROOT_DIR / "configs" / "MR_for_HER.json"
        )
        self.scaled_obs_env = ScaledObservationWrapper(self.env)
        self.scaled_act_obs_env = ScaledActionWrapper(self.scaled_obs_env)
    
    def test_obs_act_shape(self):
        scaled_obs_env_obs_shape = self.scaled_obs_env.observation_space
        scaled_obs_env_act_shape = self.scaled_obs_env.action_space
        print("Scaled obs env: ")
        print(scaled_obs_env_obs_shape)
        print(scaled_obs_env_act_shape)

        scaled_act_obs_env_obs_shape = self.scaled_act_obs_env.observation_space
        scaled_act_obs_env_act_shape = self.scaled_act_obs_env.action_space
        print("Scaled act and obs env: ")
        print(scaled_act_obs_env_obs_shape)
        print(scaled_act_obs_env_act_shape)
    
    def test_reset(self):
        scaled_obs_env_obs, scaled_obs_env_info = self.scaled_obs_env.reset()
        print("Scaled obs env: ")
        print(scaled_obs_env_obs)
        print(scaled_obs_env_info)

        scaled_act_obs_env_obs, scaled_act_obs_env_info = self.scaled_act_obs_env.reset()
        print("Scaled act and obs env: ")
        print(scaled_act_obs_env_obs)
        print(scaled_act_obs_env_info)
    
    def test_action_sample(self):
        for i in range(5):
            scaled_obs_env_act = self.scaled_obs_env.action_space.sample()
            print(scaled_obs_env_act)

            scaled_act_obs_env_act = self.scaled_act_obs_env.action_space.sample()
            print(scaled_act_obs_env_act)
    
    def test_step_1(self):
        print("In test step 1:")
        self.scaled_obs_env.reset()
        for i in range(10):
            action = self.scaled_obs_env.action_space.sample()
            next_obs, reward, terminated, truncated, info = self.scaled_obs_env.step(action)
            print(next_obs["observation"], action)

    def test_step_2(self):
        print("In test step 2:")
        self.scaled_act_obs_env.reset()
        for i in range(10):
            action = self.scaled_act_obs_env.action_space.sample()
            next_obs, reward, terminated, truncated, info = self.scaled_act_obs_env.step(action)
            print(next_obs["observation"], action)

if __name__ == "__main__":
    unittest.main()