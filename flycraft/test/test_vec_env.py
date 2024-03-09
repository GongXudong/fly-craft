import unittest
from pathlib import Path
import sys

PROJECT_ROOT_DIR = Path(__file__).parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from env import FlyCraftEnv
from flycraft.utils.sb3.vec_env_helper import get_vec_env
from utils.sb3.my_wrappers import ScaledActionWrapper, ScaledObservationWrapper


class FlyCraftEnvVecEnvTest(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.env_n = 4
        self.vec_env = get_vec_env(
            num_process=self.env_n, 
            config_file=PROJECT_ROOT_DIR / "configs" / "MR_for_HER.json"
        )
    
    def test_space_shape(self):
        obs_shape = self.vec_env.observation_space
        act_shape = self.vec_env.action_space
        print("test observation and action space shape: ")
        print(obs_shape)
        print(act_shape)
    
    def test_reset(self):
        obss = self.vec_env.reset()
        print("test reset: ")
        print(obss)
    
    def test_action_sample(self):
        print("test action sample: ")
        for i in range(5):
            acts = self.vec_env.action_space.sample()
            print(acts)
    
    def test_step(self):
        print("test step 1:")
        self.vec_env.reset()
        for i in range(10):
            action = [self.vec_env.action_space.sample() for i in range(self.env_n)]
            next_obs, reward, terminated, info = self.vec_env.step(action)
            print(next_obs["observation"], action)

if __name__ == "__main__":
    unittest.main()