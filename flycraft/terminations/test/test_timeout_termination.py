import unittest
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from terminations.timeout_termination import TimeoutTermination
from tasks.velocity_vector_control_task import VelocityVectorControlTask
from utils_common.load_config import load_config


class TimeoutTerminationTest(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        env_config = load_config(PROJECT_ROOT_DIR / "configs" / "NMR.json")

        self.timeout_termination = TimeoutTermination(
            termination_reward=-1.,
            env_config=env_config
        )
        self.state_var_type = VelocityVectorControlTask.get_state_vars()

    def test_1(self):
        tested_step_cnt = 410
        state = self.state_var_type(phi=0., theta=0., psi=0., v=200., mu=20., chi=30., p=0., h=0.)
        res = self.timeout_termination.get_termination(state=state, step_cnt=tested_step_cnt)
        self.assertTrue(res[0])
        self.assertTrue(res[1])

        res2 = self.timeout_termination.get_termination_and_reward(state=state, step_cnt=tested_step_cnt)
        self.assertTrue(res2[0])
        self.assertTrue(res2[1])
        self.assertAlmostEqual(res2[2], -1.)
    
    def test_2(self):
        tested_step_cnt = 500
        state = self.state_var_type(phi=0., theta=0., psi=0., v=200., mu=20., chi=30., p=0., h=0.)
        res = self.timeout_termination.get_termination(state=state, step_cnt=tested_step_cnt)
        self.assertTrue(res[0])
        self.assertTrue(res[1])

        res2 = self.timeout_termination.get_termination_and_reward(state=state, step_cnt=tested_step_cnt)
        self.assertTrue(res2[0])
        self.assertTrue(res2[1])
        self.assertAlmostEqual(res2[2], -1.)
    
    def test_3(self):
        tested_step_cnt = 398
        state = self.state_var_type(phi=0., theta=0., psi=0., v=200., mu=20., chi=30., p=0., h=0.)
        res = self.timeout_termination.get_termination(state=state, step_cnt=tested_step_cnt)
        self.assertFalse(res[0])
        self.assertFalse(res[1])

        res2 = self.timeout_termination.get_termination_and_reward(state=state, step_cnt=tested_step_cnt)
        self.assertFalse(res2[0])
        self.assertFalse(res2[1])
        self.assertAlmostEqual(res2[2], 0.)

if __name__ == "__main__":
    unittest.main()