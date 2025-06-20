import unittest
import numpy as np
from pathlib import Path

from flycraft.terminations.crash_termination import CrashTermination
from flycraft.tasks.velocity_vector_control_task import VelocityVectorControlTask
from flycraft.utils_common.load_config import load_config

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent


class CrashTerminationTest(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        env_config = load_config(PROJECT_ROOT_DIR / "configs" / "NMR.json")

        self.crash_termination = CrashTermination(
            h0=0., 
            termination_reward=-1.,
            is_termination_reward_based_on_steps_left=False,
            env_config=env_config
        )
        self.state_var_type = VelocityVectorControlTask.get_state_vars()

    def test_1(self):
        tested_h = -1.
        state = self.state_var_type(phi=0., theta=0., psi=0., v=200., mu=20., chi=30., p=0., h=1000.)
        next_state = self.state_var_type(phi=0., theta=0., psi=0., v=200., mu=20., chi=30., p=0., h=tested_h)
        res = self.crash_termination.get_termination(
            state=state,
            next_state=next_state,
        )
        self.assertTrue(res[0])
        self.assertTrue(res[1])

        res2 = self.crash_termination.get_termination_and_reward(
            state=state,
            next_state=next_state,
            step_cnt=100,
        )
        self.assertTrue(res2[0])
        self.assertTrue(res2[1])
        self.assertAlmostEqual(res2[2], -1.)

    def test_2(self):
        tested_h = 1.
        state = self.state_var_type(phi=0., theta=0., psi=0., v=200., mu=20., chi=30., p=0., h=1000.)
        next_state = self.state_var_type(phi=0., theta=0., psi=0., v=200., mu=20., chi=30., p=0., h=tested_h)
        res = self.crash_termination.get_termination(
            state=state,
            next_state=next_state,    
        )
        self.assertFalse(res[0])
        self.assertFalse(res[1])

        res2 = self.crash_termination.get_termination_and_reward(
            state=state,
            next_state=next_state,
            step_cnt=100,
        )
        self.assertFalse(res2[0])
        self.assertFalse(res2[1])
        self.assertAlmostEqual(res2[2], 0.)


if __name__ == "__main__":
    unittest.main()