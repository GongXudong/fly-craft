import unittest
import numpy as np
from pathlib import Path

from flycraft.terminations.extreme_state_termination import ExtremeStateTermination
from flycraft.tasks.velocity_vector_control_task import VelocityVectorControlTask
from flycraft.utils_common.load_config import load_config

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent


class CrashTerminationTest(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        env_config = load_config(PROJECT_ROOT_DIR / "configs" / "NMR.json")

        self.extreme_state_termination = ExtremeStateTermination(
            v_max=1000., 
            p_max=500.,
            termination_reward=-1.,
            is_termination_reward_based_on_steps_left=False,
            env_config=env_config
        )
        self.state_var_type = VelocityVectorControlTask.get_state_vars()

    def test_1(self):
        tested_v = 1100.
        state = self.state_var_type(phi=0., theta=0., psi=0., v=200., mu=20., chi=30., p=0., h=5000.)
        next_state = self.state_var_type(phi=0., theta=0., psi=0., v=tested_v, mu=20., chi=30., p=0., h=5000.)
        res = self.extreme_state_termination.get_termination(
            state=state,
            next_state=next_state,
        )
        self.assertTrue(res[0])
        self.assertTrue(res[1])

        res2 = self.extreme_state_termination.get_termination_and_reward(
            state=state,
            next_state=next_state,
            step_cnt=100,
        )
        self.assertTrue(res2[0])
        self.assertTrue(res2[1])
        self.assertAlmostEqual(res2[2], -1.)

    def test_2(self):
        tested_v = 999.
        state = self.state_var_type(phi=0., theta=0., psi=0., v=200., mu=20., chi=30., p=0., h=5000.)
        next_state = self.state_var_type(phi=0., theta=0., psi=0., v=tested_v, mu=20., chi=30., p=0., h=5000.)
        res = self.extreme_state_termination.get_termination(
            state=state,
            next_state=next_state,    
        )
        self.assertFalse(res[0])
        self.assertFalse(res[1])

        res2 = self.extreme_state_termination.get_termination_and_reward(
            state=state,
            next_state=next_state,
            step_cnt=100,
        )
        self.assertFalse(res2[0])
        self.assertFalse(res2[1])
        self.assertAlmostEqual(res2[2], 0.)

    def test_3(self):
        tested_p = -410.
        state = self.state_var_type(phi=0., theta=0., psi=0., v=200., mu=20., chi=30., p=200., h=5000.)
        next_state = self.state_var_type(phi=0., theta=0., psi=0., v=200., mu=20., chi=30., p=tested_p, h=5000.)
        res = self.extreme_state_termination.get_termination(
            state=state,
            next_state=next_state,
        )
        self.assertFalse(res[0])
        self.assertFalse(res[1])

        res2 = self.extreme_state_termination.get_termination_and_reward(
            state=state,
            next_state=next_state,
            step_cnt=100,
        )
        self.assertFalse(res2[0])
        self.assertFalse(res2[1])
        self.assertAlmostEqual(res2[2], 0.)
    
    def test_4(self):
        tested_p = -510.
        state = self.state_var_type(phi=0., theta=0., psi=0., v=200., mu=20., chi=30., p=-200., h=5000.)
        next_state = self.state_var_type(phi=0., theta=0., psi=0., v=200., mu=20., chi=30., p=tested_p, h=5000.)
        res = self.extreme_state_termination.get_termination(
            state=state,
            next_state=next_state,
        )
        self.assertTrue(res[0])
        self.assertTrue(res[1])

        res2 = self.extreme_state_termination.get_termination_and_reward(
            state=state,
            next_state=next_state,
            step_cnt=100,
        )
        self.assertTrue(res2[0])
        self.assertTrue(res2[1])
        self.assertAlmostEqual(res2[2], -1.)

    def test_5(self):
        tested_p = 510.
        state = self.state_var_type(phi=0., theta=0., psi=0., v=200., mu=20., chi=30., p=200., h=5000.)
        next_state = self.state_var_type(phi=0., theta=0., psi=0., v=200., mu=20., chi=30., p=tested_p, h=5000.)
        res = self.extreme_state_termination.get_termination(
            state=state,
            next_state=next_state,    
        )
        self.assertTrue(res[0])
        self.assertTrue(res[1])

        res2 = self.extreme_state_termination.get_termination_and_reward(
            state=state,
            next_state=next_state,
            step_cnt=100,
        )
        self.assertTrue(res2[0])
        self.assertTrue(res2[1])
        self.assertAlmostEqual(res2[2], -1.)


if __name__ == "__main__":
    unittest.main()