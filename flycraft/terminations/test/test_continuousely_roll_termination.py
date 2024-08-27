import unittest
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from terminations.continuousely_roll_termination import ContinuouselyRollTermination
from flycraft.tasks.velocity_vector_control_task import VelocityVectorControlTask
from utils.load_config import load_config

class ContinuouselyRollTerminationTest(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        env_config = load_config(PROJECT_ROOT_DIR / "configs" / "NMR.json")

        self.continuousely_roll_termination = ContinuouselyRollTermination(
            continuousely_roll_threshold=720., 
            termination_reward=-1.,
            is_termination_reward_based_on_steps_left=False,
            env_config=env_config
        )

        self.state_var_type = VelocityVectorControlTask.get_state_vars()

    def test_1(self):
        phi_arr = list(range(0, 180, 10)) + list(range(-180, 0, 10)) + list(range(0, 180, 10)) + list(range(-180, 0, 10)) + list(range(0, 180, 10))
        print(len(phi_arr))
        state_list = [self.state_var_type(phi=phi, theta=0., psi=0., v=200., mu=0., chi=30., p=0., h=0.) for phi in phi_arr]

        self.continuousely_roll_termination.reset()
        for i in range(len(state_list) - 1):
            state = state_list[i]
            next_state = state_list[i+1]
            res = self.continuousely_roll_termination.get_termination(state=state, next_state=next_state)
            if res[0]:
                break
        
        self.assertTrue(res[0])
        self.assertFalse(res[1])
        
        self.continuousely_roll_termination.reset()
        for i in range(len(state_list) - 1):
            state = state_list[i]
            next_state = state_list[i+1]
            res2 = self.continuousely_roll_termination.get_termination_and_reward(state=state, next_state=next_state, step_cnt=100)
            if res2[0]:
                break
        
        self.assertTrue(res2[0])
        self.assertFalse(res2[1])
        self.assertAlmostEqual(res2[2], -1.)
        

    def test_2(self):
        phi_arr = list(range(0, 180, 10)) + list(range(-180, 0, 10)) + list(range(0, 180, 10)) + [160] + list(range(-180, 0, 10)) + list(range(0, 180, 10))

        # for i in phi_arr:
        #     print(i)
        
        state_list = [self.state_var_type(phi=phi, theta=0., psi=0., v=200., mu=0., chi=30., p=0., h=0.) for phi in phi_arr]

        self.continuousely_roll_termination.reset()
        for i in range(len(state_list) - 1):
            state = state_list[i]
            next_state = state_list[i+1]
            res = self.continuousely_roll_termination.get_termination(state=state, next_state=next_state)
            if res[0]:
                break
        
        self.assertFalse(res[0])
        self.assertFalse(res[1])
        
        self.continuousely_roll_termination.reset()
        for i in range(len(state_list) - 1):
            state = state_list[i]
            next_state = state_list[i+1]
            res2 = self.continuousely_roll_termination.get_termination_and_reward(state=state, next_state=next_state, step_cnt=100)
            if res2[0]:
                break
        
        self.assertFalse(res2[0])
        self.assertFalse(res2[1])
        self.assertAlmostEqual(res2[2], 0.)


if __name__ == "__main__":
    unittest.main()