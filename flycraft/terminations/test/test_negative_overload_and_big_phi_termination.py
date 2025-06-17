import unittest
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from terminations.negative_overload_and_big_phi_termination import NegativeOverloadAndBigPhiTermination
from tasks.velocity_vector_control_task import VelocityVectorControlTask
from utils_common.load_config import load_config


class NegativeOverloadAndBigPhiTerminationTest(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        env_config = load_config(PROJECT_ROOT_DIR / "configs" / "NMR.json")

        self.negative_overload_and_big_phi_termination = NegativeOverloadAndBigPhiTermination(
            time_window=2, 
            negative_overload_threshold=0., 
            big_phi_threshold=60.,
            termination_reward=-1.,
            is_termination_reward_based_on_steps_left=False,
            env_config=env_config
        )

        self.state_var_type = VelocityVectorControlTask.get_state_vars()
    
    def test_1(self):
        """测试长度超过200, phi>60, nz<0 的情况
        """
        episode_length = 22
        phi=61.
        state_list = [self.state_var_type(phi=phi, theta=0., psi=0., v=200., mu=0., chi=30., p=0., h=0.) for i in range(episode_length)]

        self.negative_overload_and_big_phi_termination.reset()
        for i in range(episode_length-1):
            res = self.negative_overload_and_big_phi_termination.get_termination(
                state=state_list[i], 
                next_state=state_list[i+1],
                nz=-1.
            )
            if res[0]:
                break
        
        self.assertTrue(res[0])
        self.assertFalse(res[1])

        self.negative_overload_and_big_phi_termination.reset()
        for i in range(episode_length-1):
            res2 = self.negative_overload_and_big_phi_termination.get_termination_and_reward(
                state=state_list[i],
                next_state=state_list[i+1],
                nz=-1., 
                step_cnt=100
            )
            if res2[0]:
                break
        
        self.assertTrue(res2[0])
        self.assertFalse(res2[1])
        self.assertAlmostEqual(res2[2], -1.)
    
    def test_2(self):
        """测试长度超过200, phi<-60, nz<0 的情况
        """
        episode_length = 22
        phi=-61.
        state_list = [self.state_var_type(phi=phi, theta=0., psi=0., v=200., mu=0., chi=30., p=0., h=0.) for i in range(episode_length)]

        self.negative_overload_and_big_phi_termination.reset()
        for i in range(episode_length-1):
            res = self.negative_overload_and_big_phi_termination.get_termination(
                state=state_list[i], 
                next_state=state_list[i+1],
                nz=-1.
            )
            if res[0]:
                break
        
        self.assertTrue(res[0])
        self.assertFalse(res[1])

        self.negative_overload_and_big_phi_termination.reset()
        for i in range(episode_length-1):
            res2 = self.negative_overload_and_big_phi_termination.get_termination_and_reward(
                state=state_list[i], 
                next_state=state_list[i+1],
                nz=-1., 
                step_cnt=100
            )
            if res2[0]:
                break
        
        self.assertTrue(res2[0])
        self.assertFalse(res2[1])
        self.assertAlmostEqual(res2[2], -1.)

    def test_3(self):
        """测试长度不够的情况
        """
        episode_length = 20
        phi=-61.
        state_list = [self.state_var_type(phi=phi, theta=0., psi=0., v=200., mu=0., chi=30., p=0., h=0.) for i in range(episode_length)]

        self.negative_overload_and_big_phi_termination.reset()
        for i in range(episode_length-1):
            res = self.negative_overload_and_big_phi_termination.get_termination(
                state=state_list[i], 
                next_state=state_list[i+1],
                nz=-1.
            )
            if res[0]:
                break
        
        self.assertFalse(res[0])
        self.assertFalse(res[1])

        self.negative_overload_and_big_phi_termination.reset()
        for i in range(episode_length-1):
            res2 = self.negative_overload_and_big_phi_termination.get_termination_and_reward(
                state=state_list[i], 
                next_state=state_list[i+1],
                nz=-1.,
                step_cnt=100
            )
            if res2[0]:
                break
        
        self.assertFalse(res2[0])
        self.assertFalse(res2[1])
        self.assertAlmostEqual(res2[2], 0.)

    def test_4(self):
        """测试长度超过200,phi中间有一处小于60度的情况
        """
        episode_length = 22
        phi=61.
        state_list = [self.state_var_type(phi=(phi if i != int(episode_length/2) else 59), theta=0., psi=0., v=200., mu=0., chi=30., p=0., h=0.) for i in range(episode_length)]

        self.negative_overload_and_big_phi_termination.reset()
        for i in range(episode_length-1):
            res = self.negative_overload_and_big_phi_termination.get_termination(
                state=state_list[i], 
                next_state=state_list[i+1],
                nz=-1.
            )
            if res[0]:
                break
        
        self.assertFalse(res[0])
        self.assertFalse(res[1])

        self.negative_overload_and_big_phi_termination.reset()
        for i in range(episode_length-1):
            res2 = self.negative_overload_and_big_phi_termination.get_termination_and_reward(
                state=state_list[i], 
                next_state=state_list[i+1],
                nz=-1.,
                step_cnt=100
            )
            if res2[0]:
                break
        
        self.assertFalse(res2[0])
        self.assertFalse(res2[1])
        self.assertAlmostEqual(res2[2], 0.)

if __name__ == "__main__":
    unittest.main()