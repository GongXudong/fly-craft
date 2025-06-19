import unittest
import numpy as np
from pathlib import Path

from flycraft.terminations.reach_target_termination2 import ReachTargetTermination2
from flycraft.tasks.velocity_vector_control_task import VelocityVectorControlTask
from flycraft.utils_common.load_config import load_config

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent


class ReachTargetTerminationTest(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        env_config = load_config(PROJECT_ROOT_DIR / "configs" / "NMR.json")

        self.reach_target_termination = ReachTargetTermination2(
            integral_time_length=1,
            v_threshold=10., 
            angle_threshold=1.,
            termination_reward=1.,
            env_config=env_config
        )
        self.state_var_type = VelocityVectorControlTask.get_state_vars()
    
    def test_1(self):
        """积分长度达到要求，精度也达到要求，返回True
        """
        episode_length = 11
        goal_v, goal_mu, goal_chi = 200., 20., 30.
        state_list = [self.state_var_type(phi=0., theta=0., psi=0., v=200., mu=19.1, chi=30., p=0., h=0.) for i in range(episode_length)]
        res = self.reach_target_termination.get_termination(
            state=state_list[-1], 
            goal_v=goal_v, 
            goal_mu=goal_mu,
            goal_chi=goal_chi,
            state_list=state_list
        )
        self.assertTrue(res[0])
        self.assertFalse(res[1])

        res2 = self.reach_target_termination.get_termination_and_reward(
            state=state_list[-1], 
            goal_v=goal_v, 
            goal_mu=goal_mu,
            goal_chi=goal_chi,
            state_list=state_list, 
            step_cnt=100
        )
        self.assertTrue(res2[0])
        self.assertFalse(res2[1])
        self.assertAlmostEqual(res2[2], 1.)

    def test_2(self):
        """积分长度达到要求，但速度大小精度没达到要求，返回False
        """
        episode_length = 11
        goal_v, goal_mu, goal_chi = 200., 20., 30.
        state_list = [self.state_var_type(phi=0., theta=0., psi=0., v=189., mu=20., chi=30., p=0., h=0.) for i in range(episode_length)]
        res = self.reach_target_termination.get_termination(
            state=state_list[-1], 
            goal_v=goal_v, 
            goal_mu=goal_mu,
            goal_chi=goal_chi,
            state_list=state_list
        )
        self.assertFalse(res[0])
        self.assertFalse(res[1])

        res2 = self.reach_target_termination.get_termination_and_reward(
            state=state_list[-1], 
            goal_v=goal_v, 
            goal_mu=goal_mu,
            goal_chi=goal_chi,
            state_list=state_list, 
            step_cnt=100
        )
        self.assertFalse(res2[0])
        self.assertFalse(res2[1])
        self.assertAlmostEqual(res2[2], 0.)

    def test_3(self):
        """积分长度达到要求，但速度方向精度没达到要求，返回False
        """
        episode_length = 11
        goal_v, goal_mu, goal_chi = 200., 20., 30.
        state_list = [self.state_var_type(phi=0., theta=0., psi=0., v=200., mu=21.1, chi=30., p=0., h=0.) for i in range(episode_length)]
        res = self.reach_target_termination.get_termination(
            state=state_list[-1], 
            goal_v=goal_v, 
            goal_mu=goal_mu,
            goal_chi=goal_chi,
            state_list=state_list
        )
        self.assertFalse(res[0])
        self.assertFalse(res[1])

        res2 = self.reach_target_termination.get_termination_and_reward(
            state=state_list[-1], 
            goal_v=goal_v, 
            goal_mu=goal_mu,
            goal_chi=goal_chi,
            state_list=state_list, 
            step_cnt=100
        )
        self.assertFalse(res2[0])
        self.assertFalse(res2[1])
        self.assertAlmostEqual(res2[2], 0.)

    def test_4(self):
        """积分长度没达到要求，返回False
        """
        episode_length = 9
        goal_v, goal_mu, goal_chi = 200., 20., 30.
        state_list = [self.state_var_type(phi=0., theta=0., psi=0., v=200., mu=20., chi=30., p=0., h=0.) for i in range(episode_length)]
        res = self.reach_target_termination.get_termination(
            state=state_list[-1], 
            goal_v=goal_v, 
            goal_mu=goal_mu,
            goal_chi=goal_chi,
            state_list=state_list
        )
        self.assertFalse(res[0])
        self.assertFalse(res[1])

        res2 = self.reach_target_termination.get_termination_and_reward(
            state=state_list[-1], 
            goal_v=goal_v, 
            goal_mu=goal_mu,
            goal_chi=goal_chi,
            state_list=state_list, 
            step_cnt=100
        )
        self.assertFalse(res2[0])
        self.assertFalse(res2[1])
        self.assertAlmostEqual(res2[2], 0.)


if __name__ == "__main__":
    unittest.main()