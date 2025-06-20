import unittest
import numpy as np

from flycraft.rewards.sparse_reward2 import SparseReward2
from flycraft.tasks.velocity_vector_control_task import VelocityVectorControlTask


class ReachTargetReardTest(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.reach_target_reward = SparseReward2(
            step_frequence=100, integral_time_length=1, 
            v_threshold=10., mu_threshold=1., chi_threshold=1.,
            reach_target_reward=1., else_reward=0.
        )
        
        self.state_var_type = VelocityVectorControlTask.get_state_vars()
        
    def test_1(self):
        """积分长度达到要求，精度也达到要求，返回1.
        """
        goal_v, goal_mu, goal_chi = 200., 20., 30.
        state_list = [self.state_var_type(phi=0., theta=0., psi=0., v=200., mu=20., chi=30., p=0., h=0.) for i in range(110)]
        res = self.reach_target_reward.get_reward(
            state_list[-1], 
            state_list=state_list,
            goal_v=goal_v,
            goal_mu=goal_mu,
            goal_chi=goal_chi    
        )
        self.assertAlmostEqual(res, 1.)

    def test_2(self):
        """积分长度达到要求，但精度没达到要求，返回0.
        """
        goal_v, goal_mu, goal_chi = 200., 20., 30.
        state_list = [self.state_var_type(phi=0., theta=0., psi=0., v=189., mu=20., chi=30., p=0., h=0.) for i in range(110)]
        res = self.reach_target_reward.get_reward(
            state_list[-1], 
            state_list=state_list,
            goal_v=goal_v,
            goal_mu=goal_mu,
            goal_chi=goal_chi
        )
        self.assertAlmostEqual(res, 0.)
    
    def test_3(self):
        """精度达到要求，但积分长度没达到要求，返回0.
        """
        goal_v, goal_mu, goal_chi = 200., 20., 30.
        state_list = [self.state_var_type(phi=0., theta=0., psi=0., v=200., mu=20., chi=30., p=0., h=0.) for i in range(90)]
        res = self.reach_target_reward.get_reward(
            state_list[-1], 
            state_list=state_list,
            goal_v=goal_v,
            goal_mu=goal_mu,
            goal_chi=goal_chi
        )
        self.assertAlmostEqual(res, 0.)


if __name__ == "__main__":
    unittest.main()