import unittest
import numpy as np
from pathlib import Path
import sys
import math
from copy import deepcopy

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from rewards.dense_reward import DenseReward
from tasks.velocity_vector_control_task import VelocityVectorControlTask
from utils.geometry_utils import angle_of_2_3d_vectors, v_mu_chi_2_enh


class DenseReardTest(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.dense_reward = DenseReward(b=1.)
        
        self.state_var_type = VelocityVectorControlTask.get_state_vars()
        
    def test_1(self):
        """积分长度达到要求，精度也达到要求，返回1.
        """
        goal_v, goal_mu, goal_chi = 200., 20., 30.
        state = self.state_var_type(phi=0., theta=0., psi=0., v=200., mu=20., chi=30., p=0., h=0.)
        res = self.dense_reward.get_reward(
            state, 
            goal_v=goal_v,
            goal_mu=goal_mu,
            goal_chi=goal_chi,
            next_state=deepcopy(state)
        )
        self.assertAlmostEqual(res, 0.)

    def test_2(self):
        """积分长度达到要求，但精度没达到要求，返回0.
        """
        goal_v, goal_mu, goal_chi = 200., 30., 60.
        state = self.state_var_type(phi=0., theta=0., psi=0., v=200., mu=20., chi=30., p=0., h=0.)
        res = self.dense_reward.get_reward(
            state, 
            goal_v=goal_v,
            goal_mu=goal_mu,
            goal_chi=goal_chi,
            next_state=deepcopy(state)
        )
        angle = angle_of_2_3d_vectors(v_mu_chi_2_enh(state.v, state.mu, state.chi), v_mu_chi_2_enh(goal_v, goal_mu, goal_chi))
        self.assertAlmostEqual(res, -angle/180.)
    
    # def test_3(self):
    #     """精度达到要求，但积分长度没达到要求，返回0.
    #     """
    #     state_list = [self.state_var_type(phi=0., theta=0., psi=0., v=200., mu=20., chi=30., p=0., h=0., target_v=200., target_mu=20., target_chi=30.) for i in range(90)]
    #     res = self.reach_target_reward.get_reward(state_list[-1], state_list=state_list)
    #     self.assertAlmostEqual(res, 0.)


if __name__ == "__main__":
    unittest.main()