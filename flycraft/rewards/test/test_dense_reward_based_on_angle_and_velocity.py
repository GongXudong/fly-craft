import unittest
import numpy as np
from pathlib import Path
import sys
import math
from copy import deepcopy

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from rewards.dense_reward_based_on_angle_and_velocity import DenseRewardBasedOnAngleAndVelocity
from tasks.velocity_vector_control_task import VelocityVectorControlTask
from utils.geometry_utils import angle_of_2_3d_vectors, v_mu_chi_2_enh


class DenseRewardBasedOnAngleAndVelocityTest(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.dense_reward = DenseRewardBasedOnAngleAndVelocity(
            b=1.0, 
            angle_scale=180., 
            velocity_scale=100.
        )
        
        self.state_var_type = VelocityVectorControlTask.get_state_vars()
        
    def test_1(self):
        """速度方向和大小均达到目标.
        """
        goal_v, goal_mu, goal_chi = 200, 20, 30
        state = self.state_var_type(phi=0., theta=0., psi=0., v=200., mu=20., chi=30., p=0., h=0.)
        next_state = self.state_var_type(phi=0., theta=0., psi=0., v=200., mu=20., chi=30., p=0., h=0.)
        res = self.dense_reward.get_reward(
            state, 
            next_state=next_state, 
            goal_v=goal_v,
            goal_mu=goal_mu,
            goal_chi=goal_chi
        )
        print("In test 1: ", res)
        self.assertAlmostEqual(res, 0.)

    def test_2(self):
        """速度大小达到目标，方向没达到目标.
        """
        goal_v, goal_mu, goal_chi = 200, 30, 60
        state = self.state_var_type(phi=0., theta=0., psi=0., v=200., mu=20., chi=30., p=0., h=0.)
        next_state = state = self.state_var_type(phi=0., theta=0., psi=0., v=200., mu=20., chi=30., p=0., h=0.)
        res = self.dense_reward.get_reward(
            state, 
            next_state=next_state, 
            goal_v=goal_v,
            goal_mu=goal_mu,
            goal_chi=goal_chi
        )
        angle = angle_of_2_3d_vectors(v_mu_chi_2_enh(200, 20, 30), v_mu_chi_2_enh(goal_v, goal_mu, goal_chi))
        self.assertAlmostEqual(res, (-angle/180. + 0.) / 2)
    
    def test_3(self):
        """速度方向达到目标，大小没达到目标
        """
        goal_v, goal_mu, goal_chi = 200, 0, 0
        state = self.state_var_type(phi=0., theta=0., psi=0., v=200., mu=0., chi=0., p=0., h=0.)
        next_state = self.state_var_type(phi=0., theta=0., psi=0., v=180., mu=0., chi=0., p=0., h=0.)
        res = self.dense_reward.get_reward(
            state, 
            next_state=next_state, 
            goal_v=goal_v,
            goal_mu=goal_mu,
            goal_chi=goal_chi    
        )
        self.assertAlmostEqual(res, -0.1)
    
    def test_4(self):
        """速度方向达到目标，大小没达到目标，且计算速度大小奖励时触发clip
        """
        goal_v, goal_mu, goal_chi = 200, 0, 0
        state = self.state_var_type(phi=0., theta=0., psi=0., v=200., mu=0., chi=0., p=0., h=0.)
        next_state = self.state_var_type(phi=0., theta=0., psi=0., v=80., mu=0., chi=0., p=0., h=0.)
        res = self.dense_reward.get_reward(
            state, 
            next_state=next_state, 
            goal_v=goal_v,
            goal_mu=goal_mu,
            goal_chi=goal_chi
        )
        self.assertAlmostEqual(res, -0.5)

    def test_5(self):
        """速度方向和大小均没达到目标
        """
        goal_v, goal_mu, goal_chi = 200, 30, 60
        state = self.state_var_type(phi=0., theta=0., psi=0., v=200., mu=20., chi=30., p=0., h=0.)
        next_state = self.state_var_type(phi=0., theta=0., psi=0., v=180., mu=20., chi=30., p=0., h=0.)
        res = self.dense_reward.get_reward(
            state, 
            next_state=next_state, 
            goal_v=goal_v,
            goal_mu=goal_mu,
            goal_chi=goal_chi
        )
        angle = angle_of_2_3d_vectors(v_mu_chi_2_enh(200, 20, 30), v_mu_chi_2_enh(goal_v,goal_mu, goal_chi))
        self.assertAlmostEqual(res, (-angle/180. - 0.2) / 2)

    def test_6(self):
        reward = -0.3627691419451945

        desired_goal = [201.18216, 27.027822, -64.05127]
        achieved_goal = [199.85579, 5.0925595e-16, 0.0]

        next_state = self.state_var_type(
            phi=0., theta=0., psi=0., 
            v=achieved_goal[0], 
            mu=achieved_goal[1], 
            chi=achieved_goal[2], 
            p=0., h=0.
        )

        res = self.dense_reward.get_reward(
            deepcopy(next_state), 
            next_state=next_state, 
            goal_v=desired_goal[0],
            goal_mu=desired_goal[1],
            goal_chi=desired_goal[2]
        )

        print(f"In test 6: {res}")


    # def test_3(self):
    #     """精度达到要求，但积分长度没达到要求，返回0.
    #     """
    #     state_list = [self.state_var_type(phi=0., theta=0., psi=0., v=200., mu=20., chi=30., p=0., h=0., target_v=200., target_mu=20., target_chi=30.) for i in range(90)]
    #     res = self.reach_target_reward.get_reward(state_list[-1], state_list=state_list)
    #     self.assertAlmostEqual(res, 0.)


if __name__ == "__main__":
    unittest.main()