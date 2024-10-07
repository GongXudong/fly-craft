import unittest
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from rewards.ponential_reward import PonentialReward1, PonentialReward2, ScaledPonentialReward2
from tasks.velocity_vector_control_task import VelocityVectorControlTask
from utils.load_config import load_config


class PonentialBasedRewardTest(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()

        state_min = VelocityVectorControlTask.get_state_lower_bounds()
        state_max = VelocityVectorControlTask.get_state_higher_bounds()
        self.state_var_type = VelocityVectorControlTask.get_state_vars()
        self.ponential_reward_1 = PonentialReward1(
            # gamma=0.9999,
            gamma=0.99,
            v_min=state_min.v, v_max=state_max.v,
            mu_min=state_min.mu, mu_max=state_max.mu,
            chi_min=state_min.chi, chi_max=state_max.chi
        )
        self.ponential_reward_2 = PonentialReward2(
            # gamma=0.9999
            gamma=0.99,
            coef_k_for_v=10.,
            coef_k_for_mu=5.,
            coef_k_for_chi=5.,
        )
        self.scaled_ponential_reward_2 = ScaledPonentialReward2(
            scale_coef=100.,
            gamma=0.99,
            coef_k_for_v=10.,
            coef_k_for_mu=5.,
            coef_k_for_chi=5.,
        )
    
    def test_p1_1(self):
        """test PonentialReward1
        """
        goal_v, goal_mu, goal_chi = 200., 20., 30.
        state = self.state_var_type(phi=0., theta=0., psi=0., v=170, mu=10., chi=20., p=0., h=0., )
        next_state = self.state_var_type(phi=0., theta=0., psi=0., v=180, mu=15., chi=25., p=0., h=0.)
        reward = self.ponential_reward_1.get_reward(
            state=state, 
            next_state=next_state, 
            done=False,
            goal_v=goal_v,
            goal_mu=goal_mu,
            goal_chi=goal_chi,
        )
        log = self.ponential_reward_1.reward_trajectory[-1]
        print("ponential reward 1: ", log)
    
    def test_p2_1(self):
        """test PonentialReward2
        """
        goal_v, goal_mu, goal_chi = 200., 20., 30.
        state = self.state_var_type(phi=0., theta=0., psi=0., v=170, mu=10., chi=20., p=0., h=0.)
        next_state = self.state_var_type(phi=0., theta=0., psi=0., v=180, mu=15., chi=25., p=0., h=0.)
        reward = self.ponential_reward_2.get_reward(
            state=state, 
            next_state=next_state, 
            done=False,
            goal_v=goal_v,
            goal_mu=goal_mu,
            goal_chi=goal_chi
        )
        log = self.ponential_reward_2.reward_trajectory[-1]
        self.assertAlmostEqual(
            log['reward_v'], 
            self.ponential_reward_2.gamma * (1. - 2./3.) - (1. - 3./4.)
        )
        self.assertAlmostEqual(
            log['reward_mu'], 
            self.ponential_reward_2.gamma * (1. - 1./2.) - (1. - 2./3.)
        )
        self.assertAlmostEqual(
            log['reward_chi'],
            self.ponential_reward_2.gamma * (1. - 1./2.) - (1. - 2./3.)
        )
        print(reward)

    def test_p2_2(self):
        """test PonentialReward2
        """
        goal_v, goal_mu, goal_chi = 200., 20., 30.
        state = self.state_var_type(phi=0., theta=0., psi=0., v=170, mu=10., chi=20., p=0., h=0.)
        next_state = self.state_var_type(phi=0., theta=0., psi=0., v=170.1, mu=10.5, chi=20.5, p=0., h=0.)
        reward = self.ponential_reward_2.get_reward(
            state=state, 
            next_state=next_state, 
            done=False,
            goal_v=goal_v,
            goal_mu=goal_mu,
            goal_chi=goal_chi
        )
        log = self.ponential_reward_2.reward_trajectory[-1]
        print(log)
    
    def test_p2_3(self):
        """test PonentialReward2
        """
        goal_v, goal_mu, goal_chi = 200., 20., 30.
        state = self.state_var_type(phi=0., theta=0., psi=0., v=190, mu=19., chi=28., p=0., h=0.)
        next_state = self.state_var_type(phi=0., theta=0., psi=0., v=190.1, mu=19.5, chi=28.5, p=0., h=0.)
        reward = self.ponential_reward_2.get_reward(
            state=state, 
            next_state=next_state, 
            done=False,
            goal_v=goal_v,
            goal_mu=goal_mu,
            goal_chi=goal_chi
        )
        log = self.ponential_reward_2.reward_trajectory[-1]
        print(log)

    def test_scaled_p2_1(self):
        """test ScaledPonentialReward2, 数值参考test_p2_2（放大了1e6倍）
        """
        goal_v, goal_mu, goal_chi = 200., 20., 30.
        state = self.state_var_type(phi=0., theta=0., psi=0., v=170, mu=10., chi=20., p=0., h=0.)
        next_state = self.state_var_type(phi=0., theta=0., psi=0., v=170.1, mu=10.5, chi=20.5, p=0., h=0.)
        reward = self.scaled_ponential_reward_2.get_reward(
            state=state, 
            next_state=next_state, 
            done=False,
            goal_v=goal_v,
            goal_mu=goal_mu,
            goal_chi=goal_chi
        )
        log = self.scaled_ponential_reward_2.reward_trajectory[-1]
        print("scaled ponential2 1, ", log)

    def test_scaled_p2_2(self):
        """test ScaledPonentialReward2, 数值参考test_p2_3（放大了1e6倍）
        """
        goal_v, goal_mu, goal_chi = 200., 20., 30.
        state = self.state_var_type(phi=0., theta=0., psi=0., v=190, mu=19., chi=28., p=0., h=0.)
        next_state = self.state_var_type(phi=0., theta=0., psi=0., v=190.1, mu=19.5, chi=28.5, p=0., h=0.)
        reward = self.scaled_ponential_reward_2.get_reward(
            state=state, 
            next_state=next_state, 
            done=False,
            goal_v=goal_v,
            goal_mu=goal_mu,
            goal_chi=goal_chi
        )
        log = self.scaled_ponential_reward_2.reward_trajectory[-1]
        print("scaled ponential2 2, ", log)

    def test_scaled_p2_3(self):
        """test ScaledPonentialReward2, 改变了coef，数值与test_scaled_p2_2对比
        """
        self.scaled_ponential_reward_2 = ScaledPonentialReward2(
            scale_coef=100.,
            gamma=0.99,
            coef_k_for_v=50.,
            coef_k_for_mu=20.,
            coef_k_for_chi=20.,
        )
        goal_v, goal_mu, goal_chi = 200., 20., 30.
        state = self.state_var_type(phi=0., theta=0., psi=0., v=190, mu=19., chi=28., p=0., h=0.)
        next_state = self.state_var_type(phi=0., theta=0., psi=0., v=190.1, mu=19.5, chi=28.5, p=0., h=0.)
        reward = self.scaled_ponential_reward_2.get_reward(
            state=state, 
            next_state=next_state, 
            done=False,
            goal_v=goal_v,
            goal_mu=goal_mu,
            goal_chi=goal_chi
        )
        log = self.scaled_ponential_reward_2.reward_trajectory[-1]
        print("scaled ponential2 3, ", log)

if __name__ == "__main__":
    unittest.main()