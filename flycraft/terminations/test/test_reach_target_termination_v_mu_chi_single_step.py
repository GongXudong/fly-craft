import unittest
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from terminations.reach_target_termination_v_mu_chi_single_step import ReachTargetTerminationVMuChiSingleStep
from tasks.velocity_vector_control_task import VelocityVectorControlTask
from utils.load_config import load_config


class ReachTargetTerminationSingleStepTest(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        env_config = load_config(PROJECT_ROOT_DIR / "configs" / "NMR.json")

        self.reach_target_termination = ReachTargetTerminationVMuChiSingleStep(
            v_threshold=10., 
            mu_threshold=1.,
            chi_threshold=1.,
            termination_reward=1.,
            env_config=env_config
        )
        self.state_var_type = VelocityVectorControlTask.get_state_vars()
    
    def test_1(self):
        """精度达到要求，返回True
        """
        episode_length = 11
        goal_v, goal_mu, goal_chi = 200., 20., 30.
        state_list = [self.state_var_type(phi=0., theta=0., psi=0., v=200., mu=19.1, chi=30., p=0., h=0.) for i in range(episode_length)]
        res = self.reach_target_termination.get_termination(
            state=state_list[-2],
            next_state=state_list[-1],
            goal_v=goal_v,
            goal_mu=goal_mu,
            goal_chi=goal_chi,
        )
        self.assertTrue(res[0])
        self.assertFalse(res[1])

        res2 = self.reach_target_termination.get_termination_and_reward(
            state=state_list[-2],
            next_state=state_list[-1],
            goal_v=goal_v,
            goal_mu=goal_mu,
            goal_chi=goal_chi,
            step_cnt=100,
        )
        self.assertTrue(res2[0])
        self.assertFalse(res2[1])
        self.assertAlmostEqual(res2[2], 1.)

    def test_2(self):
        """精度达到要求，返回True
        """
        episode_length = 11
        goal_v, goal_mu, goal_chi = 200., 20., 30.
        state_list = [self.state_var_type(phi=0., theta=0., psi=0., v=200., mu=19.1, chi=30.9, p=0., h=0.) for i in range(episode_length)]
        res = self.reach_target_termination.get_termination(
            state=state_list[-2],
            next_state=state_list[-1],
            goal_v=goal_v,
            goal_mu=goal_mu,
            goal_chi=goal_chi,
        )
        self.assertTrue(res[0])
        self.assertFalse(res[1])

        res2 = self.reach_target_termination.get_termination_and_reward(
            state=state_list[-2],
            next_state=state_list[-1],
            goal_v=goal_v,
            goal_mu=goal_mu,
            goal_chi=goal_chi,
            step_cnt=100,
        )
        self.assertTrue(res2[0])
        self.assertFalse(res2[1])
        self.assertAlmostEqual(res2[2], 1.)

    def test_3(self):
        """速度大小精度没达到要求，返回False
        """
        episode_length = 11
        goal_v, goal_mu, goal_chi = 200., 20., 30.
        state_list = [self.state_var_type(phi=0., theta=0., psi=0., v=189., mu=20., chi=30., p=0., h=0.) for i in range(episode_length)]
        res = self.reach_target_termination.get_termination(
            state=state_list[-2],
            next_state=state_list[-1],
            goal_v=goal_v,
            goal_mu=goal_mu,
            goal_chi=goal_chi,
        )
        self.assertFalse(res[0])
        self.assertFalse(res[1])

        res2 = self.reach_target_termination.get_termination_and_reward(
            state=state_list[-2],
            next_state=state_list[-1],
            goal_v=goal_v,
            goal_mu=goal_mu,
            goal_chi=goal_chi,
            step_cnt=100,
        )
        self.assertFalse(res2[0])
        self.assertFalse(res2[1])
        self.assertAlmostEqual(res2[2], 0.)

    def test_4(self):
        """速度方向精度没达到要求，返回False
        """
        episode_length = 11
        goal_v, goal_mu, goal_chi = 200., 20., 30.
        state_list = [self.state_var_type(phi=0., theta=0., psi=0., v=200., mu=21.1, chi=30., p=0., h=0.) for i in range(episode_length)]
        res = self.reach_target_termination.get_termination(
            state=state_list[-2],
            next_state=state_list[-1],
            goal_v=goal_v,
            goal_mu=goal_mu,
            goal_chi=goal_chi,
        )
        self.assertFalse(res[0])
        self.assertFalse(res[1])

        res2 = self.reach_target_termination.get_termination_and_reward(
            state=state_list[-2],
            next_state=state_list[-1], 
            goal_v=goal_v,
            goal_mu=goal_mu,
            goal_chi=goal_chi,
            step_cnt=100,
        )
        self.assertFalse(res2[0])
        self.assertFalse(res2[1])
        self.assertAlmostEqual(res2[2], 0.)

if __name__ == "__main__":
    unittest.main()