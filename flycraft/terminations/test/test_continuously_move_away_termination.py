import unittest
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from terminations.continuousely_move_away_termination import ContinuouselyMoveAwayTermination
from flycraft.tasks.velocity_vector_control_task import VelocityVectorControlTask
from utils.load_config import load_config


class ContinuouselyMoveAwayTerminationTest(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        env_config = load_config(PROJECT_ROOT_DIR / "configs" / "NMR.json")

        self.continuousely_move_away_termination = ContinuouselyMoveAwayTermination(
            time_window=2,
            termination_reward=-1.,
            is_termination_reward_based_on_steps_left=False,
            env_config=env_config
        )

        self.state_var_type = VelocityVectorControlTask.get_state_vars()

    def test_mu_1(self):
        """测试mu和chi持续增大的情况
        """
        tmp_mu = 30.
        tmp_chi = 40.
        episode_length = 210
        goal_v, goal_mu, goal_chi = 200., 20., 30.
        state_list = [self.state_var_type(phi=0., theta=0., psi=0., v=200., mu=(tmp_mu:=tmp_mu + np.random.rand()*0.1), chi=(tmp_chi:=tmp_chi + np.random.rand()*0.1), p=0., h=0.) for i in range(episode_length)]
        
        # for item in state_list:
        #     print(item.mu)

        self.continuousely_move_away_termination.reset()
        # 正确使用方式，在整个轨迹上依次调用！！！！！
        for i in range(episode_length):
            tmp_state_list = state_list[:i+1]
            res = self.continuousely_move_away_termination.get_termination(
                state=tmp_state_list[-1], 
                goal_v=goal_v, 
                goal_mu=goal_mu,
                goal_chi=goal_chi,
                state_list=tmp_state_list
            )
            # print(self.continuousely_move_away_termination.mu_continuously_increasing_num)
            if res[0] or res[1]:
                break
        
        self.assertTrue(res[0])
        self.assertFalse(res[1])

        self.continuousely_move_away_termination.reset()
        for i in range(episode_length):
            tmp_state_list = state_list[:i+1]
            res2 = self.continuousely_move_away_termination.get_termination_and_reward(
                state=tmp_state_list[-1], 
                goal_v=goal_v,
                goal_mu=goal_mu,
                goal_chi=goal_chi,
                state_list=tmp_state_list, 
                step_cnt=100
            )
            # print(self.continuousely_move_away_termination.mu_continuously_increasing_num)
            if res2[0] or res2[1]:
                break
        
        self.assertTrue(res2[0])
        self.assertFalse(res2[1])
        self.assertAlmostEqual(res2[2], -1.)
    
    def test_mu_2(self):
        """测试mu先增大，中间一拍减小，后面又持续增大的情况
        """
        tmp_mu = 30.
        episode_length = 210
        goal_v, goal_mu, goal_chi = 200., 20., 30.
        state_list = [self.state_var_type(phi=0., theta=0., psi=0., v=200., mu=(tmp_mu:=tmp_mu + np.random.rand()*(0.1 if i != episode_length/2 else -0.1)), chi=30., p=0., h=0.) for i in range(episode_length)]
        
        # for item in state_list:
        #     print(item.mu)

        self.continuousely_move_away_termination.reset()
        # 正确使用方式，在整个轨迹上依次调用！！！！！
        for i in range(episode_length):
            tmp_state_list = state_list[:i+1]
            res = self.continuousely_move_away_termination.get_termination(
                state=tmp_state_list[-1], 
                goal_v=goal_v,
                goal_mu=goal_mu,
                goal_chi=goal_chi,
                state_list=tmp_state_list
            )
            # print(self.continuousely_move_away_termination.mu_continuously_increasing_num)
            if res[0] or res[1]:
                break
        
        self.assertFalse(res[0])
        self.assertFalse(res[1])

        self.continuousely_move_away_termination.reset()
        for i in range(episode_length):
            tmp_state_list = state_list[:i+1]
            res2 = self.continuousely_move_away_termination.get_termination_and_reward(
                state=tmp_state_list[-1], 
                goal_v=goal_v,
                goal_mu=goal_mu,
                goal_chi=goal_chi,
                state_list=tmp_state_list, 
                step_cnt=100
            )
            # print(self.continuousely_move_away_termination.mu_continuously_increasing_num)
            if res2[0] or res2[1]:
                break
        
        self.assertFalse(res2[0])
        self.assertFalse(res2[1])
        self.assertAlmostEqual(res2[2], 0.)

    def test_chi_1(self):
        """测试mu和chi持续增大的情况
        """
        tmp_mu = 30.
        tmp_chi = 40.
        episode_length = 210
        goal_v, goal_mu, goal_chi = 200., 20., 30.
        state_list = [self.state_var_type(phi=0., theta=0., psi=0., v=200., mu=(tmp_mu:=tmp_mu + np.random.rand()*0.1), chi=(tmp_chi:=tmp_chi + np.random.rand()*0.1), p=0., h=0.) for i in range(episode_length)]
        
        # for item in state_list:
        #     print(item.chi)

        self.continuousely_move_away_termination.reset()
        for i in range(episode_length):
            tmp_state_list = state_list[:i+1]
            res = self.continuousely_move_away_termination.get_termination(
                state=tmp_state_list[-1], 
                goal_v=goal_v,
                goal_mu=goal_mu,
                goal_chi=goal_chi,
                state_list=tmp_state_list
            )
            if res[0] or res[1]:
                break
        
        self.assertTrue(res[0])
        self.assertFalse(res[1])

        self.continuousely_move_away_termination.reset()
        for i in range(episode_length):
            tmp_state_list = state_list[:i+1]
            res2 = self.continuousely_move_away_termination.get_termination_and_reward(
                state=tmp_state_list[-1], 
                goal_v=goal_v,
                goal_mu=goal_mu,
                goal_chi=goal_chi,
                state_list=tmp_state_list, 
                step_cnt=100
            )
            if res2[0] or res2[1]:
                break
        
        self.assertTrue(res2[0])
        self.assertFalse(res2[1])
        self.assertAlmostEqual(res2[2], -1.)

    def test_chi_2(self):
        """测试chi先增大，中间一拍减小，后面又持续增大的情况
        """
        tmp_chi = 40.
        episode_length = 210
        goal_v, goal_mu, goal_chi = 200., 20., 30.
        state_list = [self.state_var_type(phi=0., theta=0., psi=0., v=200., mu=20., chi=(tmp_chi:=tmp_chi + np.random.rand()*(0.1 if i != episode_length/2 else -0.1)), p=0., h=0.) for i in range(episode_length)]
        
        # for item in state_list:
        #     print(item.chi)

        self.continuousely_move_away_termination.reset()
        for i in range(episode_length):
            tmp_state_list = state_list[:i+1]
            res = self.continuousely_move_away_termination.get_termination(
                state=tmp_state_list[-1], 
                goal_v=goal_v,
                goal_mu=goal_mu,
                goal_chi=goal_chi,
                state_list=tmp_state_list
            )
            if res[0] or res[1]:
                break
        
        self.assertFalse(res[0])
        self.assertFalse(res[1])

        self.continuousely_move_away_termination.reset()
        for i in range(episode_length):
            tmp_state_list = state_list[:i+1]
            res2 = self.continuousely_move_away_termination.get_termination_and_reward(
                state=tmp_state_list[-1], 
                goal_v=goal_v,
                goal_mu=goal_mu,
                goal_chi=goal_chi,
                state_list=tmp_state_list, 
                step_cnt=100
            )
            if res2[0] or res2[1]:
                break
        
        self.assertFalse(res2[0])
        self.assertFalse(res2[1])
        self.assertAlmostEqual(res2[2], 0.)

    def test_has_not_reach_length_1(self):
        episode_length = 190
        goal_v, goal_mu, goal_chi = 200., 20., 30.
        state_list = [self.state_var_type(phi=0., theta=0., psi=0., v=200., mu=20., chi=40., p=0., h=0.) for i in range(episode_length)]

        self.continuousely_move_away_termination.reset()
        for i in range(episode_length):
            tmp_state_list = state_list[:i+1]
            res = self.continuousely_move_away_termination.get_termination(
                state=tmp_state_list[-1], 
                goal_v=goal_v,
                goal_mu=goal_mu,
                goal_chi=goal_chi,
                state_list=tmp_state_list
            )
            if res[0] or res[1]:
                break
        
        self.assertFalse(res[0])
        self.assertFalse(res[1])

        self.continuousely_move_away_termination.reset()
        for i in range(episode_length):
            tmp_state_list = state_list[:i+1]
            res2 = self.continuousely_move_away_termination.get_termination_and_reward(
                state=tmp_state_list[-1], 
                goal_v=goal_v,
                goal_mu=goal_mu,
                goal_chi=goal_chi,
                state_list=tmp_state_list, 
                step_cnt=100
            )
            if res2[0] or res2[1]:
                break
        
        self.assertFalse(res2[0])
        self.assertFalse(res2[1])
        self.assertAlmostEqual(res2[2], 0.)


if __name__ == "__main__":
    unittest.main()