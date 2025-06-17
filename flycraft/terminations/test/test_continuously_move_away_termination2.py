import unittest
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from terminations.continuousely_move_away_termination2 import ContinuouselyMoveAwayTermination2
from tasks.velocity_vector_control_task import VelocityVectorControlTask
from utils_common.load_config import load_config


class ContinuouselyMoveAwayTermination2Test(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        env_config = load_config(PROJECT_ROOT_DIR / "configs" / "NMR.json")

        self.continuousely_move_away_termination = ContinuouselyMoveAwayTermination2(
            time_window=2,
            termination_reward=-1.,
            is_termination_reward_based_on_steps_left=False,
            env_config=env_config
        )

        self.state_var_type = VelocityVectorControlTask.get_state_vars()

    def test_1(self):
        """速度矢量误差持续增大
        """
        tmp_mu = 30.
        episode_length = 22
        goal_v, goal_mu, goal_chi = 110., 10., -135.
        state_list = [self.state_var_type(phi=0., theta=0., psi=0., v=110., mu=(tmp_mu:=tmp_mu + np.random.rand()*0.1), chi=-135., p=0., h=0.) for i in range(episode_length)]

        self.continuousely_move_away_termination.reset()
        # 正确使用方式，在整个轨迹上依次调用！！！！！
        for i in range(episode_length-1):
            tmp_state_list = state_list[:i+2]
            res = self.continuousely_move_away_termination.get_termination(
                state=tmp_state_list[-2],
                next_state=tmp_state_list[-1],
                goal_v=goal_v,
                goal_mu=goal_mu,
                goal_chi=goal_chi,
            )
            # print(self.continuousely_move_away_termination.mu_continuously_increasing_num)
            if res[0] or res[1]:
                break
        
        self.assertTrue(res[0])
        self.assertFalse(res[1])

        self.continuousely_move_away_termination.reset()
        for i in range(episode_length-1):
            tmp_state_list = state_list[:i+2]
            res2 = self.continuousely_move_away_termination.get_termination_and_reward(
                state=tmp_state_list[-2],
                next_state=tmp_state_list[-1], 
                goal_v=goal_v,
                goal_mu=goal_mu,
                goal_chi=goal_chi,
                state_list=tmp_state_list,
                step_cnt=100,
            )
            # print(self.continuousely_move_away_termination.mu_continuously_increasing_num)
            if res2[0] or res2[1]:
                break

        self.assertTrue(res2[0])
        self.assertFalse(res2[1])
        self.assertAlmostEqual(res2[2], -1.)

    def test_has_not_reach_length_1(self):
        episode_length = 20
        goal_v, goal_mu, goal_chi = 200., 20., 30.
        state_list = [self.state_var_type(phi=0., theta=0., psi=0., v=200., mu=20., chi=40., p=0., h=0.) for i in range(episode_length)]
        v_list = np.random.random((episode_length, 3))

        self.continuousely_move_away_termination.reset()
        for i in range(episode_length-1):
            tmp_state_list = state_list[:i+2]
            res = self.continuousely_move_away_termination.get_termination(
                state=tmp_state_list[-2],
                next_state=tmp_state_list[-1],
                goal_v=goal_v,
                goal_mu=goal_mu,
                goal_chi=goal_chi,
                state_list=tmp_state_list,
                ve=v_list[i][0],
                vn=v_list[i][1],
                vh=v_list[i][2] 
            )
            if res[0] or res[1]:
                break
        
        self.assertFalse(res[0])
        self.assertFalse(res[1])

        self.continuousely_move_away_termination.reset()
        for i in range(episode_length-1):
            tmp_state_list = state_list[:i+2]
            res2 = self.continuousely_move_away_termination.get_termination_and_reward(
                state=tmp_state_list[-2],
                next_state=tmp_state_list[-1],
                goal_v=goal_v,
                goal_mu=goal_mu,
                goal_chi=goal_chi,
                state_list=tmp_state_list,
                ve=v_list[i][0],
                vn=v_list[i][1],
                vh=v_list[i][2],
                step_cnt=100,
            )
            if res2[0] or res2[1]:
                break
        
        self.assertFalse(res2[0])
        self.assertFalse(res2[1])
        self.assertAlmostEqual(res2[2], 0.)


if __name__ == "__main__":
    unittest.main()