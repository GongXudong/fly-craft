import unittest
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from terminations.continuousely_move_away_termination2 import ContinuouselyMoveAwayTermination2
from tasks.attitude_control_task import AttitudeControlTask
from utils.load_config import load_config


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

        self.state_var_type = AttitudeControlTask.get_state_vars()

    def test_1(self):
        """速度矢量误差持续增大
        """
        tmp_mu = 30.
        episode_length = 21
        goal_v, goal_mu, goal_chi = 110., 10., -135.
        state_list = [self.state_var_type(phi=0., theta=0., psi=0., v=200., mu=(tmp_mu:=tmp_mu + np.random.rand()*0.1), chi=30., p=0., h=0.) for i in range(episode_length)]
        
        v_list = [
            [-97.89154866386451,-105.78853707128893,25.39202006327604],
            [-97.66971150347698,-105.63508726111036,25.32152198925065],
            [-97.44689373748864,-105.48130621599795,25.25380986414457],
            [-97.22316447223449,-105.327210643237,25.1885888876563],
            [-96.9985962121573,-105.17282140158395,25.12556090064589],
            [-96.77326202778633,-105.01816161783964,25.064434953477686],
            [-96.54723324032682,-104.86325505794515,25.004935519718902],
            [-96.32057756179637,-104.7081248870883,24.946808238187998],
            [-96.09335778706586,-104.55279270727479,24.8898235977733],
            [-95.86563103164045,-104.39727787321713,24.833778516453172],
            [-95.63744827548416,-104.24159718337772,24.77849655867769],
            [-95.40885443227141,-104.08576469852095,24.72382651629833],
            [-95.17988857182064,-103.92979185500882,24.66964034781024],
            [-94.95058444161141,-103.77368764155642,24.6158303878762],
            [-94.72097118078737,-103.61745880291542,24.562306374961526],
            [-94.49107383142409,-103.46111034101628,24.508992420760997],
            [-94.2609141237048,-103.30464575326336,24.455824303776296],
            [-94.03051114196255,-103.14806738405423,24.402746721728697],
            [-93.79988189476984,-102.99137673164317,24.349711217256854],
            [-93.5690418165318,-102.83457470278881,24.296674389768512],
            [-93.33800518152856,-102.67766180605986,24.24359653447312],
            [-93.1067854654811,-102.5206382429419,24.1904407174394]
        ]

        self.continuousely_move_away_termination.reset()
        # 正确使用方式，在整个轨迹上依次调用！！！！！
        for i in range(episode_length):
            tmp_state_list = state_list[:i+1]
            res = self.continuousely_move_away_termination.get_termination(
                state=tmp_state_list[-1], 
                goal_v=goal_v,
                goal_mu=goal_mu,
                goal_chi=goal_chi,
                state_list=tmp_state_list, 
                ve=v_list[i][0],
                vn=v_list[i][1],
                vh=v_list[i][2]
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
                ve=v_list[i][0],
                vn=v_list[i][1],
                vh=v_list[i][2],
                step_cnt=100,
            )
            # print(self.continuousely_move_away_termination.mu_continuously_increasing_num)
            if res2[0] or res2[1]:
                break
        
        self.assertTrue(res2[0])
        self.assertFalse(res2[1])
        self.assertAlmostEqual(res2[2], -1.)

    def test_has_not_reach_length_1(self):
        episode_length = 19
        goal_v, goal_mu, goal_chi = 200., 20., 30.
        state_list = [self.state_var_type(phi=0., theta=0., psi=0., v=200., mu=20., chi=40., p=0., h=0.) for i in range(episode_length)]
        v_list = np.random.random((episode_length, 3))

        self.continuousely_move_away_termination.reset()
        for i in range(episode_length):
            tmp_state_list = state_list[:i+1]
            res = self.continuousely_move_away_termination.get_termination(
                state=tmp_state_list[-1], 
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
        for i in range(episode_length):
            tmp_state_list = state_list[:i+1]
            res2 = self.continuousely_move_away_termination.get_termination_and_reward(
                state=tmp_state_list[-1], 
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