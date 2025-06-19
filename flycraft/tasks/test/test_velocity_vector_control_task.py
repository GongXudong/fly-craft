import unittest
import numpy as np
from pathlib import Path

from flycraft.planes.f16_plane import F16Plane
from flycraft.tasks.velocity_vector_control_task import VelocityVectorControlTask
from flycraft.tasks.goal_samplers.goal_sampler_for_velocity_vector_control import GoalSampler
from flycraft.utils_common.load_config import load_config
from flycraft.utils_common.dict_utils import update_nested_dict

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent


class AttitudeControlTaskTest(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.config: dict = load_config(PROJECT_ROOT_DIR / "configs" / "MR_for_HER.json")

        self.plane = F16Plane(self.config)
        self.task = VelocityVectorControlTask(
            plane=self.plane,
            env_config=self.config,
            my_logger=None
        )
        
    def test_1(self):
        """check reward funcs and termination funcs
        """
        self.assertEqual(len(self.task.reward_funcs), 1)
        self.assertEqual(len(self.task.termination_funcs), 6)

        print(str(self.task.reward_funcs[0]))
        for t in self.task.termination_funcs:
            print(str(t))
        
    def test_2(self):
        """reset
        """
        print("\nTest reset:")
        for i in range(3):
            self.task.reset()
            print(self.task.goal)
    
    def test_3(self):
        """get obs
        """
        print("\nTest get obs: ")
        self.plane.reset()
        print(self.task.get_obs())

    def test_is_success(self):
        """_summary_
        """
        pass

    def test_compute_reward_1(self):
        """_summary_
        """
        achieved_goal = np.array([
            [200, 0, 0],
            [250, 10, 30],
            [130, 45, 160]
        ])
        desired_goal = np.array([
            [200, 0, 0],
            [250, 10, 30],
            [130, 45, 160]
        ])
        info = [{}, {}, {}]

        res = self.task.compute_reward(achieved_goal=achieved_goal, desired_goal=desired_goal, info=info)
        res_ref = [0., 0., 0.]
        for a, b in zip(res, res_ref):
            self.assertAlmostEqual(a, b)

    def test_compute_reward_2(self):
        achieved_goal = np.array([
            [200, 0, 0],
            [250, 10, 30],
            [130, 45, 160]
        ])
        desired_goal = np.array([
            [200, 10, 10],
            [250, 13, 25],
            [133, 42, 163]
        ])
        info = [{}, {}, {}]

        res = self.task.compute_reward(achieved_goal=achieved_goal, desired_goal=desired_goal, info=info)
        print(res)
        # res_ref = [0., 0., 0.]
        # for a, b in zip(res, res_ref):
        #     self.assertAlmostEqual(a, b)

if __name__ == "__main__":
    unittest.main()