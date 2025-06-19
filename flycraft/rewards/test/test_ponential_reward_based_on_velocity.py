import unittest
import numpy as np

from flycraft.rewards.ponential_reward_based_on_velocity import PonentialRewardBasedOnVelocity
from flycraft.tasks.velocity_vector_control_task import VelocityVectorControlTask


class PonentialRewardBasedOnAngleTest(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()

        state_min = VelocityVectorControlTask.get_state_lower_bounds()
        state_max = VelocityVectorControlTask.get_state_higher_bounds()
        self.state_var_type = VelocityVectorControlTask.get_state_vars()
        self.ponential_reward = PonentialRewardBasedOnVelocity(
            b=1.,
            # gamma=0.9999,
            gamma=0.99,
            scale=100.
        )

    def test_1(self):
        goal_v, goal_mu, goal_chi = 200., 20., 30.
        state = self.state_var_type(phi=0., theta=0., psi=0., v=170, mu=10., chi=20., p=0., h=0.)
        next_state = self.state_var_type(phi=0., theta=0., psi=0., v=180, mu=15., chi=25., p=0., h=0.)
        reward = self.ponential_reward.get_reward(
            state=state, 
            next_state=next_state, 
            done=False,
            goal_v=goal_v
        )

        reward_test = - self.ponential_reward.gamma * (goal_v - next_state.v) / self.ponential_reward.scale + (goal_v - state.v) / self.ponential_reward.scale

        self.assertAlmostEqual(reward, reward_test) 


if __name__ == "__main__":
    unittest.main()