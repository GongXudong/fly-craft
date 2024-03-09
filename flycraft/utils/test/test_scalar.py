import unittest
import numpy as np
import sys
from pathlib import Path
from collections import namedtuple

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from utils.scalar import get_min_max_scalar
from tasks.attitude_control_task import AttitudeControlTask
from planes.f16_plane import F16Plane

from functools import partial

my_all_close = partial(np.allclose, atol=1.e-8)


class Test(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.scalar = get_min_max_scalar(
            feature_range=[0., 1.],
           mins=np.array([-10., -5.]),
           maxs=np.array([10., 5.])
        )

    def test_transform_1(self):
        data = np.array([5., 2.])
        res = self.scalar.transform(data.reshape((1, -1)))
        self.assertTrue(np.allclose(res, np.array([[0.75, 0.7]]), atol=1.e-8))

    def test_inverse_transform_1(self):
        data = np.array([0.75, 0.7])
        res = self.scalar.inverse_transform(data.reshape((1, -1)))
        self.assertTrue(np.allclose(res, np.array([[5., 2.]]), atol=1.e-8))

    def test_bound_1(self):
        data = np.array([5., 6.])
        res = self.scalar.transform(data.reshape((1, -1)))
        self.assertTrue(np.allclose(res, np.array([[0.75, 1.]]), atol=1.e-8))

    def test_bound_2(self):
        data = np.array([-15., 6.])
        res = self.scalar.transform(data.reshape((1, -1)))
        self.assertTrue(np.allclose(res, np.array([[0, 1.]]), atol=1.e-8))


class Test_Action_Data(unittest.TestCase):
    
    def setUp(self) -> None:
        super().setUp()
        self.scalar = get_min_max_scalar(
            mins=F16Plane.get_action_lower_bounds(),
            maxs=F16Plane.get_action_higher_bounds(),
            feature_range=[0., 1.]
        )
    
    def test_1(self):
        data = np.array([0., 1., 1.])
        res = self.scalar.inverse_transform(data.reshape((1, -1)))
        self.assertTrue(np.allclose(res, np.array([[-180., 9., 1.]]), atol=1.e-8))
    
    def test_2(self):
        data = np.array([0.5, 0., 0.5])
        res = self.scalar.inverse_transform(data.reshape((1, -1)))
        self.assertTrue(np.allclose(res, np.array([[0., -4., 0.5]]), atol=1.e-8))


class Test_Observation_Data(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.scalar = get_min_max_scalar(
            mins=AttitudeControlTask.get_state_lower_bounds(),
            maxs=AttitudeControlTask.get_state_higher_bounds(),
            feature_range=[0., 1.]
        )
    
    def test_in_bound_1(self):
        state_vars = AttitudeControlTask.get_state_vars()
        data = np.array(state_vars(phi=-90., theta=-90., psi=-180., v=0., mu=-90., chi=-180., p=-300., h=0.))
        res = self.scalar.transform(data.reshape((1, -1)))
        self.assertTrue(np.allclose(res, np.array([0.25] + [0.]*7).reshape((1, -1)), atol=1.e-8))

    def test_on_bound_1(self):
        state_vars = AttitudeControlTask.get_state_vars()
        data = np.array(state_vars(phi=-180., theta=-90., psi=-180., v=0., mu=-90., chi=-180., p=-300., h=0.))
        res = self.scalar.transform(data.reshape((1, -1)))
        self.assertTrue(np.allclose(res, np.array([0.]*8).reshape((1, -1)), atol=1.e-8))
    
    def test_exceed_bound_1(self):
        state_vars = AttitudeControlTask.get_state_vars()
        data = np.array(state_vars(phi=-190., theta=-91., psi=-180., v=0., mu=-90., chi=-180., p=-300., h=0.))
        res = self.scalar.transform(data.reshape((1, -1)))
        self.assertTrue(np.allclose(res, np.array([0.]*8), atol=1.e-8))


if __name__ == "__main__":
    unittest.main()