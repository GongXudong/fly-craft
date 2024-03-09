import unittest
import numpy as np
import sys
from pathlib import Path
from math import sqrt

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from utils.attitude_angle_calc_utils import roll2xy_on_unit_circle, RollDirection, get_roll_direction, get_roll_deg


class AttitudeAngleCalcUtilsTest(unittest.TestCase):

    def float_tuple_equal(self, t1, t2):
        self.assertAlmostEqual(t1[0], t2[0])
        self.assertAlmostEqual(t1[1], t2[1])

    def test_roll2xy_1(self):
        roll = 45
        res = roll2xy_on_unit_circle(roll)
        self.float_tuple_equal(res, (sqrt(2.)/2, -sqrt(2.)/2))
    
    def test_roll2xy_2(self):
        roll = -45
        res = roll2xy_on_unit_circle(roll)
        self.float_tuple_equal(res, (sqrt(2.)/2, sqrt(2.)/2))

    def test_roll2xy_3(self):
        roll = -135
        res = roll2xy_on_unit_circle(roll)
        self.float_tuple_equal(res, (-sqrt(2.)/2, sqrt(2.)/2))
    
    def test_roll2xy_4(self):
        roll = 135
        res = roll2xy_on_unit_circle(roll)
        self.float_tuple_equal(res, (-sqrt(2.)/2, -sqrt(2.)/2))
    
    def test_roll2xy_5(self):
        roll = 150
        res = roll2xy_on_unit_circle(roll)
        self.float_tuple_equal(res, (-sqrt(3.)/2, -1./2))

    def test_get_roll_direction_1(self):
        roll_1, roll_2 = -45, 45
        res = get_roll_direction(roll_1, roll_2)
        self.assertEqual(res, RollDirection.RIGHT)

    def test_get_roll_direction_2(self):
        roll_1, roll_2 = 5, -8
        res = get_roll_direction(roll_1, roll_2)
        self.assertEqual(res, RollDirection.LEFT)
    
    def test_get_roll_direction_3(self):
        roll_1, roll_2 = -178, 170
        res = get_roll_direction(roll_1, roll_2)
        self.assertEqual(res, RollDirection.LEFT)

    def test_get_roll_direction_4(self):
        roll_1, roll_2 = 175, -168
        res = get_roll_direction(roll_1, roll_2)
        self.assertEqual(res, RollDirection.RIGHT)

    def test_get_roll_direction_5(self):
        roll_1, roll_2 = 115, 141
        res = get_roll_direction(roll_1, roll_2)
        self.assertEqual(res, RollDirection.RIGHT)
    
    def test_get_roll_direction_6(self):
        roll_1, roll_2 = -15, -28
        res = get_roll_direction(roll_1, roll_2)
        self.assertEqual(res, RollDirection.LEFT)

    def test_get_roll_direction_7(self):
        roll_1, roll_2 = 170, -180
        res = get_roll_direction(roll_1, roll_2)
        self.assertEqual(res, RollDirection.RIGHT)
    
    def test_get_roll_direction_8(self):
        roll_1, roll_2 = 170, -15
        res = get_roll_direction(roll_1, roll_2)
        self.assertEqual(res, RollDirection.RIGHT)
    
    def test_get_roll_direction_9(self):
        roll_1, roll_2 = 170, -5
        res = get_roll_direction(roll_1, roll_2)
        self.assertEqual(res, RollDirection.LEFT)

    def test_get_roll_deg_1(self):
        roll_1, roll_2 = 170, -180
        res = get_roll_deg(roll_1, roll_2)
        self.assertAlmostEqual(res, 10.)

    def test_get_roll_deg_2(self):
        roll_1, roll_2 = 170, -15
        res = get_roll_deg(roll_1, roll_2)
        self.assertAlmostEqual(res, 175)
    
    def test_get_roll_deg_3(self):
        roll_1, roll_2 = 170, -5
        res = get_roll_deg(roll_1, roll_2)
        self.assertAlmostEqual(res, 175)

if __name__ == "__main__":
    unittest.main()