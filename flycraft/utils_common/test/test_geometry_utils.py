import unittest
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from utils_common import geometry_utils
from functools import partial

my_all_close = partial(np.allclose, atol=1.e-8)


class Test(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()

    def test_v_mu_chi_2_enh_1(self):
        self.assertTrue(
            my_all_close(geometry_utils.v_mu_chi_2_enh(20, 30, 30), [5*np.sqrt(3), 15., 10.])
        )

    def test_v_mu_chi_2_enh_2(self):
        self.assertTrue(
            my_all_close(geometry_utils.v_mu_chi_2_enh(20, 45, 45), [10., 10., 10*np.sqrt(2)])
        )

    def test_angle_of_2_3d_vectors_1(self):
        # 只有v不一样
        self.assertTrue(
            my_all_close(
                [geometry_utils.angle_of_2_velocity(v_1=210, mu_1=3, chi_1=10, v_2=220, mu_2=3, chi_2=10)],
                [0.]
            )
        )
    
    def test_angle_of_2_3d_vectors_2(self):
        # chi=0，且只有mu不一样
        # print(geometry_utils.angle_of_2_velocity(v_1=210, mu_1=3, chi_1=0, v_2=210, mu_2=5, chi_2=0))
        self.assertTrue(
            my_all_close(
                [geometry_utils.angle_of_2_velocity(v_1=210, mu_1=3, chi_1=0, v_2=210, mu_2=5, chi_2=0)],
                [2.]
            )
        )
    
    def test_angle_of_2_3d_vectors_3(self):
        # mu=0，且只有chi不一样
        # print(geometry_utils.angle_of_2_velocity(v_1=210, mu_1=0, chi_1=22, v_2=210, mu_2=0, chi_2=20))
        self.assertTrue(
            my_all_close(
                [geometry_utils.angle_of_2_velocity(v_1=210, mu_1=0, chi_1=22, v_2=210, mu_2=0, chi_2=20)],
                [2.]
            )
        )

    def test_angle_of_2_3d_vectors_4(self):
        # 只有真空速不一样的情况，角度误差应恒为0
        for i in range(10000):
            v_1 = np.random.random() * 100 + 200
            v_2 = np.random.random() * 100 + 200
            mu = np.random.random() * 85
            chi = np.random.random() * 180
            
            self.assertTrue(
                np.allclose(
                    [geometry_utils.angle_of_2_velocity(v_1=v_1, mu_1=mu, chi_1=chi, v_2=v_2, mu_2=mu, chi_2=chi)],
                    [0.],
                    atol=1e-5
                ),
            )
        


if __name__ == "__main__":
    unittest.main()