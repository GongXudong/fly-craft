import unittest
import numpy as np
from pathlib import Path
import sys
import math

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from planes.f16_plane import F16Plane
from utils_common.load_config import load_config


class GoalSamplerTest(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.env_config = load_config(PROJECT_ROOT_DIR / "configs" / "NMR.json")
        self.plane = F16Plane(self.env_config)
        self.plane.reset()
        
    def test_1(self):
        """reset
        """
        self.plane.reset()
    
    def test_2(self):
        """step with fixed action
        """
        action = [0.2, 0.3, 1]

        for i in range(100):
            state = self.plane.step(action)
            print(state)

if __name__ == "__main__":
    unittest.main()