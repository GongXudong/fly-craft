import unittest
import numpy as np
import sys
from pathlib import Path
import logging

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from utils.my_log import get_logger


class Test(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.logger: logging.Logger = get_logger()
    
    def test_1(self):
        self.logger.debug('debug信息')
        self.logger.info('info,一般的信息输出')
        self.logger.warning('waring，用来用来打印警告信息')
        self.logger.error('error，一般用来打印一些错误信息')
        self.logger.critical('critical，用来打印一些致命的错误信息，等级最高')


if __name__ == "__main__":
    unittest.main()