from math import fabs, cos, sin, radians, sqrt, acos, degrees
import numpy as np
from enum import Enum


ROLL_THRESHOLD = 1e-2

class RollDirection(Enum):
    LEFT = 1
    RIGHT = 2
    NOROLL = 3


def roll2xy_on_unit_circle(roll):
    """将滚转角转换成单位圆上的坐标

    Args:
        roll (_type_): 单位度，[-180, 180]
    """

    return cos(radians(roll)), -sin(radians(roll))


def get_roll_direction(roll_1: float, roll_2: float, roll_threshold: float=ROLL_THRESHOLD):
    """判断roll_1到roll_2，飞机的滚转方向（按小于180度的角算转向！！！）.

    思路：

    平面直角坐标系：x正向表示滚转角0，y轴正向表示滚转角-90，y轴负向表示滚转角90，x轴负向上方-180，下方180

    将滚转角表示成半径为1的圆上的点，roll_1 -> (x1, y1)，roll_2 -> (x2, y2)

    通过计算(x1, y1)与(x2, y2)的叉积判断旋转方向（右手定则）

    注意：

    该方法计算的值按小于180度的角旋转计算的，例如：170右旋转到-5，旋转了185度。此方法是按170左旋175到-5，所以会返回左滚转！！！

    Args:
        roll_1 (float): deg, [-180, 180]
        roll_2 (float): deg, [-180, 180]
    """
    
    if fabs(roll_1 - roll_2) < roll_threshold:
        return RollDirection.NOROLL
    
    x1, y1 = roll2xy_on_unit_circle(roll_1)
    x2, y2 = roll2xy_on_unit_circle(roll_2)

    cross_product = x1 * y2 - y1 * x2

    if cross_product > 0.:
        return RollDirection.LEFT
    else:
        return RollDirection.RIGHT 


def get_roll_deg(roll_1: float, roll_2: float):
    """计算roll_1到roll_2滚转的角度（按小于180度的角算转向！！！）

    Args:
        roll_1 (float): _description_
        roll_2 (float): _description_

    Returns:
        _type_: _description_
    """
    x1, y1 = roll2xy_on_unit_circle(roll_1)
    x2, y2 = roll2xy_on_unit_circle(roll_2)

    inner_product = x1 * x2 + y1 * y2

    tmp = inner_product / (sqrt(x1 * x1 + y1 * y1) * sqrt(x2 * x2 + y2 * y2))
    tmp = np.clip(tmp, a_min=-1., a_max=1.)
    
    res_rad = acos(tmp)

    return degrees(res_rad)