import numpy as np
from typing import Tuple

def angle_of_2_3d_vectors(v1, v2):
    x = np.array(v1)
    y = np.array(v2)

    # 分别计算两个向量的模：
    module_x = np.sqrt(x.dot(x))
    module_y = np.sqrt(y.dot(y))

    assert not np.any(np.allclose([module_x], [0.], atol=1e-8)), "v1的模为0！"
    assert not np.any(np.allclose([module_y], [0.], atol=1e-8)), "v2的模为0！"

    # 计算两个向量的点积
    dot_value = x.dot(y)

    # 计算夹角的cos值：
    cos_theta = dot_value / (module_x * module_y)
    # cos_theta在[-1, 1]之间
    cos_theta = np.clip(cos_theta, a_min=-1.0, a_max=1.0)

    # 求得夹角（弧度制）：
    angle_radian = np.arccos(cos_theta)

    # 转换为角度值：
    # angle_value = angle_radian * 180. / np.pi 
    angle_value = np.rad2deg(angle_radian)
    return angle_value

def angle_of_2_velocity(v_1: float, mu_1: float, chi_1: float, v_2: float, mu_2: float, chi_2: float):
    """计算两个速度矢量之间的夹角，两个速度矢量都由(v, mu, chi)描述。
    """
    current_velocity_vector = [
        v_1 * np.cos(np.deg2rad(mu_1)) * np.sin(np.deg2rad(chi_1)), 
        v_1 * np.cos(np.deg2rad(mu_1)) * np.cos(np.deg2rad(chi_1)),
        v_1 * np.sin(np.deg2rad(mu_1)),
    ]
    target_velocity_vector = [
        v_2 * np.cos(np.deg2rad(mu_2)) * np.sin(np.deg2rad(chi_2)), 
        v_2 * np.cos(np.deg2rad(mu_2)) * np.cos(np.deg2rad(chi_2)),
        v_2 * np.sin(np.deg2rad(mu_2)),
    ]
    return angle_of_2_3d_vectors(current_velocity_vector, target_velocity_vector)

def angle_of_2_velocity2(ve_1: float, vn_1: float, vh_1: float, v_2: float, mu_2: float, chi_2: float):
    """计算两个速度矢量之间的夹角，第一个速度矢量由(ve, vn, vh)描述，第二个速度矢量由(v, mu, chi)描述
    """
    current_velocity_vector = [
        ve_1,
        vn_1,
        vh_1
    ]
    target_velocity_vector = [
        v_2 * np.cos(np.deg2rad(mu_2)) * np.sin(np.deg2rad(chi_2)), 
        v_2 * np.cos(np.deg2rad(mu_2)) * np.cos(np.deg2rad(chi_2)),
        v_2 * np.sin(np.deg2rad(mu_2)),
    ]
    return angle_of_2_3d_vectors(current_velocity_vector, target_velocity_vector)

def v_mu_chi_2_enh(v: float, mu: float, chi: float) -> Tuple[float, float, float]:
    """航迹角表示的速度矢量转“东北天（east-north-height，右手坐标系）”坐标

    Args:
        v (float): 速度
        mu (float): 航迹倾斜角，单位：度
        chi (float): 航迹方位角，单位：度

    Returns:
        (float, float, float): (east, north, height)
    """
    mu_rad = np.deg2rad(mu)
    chi_rad = np.deg2rad(chi)
    return (
        v * np.cos(mu_rad) * np.sin(chi_rad),
        v * np.cos(mu_rad) * np.cos(chi_rad),
        v * np.sin(mu_rad)
    )