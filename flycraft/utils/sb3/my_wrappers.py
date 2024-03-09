from gymnasium import ObservationWrapper, ActionWrapper, Env, spaces
from sklearn.preprocessing import MinMaxScaler
from typing import TypeVar, Dict, Union, List
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT_DIR = Path(__file__).parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from utils.scalar import get_min_max_scalar
from env import FlyCraftEnv
from tasks.attitude_control_task import AttitudeControlTask
from planes.f16_plane import F16Plane

# from gymnasium core.py
ObsType = TypeVar("ObsType")
WrapperObsType = TypeVar("WrapperObsType")
ActType = TypeVar("ActType")
WrapperActType = TypeVar("WrapperActType")


class ScaledObservationWrapper(ObservationWrapper):
    
    def __init__(self, env: Env[ObsType, ActType]):
        super().__init__(env)

        # 缩放与仿真器无关，只在学习器中使用
        # 送进策略网络的观测，各分量的取值都在[0, 1]之间
        
        plane_state_mins = AttitudeControlTask.get_state_lower_bounds()
        plane_state_maxs = AttitudeControlTask.get_state_higher_bounds()
        plane_goal_mins = AttitudeControlTask.get_goal_lower_bounds()
        plane_goal_maxs = AttitudeControlTask.get_goal_higher_bounds()
        
        self.observation_space = spaces.Dict(
            dict(
                observation = spaces.Box(low=0., high=1., shape=(len(plane_state_mins),)),  # phi, theta, psi, v, mu, chi, p, h
                desired_goal = spaces.Box(low=0., high=1., shape=(len(plane_goal_mins),)),
                achieved_goal = spaces.Box(low=0., high=1., shape=(len(plane_goal_mins),)),
            )
        )

        self.state_scalar: MinMaxScaler = get_min_max_scalar(
            mins=np.array(plane_state_mins),
            maxs=np.array(plane_state_maxs),
            feature_range=(0., 1.),
        )
        self.goal_scalar: MinMaxScaler = get_min_max_scalar(
            mins=np.array(plane_goal_mins),
            maxs=np.array(plane_goal_maxs),
            feature_range=(0., 1.)
        )
    
    def scale_state(self, state_var: Union[Dict, np.ndarray]) -> Union[Dict, np.ndarray]:
        """将仿真器返回的state缩放到[0, 1]之间。
        每一步的状态是字典类型，
        包括三个键：observation，desired_goal，achieved_goal，对应的值的类型都是np.ndarray。
        """
        if isinstance(state_var, dict):
            tmp_state_var = [state_var]
            # return self.state_scalar.transform(tmp_state_var).reshape((-1))
        elif len(state_var.shape) == 2:
            tmp_state_var = state_var
            # return self.state_scalar.transform(state_var)
        else:
            raise TypeError("state_var只能是1维或者2维！")
        
        res = [
            dict(
                observation = self.state_scalar.transform(tmp_state["observation"].reshape((1,-1))).reshape((-1)),
                desired_goal = self.goal_scalar.transform(tmp_state["desired_goal"].reshape((1,-1))).reshape((-1)),
                achieved_goal = self.goal_scalar.transform(tmp_state["achieved_goal"].reshape((1,-1))).reshape((-1)),
            )
            for tmp_state in tmp_state_var
        ]

        if isinstance(state_var, dict):
            return res[0]
        else:
            return np.array(res)

    def observation(self, observation: ObsType) -> WrapperObsType:
        return self.scale_state(observation)
    
    def inverse_scale_state(self, state_var: Union[Dict, np.ndarray]) -> Union[Dict, np.ndarray]:
        """将[0, 1]之间state变回仿真器定义的原始state。用于测试！！！
        """
        if isinstance(state_var, dict):
            tmp_state_var = [state_var]
            # return self.state_scalar.inverse_transform(tmp_state_var).reshape((-1))
        elif len(state_var.shape) == 2:
            tmp_state_var = state_var
            # return self.state_scalar.inverse_transform(state_var)
        else:
            raise TypeError("state_var只能是1维或者2维！")
        
        res = [
            dict(
                observation = self.state_scalar.inverse_transform(tmp_state["observation"].reshape((1,-1))).reshape((-1)),
                desired_goal = self.goal_scalar.inverse_transform(tmp_state["desired_goal"].reshape((1,-1))).reshape((-1)),
                achieved_goal = self.goal_scalar.inverse_transform(tmp_state["achieved_goal"].reshape((1,-1))).reshape((-1)),
            )
            for tmp_state in tmp_state_var
        ]

        if isinstance(state_var, dict):
            return res[0]
        else:
            return np.array(res)

class ScaledActionWrapper(ActionWrapper):

    def __init__(self, env: Env[ObsType, ActType]):
        super().__init__(env)

        action_mins = F16Plane.get_action_lower_bounds()
        action_maxs = F16Plane.get_action_higher_bounds()

        self.action_space = spaces.Box(low=0., high=1., shape=(len(action_mins),))  # p, nz, pla

        # 策略网络输出的动作，各分量的取值都在[0, 1]之间
        self.action_scalar: MinMaxScaler = get_min_max_scalar(
            mins=np.array(action_mins),
            maxs=np.array(action_maxs),
            feature_range=(0., 1.)
        )
    
    def inverse_scale_action(self, action_var: np.ndarray) -> np.ndarray:
        """将学习器推理出的动作放大到仿真器接收的动作范围
        """
        if len(action_var.shape) == 1:
            tmp_action_var = action_var.reshape((1, -1))
            return self.action_scalar.inverse_transform(tmp_action_var).reshape((-1))
        elif len(action_var.shape) == 2:
            return self.action_scalar.inverse_transform(action_var)
        else:
            raise TypeError("action_var只能是1维或者2维！") 
    
    def action(self, action: WrapperActType) -> ActType:
        # 检查action类型
        if type(action) == np.ndarray:
            return self.inverse_scale_action(action)
        else:
            return self.inverse_scale_action(np.array(action))
    
    def scale_action(self, action_var: np.ndarray) -> np.ndarray:
        """将仿真器接收范围的action缩放到[0, 1]之间。用于测试！！！
        """
        if len(action_var.shape) == 1:
            tmp_action_var = action_var.reshape((1, -1))
            return self.action_scalar.transform(tmp_action_var).reshape((-1))
        elif len(action_var.shape) == 2:
            return self.action_scalar.transform(action_var)
        else:
            raise TypeError("action_var只能是1维或者2维！") 