from collections import namedtuple
from pathlib import Path
import sys
from copy import deepcopy
import gc
from typing import Union, Dict, Iterable

PROJECT_ROOT_DIR = Path(__file__).parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from planes.utils.f16Classes import ControlLaw, PlaneModel

class F16Plane(object):

    def __init__(self, 
        env_config: dict
    ) -> None:
        
        self.config = env_config

        self.step_frequence = self.config["task"].get("step_frequence")
        self.h0, self.v0 = self.config["task"].get("h0"), self.config["task"].get("v0")
        self.max_simulate_time = self.config["task"].get("max_simulate_time")
        self.control_mode = self.config["task"].get("control_mode", "guidance_law_mode")  # 两个值：guidance_law_mode，end_to_end_mode
        self.control_mode_guidance_law: bool = (self.control_mode == "guidance_law_mode")

        self.f16cl: ControlLaw = None
        self.f16model: PlaneModel = None

        self.state_list = []
        self.action_list = []
        self.step_cnt = 0

    def reset(self):
        if self.control_mode_guidance_law:
            self.f16cl = ControlLaw(stepTime=self.step_time)  # 控制律模型内有积分，所以也要重新初始化
        self.f16model = PlaneModel(self.h0, self.v0, stepTime=self.step_time)  # 飞机初始化，会配平，

        self.step_cnt = 0
        self.current_obs = self.f16model.getPlaneState()
        self.state_list = [self.current_obs]
        self.action_list = []

        return self.state_list[-1]

    def close(self):
        # 先回收旧的self.f16cl和self.f16model，否则会存在内存持续增长的情况！！！！！
        if self.f16cl is not None:
            # if hasattr(self.f16cl, "f16CL"):
            #     del self.f16cl.f16CL
            del self.f16cl
        if self.f16model is not None:
            # if hasattr(self.f16model, "f16Model"):
            #     del self.f16model.f16Model
            del self.f16model
        gc.collect()

    def step(self, action: Union[Dict, Iterable]) -> Dict:
        """

        Args:
            action (Union[Dict, Iterable]): {"p":xx, "nz":xx, "pla":xx} or [xx, xx, xx]

        Returns:
            Dict: {
                'lef': 0.0, 
                'npos': 202.93280838546622, 
                'epos': -0.0003178369761097856, 
                'h': 4997.945252461648, 
                'alpha': -3.8763871428920833, 
                'beta': 0.0028827414834535683, 
                'phi': -0.0007924768497675406, 
                'theta': -6.5392414504888485, 
                'psi': -0.003222236733571302, 
                'p': 0.0012092603223480781, 
                'q': -2.301811791467605, 
                'r': 0.0018719855098707125, 
                'v': 206.42898450413645, 
                'vn': 206.23466760853253, 
                've': -0.0014038370582571266, 
                'vh': -9.591781654339009, 
                'nx': 0.6544365026301127, 
                'ny': -0.00026289226844918224, 
                'nz': -0.8983464478861598, 
                'ele': -3.725121874665151, 
                'ail': -0.0017574644108460835, 
                'rud': -0.003584270151159641, 
                'thrust': 1.0, 
                'lon': 122.42499999666104, 
                'lat': 31.426828065405605, 
                'mu': -2.126962766563186, 
                'chi': -0.00031410986307750424
            }
        """
        self.step_cnt += 1
        self.action_list.append(action)

        if self.control_mode_guidance_law:
            # 使用控制律模型
            if isinstance(action, Iterable):
                guide_output = {
                    "p": action[0],
                    "nz": action[1],
                    "pla": action[2],
                    "rud": 0.
                }  # [p, nz, pla, rud], rud置为0
            elif isinstance(action, Dict):
                guide_output = {
                    "p": action["p"],
                    "nz": action["nz"],
                    "pla": action["pla"],
                    "rud": 0.
                }
            else:
                raise TypeError("plane: action must be of Iterable or Dict!")

            self.f16cl.step(guide_output, self.current_obs)
            control_law_output = self.f16cl.getOutputDict()
            self.f16model.step(control_law_output)
        else:
            # 不用控制律模型
            if isinstance(action, Iterable):
                end2end_action = {
                    "ail": action[0],
                    "ele": action[1],
                    "rud": action[2],
                    "pla": action[3]
                }
            elif isinstance(action, Dict):
                end2end_action = action
            else:
                raise TypeError("plane: action must be of Iterable or Dict!")
            
            self.f16model.step(end2end_action)
        
        next_obs = self.f16model.getPlaneState()

        self.state_list.append(next_obs)
        self.current_obs = deepcopy(next_obs)
        return self.state_list[-1]

    @property
    def step_time(self):
        return 1. / self.step_frequence

    @staticmethod
    def get_action_vars(control_mode: str="guidance_law_mode"):
        if control_mode == "guidance_law_mode":
            return namedtuple("action_vars", ["p", "nz", "pla"])
        elif control_mode == "end_to_end_mode":
            return namedtuple("action_vars", ["ail", "ele", "rud", "pla"])
        else:
            raise TypeError("plane: control_mode must be guidance_law_mode or end_to_end_mode!")
    
    @staticmethod
    def get_action_lower_bounds(control_mode: str="guidance_law_mode"):
        action_vars_type = F16Plane.get_action_vars(control_mode)
        
        if control_mode == "guidance_law_mode":
            return action_vars_type(p=-180., nz=-4., pla=0.)
        elif control_mode == "end_to_end_mode":
            return action_vars_type(ail=-21.5, ele=-25., rud=-30, pla=0.)
        else:
            raise TypeError("plane: control_mode must be guidance_law_mode or end_to_end_mode!")

    @staticmethod
    def get_action_higher_bounds(control_mode: str="guidance_law_mode"):
        action_vars_type = F16Plane.get_action_vars(control_mode)

        if control_mode == "guidance_law_mode":
            return action_vars_type(p=180., nz=9., pla=1.)
        elif control_mode == "end_to_end_mode":
            return action_vars_type(ail=21.5, ele=25., rud=30, pla=1.)
        else:
            raise TypeError("plane: control_mode must be guidance_law_mode or end_to_end_mode!")
