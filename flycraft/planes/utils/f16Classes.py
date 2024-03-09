import os
import sys
from pathlib import Path
from math import radians, sin, cos, tan, atan2, asin
from ctypes import CDLL, c_char_p, c_double, c_void_p, c_int

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from planes.utils.f16_utils import currentOps, currentArchi, pythonVersion
from planes.utils.f16_utils.util import Mach2TAS


def checkPlatform():
    if currentOps == 'Windows':
        suffix = '.dll'
        if currentArchi == '32':
            end = 'x86'
        else:
            end = 'x64'
    else:
        suffix = '.so'
        end = 'ubuntu'
    path = 'libs/plane/f16/' + end
    return path, suffix

class Guide(object):
    def __init__(self, planeType=None):
        self.planeType = planeType
        self.initDll()
        self.initLog()
    
    def __del__(self):
        self.log.close()
        
    def initDll(self):
        path, ends = checkPlatform()
        # dllGuide = os.path.join(path, "Guide1" + ends)
        dllGuide = os.path.join(PROJECT_ROOT_DIR, "planes", "utils", path, "Guide1" + ends)
        self.f16Guide = CDLL(dllGuide)
        self.initArgs()
    
    def initLog(self):
        # self.log = open('logs/guideInOut.csv', 'w')
        self.log = open(os.path.join(PROJECT_ROOT_DIR, "planes", "utils", "logs", "guideInOut.csv"), 'w')
        usefulKeys = ['cmdMu', 'mu', 'dchi', 'phi', 'cmdPhi', 'cmdNz']
        self.log.write(','.join(usefulKeys))
        self.log.write('\n')
    
    def initArgs(self):
        c_double_array4 = c_double*4
        c_double_array19 = c_double*19
        self.inputGuide = c_double_array19()
        self.outputGuide = c_double_array4()
        self.outputDict = {}
    
    def step(self, cmdDict, planeStateDict):
        self.setInput(cmdDict, planeStateDict)
        self.f16Guide.Guide_OneStep(self.inputGuide)
        self.f16Guide.Guide_GetOutput(self.outputGuide)
    
    def setWCmd(self, cmdDict):
        self.inputGuide[0] = cmdDict['v']       # Cmd_V, m/s
        self.inputGuide[1] = cmdDict['mu']         # Cmd_Vtheta, deg
        self.inputGuide[2] = cmdDict['chi']        # Cmd_Vpsi, deg
        self.inputGuide[3] = 0         # K_switch, proportion of LV cmd, 0 is only W, 1 is only LV
        self.inputGuide[4] = 1         # Nz_c_LV
        self.inputGuide[5] = 0         # Phi_c_LV
        self.inputGuide[6] = 0         # Thr_c_LV
        self.inputGuide[7] = 1         # Mode, 1W, 2LV, 3Best turn
        self.log.write(str(cmdDict['mu']))
    
    def setPlaneState(self, planeStateDict):
        self.inputGuide[8] = planeStateDict['psi']          #psi_p
        self.inputGuide[9] = planeStateDict['theta']        #pit_p
        self.inputGuide[10] = planeStateDict['phi']         #bank_p
        self.inputGuide[11] = planeStateDict['v']       #VIAS_p
        self.inputGuide[12] = planeStateDict['vh']      #Vh 
        self.inputGuide[13] = planeStateDict['vn']      #Vn Vx
        self.inputGuide[14] = planeStateDict['ve']      #Ve Vy
        self.inputGuide[15] = planeStateDict['p']       #p
        self.inputGuide[16] = planeStateDict['h']       #h
        self.inputGuide[17] = 0                  #cmdP
        self.inputGuide[18] = 0                  #cmdRudder

    def setInput(self, cmdDict, planeStateDict):
        self.setWCmd(cmdDict)
        self.setPlaneState(planeStateDict)
    
    def getOutputDict(self):
        self.outputDict['p'] = self.outputGuide[0]      #roll rate: -180~180 deg/s
        self.outputDict['nz'] = self.outputGuide[1]     #g load: -3~8
        self.outputDict['pla'] = self.outputGuide[2]    #throttle: 0~1.0
        self.outputDict['rud'] = self.outputGuide[3]    #RUD
        return self.outputDict

# stepTime = 0.01

class ControlLaw(object):
    def __init__(self, type='f16', stepTime=0.01):
        self.stepTime = stepTime
        self.initDll()
        self.initArgs()
        self.initDllFuncTypes()
        self.initControlLaw()
    
    def __del__(self):
        self.f16CL.clExit(self.clPtr)
    
    def initDll(self):
        path, ends = checkPlatform()
        # dllCL = os.path.join(path, "f16cl" + ends)
        dllCL = os.path.join(PROJECT_ROOT_DIR, "planes", "utils", path, "f16cl" + ends)
        self.f16CL = CDLL(dllCL)
    
    def initArgs(self):
        c_double_array4 = c_double*4
        c_double_array10 = c_double*10
        self.inState = c_double_array10()
        self.inCmd = c_double_array4()
        self.output = c_double_array4()
        self.outputDict = {}
    
    def initDllFuncTypes(self):
        self.f16CL.init.argtypes = [c_double, c_char_p]
        self.f16CL.init.restype = c_void_p
        self.f16CL.step.argtypes = [c_void_p, c_double*4, c_double*10, c_double*4]
        self.f16CL.setLog.argtypes = [c_void_p, c_char_p]
    
    def initControlLaw(self):
        clParamDir = os.path.join(PROJECT_ROOT_DIR, "planes", "utils", "f16", "lib", "cl")
        clParamDir = bytes(clParamDir, 'utf-8')
        self.clPtr = self.f16CL.init(self.stepTime, clParamDir)
    
    def setInState(self, planeStateDict):
        self.inState[0] = planeStateDict['h']
        self.inState[1] = planeStateDict['v']
        self.inState[2] = planeStateDict['alpha']
        self.inState[3] = planeStateDict['beta']
        self.inState[4] = planeStateDict['phi']
        self.inState[5] = planeStateDict['theta']
        self.inState[6] = planeStateDict['p']
        self.inState[7] = planeStateDict['q']
        self.inState[8] = planeStateDict['r']
        self.inState[9] = planeStateDict['nz']
    
    def setInCmd(self, cmdDict):
        self.inCmd[0] = cmdDict['p']      # roll rate: -85~85 deg/s
        self.inCmd[1] = cmdDict['nz']     # g load: -3~8
        self.inCmd[2] = cmdDict['rud']
        self.inCmd[3] = cmdDict['pla']    # throttle: 0~1.0
    
    def step(self, cmdDict, planeStateDict):
        self.setInCmd(cmdDict)
        self.setInState(planeStateDict)
        self.f16CL.step(self.clPtr, self.inCmd, self.inState, self.output)
    
    def getOutputDict(self):
        self.outputDict['ail'] = self.output[0]
        self.outputDict['ele'] = self.output[1]
        self.outputDict['rud'] = self.output[2]
        self.outputDict['pla'] = self.output[3]
        return self.outputDict
    
    def setLog(self, logName):
        logName = bytes(logName, 'utf-8')
        self.f16CL.setLog(self.clPtr, logName)

class PlaneModel(object):
    def __init__(self, h0, v0, stepTime=0.01):
        self.stepTime = stepTime
        self.initDll()
        self.initArgs()
        self.initDllFuncTypes()
        self.initModel(h0, v0)
    
    def __del__(self):
        pass
        # 下面一行原本是注释状态，取消注释后，程序无法正常运行！！！！！！！！！！！！！！
        # self.f16Model.deletePtr(self.planePtr)
    
    def initDll(self):
        path, ends = checkPlatform()
        # dllModel = os.path.join(path, "f16model" + ends)
        dllModel = os.path.join(PROJECT_ROOT_DIR, "planes", "utils", path, "f16model" + ends)
        self.f16Model = CDLL(dllModel)
    
    def initArgs(self):
        c_double_array4 = c_double*4
        c_double_array31 = c_double*31
        self.inputAct = c_double_array4()
        self.output = c_double_array31()
        self.stateDict = {}
        self.stateKeys = ['lef']
        self.stateKeys += ['npos', 'epos', 'h']
        self.stateKeys += ['alpha', 'beta']
        self.stateKeys += ['phi', 'theta', 'psi']
        self.stateKeys += ['p', 'q', 'r']
        self.stateKeys += ['v', 'vn', 've', 'vh']
        self.stateKeys += ['nx', 'ny', 'nz']
        self.stateKeys += ['ele', 'ail', 'rud', 'thrust']
        self.stateKeys += ['ele', 'ail', 'rud', 'thrust']
        self.stateKeys += ['lon', 'lat']
        self.stateKeys += ['mu', 'chi']
    
    def initDllFuncTypes(self):
        self.f16Model.init.argtypes = [c_double, c_double, c_double, c_int, c_char_p]
        self.f16Model.init.restype = c_void_p
        self.f16Model.setlla.argtypes = [c_void_p, c_double, c_double]
        self.f16Model.setPos.argtypes = [c_void_p, c_double, c_double]
        self.f16Model.setPhi.argtypes = [c_void_p, c_double]
        self.f16Model.setAttitude.argtypes = [c_void_p, c_double, c_double, c_double]
        self.f16Model.stepsim.argtypes = [c_void_p, c_double, c_double, c_double, c_double, c_double*31]
        self.f16Model.getStates.argtypes = [c_void_p, c_double*31]
        self.f16Model.getMaxThrust.argtypes = [c_void_p, c_double, c_double]
        self.f16Model.getMaxThrust.restype = c_double
        self.f16Model.saveStates.argtypes = [c_void_p]
        self.f16Model.loadStates.argtypes = [c_void_p]
        self.f16Model.loadPlaneInfo.argtypes = [c_void_p, c_double*14, c_double*4]
    
    def initModel(self, h0, v0):
        ''' init Plane model '''
        if v0 < 10:
            v0 = Mach2TAS(v0, h0)
        self.h0 = h0
        self.v0 = v0
        self.lon = 122.425
        self.lat = 31.425
        
        self.planeAirLib = os.path.join(PROJECT_ROOT_DIR, "planes", "utils", "f16", "lib", "f16_air.lib")
        libsPath = c_char_p(bytes(self.planeAirLib, 'utf-8'))
        self.planePtr = self.f16Model.init(self.h0, self.v0, self.stepTime, 1, libsPath)
        self.setLonLat(self.lon, self.lat)
    
    def setLonLat(self, lon, lat):
        self.f16Model.setlla(self.planePtr, lon, lat)
        self.f16Model.getStates(self.planePtr, self.output)
    
    def setPlaneAttitude(self, phi, theta, psi):
        self.f16Model.setAttitude(self.planePtr, phi, theta, psi)
        self.f16Model.getStates(self.planePtr, self.output)
    
    def setLog(self, logName):
        logName = bytes(logName, 'utf-8')
        self.f16Model.setLog(self.planePtr, logName)
    
    def setInAct(self, actDict):
        self.inputAct[0] = actDict['ail']
        self.inputAct[1] = actDict['ele']
        self.inputAct[2] = actDict['rud']
        self.inputAct[3] = actDict['pla']
    
    def step(self, actDict):
        self.setInAct(actDict)
        self.f16Model.stepsim(self.planePtr, self.inputAct[0], self.inputAct[1], self.inputAct[2], self.inputAct[3], self.output)

    def getPlaneState(self):
        for index, key in enumerate(self.stateKeys):
            self.stateDict[key] = self.output[index]
        return self.stateDict
