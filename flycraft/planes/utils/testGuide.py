import os
import sys
from pathlib import Path
from math import radians, sin, cos

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from planes.utils.f16Classes import Guide, ControlLaw, PlaneModel


def testGuide():
    gtmp = Guide()
    cmds = {}
    cmds['v'] = 180
    cmds['mu'] = 0
    cmds['chi'] = 30
    
    stateDict = {}
    stateDict['phi'] = 0
    stateDict['theta'] = 10
    stateDict['psi'] = 0
    stateDict['v'] = 160
    stateDict['mu'] = 0
    stateDict['chi'] = 0
    stateDict['vn'] = stateDict['v'] * cos(radians(stateDict['mu'])) * cos(radians(stateDict['chi']))
    stateDict['ve'] = stateDict['v'] * cos(radians(stateDict['mu'])) * sin(radians(stateDict['chi']))
    stateDict['vh'] = stateDict['v'] * sin(radians(stateDict['mu']))
    stateDict['p'] = 0
    stateDict['h'] = 1000
    for i in range(10):
        gtmp.step(cmds, stateDict)

        #################################
        acts = gtmp.getOutputDict()
        for key in acts:
            print(key, acts[key], end=',')
        print()
        #################################

        for i in range(4):
            print(gtmp.outputGuide[i], end = ',')
        print()

def testCL():
    f16cl = ControlLaw()
    cmds = {}
    cmds['p'] = 180
    cmds['nz'] = 1
    cmds['rud'] = 0
    cmds['pla'] = 1
    
    stateDict = {}
    stateDict['v'] = 160
    stateDict['h'] = 1000
    stateDict['alpha'] = 10
    stateDict['beta'] = 0
    stateDict['phi'] = 0
    stateDict['theta'] = 10
    stateDict['p'] = 0
    stateDict['q'] = 0
    stateDict['r'] = 0
    stateDict['nz'] = 1
    f16cl.step(cmds, stateDict)
    acts = f16cl.getOutputDict()
    for key in acts:
        print(key, acts[key], end = ',')
    print()

def testPlane():
    h0 = 1000
    v0 = 200
    f16model = PlaneModel(h0, v0)
    stsDict = f16model.getPlaneState()  
    # stsDict: lef, npos, epos, h, alpha, beta, phi, theta, psi, p, q, r, v, vn, ve, vh, nx, ny, nz, ele, ail, rud, thrust, lon, lat, mu, chi
    
    # for k in stsDict.keys():
    #     print(k, end=', ')
    # print()
    showKeys = ['p', 'q', 'r']
    for key in showKeys:
        print(stsDict[key], end = ',')
    print()
    
    cmds = {}
    cmds['ail'] = 20
    cmds['ele'] = -1
    cmds['rud'] = 0
    cmds['pla'] = 1
    for i in range(10):
        f16model.step(cmds)
        stsDict = f16model.getPlaneState()
        for key in showKeys:
            print(stsDict[key], end = ',')
        print()

def testFullStep():
    h0 = 1000
    v0 = 200
    gtmp = Guide()
    f16cl = ControlLaw()
    f16model = PlaneModel(h0, v0)
    # f16model.setLog('utils/f16trace4.csv')  # 设置日志名
    stsDict = f16model.getPlaneState()
    
    wCmds = {}
    # wCmds['v'] = 180
    # wCmds['mu'] = 20
    # wCmds['chi'] = 30
    wCmds['v'] = 180
    wCmds['mu'] = -90
    wCmds['chi'] = 0
    for i in range(12000):
        gtmp.step(wCmds, stsDict)
        gout = gtmp.getOutputDict()
        f16cl.step(gout, stsDict)
        clout = f16cl.getOutputDict()
        f16model.step(clout)
        stsDict = f16model.getPlaneState()

if __name__ == '__main__':
    # testGuide()
    # testCL()
    # testPlane()
    testFullStep()