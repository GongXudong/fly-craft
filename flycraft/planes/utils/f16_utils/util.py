

from math import sqrt

R = 287.05
T0 = 288.15
gamma = 1.4

def temperature(altt=4000):
    temp = T0 - 0.0065 * altt
    if altt >= 11000.0:
       temp = 216.65
    return temp


def airSpeed(altt=0):
    return sqrt(gamma * R * temperature(altt))

def TAS2Mach(TAS, altt=4000):
    mach = TAS / airSpeed(altt)
    return mach
    
def Mach2TAS(mach, altt=4000):
    TAS = mach * airSpeed(altt)
    return TAS