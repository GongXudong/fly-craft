import platform
import os
import sys

currentOps = ""
currentArchi = ""
pythonVersion = sys.version_info.major
esPath = "/home/pi/champ/pythonChamp/Strategy/integrator"

if platform.system() == "Windows":
    currentOps = "Windows"
elif os.path.exists(esPath):
    currentOps = "Rasberrypi"
else:
    currentOps = "Ubuntu"

if platform.architecture()[0] == "64bit":
    currentArchi = "64"
else:
    currentArchi = "32"
    
