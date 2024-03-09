import os
from gymnasium.envs.registration import register

with open(os.path.join(os.path.dirname(__file__), "version.txt"), "r") as file_handler:
    __version__ = file_handler.read().strip()

register(
    id="FlyCraft-v0",
    entry_point="flycraft.env:FlyCraftEnv",
)