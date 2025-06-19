from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, List

from flycraft.planes.f16_plane import F16Plane
from flycraft.terminations.termination_base import TerminationBase


class Task(ABC):

    def __init__(self, plane: F16Plane) -> None:
        self.plane: F16Plane = plane
        self.goal: np.ndarray = None

    @abstractmethod
    def reset(self) -> None:
        """Reset the task: sample a new goal."""

    @abstractmethod
    def get_obs(self) -> np.ndarray:
        """Return the observation associated to the task."""

    @abstractmethod
    def get_achieved_goal(self) -> np.ndarray:
        """Return the achieved goal."""

    def get_goal(self) -> np.ndarray:
        """Return the current goal."""
        if self.goal is None:
            raise RuntimeError("No goal yet, call reset() first")
        else:
            return self.goal.copy()

    @abstractmethod
    def get_reach_target_terminations(self) -> List[TerminationBase]:
        """Return the list of reach target terminations associated to the task."""

    def is_reach_target_termination(self, tmnt_obj: TerminationBase) -> bool:
        """Judge whether tmnt_obj belongs to a reach target termination.
        """
        return any(
            [isinstance(tmnt_obj, rt_tmnt) for rt_tmnt in self.get_reach_target_terminations()]
        )

    @abstractmethod
    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any]={}) -> np.ndarray:
        """Returns whether the achieved goal match the desired goal."""
    
    @abstractmethod
    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}) -> np.ndarray:
        """Compute reward associated to the achieved and the desired goal."""
