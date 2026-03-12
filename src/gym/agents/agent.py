"""
Abstract base class for agents.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Any
from src.gym.observation import Observation


class Agent(ABC):
    """Abstract base class for all agents."""

    def __init__(self, env: Optional[Any] = None):
        self.env = env

    @abstractmethod
    def predict(self, observation: Observation) -> Any:
        """
        Get the next action based on current observation.

        Returns:
            action: int (discrete action 0-15) or dict of booleans {accel, brake, left, right}
        """
        pass
