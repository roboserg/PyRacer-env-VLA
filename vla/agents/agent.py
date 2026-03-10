"""
Abstract base class for agents.
Adheres to StableBaselines3-like 'predict' interface.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Any
from vla.observation import Observation


class Agent(ABC):
    """Abstract base class for all agents. Mirrors the SB3 Model interface."""

    def __init__(self, env: Optional[Any] = None):
        self.env = env
        self._should_quit = False

    @abstractmethod
    def predict(
        self,
        observation: Observation,
        state: Optional[Tuple[Any, ...]] = None,
        episode_start: Optional[Any] = None,
        deterministic: bool = False,
    ) -> Tuple[Any, Optional[Tuple[Any, ...]]]:
        """
        Get the next action based on current observation.
        Mirrors SB3 model.predict() signature.

        Returns:
            Tuple of (action, next_state)
            action: Can be int (discrete action) or dict (boolean actions)
        """
        pass

    @property
    def should_quit(self) -> bool:
        """Whether the agent has signaled to quit the game."""
        return self._should_quit

    def reset(self):
        """Called when game resets. Optional for subclasses to override."""
        self._should_quit = False
