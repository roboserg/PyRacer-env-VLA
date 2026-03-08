"""
Abstract base class for controllers.
Controllers define how input (human or bot) is translated to game actions.
"""

from abc import ABC, abstractmethod
from vla.observation import Observation


class Controller(ABC):
    """Abstract base class for all controllers."""

    @abstractmethod
    def get_action(self, observation: Observation) -> dict:
        """
        Get the next action based on current observation.

        Args:
            observation: Observation object with current game state

        Returns:
            dict with keys: {"accel": bool, "brake": bool, "left": bool, "right": bool}
        """
        pass

    @property
    def should_quit(self) -> bool:
        """Whether the controller has signaled to quit the game."""
        return False

    def reset(self):
        """Called when game resets. Optional for subclasses to override."""
        pass
