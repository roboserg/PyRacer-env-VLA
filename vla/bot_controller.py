"""
Bot controller - generates actions via random selection or learned policy.
Currently implements random action selection from the 4-action discrete space.
"""

import random
from vla.controller import Controller
from vla.observation import Observation


class BotController(Controller):
    """Bot controller that selects random actions from action space."""

    # Discrete action space: 4 possible actions
    ACTION_SPACE = [
        {"accel": True, "brake": False, "left": False, "right": False},  # accelerate
        {"accel": False, "brake": True, "left": False, "right": False},  # brake
        {"accel": False, "brake": False, "left": True, "right": False},  # steer left
        {"accel": False, "brake": False, "left": False, "right": True},  # steer right
    ]

    def __init__(self, seed=None, strategy="random"):
        """
        Initialize bot controller.

        Args:
            seed: Random seed for reproducibility
            strategy: "random" for now, extensible for future policies
        """
        self.rng = random.Random(seed)
        self.strategy = strategy
        self._should_quit = False

    def get_action(self, observation: Observation) -> dict:
        """
        Get next action based on strategy.

        Args:
            observation: Current game observation

        Returns:
            dict with discrete action: {"accel", "brake", "left", "right"}
        """
        if self.strategy == "random":
            return self.rng.choice(self.ACTION_SPACE)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    @property
    def should_quit(self) -> bool:
        """Bot doesn't quit unless explicitly set."""
        return self._should_quit

    def reset(self):
        """Reset bot state for new game."""
        pass
