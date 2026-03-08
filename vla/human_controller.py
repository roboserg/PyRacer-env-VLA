"""
Human controller - translates keyboard input to game actions.
Uses pygame.key.get_pressed() for non-blocking input.
"""

import pygame
from vla.controller import Controller
from vla.observation import Observation


class HumanController(Controller):
    """Controller that maps keyboard input to game actions."""

    def __init__(self):
        """Initialize human controller."""
        self._should_quit = False

    def get_action(self, observation: Observation) -> dict:
        """
        Get action from keyboard input.

        Processes pygame events (non-blocking) and returns current key state.
        Sets should_quit flag if ESC is pressed.

        Args:
            observation: Current game observation (unused by human controller)

        Returns:
            dict with action keys: accel, brake, left, right (all bool)
        """
        # Process events to update quit flag
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._should_quit = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._should_quit = True

        # Get current key state (non-blocking)
        keys = pygame.key.get_pressed()
        action = {
            "accel": keys[pygame.K_UP],
            "brake": keys[pygame.K_DOWN],
            "left": keys[pygame.K_LEFT],
            "right": keys[pygame.K_RIGHT],
        }

        return action

    @property
    def should_quit(self) -> bool:
        """Whether user has pressed ESC or closed window."""
        return self._should_quit

    def reset(self):
        """Reset quit flag when game resets."""
        self._should_quit = False
