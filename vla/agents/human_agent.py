"""
Human agent - translates keyboard input to game actions.
Uses pygame.key.get_pressed() for non-blocking input.
"""

from __future__ import annotations

from typing import Optional, Tuple, Any, Dict
import pygame
from vla.agents.agent import Agent
from vla.observation import Observation


class HumanAgent(Agent):
    """Agent that maps keyboard input to game actions."""

    def __init__(self, env: Optional[Any] = None):
        super().__init__(env)

    def predict(
        self,
        observation: Observation,
        state: Optional[Tuple[Any, ...]] = None,
        episode_start: Optional[Any] = None,
        deterministic: bool = False,
    ) -> Tuple[Dict[str, bool], Optional[Tuple[Any, ...]]]:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._should_quit = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._should_quit = True

        keys = pygame.key.get_pressed()
        action = {
            "accel": bool(keys[pygame.K_UP]),
            "brake": bool(keys[pygame.K_DOWN]),
            "left": bool(keys[pygame.K_LEFT]),
            "right": bool(keys[pygame.K_RIGHT]),
        }

        return action, state
