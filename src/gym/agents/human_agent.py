"""
Human agent - translates keyboard input to game actions.
Uses pygame.key.get_pressed() for non-blocking input.
"""

from __future__ import annotations

from typing import Optional, Any, Dict
import pygame
from src.gym.agents.agent import Agent
from src.gym.observation import Observation


class HumanAgent(Agent):
    """Agent that maps keyboard input to game actions."""

    def __init__(self, env: Optional[Any] = None):
        super().__init__(env)

    def predict(self, observation: Observation) -> Dict[str, bool]:
        keys = pygame.key.get_pressed()
        return {
            "accel": bool(keys[pygame.K_UP]),
            "brake": bool(keys[pygame.K_DOWN]),
            "left": bool(keys[pygame.K_LEFT]),
            "right": bool(keys[pygame.K_RIGHT]),
        }
