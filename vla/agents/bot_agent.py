"""
Bot agent - simple rule-based agent that steers toward track center.
Uses observation data to make decisions: stays on road, accelerates, steers by offset.
"""

from typing import Optional, Tuple, Any, Dict
import pygame
from vla.agents.agent import Agent
from vla.observation import Observation

# How far from center (px) before corrective steering kicks in
STEER_THRESHOLD = 15
# Speed above which braking is applied when off-road
OFF_ROAD_BRAKE_SPEED = 3.0


class BotAgent(Agent):
    """
    Rule-based agent that keeps the car on the road by steering toward center.

    Strategy:
    - Always accelerate
    - Steer left/right proportionally to offset from track center
    - Brake when off-road at high speed to recover
    """

    def __init__(self, env: Optional[Any] = None):
        super().__init__(env)

    def predict(
        self,
        observation: Observation,
        state: Optional[Tuple[Any, ...]] = None,
        episode_start: Optional[Any] = None,
        deterministic: bool = False,
    ) -> Tuple[Dict[str, bool], Optional[Tuple[Any, ...]]]:
        offset = observation.car_offset_from_center
        speed = observation.speed
        on_road = observation.on_road

        accel = True
        brake = False
        left = False
        right = False

        # Steer back toward center based on lateral offset
        if offset > STEER_THRESHOLD:
            left = True
        elif offset < -STEER_THRESHOLD:
            right = True

        # Brake to recover when off-road at speed
        if not on_road and speed > OFF_ROAD_BRAKE_SPEED:
            brake = True
            accel = False

        # Check for quit events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._should_quit = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self._should_quit = True

        return {"accel": accel, "brake": brake, "left": left, "right": right}, state
