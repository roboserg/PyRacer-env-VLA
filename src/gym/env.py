"""
GameEnvironment - Gymnasium wrapper for the PyRacer game.
Manages game state and recording. Agent is external (SB3-style loop).
"""

import pygame
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from PIL import Image
from typing import Optional, Tuple
from data.gameFiles.game import Game
from src.gym.recorder import Recorder
from src.gym.observation import Observation


class GameEnvironment(gym.Env):
    """Gymnasium environment wrapping the PyRacer game."""

    metadata = {"render_modes": ["human"]}

    action_space = spaces.Discrete(16)

    def __init__(self, recorder: Optional[Recorder] = None):
        """
        Initialize game environment.

        Args:
            recorder: Optional Recorder instance for data collection
        """
        super().__init__()
        self.game = Game()
        self.recorder = recorder

        self.current_observation = None

        self.frame_count = 0
        self.iteration_count = 0

        self._step_count = 0
        self._max_steps = 5000

    def _create_observation(self) -> Observation:
        """Create observation object from current game state."""
        surface = self.game.display
        frame = Image.frombytes("RGB", surface.get_size(), pygame.image.tostring(surface, "RGB"))

        car = self.game.map.car
        map_obj = self.game.map

        car_x = car.position_int
        car_offset_from_center = car_x - 240
        on_road = abs(car_offset_from_center) < 80

        observation = Observation(
            frame=frame,
            speed=car.speed,
            position=car.distance,
            car_x=car_x,
            car_y=car.current_draw_y,
            map_obj=map_obj,
            on_road=on_road,
            car_offset_from_center=car_offset_from_center,
        )

        return observation

    def _decode_action(self, action: int) -> dict:
        """Decode discrete action (0-15) to 4 boolean actions."""
        return {
            "accel": bool(action & 1),
            "brake": bool(action & 2),
            "left": bool(action & 4),
            "right": bool(action & 8),
        }

    def _encode_action(self, action_dict: dict) -> int:
        """Encode 4 boolean actions to discrete action (0-15)."""
        action = 0
        if action_dict.get("accel", False):
            action |= 1
        if action_dict.get("brake", False):
            action |= 2
        if action_dict.get("left", False):
            action |= 4
        if action_dict.get("right", False):
            action |= 8
        return action

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Observation, dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        self.game.playing = True
        self.game.reset()

        self._step_count = 0
        self.frame_count = 0
        self.iteration_count = 0

        self.game.render()
        self.current_observation = self._create_observation()

        info = {"speed": self.game.map.car.speed}
        return self.current_observation, info

    def step(self, action) -> Tuple[Observation, float, bool, bool, dict]:
        """
        Execute one game step.

        Args:
            action: Discrete action (0-15) or dict of 4 boolean buttons

        Returns:
            observation: Observation dataclass
            reward: Current speed (float)
            terminated: Whether episode ended (game complete)
            truncated: Whether episode was truncated (max steps or stalled)
            info: Additional info dict
        """
        self.iteration_count += 1
        self._step_count += 1

        if isinstance(action, (int, np.integer)):
            action_dict = self._decode_action(int(action))
        else:
            action_dict = action

        self.game.get_dt()

        self.game.get_events()

        if self.game.dt > 0.05:
            self.game.dt = 0.05

        self.game.actions["accel"] = action_dict["accel"]
        self.game.actions["brake"] = action_dict["brake"]
        self.game.actions["left"] = action_dict["left"]
        self.game.actions["right"] = action_dict["right"]

        if self.game.countdown > 0:
            self.game.count_down()
        else:
            self.game.update()

        self.game.render()

        self.current_observation = self._create_observation()

        self.frame_count += 1
        should_record = self.frame_count % 6 == 0

        if should_record and self.recorder:
            self.recorder.record_frame(
                self.current_observation, action_dict, frame_idx=self.frame_count
            )

        car = self.game.map.car
        off_road_penalty = -0.5 if not self.current_observation.on_road else 0.0
        reward = float(car.speed) + off_road_penalty

        terminated = False
        truncated = self._step_count >= self._max_steps

        info = {
            "should_quit": not self.game.playing,
            "speed": self.game.map.car.speed,
        }

        return self.current_observation, reward, terminated, truncated, info

    def render(self, mode: str = "human"):
        """Render the environment to the screen."""
        if mode == "human":
            pygame.display.flip()

    def close(self):
        """Clean up environment resources."""
        if self.recorder:
            self.recorder.save_metadata()
        self.game.playing = False
        pygame.quit()
