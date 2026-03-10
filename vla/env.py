"""
GameEnvironment - Main orchestrator for the VLA game environment.
Manages game state, recording, and the main game loop.
"""

import time
import pygame
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from PIL import Image
from typing import Any, Optional, Tuple, Dict
from data.gameFiles.game import Game
from vla.agents.agent import Agent
from vla.recorder import Recorder
from vla.observation import Observation


class GameEnvironment(gym.Env):
    """Main game environment that coordinates game, agent, and recording."""

    metadata = {"render_modes": ["human"]}

    action_space = spaces.Discrete(16)

    observation_space = spaces.Dict(
        {
            "image": spaces.Box(0, 255, (270, 480, 3), dtype=np.uint8),
            "speed": spaces.Box(0.0, 1.5, (1,), dtype=np.float32),
            "offset": spaces.Box(-240.0, 240.0, (1,), dtype=np.float32),
            "on_road": spaces.Box(0.0, 1.0, (1,), dtype=np.float32),
        }
    )

    def __init__(self, agent: Agent, recorder: Optional[Recorder] = None):
        """
        Initialize game environment.

        Args:
            agent: Agent instance (HumanAgent, BotAgent, etc.)
            recorder: Optional Recorder instance for data collection
        """
        super().__init__()
        self.game = Game()
        self.agent = agent
        self.recorder = recorder

        # Link agent to this environment (SB3 style)
        if self.agent:
            self.agent.env = self

        self.current_observation = None

        self.frame_count = 0
        self.iteration_count = 0

        self._step_count = 0
        self._zero_speed_counter = 0
        self._max_steps = 5000

    def _create_observation(self) -> Observation:
        """Create observation object from current game state."""
        frame = self.game.display.copy()

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

    def _get_observation(self) -> Dict[str, Any]:
        """Get observation in Gymnasium format (dict with image and speed)."""
        frame = self.game.display.copy()
        pil_image = Image.frombytes(
            "RGB", frame.get_size(), pygame.image.tostring(frame, "RGB")
        )
        image_array = np.array(pil_image, dtype=np.uint8)

        car = self.game.map.car
        car_offset = car.position_int - 240
        speed = np.array([car.speed], dtype=np.float32)
        offset = np.array([car_offset], dtype=np.float32)
        on_road = np.array([1.0 if abs(car_offset) < 80 else 0.0], dtype=np.float32)

        return {"image": image_array, "speed": speed, "offset": offset, "on_road": on_road}

    def _decode_action(self, action: int) -> Dict[str, bool]:
        """Decode discrete action (0-15) to 4 boolean actions."""
        return {
            "accel": bool(action & 1),
            "brake": bool(action & 2),
            "left": bool(action & 4),
            "right": bool(action & 8),
        }

    def _encode_action(self, action_dict: Dict[str, bool]) -> int:
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
    ) -> Tuple[Dict[str, Any], dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        self.game.playing = True
        self.game.reset()
        self.agent.reset()

        self._step_count = 0
        self._zero_speed_counter = 0
        self.frame_count = 0
        self.iteration_count = 0

        self.game.render()
        self.current_observation = self._create_observation()

        observation = self._get_observation()
        info = {
            "speed": self.game.map.car.speed,
        }

        return observation, info

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, dict]:
        """
        Execute one game step.

        Args:
            action: Discrete action (0-15) encoding 4 boolean buttons

        Returns:
            observation: Dict with "image" and "speed"
            reward: Current speed (float)
            terminated: Whether episode ended (game complete)
            truncated: Whether episode was truncated (max steps or stalled)
            info: Additional info dict
        """
        self.iteration_count += 1
        self._step_count += 1

        action_dict = self._decode_action(action)

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

        # Endless track has no natural end; terminated is always False
        terminated = False

        if self.game.map.car.speed < 0.1:
            self._zero_speed_counter += 1
        else:
            self._zero_speed_counter = 0

        truncated = (
            self._step_count >= self._max_steps or self._zero_speed_counter >= 200
        )

        info = {
            "should_quit": self.agent.should_quit,
            "speed": self.game.map.car.speed,
        }

        observation = self._get_observation()

        return observation, reward, terminated, truncated, info

    def run(self, max_steps: Optional[int] = None, verbose: bool = True) -> dict:
        """
        Run the game environment main loop.

        Args:
            max_steps: Optional maximum number of steps (frames)
            verbose: Print progress information

        Returns:
            stats dict with final game statistics
        """
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Starting game environment (Synchronous Mode)")
            print(f"Agent: {self.agent.__class__.__name__}")
            print(f"Recording: {self.recorder is not None}")
            print(f"{'=' * 60}\n")

        self.game.playing = True
        self.game.reset()
        self.agent.reset()

        self.current_observation = self._create_observation()

        step_count = 0
        state = None
        try:
            print_interval = 10

            while self.game.playing:
                # Get action from agent (blocking) - SB3 style predict
                action, state = self.agent.predict(
                    self.current_observation, state=state
                )

                # Decode action if it's an integer (SB3 style)
                if isinstance(action, (int, np.integer)):
                    action_dict = self._decode_action(int(action))
                else:
                    action_dict = action

                self.game.get_dt()

                self.game.get_events()

                if self.game.dt > 0.05:
                    self.game.dt = 0.05

                # Update game.actions dict with current action
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
                        self.current_observation,
                        action_dict,
                        frame_idx=self.frame_count,
                    )

                info = {
                    "should_quit": self.agent.should_quit,
                    "speed": self.game.map.car.speed,
                }

                if info["should_quit"]:
                    if verbose:
                        print("\n✓ User quit")
                    break

                if max_steps and step_count >= max_steps:
                    if verbose:
                        print(f"\n✓ Max steps reached ({max_steps})")
                    break

                if (
                    not self.recorder
                    and step_count % print_interval == 0
                    and step_count > 0
                ):
                    fps = self.game.clock.get_fps()
                    if verbose:
                        print(
                            f"  Step {step_count}: Speed={info['speed']:.2f}, "
                            f"Time={self.game.total_time:.1f}s, "
                            f"Inference FPS={fps:.1f}"
                        )

                step_count += 1

            if verbose:
                print(f"\n{'=' * 60}")
                print(f"Game Statistics")
                print(f"{'=' * 60}")
                print(f"Total steps: {step_count}")
                print(f"Total frames: {self.frame_count}")
                print(
                    f"Recorded frames: {self.recorder.frame_count if self.recorder else 0}"
                )
                print(f"Final speed: {self.game.map.car.speed:.2f}")
                print(f"Total time: {self.game.total_time:.2f}s")
                print(f"{'=' * 60}\n")

            stats = {
                "steps": step_count,
                "frames": self.frame_count,
                "recorded_frames": self.recorder.frame_count if self.recorder else 0,
                "final_speed": self.game.map.car.speed,
                "total_time": self.game.total_time,
            }
            return stats

        finally:
            if self.recorder:
                self.recorder.save_metadata()
            self.game.playing = False

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
