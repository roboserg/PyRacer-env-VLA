"""
GameEnvironment - Main orchestrator for the VLA game environment.
Manages game state, recording, and the main game loop.
"""

import time
import pygame
from typing import Optional, Tuple
from data.gameFiles.game import Game
from vla.controller import Controller
from vla.recorder import Recorder
from vla.observation import Observation


class GameEnvironment:
    """Main game environment that coordinates game, controller, and recording."""

    def __init__(self, controller: Controller, recorder: Optional[Recorder] = None):
        """
        Initialize game environment.

        Args:
            controller: Controller instance (HumanController or BotController)
            recorder: Optional Recorder instance for data collection
        """
        self.game = Game()
        self.controller = controller
        self.recorder = recorder

        self.current_observation = None

        # Recording state
        self.frame_count = 0
        self.iteration_count = 0

    def _create_observation(self) -> Observation:
        """Create observation object from current game state."""
        # Get current display surface (for recording)
        frame = self.game.display.copy()

        # Extract car and map data
        car = self.game.map.car
        map_obj = self.game.map

        observation = Observation(
            frame=frame,
            speed=car.speed,
            lap=self.game.map.lap,
            lap_time=self.game.lap_time,
            position=car.distance,
            car_x=car.position_int,
            car_y=220,  # Car is always drawn at y=220
            map_obj=map_obj,
        )

        return observation

    def step(self) -> Tuple[Observation, dict]:
        """
        Execute one game step.
        """
        self.iteration_count += 1

        # Create observation from current state (if not already created)
        if self.current_observation is None:
            self.game.render() # Ensure something is drawn
            self.current_observation = self._create_observation()

        # Get action from controller (blocking)
        # This will now block the main loop, ensuring we wait for the model
        action = self.controller.get_action(self.current_observation)

        # Get delta time AFTER inference to reset the clock
        self.game.get_dt()
        
        # Cap dt to prevent physics glitches
        if self.game.dt > 0.05:
            self.game.dt = 0.05

        # Update game.actions dict with current action
        self.game.actions["accel"] = action["accel"]
        self.game.actions["brake"] = action["brake"]
        self.game.actions["left"] = action["left"]
        self.game.actions["right"] = action["right"]

        # Update game logic
        if self.game.countdown > 0:
            self.game.count_down()
        else:
            self.game.update()

        # Render
        self.game.render()

        # Update observation for the NEXT step
        self.current_observation = self._create_observation()

        # Record frame (every 6th frame = 10 FPS at 60 FPS game)
        self.frame_count += 1
        should_record = self.frame_count % 6 == 0

        if should_record and self.recorder:
            self.recorder.record_frame(self.current_observation, action)

        # Check for quit signals
        info = {
            "should_quit": self.controller.should_quit,
            "game_complete": self.game.complete,
            "lap": self.game.map.lap,
            "speed": self.game.map.car.speed,
        }

        return self.current_observation, info

    def run(self, max_laps: int = 2, max_steps: int = None, verbose: bool = True) -> dict:
        """
        Run the game environment main loop.

        Args:
            max_laps: Stop after this many laps
            max_steps: Optional maximum number of steps (frames)
            verbose: Print progress information

        Returns:
            stats dict with final game statistics
        """
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Starting game environment (Synchronous Mode)")
            print(f"Controller: {self.controller.__class__.__name__}")
            print(f"Recording: {self.recorder is not None}")
            print(f"{'=' * 60}\n")

        # Initialize game
        self.game.playing = True
        self.game.reset()
        self.controller.reset()

        # Create initial observation
        self.current_observation = self._create_observation()

        try:
            step_count = 0
            print_interval = 10  # Print stats more frequently in sync mode

            while self.game.playing:
                # Run one step
                obs, info = self.step()

                # Check quit conditions
                if info["should_quit"]:
                    if verbose:
                        print("\n✓ User quit")
                    break

                if info["game_complete"] or info["lap"] >= max_laps:
                    if verbose:
                        print("\n✓ Race complete")
                    break

                if max_steps and step_count >= max_steps:
                    if verbose:
                        print(f"\n✓ Max steps reached ({max_steps})")
                    break

                # Print progress
                if step_count % print_interval == 0 and step_count > 0:
                    fps = self.game.clock.get_fps()
                    if verbose:
                        print(
                            f"  Step {step_count}: Speed={info['speed']:.2f}, "
                            f"Lap={info['lap']}, Time={self.game.lap_time:.1f}s, "
                            f"Inference FPS={fps:.1f}"
                        )

                step_count += 1

            # Print final stats
            if verbose:
                print(f"\n{'=' * 60}")
                print(f"Game Statistics")
                print(f"{'=' * 60}")
                print(f"Total steps: {step_count}")
                print(f"Total frames: {self.frame_count}")
                print(f"Recorded frames: {self.recorder.frame_count if self.recorder else 0}")
                print(f"Final speed: {self.game.map.car.speed:.2f}")
                print(f"Laps completed: {self.game.map.lap}")
                print(f"Final lap time: {self.game.lap_time:.2f}s")
                if self.game.map.lap_times:
                    print(f"Lap times: {[round(t, 2) for t in self.game.map.lap_times]}")
                print(f"{'=' * 60}\n")

            # Save recording metadata
            if self.recorder:
                self.recorder.save_metadata()

            stats = {
                "steps": step_count,
                "frames": self.frame_count,
                "recorded_frames": self.recorder.frame_count if self.recorder else 0,
                "final_speed": self.game.map.car.speed,
                "laps": self.game.map.lap,
                "lap_times": self.game.map.lap_times,
            }

            return stats

        finally:
            self.game.playing = False
