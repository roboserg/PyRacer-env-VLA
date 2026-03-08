"""
Recorder module for capturing game frames and metadata.
Saves frames as JPEG images and metadata as JSONL format.
"""

import os
import json
import pygame
import time
from datetime import datetime
from pathlib import Path
from vla.observation import Observation

# Default recording directory (relative to project root)
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


class Recorder:
    """Records game frames and actions as a dataset."""

    def __init__(self, output_dir: str = None, enabled: bool = True):
        """
        Initialize recorder.

        Args:
            output_dir: Base path where recordings are saved
                       Defaults to /vla/data/recordings/
            enabled: Whether recording is active
        """
        if output_dir is None:
            output_dir = os.path.join(DEFAULT_DATA_DIR, "recordings")

        self.enabled = enabled
        self.frame_count = 0
        self.metadata_entries = []
        self.last_annotation = ""
        self.last_obs = None
        self.start_time = None  # Set on first frame record
        self._last_steer_left = False
        self._last_steer_right = False

        if self.enabled:
            # Create timestamped recording directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = Path(output_dir) / timestamp
            self.images_dir = self.output_dir / "images"
            self.images_dir.mkdir(parents=True, exist_ok=True)
            print(f"✓ Recorder initialized: {self.output_dir}")

    def record_frame(self, observation: Observation, action: dict, text: str = "") -> None:
        """
        Record a single frame with action and annotation.

        Called every Nth frame (e.g., every 6th frame for 10 FPS recording).

        Args:
            observation: Current game observation with frame surface
            action: Action dict {"accel", "brake", "left", "right"}
            text: Optional manual annotation text
        """
        if not self.enabled:
            return

        self.frame_count += 1

        # Generate annotation from game state (only if state changed)
        if text:
            annotation = text
        else:
            annotation = self._generate_annotation(observation, action)

        # Only record if annotation changed or this is first frame
        if annotation != self.last_annotation or self.frame_count == 1:
            # Initialize start_time on first recorded frame
            if self.start_time is None:
                self.start_time = time.time()

            # Save frame image
            frame_name = f"frame_{self.frame_count:05d}.jpg"
            frame_path = self.images_dir / frame_name

            # Convert pygame surface to JPEG
            pygame.image.save(observation.frame, str(frame_path))

            # Normalize action to [0, 1] range
            action_vector = [
                float(action.get("accel", False)),
                float(action.get("brake", False)),
                float(action.get("left", False)),
                float(action.get("right", False)),
            ]

            # Calculate timestamp relative to start of recording
            elapsed_time = time.time() - self.start_time

            # Create metadata entry with timestamp
            entry = {
                "frame": frame_name,
                "timestamp": round(elapsed_time, 2),
                "speed": round(observation.speed, 2),
                "action": action_vector,
                "text": annotation,
            }

            self.metadata_entries.append(entry)
            self.last_annotation = annotation

            # Print progress
            if self.frame_count % 10 == 0:
                print(f"  Recorded {self.frame_count} frames | {annotation}")

        self.last_obs = observation

    def save_metadata(self) -> None:
        """Write metadata entries as JSONL file."""
        if not self.enabled:
            return

        output_path = Path(self.output_dir)
        metadata_path = output_path / "metadata.jsonl"

        with open(str(metadata_path), "w") as f:
            for entry in self.metadata_entries:
                json.dump(entry, f)
                f.write("\n")

        print(f"\n✓ Saved {len(self.metadata_entries)} metadata entries to {metadata_path}")
        print(f"✓ Total frames recorded: {self.frame_count}")
        print(f"✓ Dataset location: {self.output_dir}")

    def _generate_annotation(self, observation: Observation, action: dict) -> str:
        """
        Generate rich text annotation from game state.

        Uses multiple aspects of game state to create descriptive text.

        Args:
            observation: Current observation
            action: Current action dict

        Returns:
            Descriptive string
        """
        parts = []

        # ===== SPEED ANALYSIS (0-1 normalized) =====
        speed = observation.speed
        if speed < 0.01:
            parts.append("Stationary")
        elif speed < 0.2:
            parts.append("Barely moving")
        elif speed < 0.4:
            parts.append("Crawling")
        elif speed < 0.6:
            parts.append("Slow")
        elif speed < 0.75:
            parts.append("Moderate pace")
        elif speed < 0.9:
            parts.append("Good speed")
        elif speed < 0.98:
            parts.append("High speed")
        else:
            parts.append("Max speed")

        # ===== ACTION ANALYSIS =====
        accel = action.get("accel", False)
        brake = action.get("brake", False)
        left = action.get("left", False)
        right = action.get("right", False)

        # Combine steering and throttle
        if accel and brake:
            parts.append("Conflicting inputs")
        elif accel and left:
            parts.append("Accelerating left")
        elif accel and right:
            parts.append("Accelerating right")
        elif accel:
            parts.append("Full throttle")
        elif brake and left:
            parts.append("Braking while turning left")
        elif brake and right:
            parts.append("Braking while turning right")
        elif brake:
            parts.append("Hard braking")
        elif left and right:
            parts.append("Steering conflict")
        elif left:
            parts.append("Turning left")
        elif right:
            parts.append("Turning right")
        else:
            parts.append("Coasting")

        # ===== ROAD POSITION ANALYSIS =====
        offset = observation.car_offset_from_center
        on_road = observation.on_road

        if not on_road:
            if offset > 100:
                parts.append("Off road on right")
            elif offset < -100:
                parts.append("Off road on left")
            else:
                parts.append("Approaching edge")
        else:
            if offset > 60:
                parts.append("Hugging right edge")
            elif offset < -60:
                parts.append("Hugging left edge")
            elif offset > 30:
                parts.append("Drifting right")
            elif offset < -30:
                parts.append("Drifting left")
            else:
                parts.append("Centered on track")

        # ===== SPEED TREND ANALYSIS =====
        if self.last_obs:
            speed_delta = observation.speed - self.last_obs.speed
            if speed_delta > 0.05:
                parts.append("Accelerating")
            elif speed_delta < -0.05:
                parts.append("Decelerating")

        # ===== TRACK CURVATURE ANALYSIS =====
        try:
            curvature = getattr(observation.map_obj, "curvature", 0)
            if curvature is not None:
                if abs(curvature) < 0.02:
                    parts.append("On straight")
                elif abs(curvature) < 0.1:
                    parts.append("Gentle curve")
                elif abs(curvature) < 0.3:
                    parts.append("Curving")
                else:
                    parts.append("Sharp turn")
        except:
            pass

        # ===== SPEED VS CURVE SAFETY =====
        try:
            curvature = getattr(observation.map_obj, "curvature", 0)
            if curvature is not None and speed > 0.5:
                curve_severity = abs(curvature) * speed
                if curve_severity > 0.2:
                    parts.append("Too fast for turn")
                elif speed > 0.85 and curve_severity < 0.02:
                    parts.append("Room to accelerate")
                elif speed > 0.8 and curve_severity < 0.05:
                    parts.append("Pushing hard")
        except:
            pass

        # ===== RECOVERY ANALYSIS =====
        if self.last_obs and not self.last_obs.on_road and observation.on_road:
            parts.append("Recovering to track")

        # ===== STEERING CONSISTENCY =====
        if self.last_obs and self.frame_count > 5:
            prev_left = getattr(self, "_last_steer_left", False)
            prev_right = getattr(self, "_last_steer_right", False)
            if left and prev_left:
                parts.append("Holding left")
            elif right and prev_right:
                parts.append("Holding right")

        # Store current steering for next frame
        self._last_steer_left = left
        self._last_steer_right = right

        # Remove duplicates while preserving order
        seen = set()
        unique_parts = []
        for part in parts:
            if part not in seen:
                seen.add(part)
                unique_parts.append(part)

        return ", ".join(unique_parts)
