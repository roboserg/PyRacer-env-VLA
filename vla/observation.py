"""
Observation class that encapsulates game state and frame data.
Tracks state changes for text annotation generation.
"""

import pygame


class Observation:
    """Represents a single observation from the game environment."""

    def __init__(self, frame, speed, lap, lap_time, position, car_x, car_y, map_obj):
        """
        Args:
            frame: pygame.Surface of the game display
            speed: float, current car speed
            lap: int, current lap number
            lap_time: float, time in current lap
            position: tuple (x, y), car position relative to start
            car_x: float, car x-coordinate on screen
            car_y: float, car y-coordinate on screen
            map_obj: Map object with road/collision info
        """
        self.frame = frame  # pygame.Surface (480x270)
        self.speed = speed
        self.lap = lap
        self.lap_time = lap_time
        self.position = position
        self.car_x = car_x
        self.car_y = car_y
        self.map_obj = map_obj

        # Calculate derived state
        self._update_derived_state()

    def _update_derived_state(self):
        """Calculate derived state from raw values."""
        # Road center is roughly at x=240 (DISPLAY_W/2)
        self.car_offset_from_center = self.car_x - 240

        # Estimate if car is on road or off-road
        # This is approximate - a better approach would query the Map object
        self.on_road = abs(self.car_offset_from_center) < 80

    def to_dict(self):
        """Convert observation to dict for logging."""
        return {
            "speed": round(self.speed, 2),
            "lap": self.lap,
            "lap_time": round(self.lap_time, 2),
            "position": self.position,
            "car_x": round(self.car_x, 2),
            "car_y": round(self.car_y, 2),
            "on_road": self.on_road,
            "car_offset": round(self.car_offset_from_center, 2),
        }

    def state_changed(self, prev_obs):
        """Check if critical game state has changed significantly."""
        if prev_obs is None:
            return True

        # Consider state changed if:
        # - Speed changed by more than 0.5
        # - Lap changed
        # - On-road status changed
        # - Position changed significantly
        speed_delta = abs(self.speed - prev_obs.speed)
        lap_changed = self.lap != prev_obs.lap
        road_status_changed = self.on_road != prev_obs.on_road
        pos_delta = abs(self.car_offset_from_center - prev_obs.car_offset_from_center)

        return speed_delta > 0.5 or lap_changed or road_status_changed or pos_delta > 10
