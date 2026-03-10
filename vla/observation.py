from dataclasses import dataclass
from typing import Any


@dataclass
class Observation:
    """Represents a single observation from the game environment."""

    frame: Any
    speed: float
    position: Any
    car_x: float
    car_y: float
    map_obj: Any
    on_road: bool
    car_offset_from_center: float
