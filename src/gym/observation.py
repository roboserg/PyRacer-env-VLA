from dataclasses import dataclass
from typing import Any
from PIL import Image


@dataclass
class Observation:
    """Represents a single observation from the game environment."""

    frame: Image.Image
    speed: float
    position: Any
    car_x: float
    car_y: float
    map_obj: Any
    on_road: bool
    car_offset_from_center: float
