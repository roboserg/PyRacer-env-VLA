"""
VLA (Vision-Language-Action) Module for PyRacer
Provides modular game environment for human play, bot play, and data recording.
"""

__all__ = [
    "Observation",
    "Controller",
    "HumanController",
    "BotController",
    "Recorder",
    "GameEnvironment",
]


def __getattr__(name):
    if name == "Observation":
        from vla.observation import Observation

        return Observation
    if name == "Controller":
        from vla.controller import Controller

        return Controller
    if name == "HumanController":
        from vla.human_controller import HumanController

        return HumanController
    if name == "BotController":
        from vla.bot_controller import BotController

        return BotController
    if name == "Recorder":
        from vla.recorder import Recorder

        return Recorder
    if name == "GameEnvironment":
        from vla.environment import GameEnvironment

        return GameEnvironment
    raise AttributeError(f"module 'vla' has no attribute '{name}'")
