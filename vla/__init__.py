"""
VLA (Vision-Language-Action) Module for PyRacer
Provides modular game environment for human play, bot play, and data recording.
"""

__all__ = [
    "Observation",
    "Agent",
    "HumanAgent",
    "RandomAgent",
    "BotAgent",
    "VLAAgent",
    "Recorder",
    "GameEnvironment",
]


def __getattr__(name):
    if name == "Observation":
        from vla.env import Observation
        return Observation
    if name == "Agent":
        from vla.agents.agent import Agent
        return Agent
    if name == "HumanAgent":
        from vla.agents.human_agent import HumanAgent
        return HumanAgent
    if name == "RandomAgent":
        from vla.agents.random_agent import RandomAgent
        return RandomAgent
    if name == "BotAgent":
        from vla.agents.bot_agent import BotAgent
        return BotAgent
    if name == "VLAAgent":
        from vla.agents.vla_agent import VLAAgent
        return VLAAgent
    if name == "Recorder":
        from vla.recorder import Recorder
        return Recorder
    if name == "GameEnvironment":
        from vla.env import GameEnvironment
        return GameEnvironment
    raise AttributeError(f"module 'vla' has no attribute '{name}'")
