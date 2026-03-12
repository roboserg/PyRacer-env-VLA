"""
Random agent - selects actions uniformly at random from the discrete action space.
"""

import random
from typing import Optional, Any
from src.gym.agents.agent import Agent
from src.gym.observation import Observation


class RandomAgent(Agent):
    """Agent that selects random actions from the full Discrete(16) action space."""

    def __init__(self, env: Optional[Any] = None, seed: Optional[int] = None):
        super().__init__(env)
        self.rng = random.Random(seed)

    def predict(self, observation: Observation) -> int:
        return self.rng.randint(0, 15)
