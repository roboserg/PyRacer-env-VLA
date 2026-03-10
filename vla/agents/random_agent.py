"""
Random agent - selects actions uniformly at random from the discrete action space.
"""

import random
from typing import Optional, Tuple, Any
from vla.agents.agent import Agent
from vla.observation import Observation


class RandomAgent(Agent):
    """Agent that selects random actions from the 4-button action space."""

    # Maps action index to Discrete(16) encoding: accel=1, brake=2, left=4, right=8
    ACTION_MAPPING = [1, 2, 4, 8]

    def __init__(self, env: Optional[Any] = None, seed: Optional[int] = None):
        super().__init__(env)
        self.rng = random.Random(seed)

    def predict(
        self,
        observation: Observation,
        state: Optional[Tuple[Any, ...]] = None,
        episode_start: Optional[Any] = None,
        deterministic: bool = False,
    ) -> Tuple[int, Optional[Tuple[Any, ...]]]:
        action_idx = self.rng.randint(0, len(self.ACTION_MAPPING) - 1)
        return self.ACTION_MAPPING[action_idx], state
