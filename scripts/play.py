#!/usr/bin/env python3
"""
Human play script. Optionally records gameplay to a dataset.

Usage:
    python scripts/play.py                # human play
    python scripts/play.py --record       # human play + recording
    python scripts/play.py --max-steps N  # limit steps
"""

import argparse
import pygame

from src.gym.agents.human_agent import HumanAgent
from src.gym.env import GameEnvironment
from src.gym.recorder import Recorder


def main():
    parser = argparse.ArgumentParser(description="Play PyRacer as a human")
    parser.add_argument("--record", action="store_true", help="Record gameplay to dataset")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum number of steps")
    args = parser.parse_args()

    pygame.display.init()
    pygame.font.init()

    agent = HumanAgent()
    recorder = Recorder(enabled=True) if args.record else None
    env = GameEnvironment(recorder=recorder)

    try:
        env.reset()
        state = None
        step = 0
        while True:
            action, state = agent.predict(env.current_observation, state=state)
            _, reward, terminated, truncated, info = env.step(action)
            step += 1
            if terminated or truncated or info["should_quit"]:
                break
            if args.max_steps and step >= args.max_steps:
                break
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        env.close()


if __name__ == "__main__":
    main()
