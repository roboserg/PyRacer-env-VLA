#!/usr/bin/env python3
"""
Eval script. Runs a bot or trained VLA model on the game.

Usage:
    python scripts/eval.py                          # VLA agent, default model dir
    python scripts/eval.py --agent bot              # rule-based bot baseline
    python scripts/eval.py --model-dir models/foo   # specify model for VLA
    python scripts/eval.py --max-steps 1000
"""

import argparse
import pygame

from src.gym.env import GameEnvironment


def main():
    parser = argparse.ArgumentParser(description="Evaluate an agent in PyRacer")
    parser.add_argument("--agent", choices=["vla", "bot"], default="vla", help="Agent type")
    parser.add_argument("--model-dir", default="./models", help="Model directory (VLA only)")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum number of steps")
    args = parser.parse_args()

    pygame.display.init()
    pygame.font.init()

    if args.agent == "vla":
        from src.vla.model import get_model_and_processor
        from src.vla.vla_agent import VLAAgent
        model, processor = get_model_and_processor(args.model_dir)
        agent = VLAAgent(model=model, processor=processor)
    else:
        from src.gym.agents.bot_agent import BotAgent
        agent = BotAgent()

    env = GameEnvironment()

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
