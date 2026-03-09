#!/usr/bin/env python3
"""
Example: Bot player (random agent).

Demonstrates how a bot can play the game using the VLA environment.
The bot makes random decisions from the action space.

Run with: python3 bot_play_random.py
"""

from vla.env import GameEnvironment
from vla.agents.random_agent import RandomAgent


def main():
    print("\n" + "=" * 60)
    print("PyRacer - Bot Play (Random Agent)")
    print("=" * 60 + "\n")

    # Create random bot controller
    controller = RandomAgent(seed=42)

    # Create environment (no recording for this example)
    env = GameEnvironment(controller, recorder=None)

    # Run for a session (e.g. 1000 steps)
    stats = env.run(max_steps=1000)

    print("✓ Bot race complete!")
    print(f"  Final speed: {stats['final_speed']:.2f}")
    print(f"  Steps: {stats['steps']}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n✗ Interrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
