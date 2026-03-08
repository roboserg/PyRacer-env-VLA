#!/usr/bin/env python3
"""
Example: Test recording with short bot session.

Quick test to verify recording functionality works correctly.
Records ~5 seconds of gameplay with bot player.

Run with: python3 test_recording.py
"""

from vla.environment import GameEnvironment
from vla.bot_controller import BotController
from vla.recorder import Recorder


def main():
    print("\n" + "=" * 60)
    print("PyRacer - Test Recording (Bot, 5 seconds)")
    print("=" * 60 + "\n")

    # Create recorder (saves to /vla/data/ by default)
    recorder = Recorder(enabled=True)

    # Create random bot
    controller = BotController(seed=42)

    # Create environment
    env = GameEnvironment(controller, recorder)

    # Run for short duration (300 frames = 5 seconds at 60 FPS)
    stats = env.run(max_steps=300, verbose=True)

    print("\n" + "=" * 60)
    print("Recording Test Complete!")
    print("=" * 60)
    print(f"Dataset saved to: /vla/data/")
    print(f"Total frames recorded: {stats['recorded_frames']}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n✗ Interrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
