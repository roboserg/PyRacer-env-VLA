#!/usr/bin/env python3
"""
Example: Human player with data recording.

Records frames and player actions to a dataset directory.
This is useful for collecting VLA (Vision-Language-Action) training data.

Run with: python3 human_play_record.py

Controls:
- UP arrow:    Accelerate
- DOWN arrow:  Brake
- LEFT arrow:  Steer left
- RIGHT arrow: Steer right
- ESC:         Quit
"""

from vla.env import GameEnvironment
from vla.agents.human_agent import HumanAgent
from vla.recorder import Recorder


def main():
    print("\n" + "=" * 60)
    print("PyRacer - Human Play with Data Recording")
    print("=" * 60)
    print("\nControls:")
    print("  UP arrow:    Accelerate")
    print("  DOWN arrow:  Brake")
    print("  LEFT arrow:  Steer left")
    print("  RIGHT arrow: Steer right")
    print("  ESC:         Quit\n")

    # Create recorder (saves to /vla/data/ by default)
    recorder = Recorder(enabled=True)

    # Create human agent
    agent = HumanAgent()

    # Create environment
    env = GameEnvironment(agent, recorder)

    # Run for a session (stop with ESC)
    stats = env.run()

    print("\n✓ Session complete!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n✗ Interrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
