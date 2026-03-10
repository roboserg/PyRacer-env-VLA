#!/usr/bin/env python3
"""
Play script - run the game with human keyboard control.
Run with: python vla/play.py
"""

import argparse
import pygame

from vla.agents.human_agent import HumanAgent
from vla.env import GameEnvironment


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Play PyRacer")
    parser.add_argument(
        "--max-steps", type=int, default=5000, help="Maximum number of steps to play"
    )
    return parser.parse_args()


def play(env: GameEnvironment, max_steps: int = 5000):
    """Play with human keyboard input using Gym interface."""
    print("=" * 60)
    print("PyRacer - Human Play Mode")
    print("=" * 60)
    print("Controls: Arrow keys (UP=accel, DOWN=brake, LEFT/RIGHT=steer)")
    print("Press ESC to quit early\n")

    obs, info = env.reset()
    total_reward = 0
    step_count = 0

    running = True
    while running:
        # Get human action from keyboard
        keys = pygame.key.get_pressed()

        # Map arrow keys to discrete action (0-15)
        accel = 1 if keys[pygame.K_UP] else 0
        brake = 2 if keys[pygame.K_DOWN] else 0
        left = 4 if keys[pygame.K_LEFT] else 0
        right = 8 if keys[pygame.K_RIGHT] else 0
        action = accel + brake + left + right

        # Check for quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1

        # Render
        env.render()

        if step_count >= max_steps:
            print(f"\nMax steps reached ({max_steps})")
            break

    print(f"\nGame Over!")
    print(f"Total steps: {step_count}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Average reward: {total_reward / max(step_count, 1):.2f}")
    print(f"Final speed: {info['speed']:.2f}")


def main():
    args = parse_args()

    pygame.display.init()
    pygame.font.init()

    controller = HumanAgent()
    env = GameEnvironment(controller=controller, recorder=None)

    try:
        play(env, args.max_steps)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
    finally:
        env.close()


if __name__ == "__main__":
    main()
