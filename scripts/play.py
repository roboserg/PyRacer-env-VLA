#!/usr/bin/env python3
"""
Play script - immediately starts racing with human input using Gym environment.
Run with: python scripts/play.py

Controls:
- UP arrow: Accelerate
- DOWN arrow: Brake
- LEFT arrow: Steer left
- RIGHT arrow: Steer right
- ESC: Quit
"""

import pygame
import argparse

from vla.agents.human_agent import HumanAgent
from vla.env import GameEnvironment


def play(env: GameEnvironment, max_steps: int = None):
    print("PyRacer - Human Play Mode (Gym Environment)")
    print("Controls: Arrow keys (UP=accel, DOWN=brake, LEFT/RIGHT=steer)")
    print("Press ESC to quit early\n")

    obs, info = env.reset()
    total_reward = 0
    step_count = 0
    speed_history = []

    running = True
    while running:
        keys = pygame.key.get_pressed()

        accel = 1 if keys[pygame.K_UP] else 0
        brake = 2 if keys[pygame.K_DOWN] else 0
        left = 4 if keys[pygame.K_LEFT] else 0
        right = 8 if keys[pygame.K_RIGHT] else 0
        action = accel + brake + left + right

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        speed_history.append(info["speed"])

        if terminated or truncated:
            break

        if max_steps and step_count >= max_steps:
            print(f"Max steps reached ({max_steps})")
            break

        if step_count % 60 == 0 and step_count > 0:
            avg_speed = sum(speed_history) / len(speed_history)
            print(
                f"  Step {step_count}: Speed={info['speed']:.2f}, "
                f"Avg={avg_speed:.2f}, Time={env.game.total_time:.2f}s"
            )

    print(f"\nRace completed!")
    print(f"Total steps: {step_count}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Average speed: {total_reward / max(step_count, 1):.2f}")
    print(f"Final speed: {info['speed']:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Play PyRacer")
    parser.add_argument(
        "--max-steps", type=int, default=None, help="Maximum number of steps"
    )
    args = parser.parse_args()

    pygame.display.init()
    pygame.font.init()

    controller = HumanAgent()
    env = GameEnvironment(controller=controller, recorder=None)

    try:
        play(env, args.max_steps)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        env.close()


if __name__ == "__main__":
    main()
