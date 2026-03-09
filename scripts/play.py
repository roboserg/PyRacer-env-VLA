#!/usr/bin/env python3
"""
Play script - immediately starts racing with human input.
Run with: python3 play.py

Controls:
- UP arrow: Accelerate
- DOWN arrow: Brake
- LEFT arrow: Steer left
- RIGHT arrow: Steer right
- ESC: Quit
"""

from data.gameFiles.game import Game
import pygame


def play():
    # Initialize game
    g = Game()
    print("✓ Game initialized")
    print("✓ Racing game starting immediately - no menus!")
    print("\nControls:")
    print("  UP arrow:    Accelerate")
    print("  DOWN arrow:  Brake")
    print("  LEFT arrow:  Steer left")
    print("  RIGHT arrow: Steer right")
    print("  ESC:         Quit\n")

    # Skip menus and start playing
    g.playing = True
    g.reset()
    g.countdown = 0  # Skip countdown
    print("✓ Game reset - race started immediately!")
    print(f"✓ Initial speed: {g.map.car.speed:.2f}")

    iteration = 0
    speed_history = []  # Track speed over time for averaging

    # Main game loop with human input
    while g.playing:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                g.playing = False
                break
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    g.playing = False
                    break

        # Get delta time
        g.get_dt()

        # Handle keyboard input - map to game actions
        keys = pygame.key.get_pressed()
        g.actions["accel"] = keys[pygame.K_UP]
        g.actions["brake"] = keys[pygame.K_DOWN]
        g.actions["left"] = keys[pygame.K_LEFT]
        g.actions["right"] = keys[pygame.K_RIGHT]

        # Update and render
        g.update()
        g.render()

        # Track speed for averaging
        speed_history.append(g.map.car.speed)

        # Keep only last 3600 entries (60 seconds at 60 FPS)
        if len(speed_history) > 3600:
            speed_history.pop(0)

        # Print progress every 60 iterations (1 second at 60 FPS)
        if iteration % 60 == 0 and iteration > 0:
            fps = g.clock.get_fps()

            # Calculate average speeds
            avg_speed_overall = (
                sum(speed_history) / len(speed_history) if speed_history else 0
            )
            last_60_seconds = (
                speed_history[-3600:] if len(speed_history) >= 3600 else speed_history
            )
            avg_speed_60s = (
                sum(last_60_seconds) / len(last_60_seconds) if last_60_seconds else 0
            )

            print(
                f"  Iteration {iteration}: Speed={g.map.car.speed:.2f}, "
                f"Avg={avg_speed_overall:.2f}, Avg60s={avg_speed_60s:.2f}, "
                f"Time={g.total_time:.2f}s, FPS={fps:.1f}"
            )

        iteration += 1

        # Stop after if game marks as complete
        if g.complete:
            g.playing = False

    # Print final stats
    print(f"\n✓ Race completed!")
    print(f"✓ Total iterations: {iteration}")
    print(f"✓ Final speed: {g.map.car.speed:.2f}")
    if speed_history:
        avg_speed_overall = sum(speed_history) / len(speed_history)
        print(f"✓ Average speed (overall): {avg_speed_overall:.2f}")
    print(f"✓ Total time: {g.total_time:.2f}s")


if __name__ == "__main__":
    try:
        play()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
