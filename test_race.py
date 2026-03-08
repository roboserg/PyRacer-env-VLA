#!/usr/bin/env python3
"""
Test script to automatically start a race and drive the car forward.
Run with: python3 test_race.py
"""

from data.gameFiles.game import Game
import pygame


def test_race():
    # Initialize game
    g = Game()
    print("✓ Game initialized")

    # Start the race (simulate clicking Start button)
    g.current_menu = g.main_menu
    g.actions["start"] = True
    g.current_menu.handle_input()
    g.playing = True
    print("✓ Race started")

    # Initialize game state
    g.reset()
    print("✓ Game reset - ready to race")
    print(f"✓ Countdown: {g.countdown}")

    # Run the game loop with automatic acceleration
    iteration = 0
    max_iterations = 600  # Run for ~10 seconds at 60 FPS

    while g.playing and iteration < max_iterations:
        # Handle events (for window close, etc.)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                g.playing = False
                break

        # Get delta time
        g.get_dt()

        # Skip countdown by setting it to 0 after first iteration
        if iteration == 0:
            g.countdown = 0
            print("✓ Skipping countdown - race started immediately")

        # Automatically accelerate
        g.actions["accel"] = True
        g.actions["left"] = False
        g.actions["right"] = False
        g.actions["brake"] = False

        # Add some steering after we're moving
        if iteration > 60:  # After 1 second
            if iteration % 120 < 60:
                g.actions["left"] = True
            else:
                g.actions["right"] = True

        # Update and render
        g.update()
        g.render()

        # Print progress every 60 iterations (1 second)
        if iteration % 60 == 0:
            fps = g.clock.get_fps()
            print(
                f"  Iteration {iteration}: Speed={g.map.car.speed:.2f}, "
                f"Lap={g.map.lap}, Time={g.lap_time:.2f}s, FPS={fps:.1f}"
            )

        iteration += 1

    print(f"\n✓ Test completed!")
    print(f"✓ Total iterations: {iteration}")
    print(f"✓ Final speed: {g.map.car.speed:.2f}")
    print(f"✓ Lap time: {g.lap_time:.2f}s")
    print(f"✓ Current lap: {g.map.lap}")
    print(f"✓ No crashes - game is stable!")


if __name__ == "__main__":
    try:
        test_race()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
