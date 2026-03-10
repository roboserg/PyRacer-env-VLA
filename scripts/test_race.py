#!/usr/bin/env python3
"""
Test script to automatically start a race and drive the car forward using the Gym environment.
Run with: python3 test_race.py
"""

from vla import GameEnvironment, Agent, Observation
import pygame
import sys

class TestAgent(Agent):
    """
    A simple agent for testing that accelerates and steers left/right periodically.
    Replicates the original logic from the non-Gym test script.
    """

    def __init__(self, env=None):
        super().__init__(env)
        self.iteration = 0

    def predict(
        self,
        observation: Observation,
        state: Optional[Tuple[Any, ...]] = None,
        episode_start: Optional[Any] = None,
        deterministic: bool = False,
    ) -> Tuple[Dict[str, bool], Optional[Tuple[Any, ...]]]:
        self.iteration += 1

        # Automatically accelerate
        accel = True
        brake = False
        left = False
        right = False

        # Add some steering after we're moving
        if self.iteration > 60:  # After 1 second (approx at 60 FPS)
            if (self.iteration // 60) % 2 == 0:  # Alternate every second
                left = True
            else:
                right = True

        # Check for quit events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._should_quit = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._should_quit = True

        action_dict = {
            "accel": accel,
            "brake": brake,
            "left": left,
            "right": right,
        }
        
        # Encode to int
        action = 0
        if accel: action |= 1
        if brake: action |= 2
        if left: action |= 4
        if right: action |= 8
        
        return action, state

    def reset(self):
        super().reset()
        self.iteration = 0


def test_race():
    # Initialize environment with our test agent
    agent = TestAgent()
    env = GameEnvironment(agent=agent, recorder=None)
    print("✓ Gym Environment initialized with TestAgent")

    # Run the environment for 600 steps (~10 seconds at 60 FPS)
    print("✓ Starting race test (running for 600 steps)...")
    stats = env.run(max_steps=600, verbose=True)

    print(f"\n✓ Test completed!")
    print(f"✓ Total steps: {stats['steps']}")
    print(f"✓ Final speed: {stats['final_speed']:.2f}")
    print(f"✓ Total time: {stats['total_time']:.2f}s")
    print(f"✓ No crashes - Gym environment is stable!")

    env.close()


if __name__ == "__main__":
    try:
        test_race()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
