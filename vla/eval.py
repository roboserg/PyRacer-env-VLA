#!/usr/bin/env python3
"""
Eval script - runs the game with a trained VLA model.
Run with: python vla/eval.py

This loads the model from ./smolvla-racer-final and uses VLAController
to play the game autonomously.
"""

import torch
import sys
import pygame
from vla.vla_controller import VLAController
from vla.environment import GameEnvironment
from vla.utils import load_model_and_processor, MODEL_DIR

pygame.init()
pygame.mixer.init()

def eval():
    """Run evaluation with the trained VLA model."""
    print("=" * 60)
    print("PyRacer VLA Evaluation")
    print("=" * 60)
    print(f"Model directory: {MODEL_DIR}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        model, processor = load_model_and_processor(MODEL_DIR, device=device)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Please train the model first with: python vla/train.py")
        sys.exit(1)

    controller = VLAController(
        model=model, processor=processor, device=device, max_length=128
    )

    print(f"\nStarting game with {controller.__class__.__name__}...")
    print("Press ESC to quit early\n")

    env = GameEnvironment(controller=controller, recorder=None)

    stats = env.run(max_laps=2, verbose=True)

    print("\nEvaluation complete!")
    return stats


if __name__ == "__main__":
    try:
        eval()
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
