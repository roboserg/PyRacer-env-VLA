#!/usr/bin/env python3
"""
Eval script. Runs a bot or trained VLA model on the game.

Usage:
    python scripts/eval.py                          # VLA agent, default model dir
    python scripts/eval.py --bot                    # rule-based bot baseline
    python scripts/eval.py --bot --record           # bot play + record dataset
    python scripts/eval.py --model-dir models/foo   # specify model for VLA
    python scripts/eval.py --max-steps 1000
    python scripts/eval.py --interval 6             # blocking VLA inference every N frames (default: threaded/non-blocking)
"""

import argparse
import json
import os
import time
import pygame

from src.gym.env import GameEnvironment
from src.gym.recorder import Recorder
from src.gym.agents.bot_agent import BotAgent


def main():
    parser = argparse.ArgumentParser(description="Evaluate an agent in PyRacer")
    parser.add_argument("--bot", action="store_true", help="Run rule-based bot instead of VLA model")
    parser.add_argument("--model-dir", default=None, help="Model directory (VLA only, defaults to latest run in ./models)")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum number of steps")
    parser.add_argument("--record", action="store_true", help="Record gameplay to dataset")
    parser.add_argument("--interval", type=int, default=None, help="Run VLA inference every N frames in blocking mode (default: non-blocking threaded inference)")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature for VLA inference (default: agent class default, e.g. 0.7 for CoT)")
    args = parser.parse_args()

    # Conditionally import VLA modules only if not using bot
    if not args.bot:
        from src.vla.model import get_model_and_processor
        from src.vla.vla_agent import AGENT_REGISTRY, TwoTokenVLAAgent

    if args.model_dir is None and not args.bot:
        runs = sorted(
            (d for d in os.listdir("./models") if os.path.isdir(os.path.join("./models", d))),
            reverse=True,
        )
        if not runs:
            raise FileNotFoundError("No model runs found in ./models")
        args.model_dir = os.path.join("./models", runs[0])
        print(f"Using latest model: {args.model_dir}")

    pygame.display.init()
    pygame.font.init()

    if not args.bot:
        config_path = os.path.join(args.model_dir, "vla_config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                agent_cls = AGENT_REGISTRY.get(json.load(f)["agent_class"], TwoTokenVLAAgent)
        else:
            agent_cls = TwoTokenVLAAgent

        model, processor = get_model_and_processor(args.model_dir, agent_cls)
        agent = agent_cls(model=model, processor=processor, temperature=args.temperature)
    else:
        agent = BotAgent()

    recorder = Recorder(enabled=True, suffix="bot" if args.bot else "vla") if args.record else None
    env = GameEnvironment(recorder=recorder)

    threaded = not args.bot and args.interval is None
    interval = args.interval or 1

    is_vla = not args.bot

    try:
        env.reset()
        action = 0
        step = 0
        inferring = False
        last_step_time = time.perf_counter()
        last_predict_time = time.perf_counter()
        game_fps = 0.0
        while True:
            if threaded:
                if inferring:
                    result = agent.poll_predict()
                    if result is not None:
                        action = result
                        inferring = False
                        now = time.perf_counter()
                        elapsed = now - last_predict_time
                        predict_fps = 1.0 / elapsed if step > 0 else 0.0
                        last_predict_time = now
                        a = action if isinstance(action, dict) else {}
                        print(
                            f"frame:{step:>6} | game_fps:{game_fps:>5.1f} | predict_fps:{predict_fps:>5.1f}"
                            f" | infer:{agent.last_inference_time_ms:>6.0f}ms"
                            f" | {repr(agent.last_output_text)}"
                            f" | A:{int(a.get('accel', False))} B:{int(a.get('brake', False))}"
                            f" L:{int(a.get('left', False))} R:{int(a.get('right', False))}"
                        )
                else:
                    agent.start_predict(env.current_observation)
                    inferring = True
            else:
                if step % interval == 0:
                    action = agent.predict(env.current_observation)
                    if is_vla:
                        a = action if isinstance(action, dict) else {}
                        print(
                            f"frame:{step:>6} | game_fps:{game_fps:>5.1f}"
                            f" | infer:{agent.last_inference_time_ms:>6.0f}ms"
                            f" | {repr(agent.last_output_text)}"
                            f" | A:{int(a.get('accel', False))} B:{int(a.get('brake', False))}"
                            f" L:{int(a.get('left', False))} R:{int(a.get('right', False))}"
                        )

            _, reward, terminated, truncated, info = env.step(action)
            now = time.perf_counter()
            game_fps = 1.0 / (now - last_step_time)
            last_step_time = now
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
