#!/usr/bin/env python3
"""
Benchmark model inference speed for different max_new_tokens values.

Usage:
    python scripts/bench.py
    python scripts/bench.py --model-dir models/2_new_gameplay
    python scripts/bench.py --runs 20
"""

import argparse
import time
import torch
from PIL import Image
from src.vla.model import get_model_and_processor

DEFAULT_MODEL_DIR = "models/2_new_gameplay"
SAMPLE_IMAGE = "recordings/20260311_212337-bot/images/frame_00006.png"
MAX_TOKENS_VALUES = [2, 4, 8]


def bench_infer(model, processor, image, max_new_tokens, runs):
    device = model.device

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Action:"},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    if not prompt.endswith(" "):
        prompt += " "

    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
        do_resize=True,
        size={"longest_edge": 384},
    ).to(device)

    # Warmup
    with torch.inference_mode():
        model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)

    times = []
    for _ in range(runs):
        start = time.perf_counter()
        with torch.inference_mode():
            model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
        times.append((time.perf_counter() - start) * 1000)

    return times


def main():
    parser = argparse.ArgumentParser(description="Benchmark VLA inference speed")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--image", default=SAMPLE_IMAGE)
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()

    model, processor = get_model_and_processor(args.model_dir)
    image = Image.open(args.image).convert("RGB")

    print(f"\nBenchmarking {args.runs} runs per setting on {args.image}\n")
    print(f"{'max_new_tokens':>15} {'avg':>10} {'min':>10} {'max':>10} {'fps':>8}")
    print("-" * 58)

    for max_tokens in MAX_TOKENS_VALUES:
        times = bench_infer(model, processor, image, max_tokens, args.runs)
        avg = sum(times) / len(times)
        fps = 1000 / avg
        print(f"{max_tokens:>15} {avg:>9.1f}ms {min(times):>9.1f}ms {max(times):>9.1f}ms {fps:>7.1f}")


if __name__ == "__main__":
    main()
