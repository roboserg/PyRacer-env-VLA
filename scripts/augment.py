#!/usr/bin/env python3
"""
Augment a recorded dataset with varied chain-of-thought annotations.

Uses a VLM to look at each frame image alongside the original synthetic
annotation and action, then generate a natural, varied driving thought
suitable for CoT training.

Usage:
    python scripts/augment.py                                    # augment latest recording
    python scripts/augment.py --recording 20260311_212337-bot    # specific recording
    python scripts/augment.py --limit 20                         # test on first 20 frames
    python scripts/augment.py --model smolvlm                    # use SmolVLM instead of Qwen
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

DATASET_DIR = "recordings"

MODELS = {
    "qwen": "Qwen/Qwen3-VL-8B-Instruct",
    "qwen4b": "Qwen/Qwen3-VL-4B-Instruct",
    "smolvlm": "HuggingFaceTB/SmolVLM-Instruct",
}

ACTION_LABELS = {
    (True, False, False, False): "accelerating straight",
    (True, False, True, False):  "accelerating and steering left",
    (True, False, False, True):  "accelerating and steering right",
    (False, True, False, False): "braking",
    (False, True, True, False):  "braking and steering left",
    (False, True, False, True):  "braking and steering right",
    (False, False, True, False): "coasting and steering left",
    (False, False, False, True): "coasting and steering right",
    (False, False, False, False): "coasting with no input",
    (True, False, True, True):   "accelerating with conflicting steering",
    (True, True, False, False):  "accelerating and braking simultaneously",
}


def action_to_text(action_vec):
    key = (action_vec[0] > 0.5, action_vec[1] > 0.5,
           action_vec[2] > 0.5, action_vec[3] > 0.5)
    return ACTION_LABELS.get(key, "unknown input")


FEW_SHOT_EXAMPLES = [
    ("Max speed, Full throttle, Centered on track, On straight",
     "accelerating straight",
     "Road is straight and clear ahead, centered in my lane at top speed, keep the throttle pinned"),
    ("High speed, Accelerating left, Drifting right, Sharp turn",
     "accelerating and steering left",
     "Sharp left bend coming up fast, car drifting toward the right edge, cutting left hard while keeping speed"),
    ("Good speed, Full throttle, Centered on track, Gentle curve",
     "accelerating straight",
     "Gentle curve ahead but well centered, plenty of room to stay on throttle through this one"),
    ("Max speed, Accelerating right, Off road on left, Sharp turn, Too fast for turn",
     "accelerating and steering right",
     "Overshot the corner and slid off the left side, steering right to recover back onto the track"),
    ("Barely moving, Accelerating right, Centered on track, On straight",
     "accelerating and steering right",
     "Just starting out barely rolling, road looks straight, pushing throttle and drifting right toward the racing line"),
    ("Crawling, Full throttle, Centered on track, Gentle curve",
     "accelerating straight",
     "Still building speed from a slow start, gentle bend ahead but no need to steer yet at this low pace"),
    ("Max speed, Accelerating left, Approaching edge, Sharp turn, Too fast for turn",
     "accelerating and steering left",
     "Flying into a sharp left turn way too fast, approaching the right edge, have to steer hard and hope I hold the road"),
    ("High speed, Accelerating left, Centered on track, Curving, Too fast for turn",
     "accelerating and steering left",
     "Road curving left at high speed, still centered for now but this speed is risky, steering into the curve"),
]

FEW_SHOT_BLOCK = "\n".join(
    f"State: {s}\nAction: {a}\nThought: {t}\n"
    for s, a, t in FEW_SHOT_EXAMPLES
)

SYSTEM_PROMPT = (
    "You narrate a racing game driver's inner monologue. "
    "Given a screenshot, state description, and chosen action, write what the driver "
    "observes and thinks. Describe what you SEE: road shape ahead, car position on "
    "track, speed sensation, upcoming curves. Write exactly 1-2 sentences, 10-25 words. "
    "Never mention game mechanics, UI elements, or anything not visible on the road."
)


def load_model(model_key):
    model_id = MODELS[model_key]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_id} on {device}...")

    if model_key in ("qwen", "qwen4b"):
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        processor = AutoProcessor.from_pretrained(model_id)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        from transformers import AutoProcessor, SmolVLMForConditionalGeneration
        processor = AutoProcessor.from_pretrained(model_id)
        model = SmolVLMForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to(device)

    model.eval()
    return model, processor, model_key


def generate_thought(model, processor, model_key, image, item):
    """Ask the VLM to produce a varied driving thought given the frame context."""
    action_vec = item["action"]
    action_text = action_to_text(action_vec)
    speed_pct = int(item.get("speed", 0) * 100)
    synthetic_text = item["text"]

    # Build rich state line from raw fields when available
    state_parts = [synthetic_text]
    if "car_offset" in item:
        offset = item["car_offset"]
        direction = "right" if offset > 0 else "left"
        state_parts.append(f"offset {abs(offset):.0f}px {direction} of center")
    if "on_road" in item:
        state_parts.append("on road" if item["on_road"] else "OFF ROAD")
    if item.get("curvature") is not None:
        state_parts.append(f"curvature {item['curvature']:.3f}")

    state_line = " | ".join(state_parts)

    user_text = (
        f"Examples:\n{FEW_SHOT_BLOCK}\n"
        f"Now write a thought for this frame:\n"
        f"State: {state_line}\n"
        f"Action: {action_text}\n"
        f"Speed: {speed_pct}%\n"
        f"Thought:"
    )

    if model_key in ("qwen", "qwen4b"):
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_text},
                ],
            },
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": SYSTEM_PROMPT + "\n\n" + user_text},
                ],
            },
        ]
        chat_text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = processor(
            text=chat_text, images=image, return_tensors="pt",
            do_resize=True, size={"longest_edge": 384},
        ).to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.9,
            top_p=0.92,
            repetition_penalty=1.2,
            use_cache=True,
        )

    # Decode only the new tokens
    new_ids = generated_ids[0][inputs["input_ids"].shape[1]:]
    response = processor.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    # Clean up
    response = response.replace('"', '').replace("'", "")
    # Stop at boundaries if the model generates another example
    for stop in ["State:", "Action:", "Now ", "\n\n"]:
        if stop in response:
            response = response[:response.index(stop)]
    # Keep up to 2 sentences
    sentences = response.split(".")
    if len(sentences) > 3:
        response = ".".join(sentences[:2]).strip()
    response = response.strip().rstrip(".")
    # Truncate if too long
    words = response.split()
    if len(words) > 30:
        response = " ".join(words[:30])
    if not response:
        response = synthetic_text  # fallback to original

    return response


def main():
    parser = argparse.ArgumentParser(description="Augment dataset with VLM-generated CoT annotations")
    parser.add_argument("--recording", default=None, help="Recording directory name (default: latest)")
    parser.add_argument("--limit", type=int, default=None, help="Only process first N frames (for testing)")
    parser.add_argument("--suffix", default="cot", help="Suffix for output directory (default: cot)")
    parser.add_argument("--model", default="qwen", choices=list(MODELS.keys()),
                        help="VLM to use for augmentation (default: qwen = Qwen3-VL-8B-Instruct)")
    args = parser.parse_args()

    # Resolve source recording
    if args.recording:
        src_dir = os.path.join(DATASET_DIR, args.recording)
    else:
        recordings = sorted(
            d for d in os.listdir(DATASET_DIR)
            if os.path.isdir(os.path.join(DATASET_DIR, d))
        )
        if not recordings:
            raise FileNotFoundError(f"No recordings found in {DATASET_DIR}")
        src_dir = os.path.join(DATASET_DIR, recordings[-1])

    print(f"Source recording: {src_dir}")

    jsonl_path = os.path.join(src_dir, "metadata.jsonl")
    img_dir = os.path.join(src_dir, "images")

    data = [json.loads(line) for line in open(jsonl_path)]
    print(f"Loaded {len(data)} frames")

    if args.limit:
        data = data[:args.limit]
        print(f"Limited to {len(data)} frames")

    # Create output directory
    src_name = os.path.basename(src_dir)
    out_name = f"{src_name}-{args.suffix}"
    out_dir = os.path.join(DATASET_DIR, out_name)
    out_img_dir = os.path.join(out_dir, "images")

    # Set up output: symlink images to avoid copying
    if os.path.exists(out_img_dir):
        if os.path.islink(out_img_dir):
            os.unlink(out_img_dir)
        else:
            import shutil
            shutil.rmtree(out_img_dir)
    os.makedirs(out_dir, exist_ok=True)
    os.symlink(os.path.abspath(img_dir), out_img_dir)
    print(f"Output directory: {out_dir}")
    print(f"Symlinked images from source")

    # Load model
    model, processor, model_key = load_model(args.model)

    # Process each frame
    augmented_data = []
    start_time = time.time()

    for i, item in enumerate(tqdm(data, desc="Augmenting")):
        image = Image.open(os.path.join(img_dir, item["frame"])).convert("RGB")

        thought = generate_thought(model, processor, model_key, image, item)

        new_item = dict(item)
        new_item["text"] = thought
        new_item["original_text"] = item["text"]
        augmented_data.append(new_item)

        if i < 10 or (i + 1) % 50 == 0:
            tqdm.write(f"  [{i}] original: {item['text']}")
            tqdm.write(f"  [{i}]  augment: {thought}")

    elapsed = time.time() - start_time
    fps = len(data) / elapsed
    print(f"\nProcessed {len(data)} frames in {elapsed:.1f}s ({fps:.1f} frames/s)")

    # Save augmented metadata
    out_jsonl = os.path.join(out_dir, "metadata.jsonl")
    with open(out_jsonl, "w") as f:
        for entry in augmented_data:
            json.dump(entry, f)
            f.write("\n")

    print(f"Saved augmented dataset to {out_jsonl}")

    # Show diversity stats
    from collections import Counter
    texts = [e["text"] for e in augmented_data]
    unique = len(set(texts))
    print(f"\nDiversity: {unique}/{len(texts)} unique texts "
          f"(was {len(set(e['original_text'] for e in augmented_data))}/{len(texts)})")


if __name__ == "__main__":
    main()
