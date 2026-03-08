import torch
from vla.dataset import RacingVLADataset
import os
import sys
from PIL import Image
from time import perf_counter
from vla.utils import load_model_and_processor, MODEL_DIR

# 1. Load Model & Processor
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    model, processor = load_model_and_processor(MODEL_DIR, device=device)
except FileNotFoundError as e:
    print(f"ERROR: {e}")
    sys.exit(1)

tokenizer = processor.tokenizer

# 2. Load latest recording
recordings_dir = "vla/data/recordings"
recordings = sorted([d for d in os.listdir(recordings_dir) if os.path.isdir(os.path.join(recordings_dir, d))])
if not recordings:
    raise FileNotFoundError(f"No recordings found in {recordings_dir}")

latest_recording = os.path.join(recordings_dir, recordings[-1])
jsonl_file = os.path.join(latest_recording, "metadata.jsonl")
img_dir = os.path.join(latest_recording, "images")

# 3. Load dataset
dataset = RacingVLADataset(jsonl_file, img_dir, processor, tokenizer, augment=False)
print(f"Loaded dataset with {len(dataset)} frames.")

# 4. Run inference on 10 random samples
num_samples = 10
indices = torch.randint(0, len(dataset), (num_samples,)).tolist()

print(f"Running inference on {num_samples} random samples...")

for i, idx in enumerate(indices):
    item = dataset.data[idx] 
    image_path = os.path.join(dataset.img_dir, item["frame"])
    raw_image = Image.open(image_path).convert("RGB")

    # Create the prompt
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

    inputs = processor(text=prompt, images=raw_image, return_tensors="pt", size={"longest_edge": 384}).to(device)

    if i == 0:
        print(f"Total tokens for one frame: {inputs.input_ids.shape[1]}")
        # Warmup run
        _ = model.generate(**inputs, max_new_tokens=5, use_cache=True)

    # Actual timed run
    start = perf_counter()
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=30, use_cache=True)
    end = perf_counter()

    response = processor.decode(generated_ids[0], skip_special_tokens=True)
    
    # Ground Truth Action
    action = item["action"]
    # recorder.py order: [0:accel, 1:brake, 2:left, 3:right]
    gt_str = f"<FWD_{int(action[0])}> <BRK_{int(action[1])}> <LFT_{int(action[2])}> <RGT_{int(action[3])}>"

    print(f"\n--- [{i+1}/{num_samples}] Testing Frame: {item['frame']} ---")
    print(f"Inference time: {end - start:.3f}s")
    print(f"GT Action: {gt_str}")
    print(f"Model Out: {response}")
