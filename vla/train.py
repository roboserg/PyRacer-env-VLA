import torch
import os
import random
import re
from PIL import Image
from torch.utils.data import Subset
from transformers import (
    AutoProcessor,
    SmolVLMForConditionalGeneration,
    TrainerCallback,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from vla.dataset import RacingVLADataset

torch.set_float32_matmul_precision("high")

# --- 1. SETUP & MODEL INITIALIZATION ---
# If not None, load model from this path for fine-tuning
FINE_TUNE_MODEL_PATH = "./models/smolvla-racer-final"
MODEL_ID = "HuggingFaceTB/SmolVLM-Instruct"

# Load Processor and Tokenizer
load_path = FINE_TUNE_MODEL_PATH if FINE_TUNE_MODEL_PATH else MODEL_ID
print(f"Loading processor and model from: {load_path}")

processor = AutoProcessor.from_pretrained(load_path)
tokenizer = processor.tokenizer

# Define and add action tokens (Must match dataset.py)
action_tokens = [
    "<FWD_0>",
    "<FWD_1>",
    "<LFT_0>",
    "<LFT_1>",
    "<RGT_0>",
    "<RGT_1>",
    "<BRK_0>",
    "<BRK_1>",
]
# add_tokens returns number of newly added tokens
num_added = tokenizer.add_tokens(action_tokens)
print(f"Added {num_added} new action tokens.")

# --- TOKEN VERIFICATION ---
for token in action_tokens:
    ids = tokenizer.encode(token, add_special_tokens=False)
    assert len(ids) == 1, (
        f"CRITICAL: Token {token} was split into {len(ids)} pieces: {ids}. This will break the VLA action head."
    )
print("Token Verification Passed: All actions are single tokens.")

# Load Model in BF16 for 4090 Performance
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SmolVLMForConditionalGeneration.from_pretrained(
    load_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",  # Fast Attention
).to(device)

# CRITICAL: Always resize to match current tokenizer length
model.resize_token_embeddings(len(tokenizer))

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params / 1e6:.2f}M")

# --- 2. DATASET DEFINITION ---
recordings_dir = "vla/data/recordings"
recordings = sorted(
    [
        d
        for d in os.listdir(recordings_dir)
        if os.path.isdir(os.path.join(recordings_dir, d))
    ]
)
if not recordings:
    raise FileNotFoundError(f"No recordings found in {recordings_dir}")

latest_recording = os.path.join(recordings_dir, recordings[-1])
print(f"Using latest recording for training: {latest_recording}")
jsonl_file = os.path.join(latest_recording, "metadata.jsonl")
img_dir = os.path.join(latest_recording, "images")

full_dataset = RacingVLADataset(
    jsonl_file,
    img_dir,
    processor,
    tokenizer,
)
print(f"Total dataset size: {len(full_dataset)} frames")

# --- 3. FREEZE STRATEGY ---
# Assertive unfreezing: Ensure everything is trainable first
model.requires_grad_(True)

# Freeze the Vision Backbone
for param in model.model.vision_model.parameters():
    param.requires_grad = False

# Unfreeze the last 6 blocks of the vision model for task-specific adaptation
for param in model.model.vision_model.encoder.layers[-6:].parameters():
    param.requires_grad = True

print("Freeze Strategy: Vision=Frozen, Connector=Trainable, LLM=Trainable")

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")


# --- 4. INFERENCE LOGIC & CALLBACK ---
def run_inference(model, processor, dataset, num_samples=3):
    """Run inference on random samples using the optimized 108-token path."""
    model.eval()
    indices = random.sample(range(len(dataset)), num_samples)

    for idx in indices:
        item = dataset.data[idx]
        raw_image = Image.open(os.path.join(dataset.img_dir, item["frame"])).convert(
            "RGB"
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Action:"},
                ],
            }
        ]
        prompt = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        if not prompt.endswith(" "):
            prompt += " "

        inputs = processor(
            text=prompt,
            images=raw_image,
            return_tensors="pt",
            do_resize=True,
            size={"longest_edge": 384},
        ).to(device)

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                temperature=0.0,
            )

        response = processor.decode(generated_ids[0], skip_special_tokens=True)
        action = item["action"]
        # recorder.py order: [0:accel, 1:brake, 2:left, 3:right]
        gt_str = f"<FWD_{int(action[0])}> <BRK_{int(action[1])}> <LFT_{int(action[2])}> <RGT_{int(action[3])}>"

        print(f"\n--- Frame: {item['frame']} ---")
        print(f"GT Action: {gt_str}")
        print(f"Model Out: {response}")

    model.train()


class InferenceCallback(TrainerCallback):
    """Custom callback to trigger inference every N steps."""

    def __init__(self, model, processor, dataset, every_n_steps=10):
        self.model = model
        self.processor = processor
        self.dataset = dataset
        self.every_n_steps = every_n_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % self.every_n_steps == 0:
            print(f"\n=== MID-TRAIN INFERENCE (Step {state.global_step}) ===")
            run_inference(self.model, self.processor, self.dataset, num_samples=5)


# --- 5. TRAINING CONFIGURATION ---
training_args = TrainingArguments(
    output_dir="./models/smolvla-racer-final",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    num_train_epochs=1,
    bf16=True,
    logging_first_step=True,
    logging_steps=2,
    save_strategy="no",
    save_steps=200,
    save_total_limit=None,
    optim="adamw_torch_fused",
    gradient_checkpointing=False,
    remove_unused_columns=False,
    report_to="none",
)



trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=full_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    callbacks=[InferenceCallback(model, processor, full_dataset, every_n_steps=30)],
)

# --- 6. EXECUTION ---
print("Starting full VLA Training...")
try:
    trainer.train()
except KeyboardInterrupt:
    print("\n\n!!! Training interrupted by user (Ctrl+C) !!!")
    print("Saving current progress and running final evaluation...")

print("\n=== FINAL MODEL EVALUATION ===")
run_inference(model, processor, full_dataset, num_samples=10)

print("\n=== SAVE MODEL ===")
trainer.save_model("./models/smolvla-racer-final")
processor.save_pretrained("./models/smolvla-racer-final")
print("Training Complete. Model saved to ./models/smolvla-racer-final")
