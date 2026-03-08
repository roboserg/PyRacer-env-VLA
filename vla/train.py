import torch
import os
import random
from PIL import Image
from transformers import (
    TrainerCallback,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from tqdm import tqdm
from vla.dataset import RacingVLADataset
from vla.model import get_model_and_processor, infer, post_process_output

torch.set_float32_matmul_precision("high")

MODEL_DIR = "./models/2_new_gameplay"

def get_dataset(processor, tokenizer):
    """Identifies the latest recording and returns a RacingVLADataset."""
    # --- 1. IDENTIFY RECORDING ---
    recordings_dir = "vla/data/recordings"
    recordings = sorted([d for d in os.listdir(recordings_dir) if os.path.isdir(os.path.join(recordings_dir, d))])
    if not recordings:
        raise FileNotFoundError(f"No recordings found in {recordings_dir}")

    latest_recording = os.path.join(recordings_dir, recordings[-1])
    print(f"Using latest recording for training: {latest_recording}")
    jsonl_file = os.path.join(latest_recording, "metadata.jsonl")
    img_dir = os.path.join(latest_recording, "images")

    # --- 2. INITIALIZE DATASET ---
    dataset = RacingVLADataset(jsonl_file, img_dir, processor, tokenizer)

    # --- 3. ACTION DISTRIBUTION ANALYSIS ---
    dataset.print_stats()

    return dataset

def eval_model(model, processor, dataset, num_samples=10):
    """Run inference on random samples from dataset and calculate accuracy."""
    model.eval()
    indices = random.sample(range(len(dataset)), num_samples)
    correct = 0

    for i, idx in enumerate(tqdm(indices, desc="Evaluating")):
        item = dataset.data[idx]
        raw_image = Image.open(os.path.join(dataset.img_dir, item["frame"])).convert("RGB")
        response = infer(model, processor, raw_image)

        # Ground Truth Action [accel, brake, left, right]
        gt_action_vec = item["action"]
        
        # Predicted action
        pred_action = post_process_output(response)
        
        # Compare exactly
        match = (
            bool(gt_action_vec[0]) == pred_action["accel"] and
            bool(gt_action_vec[1]) == pred_action["brake"] and
            bool(gt_action_vec[2]) == pred_action["left"] and
            bool(gt_action_vec[3]) == pred_action["right"]
        )
        if match:
            correct += 1

        # Only print first 5 examples
        if i < 5:
            gt_str = f"<FWD_{int(gt_action_vec[0])}> <BRK_{int(gt_action_vec[1])}> <LFT_{int(gt_action_vec[2])}> <RGT_{int(gt_action_vec[3])}>"
            tqdm.write(f"\n--- Frame: {item['frame']} ---")
            tqdm.write(f"GT Action: {gt_str}")
            tqdm.write(f"Model Out: {response}")
            tqdm.write(f"Match: {'✓' if match else '✗'}")
        elif i == 5:
            tqdm.write(f"\n... (skipping detail for remaining {num_samples - 5} samples)")

    accuracy = (correct / num_samples) * 100
    print(f"\n>>> Evaluation Accuracy: {accuracy:.2f}% ({correct}/{num_samples})")
    model.train()

class InferenceCallback(TrainerCallback):
    """Custom callback to trigger inference every N steps."""
    def __init__(self, model, processor, dataset, every_n_steps=10, num_samples=20):
        self.model = model
        self.processor = processor
        self.dataset = dataset
        self.every_n_steps = every_n_steps
        self.num_samples = num_samples

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % self.every_n_steps == 0:
            print(f"\n=== MID-TRAIN INFERENCE (Step {state.global_step}) ===")
            eval_model(
                self.model, self.processor, self.dataset, num_samples=self.num_samples
            )

def main():
    # --- 1. SETUP & MODEL INITIALIZATION ---
    print(f"Training Model. Output Directory: {MODEL_DIR}")
    
    model, processor = get_model_and_processor(MODEL_DIR)
    tokenizer = processor.tokenizer

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")

    # --- 2. DATASET DEFINITION ---
    full_dataset = get_dataset(processor, tokenizer)

    # --- 3. FREEZE STRATEGY ---
    model.requires_grad_(True)
    # Freeze the Vision Backbone
    for param in model.model.vision_model.parameters():
        param.requires_grad = False
    # Unfreeze the last 12 blocks of the vision model
    for param in model.model.vision_model.encoder.layers[-12:].parameters():
        param.requires_grad = True

    print("Freeze Strategy: Vision=Frozen (last 12 layers un-frozen), Connector=Trainable, LLM=Trainable")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")

    # --- 4. TRAINING CONFIGURATION ---
    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        num_train_epochs=3,
        bf16=True,
        logging_first_step=True,
        logging_steps=2,
        save_strategy="no",
        optim="adamw_torch_fused",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=full_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        callbacks=[InferenceCallback(model, processor, full_dataset, every_n_steps=100, num_samples=50)],
    )

    # --- 5. EXECUTION ---
    print("Starting full VLA Training...")
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\n!!! Training interrupted by user (Ctrl+C) !!!")

    print("\n=== FINAL MODEL EVALUATION ===")
    eval_model(model, processor, full_dataset, num_samples=30)

    print("\n=== SAVE MODEL ===")
    trainer.save_model(MODEL_DIR)
    processor.save_pretrained(MODEL_DIR)
    print(f"Training Complete. Model saved to {MODEL_DIR}")

if __name__ == "__main__":
    main()
