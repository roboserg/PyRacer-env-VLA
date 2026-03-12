import torch
import os
import json
import random
import argparse
from datetime import datetime
from PIL import Image
from transformers import (
    TrainerCallback,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from tqdm import tqdm
from src.vla.vla_agent import TwoTokenVLAAgent, CoTVLAAgent, AGENT_REGISTRY
from src.vla.dataset import VLADataset
from src.vla.model import get_model_and_processor, run_inference

torch.set_float32_matmul_precision("high")


class TrainingConfig:
    models_root = "./models"
    run_name = "cot_action_loss"
    dataset_dir = "recordings"
    agent_cls = "CoTVLAAgent"
    num_epochs = 5
    learning_rate = 5e-5
    eval_steps = 100
    action_loss_weight = 5.0   # multiplier for throttle + steer tokens
    thought_loss_weight = 0.1  # downweight thought text to avoid memorizing dataset reasoning


class WeightedActionTrainer(Trainer):
    def __init__(self, *args, action_weight=5.0, thought_weight=0.2, num_action_tokens=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_weight = action_weight
        self.thought_weight = thought_weight
        self.num_action_tokens = num_action_tokens

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)

        # Per-token cross-entropy (causal LM shift: predict next token)
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        per_token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.shape)

        # Structural + action tokens: up-weighted; thought *content*: de-emphasised
        # Layout: [MASKED] <thought> ...content... </thought> <THROTTLE_X> <STEER_X> <EOS>
        weights = torch.ones_like(shift_labels, dtype=torch.float32)
        for b in range(shift_labels.shape[0]):
            unmasked = (shift_labels[b] != -100).nonzero(as_tuple=True)[0]
            if len(unmasked) > self.num_action_tokens + 3:
                # Fine-grained: <thought> and </thought> get action_weight,
                # thought content gets thought_weight, actions + EOS get action_weight
                weights[b, unmasked[0]] = self.action_weight           # <thought>
                weights[b, unmasked[1:-4]] = self.thought_weight       # thought content
                weights[b, unmasked[-4]] = self.action_weight          # </thought>
                weights[b, unmasked[-3:-1]] = self.action_weight       # throttle + steer
                weights[b, unmasked[-1]] = self.action_weight          # EOS
            elif len(unmasked) > self.num_action_tokens:
                # Fallback: not enough tokens for structural split
                action_pos = unmasked[-(self.num_action_tokens + 1):-1]
                thought_pos = unmasked[:-(self.num_action_tokens + 1)]
                weights[b, action_pos] = self.action_weight
                weights[b, thought_pos] = self.thought_weight

        total = (shift_labels != -100).float().sum()
        loss = (per_token_loss * weights).sum() / total

        return (loss, outputs) if return_outputs else loss


def _resolve_run_dir() -> str:
    existing = [d for d in os.listdir(TrainingConfig.models_root) if os.path.isdir(os.path.join(TrainingConfig.models_root, d))] if os.path.exists(TrainingConfig.models_root) else []
    index = len(existing) + 1
    suffix = TrainingConfig.run_name if TrainingConfig.run_name else datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(TrainingConfig.models_root, f"{index:03d}_{suffix}")

def _find_run_dir(run_name: str) -> str:
    """Find an existing run directory by exact name or suffix match (e.g. '001_cot' or 'cot')."""
    if not os.path.exists(TrainingConfig.models_root):
        raise FileNotFoundError(f"Models root {TrainingConfig.models_root!r} does not exist.")
    candidates = [
        d for d in os.listdir(TrainingConfig.models_root)
        if os.path.isdir(os.path.join(TrainingConfig.models_root, d))
        and (d == run_name or d.endswith(f"_{run_name}"))
    ]
    if not candidates:
        raise FileNotFoundError(f"No run matching {run_name!r} found in {TrainingConfig.models_root}.")
    if len(candidates) > 1:
        raise ValueError(f"Multiple runs match {run_name!r}: {candidates}. Use the full directory name.")
    return os.path.join(TrainingConfig.models_root, candidates[0])

def get_dataset(processor, tokenizer, agent_cls=None):
    """Identifies the latest recording and returns a VLADataset."""
    recordings = sorted([d for d in os.listdir(TrainingConfig.dataset_dir) if os.path.isdir(os.path.join(TrainingConfig.dataset_dir, d))])
    if not recordings:
        raise FileNotFoundError(f"No recordings found in {TrainingConfig.dataset_dir}")

    latest_recording = os.path.join(TrainingConfig.dataset_dir, recordings[-1])
    print(f"Using latest recording for dataset: {latest_recording}")
    jsonl_file = os.path.join(latest_recording, "metadata.jsonl")
    img_dir = os.path.join(latest_recording, "images")

    dataset = VLADataset(jsonl_file, img_dir, processor, tokenizer, agent_cls=agent_cls)
    dataset.print_stats()
    dataset.debug_sample()

    return dataset

def eval_model(agent, dataset, num_samples=10):
    """Run inference on random samples from dataset and calculate accuracy."""
    agent_cls = type(agent)
    agent.model.eval()
    num_samples = min(num_samples, len(dataset))
    indices = random.sample(range(len(dataset)), num_samples)
    correct = 0

    for i, idx in enumerate(tqdm(indices, desc="Evaluating")):
        item = dataset.data[idx]
        raw_image = Image.open(os.path.join(dataset.img_dir, item["frame"])).convert("RGB")
        response, _ = run_inference(
            agent.model, agent.processor, raw_image,
            agent_cls.build_inference_prompt(), agent_cls.max_new_tokens,
            temperature=agent.temperature,
        )

        gt_action_vec = item["action"]
        pred_action = agent_cls.decode_action(response)

        match = (
            bool(gt_action_vec[0]) == pred_action["accel"] and
            bool(gt_action_vec[1]) == pred_action["brake"] and
            bool(gt_action_vec[2]) == pred_action["left"] and
            bool(gt_action_vec[3]) == pred_action["right"]
        )
        if match:
            correct += 1

        if i < 5:
            gt_str = agent_cls.encode_for_stats(gt_action_vec, item)
            pred_str = agent_cls.encode_for_stats(
                [pred_action["accel"], pred_action["brake"], pred_action["left"], pred_action["right"]]
            )
            gt_raw = agent_cls.encode_action(gt_action_vec, item)
            tqdm.write(f"\n--- Frame: {item['frame']} ---")
            tqdm.write(f"GT:          {gt_raw}")
            tqdm.write(f"Model Out:   {response}")
            tqdm.write(f"GT Action:   {gt_str}")
            tqdm.write(f"Pred Action: {pred_str}")
            tqdm.write(f"Match: {'✓' if match else '✗'}")
        elif i == 5:
            tqdm.write(f"\n... (skipping detail for remaining {num_samples - 5} samples)")

    accuracy = (correct / num_samples) * 100
    print(f"\n>>> Evaluation Accuracy: {accuracy:.2f}% ({correct}/{num_samples})")
    return accuracy

class InferenceCallback(TrainerCallback):
    """Custom callback to trigger inference every N steps."""
    def __init__(self, agent, dataset, every_n_steps=10, num_samples=20):
        self.agent = agent
        self.dataset = dataset
        self.every_n_steps = every_n_steps
        self.num_samples = num_samples

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % self.every_n_steps == 0:
            print(f"\n=== MID-TRAIN INFERENCE (Step {state.global_step}) ===")
            eval_model(self.agent, self.dataset, num_samples=self.num_samples)
            self.agent.model.train()

def main():
    parser = argparse.ArgumentParser(description="Train or evaluate the VLA model")
    parser.add_argument("--eval", action="store_true", help="Only evaluate the model")
    parser.add_argument("--samples", type=int, default=50, help="Number of samples for evaluation")
    parser.add_argument("--agent-cls", default=TrainingConfig.agent_cls,
                        choices=list(AGENT_REGISTRY.keys()),
                        help="Agent class to train (default: CoTVLAAgent)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Sampling temperature for mid-train eval inference (default: agent class default, e.g. 0.7 for CoT)")
    parser.add_argument("--run", metavar="RUN_NAME",
                        help="Run name: resumes if an existing run matches, otherwise creates a new run with this name")
    args = parser.parse_args()

    # --- 1. SETUP & MODEL INITIALIZATION ---
    # Try to find an existing run to resume; if not found, create a new one
    model_dir = None
    if args.run:
        try:
            model_dir = _find_run_dir(args.run)
            config_path = os.path.join(model_dir, "vla_config.json")
            if os.path.exists(config_path):
                with open(config_path) as f:
                    saved_cls_name = json.load(f).get("agent_class")
                if saved_cls_name and saved_cls_name in AGENT_REGISTRY:
                    agent_cls = AGENT_REGISTRY[saved_cls_name]
                    print(f"Resuming run {model_dir!r} with agent class {saved_cls_name} (from vla_config.json)")
                else:
                    agent_cls = AGENT_REGISTRY[args.agent_cls]
                    print(f"Resuming run {model_dir!r} (vla_config.json missing agent_class, using --agent-cls)")
            else:
                agent_cls = AGENT_REGISTRY[args.agent_cls]
                print(f"Resuming run {model_dir!r} (no vla_config.json found, using --agent-cls)")
        except FileNotFoundError:
            TrainingConfig.run_name = args.run

    if model_dir is None:
        agent_cls = AGENT_REGISTRY[args.agent_cls]
        model_dir = _resolve_run_dir()

    print(f"Run dir: {model_dir}")

    model, processor = get_model_and_processor(model_dir, agent_cls)
    agent = agent_cls(model=model, processor=processor, temperature=args.temperature)
    tokenizer = processor.tokenizer

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f"Training prompt: {agent_cls.build_training_prompt({'text': '<observation>'})!r}")
    print(f"Inference prompt: {agent_cls.build_inference_prompt()!r}")

    # --- 2. DATASET DEFINITION ---
    full_dataset = get_dataset(processor, tokenizer, agent_cls=agent_cls)

    if args.eval:
        print("\n=== RUNNING STANDALONE EVALUATION ===")
        eval_model(agent, full_dataset, num_samples=args.samples)
        return

    # --- 3. FREEZE STRATEGY ---
    model.requires_grad_(True)
    for param in model.model.vision_model.parameters():
        param.requires_grad = False
    for param in model.model.vision_model.encoder.layers[-12:].parameters():
        param.requires_grad = True

    print("Freeze Strategy: Vision=Frozen (last 12 layers un-frozen), Connector=Trainable, LLM=Trainable")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")

    # --- 4. TRAINING CONFIGURATION ---
    training_args = TrainingArguments(
        output_dir=model_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=TrainingConfig.learning_rate,
        num_train_epochs=TrainingConfig.num_epochs,
        bf16=True,
        logging_first_step=True,
        logging_steps=2,
        save_strategy="no",
        optim="adamw_torch_fused",
        report_to="none",
    )

    trainer_cls = WeightedActionTrainer if agent_cls is CoTVLAAgent else Trainer
    trainer_kwargs = {"action_weight": TrainingConfig.action_loss_weight, "thought_weight": TrainingConfig.thought_loss_weight, "num_action_tokens": 2} if agent_cls is CoTVLAAgent else {}

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=full_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        callbacks=[InferenceCallback(agent, full_dataset, every_n_steps=TrainingConfig.eval_steps, num_samples=20)],
        **trainer_kwargs,
    )

    # --- 5. EXECUTION ---
    print("Starting full VLA Training...")
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\n!!! Training interrupted by user (Ctrl+C) !!!")

    print("\n=== FINAL MODEL EVALUATION ===")
    eval_model(agent, full_dataset, num_samples=args.samples)

    print("\n=== SAVE MODEL ===")
    trainer.save_model(model_dir)
    processor.save_pretrained(model_dir)

    # Save agent config so eval.py knows which agent class was used
    with open(os.path.join(model_dir, "vla_config.json"), "w") as f:
        json.dump({"agent_class": agent_cls.__name__}, f)

    print(f"Training Complete. Model saved to {model_dir}")

if __name__ == "__main__":
    main()
