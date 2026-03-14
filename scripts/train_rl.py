#!/usr/bin/env python3
"""
GRPO (Group Relative Policy Optimization) training for VLA agents.
Uses verifiable rewards from the game environment — no learned reward model.

The model plays the game live. At each decision point, G candidate action
sequences are generated and scored with a rule-based reward. The policy is
updated to favor higher-reward actions within each group.

Usage:
    python scripts/train_rl.py --checkpoint models/001_cot_action_loss
    python scripts/train_rl.py --checkpoint models/001_cot_action_loss --group-size 8
    python scripts/train_rl.py --checkpoint models/001_cot_action_loss --run grpo_v1
"""

import torch
import copy
import os
import re
import json
import argparse
import pygame
import trackio

from src.gym.env import GameEnvironment
from src.vla.model import get_model_and_processor
from src.vla.vla_agent import AGENT_REGISTRY

torch.set_float32_matmul_precision("high")


class RLConfig:
    models_root = "./models"
    run_name = "grpo"
    agent_cls = "GRPOVLAAgent"
    num_episodes = 100
    max_steps_per_episode = 700     # ~10s at 60fps
    decision_interval = 20          # decide every N game frames
    group_size = 3                  # G candidates per decision point
    gradient_accumulation = 8       # accumulate over this many decision points before optimizer step
    learning_rate = 3e-6
    temperature = 0.75
    clip_eps = 0.2                  # PPO-style ratio clipping
    kl_coeff = 0.0                  # KL penalty vs reference (0 = disabled)
    max_grad_norm = 1.0
    save_every = 10                 # save every N episodes
    log_every = 1                   # print per-episode stats
    print_output_every = 2         # print best candidate text every N decisions (0 = off)


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

class VerbosityPenalty:
    """Penalizes thought tokens beyond a target length. Set enabled=False to disable."""
    enabled = True
    max_words = 14          # no penalty up to this many words
    penalty_per_word = 0.02 # reward deducted per word over the limit

    @classmethod
    def compute(cls, text: str) -> float:
        if not cls.enabled:
            return 0.0
        length = len(_extract_thought(text).split())
        return -cls.penalty_per_word * max(0, length - cls.max_words)


def compute_reward(obs, action_dict: dict, text: str = "") -> float:
    r = 0.0
    r += obs.speed                                      # go fast (0–1.5)
    r += 1.0 if obs.on_road else -2.0                  # stay on road
    r -= abs(obs.car_offset_from_center) / 80.0        # stay centered (road edge ~80px)
    r += VerbosityPenalty.compute(text)                 # keep thoughts concise
    return r


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def prepare_vlm_input(processor, image, prompt):
    """Build model-ready tensors from a PIL image + text prompt."""
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
    ]
    prompt_text = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )
    if not prompt_text.endswith(" "):
        prompt_text += " "
    return processor(
        text=prompt_text, images=image, return_tensors="pt",
        do_resize=True, size={"longest_edge": 224},
    )


def generate_candidates(model, processor, image, prompt, agent_cls, num_candidates, temperature):
    """
    Generate G candidate action sequences for a single frame.
    Returns (candidates_list, base_inputs, prompt_len).
    Each candidate is a dict with keys: text, action, full_ids.
    """
    inputs = prepare_vlm_input(processor, image, prompt)
    device = model.device
    inputs_device = {k: v.to(device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    candidates = []
    with torch.no_grad():
        for _ in range(num_candidates):
            generated = model.generate(
                **inputs_device,
                max_new_tokens=agent_cls.max_new_tokens,
                do_sample=True,
                temperature=temperature,
                use_cache=True,
            )
            gen_ids = generated[0, prompt_len:]
            text = processor.decode(gen_ids, skip_special_tokens=True)
            action = agent_cls.decode_action(text)
            candidates.append({
                "text": text,
                "action": action,
                "full_ids": generated,       # [1, prompt_len + gen_len]
            })

    return candidates, inputs, prompt_len


def compute_token_log_probs(model, full_ids, base_inputs, prompt_len):
    """
    Forward pass to compute the sum of log-probs for generated tokens.
    Returns a scalar tensor WITH gradient attached.
    """
    device = model.device
    fwd = {k: v.to(device) for k, v in base_inputs.items()}
    fwd["input_ids"] = full_ids.to(device)
    fwd["attention_mask"] = torch.ones_like(full_ids, device=device)

    outputs = model(**fwd)
    logits = outputs.logits  # [1, seq_len, vocab]

    # Causal shift: logit at position t predicts token at t+1
    gen_logits = logits[0, prompt_len - 1:-1, :]
    gen_targets = full_ids[0, prompt_len:].to(device)

    log_probs = torch.nn.functional.log_softmax(gen_logits, dim=-1)
    token_log_probs = log_probs.gather(-1, gen_targets.unsqueeze(-1)).squeeze(-1)

    return token_log_probs.sum()


# ---------------------------------------------------------------------------
# Thought drift monitoring (options 1, 2, 3)
# ---------------------------------------------------------------------------

def _extract_thought(text: str) -> str:
    m = re.search(r'<thought>(.*?)</thought>', text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()

def _word_set(thought: str) -> set:
    return set(re.findall(r'\b\w+\b', thought.lower()))

def _jaccard(a: set, b: set) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 1.0

def collect_reference_frames(env, agent_cls, n=6, skip_steps=25):
    """Collect n varied frames from a fresh episode for drift comparisons."""
    frames = []
    obs, _ = env.reset()
    for _ in range(n):
        for _ in range(skip_steps):
            obs, _, terminated, truncated, info = env.step(agent_cls.default_action)
            if terminated or truncated or info["should_quit"]:
                obs, _ = env.reset()
        frames.append(obs.frame)
    env.reset()  # leave env in clean state
    return frames

def generate_reference_outputs(model, processor, frames, prompt, agent_cls):
    """Greedy inference on fixed frames. Returns list of raw output texts."""
    model.eval()
    outputs = []
    with torch.no_grad():
        for frame in frames:
            cands, _, _ = generate_candidates(model, processor, frame, prompt, agent_cls, 1, 0.1)
            outputs.append(cands[0]["text"])
    model.train()
    return outputs

def print_drift_report(episode, sft_outputs, cur_outputs, baseline_vocab):
    thoughts_sft = [_extract_thought(t) for t in sft_outputs]
    thoughts_cur = [_extract_thought(t) for t in cur_outputs]

    # Option 2: thought length
    avg_sft = sum(len(t.split()) for t in thoughts_sft) / len(thoughts_sft)
    avg_cur = sum(len(t.split()) for t in thoughts_cur) / len(thoughts_cur)

    # Option 3: vocabulary Jaccard vs SFT baseline
    cur_vocab = set()
    for t in thoughts_cur:
        cur_vocab |= _word_set(t)
    j = _jaccard(baseline_vocab, cur_vocab)

    print(f"\n  === Thought Drift (ep {episode}) ===")
    print(f"  Thought length : SFT={avg_sft:.1f}w  now={avg_cur:.1f}w")
    print(f"  Vocab Jaccard  : {j:.3f}  (SFT={len(baseline_vocab)} words, now={len(cur_vocab)} words)")

    trackio.log({
        "drift/jaccard":         j,
        "drift/thought_len_sft": avg_sft,
        "drift/thought_len_now": avg_cur,
        "drift/vocab_size_sft":  len(baseline_vocab),
        "drift/vocab_size_now":  len(cur_vocab),
    }, step=episode)

    # Option 1: side-by-side for first 3 reference frames
    for i, (sft_t, cur_t) in enumerate(zip(thoughts_sft[:3], thoughts_cur[:3])):
        print(f"  [{i+1}] SFT : {sft_t!r}")
        print(f"       Now : {cur_t!r}")
    print()


# ---------------------------------------------------------------------------
# Run directory helpers (mirrors train.py)
# ---------------------------------------------------------------------------

def _resolve_run_dir(models_root, run_name):
    existing = (
        [d for d in os.listdir(models_root) if os.path.isdir(os.path.join(models_root, d))]
        if os.path.exists(models_root) else []
    )
    index = len(existing) + 1
    return os.path.join(models_root, f"{index:03d}_{run_name}")


def _save_checkpoint(model, processor, agent_cls, run_dir):
    os.makedirs(run_dir, exist_ok=True)
    model.save_pretrained(run_dir)
    processor.save_pretrained(run_dir)
    with open(os.path.join(run_dir, "vla_config.json"), "w") as f:
        json.dump({"agent_class": agent_cls.__name__}, f)
    print(f"  *** Checkpoint saved → {run_dir} ***")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GRPO training for VLA agents")
    parser.add_argument("--checkpoint", metavar="PATH",
                        help="SFT checkpoint to initialise from (required for first run)")
    parser.add_argument("--run", metavar="NAME", help="Run name for output directory")
    parser.add_argument("--agent-cls", default=RLConfig.agent_cls,
                        choices=list(AGENT_REGISTRY.keys()))
    parser.add_argument("--episodes", type=int, default=RLConfig.num_episodes)
    parser.add_argument("--max-steps", type=int, default=RLConfig.max_steps_per_episode)
    parser.add_argument("--group-size", type=int, default=RLConfig.group_size)
    parser.add_argument("--grad-accum", type=int, default=RLConfig.gradient_accumulation)
    parser.add_argument("--lr", type=float, default=RLConfig.learning_rate)
    parser.add_argument("--temperature", type=float, default=RLConfig.temperature)
    parser.add_argument("--kl-coeff", type=float, default=RLConfig.kl_coeff)
    parser.add_argument("--save-every", type=int, default=RLConfig.save_every)
    parser.add_argument("--print-output-every", type=int, default=RLConfig.print_output_every,
                        help="Print best candidate text every N decisions per episode (0 = off)")
    args = parser.parse_args()

    if args.run:
        RLConfig.run_name = args.run

    agent_cls = AGENT_REGISTRY[args.agent_cls]

    # --- Output directory ---
    os.makedirs(RLConfig.models_root, exist_ok=True)
    run_dir = _resolve_run_dir(RLConfig.models_root, RLConfig.run_name)
    print(f"Run dir: {run_dir}")

    # --- Model ---
    checkpoint = args.checkpoint
    model, processor = get_model_and_processor(checkpoint, agent_cls)
    model.train()
    device = model.device

    # Freeze vision encoder (same strategy as SFT)
    for param in model.model.vision_model.parameters():
        param.requires_grad = False
    for param in model.model.vision_model.encoder.layers[-12:].parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable / 1e6:.2f}M")

    # Optional frozen reference model for KL penalty
    ref_model = None
    if args.kl_coeff > 0:
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        ref_model.requires_grad_(False)
        print("Reference model created for KL penalty")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
    )

    # --- Game environment ---
    pygame.display.init()
    pygame.font.init()
    env = GameEnvironment()

    prompt = agent_cls.build_inference_prompt()
    G = args.group_size
    grad_accum = args.grad_accum

    print(f"\nGRPO Training")
    print(f"  episodes={args.episodes}, G={G}, grad_accum={grad_accum}")
    print(f"  temperature={args.temperature}, lr={args.lr}, kl_coeff={args.kl_coeff}")
    print(f"  decision_interval={RLConfig.decision_interval}, max_steps={args.max_steps}")
    print()

    # --- Drift monitoring setup ---
    print("Collecting reference frames for drift monitoring...")
    ref_frames = collect_reference_frames(env, agent_cls)
    sft_outputs = generate_reference_outputs(model, processor, ref_frames, prompt, agent_cls)
    sft_vocab = set()
    for t in sft_outputs:
        sft_vocab |= _word_set(_extract_thought(t))
    print(f"  {len(ref_frames)} reference frames captured, SFT vocab={len(sft_vocab)} words")
    print()

    # --- Trackio init ---
    trackio.init(
        project="pyracer-grpo",
        name=RLConfig.run_name,
        config={
            "episodes": args.episodes,
            "group_size": G,
            "grad_accum": grad_accum,
            "lr": args.lr,
            "temperature": args.temperature,
            "kl_coeff": args.kl_coeff,
            "decision_interval": RLConfig.decision_interval,
            "max_steps": args.max_steps,
            "verbosity_penalty": VerbosityPenalty.enabled,
        },
    )

    # --- Training loop ---
    optimizer.zero_grad()
    reward_history = []   # for rolling average (option 1)
    on_road_history = []  # for rolling on-road % (option 3)
    ROLLING_WINDOW = 10
    global_step = 0

    for episode in range(args.episodes):
        print(f"\n--- Episode {episode + 1}/{args.episodes} ---")
        obs, _ = env.reset()
        action_dict = agent_cls.default_action

        episode_rewards = []
        episode_decisions = 0
        episode_on_road = 0
        accum_count = 0     # counts decision points since last optimizer.step()
        episode_loss = 0.0
        step_loss = 0.0
        episode_thoughts = []

        for step in range(args.max_steps):
            # --- Decision point ---
            if step % RLConfig.decision_interval == 0:
                candidates, base_inputs, prompt_len = generate_candidates(
                    model, processor, obs.frame, prompt, agent_cls, G, args.temperature,
                )

                # Score each candidate
                rewards = [compute_reward(obs, c["action"], c["text"]) for c in candidates]

                # GRPO advantages: normalise within group
                mean_r = sum(rewards) / G
                std_r = (sum((r - mean_r) ** 2 for r in rewards) / G) ** 0.5 + 1e-8
                advantages = [(r - mean_r) / std_r for r in rewards]

                # Compute per-candidate loss and accumulate gradients
                scale = 1.0 / (grad_accum * G)
                for i, cand in enumerate(candidates):
                    log_prob = compute_token_log_probs(
                        model, cand["full_ids"], base_inputs, prompt_len,
                    )

                    loss = -advantages[i] * log_prob * scale

                    if ref_model is not None:
                        with torch.no_grad():
                            ref_lp = compute_token_log_probs(
                                ref_model, cand["full_ids"], base_inputs, prompt_len,
                            )
                        kl = (log_prob - ref_lp) * scale
                        loss = loss + args.kl_coeff * kl

                    loss.backward()
                    episode_loss += loss.item()
                    step_loss += loss.item()

                accum_count += 1

                # Optimizer step when we've accumulated enough
                if accum_count >= grad_accum:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), RLConfig.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    accum_count = 0
                    global_step += 1
                    trackio.log({"train/loss_step": step_loss}, step=global_step)
                    step_loss = 0.0

                # Execute the best candidate action in the game
                best_idx = max(range(G), key=lambda j: rewards[j])
                action_dict = candidates[best_idx]["action"]
                episode_rewards.append(rewards[best_idx])
                episode_decisions += 1
                episode_on_road += int(obs.on_road)
                episode_thoughts.append(_extract_thought(candidates[best_idx]["text"]))

                if args.print_output_every > 0 and episode_decisions % args.print_output_every == 0:
                    avg_r = sum(episode_rewards) / len(episode_rewards)
                    print(f"  [dec {episode_decisions}] avg_reward={avg_r:.3f} | best(r={rewards[best_idx]:.2f}): {candidates[best_idx]['text']!r}")

            # --- Step the game ---
            obs, _, terminated, truncated, info = env.step(action_dict)
            if terminated or truncated or info["should_quit"]:
                break

        # Flush remaining accumulated gradients
        if accum_count > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), RLConfig.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        # --- Logging ---
        avg_r = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0
        on_road_pct = 100.0 * episode_on_road / episode_decisions if episode_decisions else 0
        avg_thought_len = sum(len(t.split()) for t in episode_thoughts) / len(episode_thoughts) if episode_thoughts else 0

        reward_history.append(avg_r)
        on_road_history.append(on_road_pct)
        rolling_reward = sum(reward_history[-ROLLING_WINDOW:]) / min(len(reward_history), ROLLING_WINDOW)
        rolling_on_road = sum(on_road_history[-ROLLING_WINDOW:]) / min(len(on_road_history), ROLLING_WINDOW)

        if (episode + 1) % RLConfig.log_every == 0:
            print(
                f"ep {episode + 1:>4}/{args.episodes}"
                f" | avg_r: {avg_r:>6.3f}  roll{ROLLING_WINDOW}: {rolling_reward:>6.3f}"
                f" | on_road: {on_road_pct:>5.1f}%  roll{ROLLING_WINDOW}: {rolling_on_road:>5.1f}%"
                f" | thought_len: {avg_thought_len:>4.1f}w"
                f" | loss: {episode_loss:>8.4f}"
            )
            trackio.log({
                "reward/episode":      avg_r,
                "reward/rolling10":    rolling_reward,
                "on_road/episode_pct": on_road_pct,
                "on_road/rolling10":   rolling_on_road,
                "thought/avg_len":     avg_thought_len,
                "train/loss":          episode_loss,
                "train/decisions":     episode_decisions,
            }, step=episode + 1)

        # --- Checkpoint ---
        if (episode + 1) % args.save_every == 0:
            cur_outputs = generate_reference_outputs(model, processor, ref_frames, prompt, agent_cls)
            print_drift_report(episode + 1, sft_outputs, cur_outputs, sft_vocab)
            _save_checkpoint(model, processor, agent_cls, run_dir)

    # Final save
    _save_checkpoint(model, processor, agent_cls, run_dir)
    print(f"\nTraining complete. Model saved to {run_dir}")
    trackio.finish()
    env.close()


if __name__ == "__main__":
    main()
