import torch
import json
from abc import ABC, abstractmethod
from PIL import Image
from torch.utils.data import Dataset
from collections import Counter

class RacingDatasetBase(Dataset, ABC):
    def __init__(self, jsonl_file, img_dir):
        self.data = [json.loads(line) for line in open(jsonl_file, "r")]
        self.img_dir = img_dir

    def __len__(self):
        return len(self.data)

    def print_stats(self):
        """Calculates and prints statistics for the current dataset instance."""
        print(f"Total dataset size: {len(self)} frames")
        stats = {"accel": 0, "brake": 0, "left": 0, "right": 0, "any_action": 0}
        action_sets = []

        for item in self.data:
            a = item["action"]
            if a[0] > 0.5:
                stats["accel"] += 1
            if a[1] > 0.5:
                stats["brake"] += 1
            if a[2] > 0.5:
                stats["left"] += 1
            if a[3] > 0.5:
                stats["right"] += 1
            if any(v > 0.5 for v in a):
                stats["any_action"] += 1
            action_sets.append(self._encode_for_stats(a, item))

        print("\nDataset Individual Action Stats:")
        for k, v in stats.items():
            pct = (v / len(self)) * 100
            print(f"  {k:10}: {v:5} ({pct:.1f}%)")

        print("\nDataset Action Set Distribution (Frequency of combinations):")
        set_counts = Counter(action_sets)
        for action_set, count in set_counts.most_common():
            pct = (count / len(self)) * 100
            print(f"  {action_set}: {count:5} ({pct:.1f}%)")
        print()

    @abstractmethod
    def _encode_for_stats(self, actions, item) -> str:
        pass

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(f"{self.img_dir}/{item['frame']}").convert("RGB")
        actions = list(item["action"])

        return self.build_sample(image, actions, item, idx)

    @abstractmethod
    def build_sample(self, image, actions, item, idx) -> dict:
        """Return {pixel_values, input_ids, labels}."""


class VLADataset(RacingDatasetBase):
    def __init__(self, jsonl_file, img_dir, processor, tokenizer, agent_cls=None):
        super().__init__(jsonl_file, img_dir)
        self.processor = processor
        self.tokenizer = tokenizer
        if agent_cls is None:
            from src.vla.vla_agent import TwoTokenVLAAgent
            agent_cls = TwoTokenVLAAgent
        self.agent_cls = agent_cls

    def _encode_for_stats(self, actions, item):
        return self.agent_cls.encode_for_stats(actions, item)

    def debug_sample(self, idx=0):
        """Print a label X-ray for one sample to verify tokenisation before training."""
        sample = self[idx]
        labels = sample["labels"]
        active_labels = labels[labels != -100]
        item = self.data[idx]
        action_str = self.agent_cls.encode_action(list(item["action"]), item)
        print("\n" + "=" * 50)
        print(f"[DEBUG] Action String:  {action_str}")
        print(f"[DEBUG] Decoded Labels: {repr(self.tokenizer.decode(active_labels))}")
        print(f"[DEBUG] Label IDs:      {active_labels.tolist()}")
        print(f"[DEBUG] Prompt Len:     {(labels == -100).sum().item()}")
        print(f"[DEBUG] Total Len:      {len(labels)}")
        print("=" * 50 + "\n")

    def build_sample(self, image, actions, item, idx):
        action_str = self.agent_cls.encode_action(actions, item)
        prompt_text_content = self.agent_cls.build_training_prompt(item)

        # --- BUILD CONVERSATION ---
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text_content},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": action_str}],
            },
        ]

        # --- PROCESS FOR SMOLVLM ---
        prompt_text = self.processor.apply_chat_template(messages[:1], add_generation_prompt=True)
        if not prompt_text.endswith(" "):
            prompt_text += " "

        full_text = prompt_text + action_str + self.tokenizer.eos_token

        inputs = self.processor(
            text=full_text, images=image, return_tensors="pt",
            do_resize=True, size={"longest_edge": 384},
        )
        prompt_inputs = self.processor(
            text=prompt_text, images=image, return_tensors="pt",
            do_resize=True, size={"longest_edge": 384},
        )

        input_ids = inputs["input_ids"].squeeze(0)
        pixel_values = inputs["pixel_values"].squeeze(0)
        prompt_len = prompt_inputs["input_ids"].shape[1]

        # --- LABEL MASKING ---
        labels = input_ids.clone()
        labels[:prompt_len] = -100

        return {
            "pixel_values": pixel_values.detach().clone().to(torch.float32),
            "input_ids": input_ids.detach().clone().to(torch.long),
            "labels": labels.detach().clone().to(torch.long),
        }


TwoTokenDataset = VLADataset
RacingVLADataset = VLADataset