import torch
import json
import random
from PIL import Image, ImageOps
from torch.utils.data import Dataset


class RacingVLADataset(Dataset):
    def __init__(self, jsonl_file, img_dir, processor, tokenizer, augment=True):
        self.data = [json.loads(line) for line in open(jsonl_file, "r")]
        self.img_dir = img_dir
        self.processor = processor
        self.tokenizer = tokenizer
        self.augment = augment  # Enable flipping

        # Mapping index to our special tokens
        self.action_map = ["FWD", "BRK", "LFT", "RGT"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(f"{self.img_dir}/{item['frame']}").convert("RGB")

        # Pull values from JSON
        actions = list(item["action"])  # [fwd, lft, rgt, brk]
        thought_text = item.get("text", "Unknown state")
        speed_val = item["speed"]

        # --- 1. DATA AUGMENTATION (FLIPPING) ---
        do_flip = self.augment and random.random() > 0.5
        if do_flip:
            image = ImageOps.mirror(image)
            # recorder.py order: [accel, brake, left, right]
            # Swap Left (idx 2) and Right (idx 3)
            actions[2], actions[3] = actions[3], actions[2]

            # Robust Text Swap using a dictionary
            swap_map = {
                "left": "right",
                "right": "left",
                "Left": "Right",
                "Right": "Left",
            }
            # We use a regex or a simple split/join to avoid double-replacing
            words = thought_text.split()
            new_words = [swap_map.get(w.strip(",."), w) for w in words]
            thought_text = " ".join(new_words)

        # --- 2. TOKEN CONVERSION ---
        # Note: Your action_map was ["FWD", "LFT", "RGT", "BRK"]
        # Ensure this order matches your JSON action list index exactly!
        action_tokens = []
        for i, val in enumerate(actions):
            state = "1" if val > 0.5 else "0"
            action_tokens.append(f"<{self.action_map[i]}_{state}>")
        action_str = " ".join(action_tokens)

        # --- 3. BUILD CONVERSATION (Strict Action) ---
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Action:"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": action_str,
                    }
                ],
            },
        ]

        # --- 4. PROCESS FOR SMOLVLM ---
        # 1. Get the prompt only (to find its length)
        prompt_text = self.processor.apply_chat_template(
            messages[:1], add_generation_prompt=True
        )
        if not prompt_text.endswith(" "):
            prompt_text += " "
            
        # 2. Get the full conversation
        # We reconstruct it manually to ensure the space is handled correctly
        full_text = prompt_text + action_str + self.tokenizer.eos_token

        # 3. Process both to ensure token alignment
        inputs = self.processor(
            text=full_text, 
            images=image, 
            return_tensors="pt", 
            do_resize=True, 
            size={"longest_edge": 384}
        )
        
        prompt_inputs = self.processor(
            text=prompt_text, 
            images=image, 
            return_tensors="pt", 
            do_resize=True, 
            size={"longest_edge": 384}
        )

        input_ids = inputs["input_ids"].squeeze(0)
        pixel_values = inputs["pixel_values"].squeeze(0)
        prompt_len = prompt_inputs["input_ids"].shape[1]

        # --- 5. LABEL MASKING ---
        labels = input_ids.clone()
        labels[:prompt_len] = -100

        # --- DEBUG X-RAY ---
        if idx == 0:
            active_labels = labels[labels != -100]
            print(f"\n" + "="*50)
            print(f"[DEBUG] Action String:  {action_str}")
            print(f"[DEBUG] Decoded Labels: {repr(self.tokenizer.decode(active_labels))}")
            print(f"[DEBUG] Label IDs:      {active_labels.tolist()}")
            print(f"[DEBUG] Prompt Len:     {prompt_len}")
            print(f"[DEBUG] Total Len:      {len(input_ids)}")
            print("="*50 + "\n")
            
        return {
            "pixel_values": pixel_values.detach().clone().to(torch.float32),
            "input_ids": input_ids.detach().clone().to(torch.long),
            "labels": labels.detach().clone().to(torch.long)
        }