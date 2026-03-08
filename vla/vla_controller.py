import torch
import re
import pygame
import numpy as np
import sys
from PIL import Image
from vla.controller import Controller


class VLAController(Controller):
    def __init__(self, model, processor, device="cuda", max_length=128):
        self.model = model
        self.processor = processor
        self.device = device
        self.max_length = max_length

        # Setup Static Cache for faster inference
        self.model.generation_config.cache_implementation = "static"

        self.model.eval()
        self.prompt_text = "Action:"

    def get_action(self, observation) -> dict:
        try:
            frame = observation.frame
            image = Image.fromarray(
                pygame.surfarray.array3d(frame).transpose(1, 0, 2)
            ).convert("RGB")
            
            # DEBUG: Save image occasionally to verify input
            if getattr(self, "step_count", 0) % 20 == 0:
                image.save("vla_debug_input.jpg")
            self.step_count = getattr(self, "step_count", 0) + 1

            # Standard chat template logic
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": self.prompt_text},
                    ],
                }
            ]
            prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            if not prompt.endswith(" "):
                prompt += " "

            # Force the 108-token 'squeezed' patch
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt",
                do_resize=True,
                size={"longest_edge": 384},
            ).to(self.device)

            with torch.inference_mode():
                # generate() automatically uses the static cache we configured in __init__
                generated_ids = self.model.generate(
                    **inputs, max_new_tokens=32, do_sample=False, use_cache=True
                )

            output_text = self.processor.decode(generated_ids[0], skip_special_tokens=True)
            
            # Parse actions with Regex for robustness
            accel = bool(re.search(r"<FWD_1>", output_text))
            brake = bool(re.search(r"<BRK_1>", output_text))
            left = bool(re.search(r"<LFT_1>", output_text))
            right = bool(re.search(r"<RGT_1>", output_text))

            # Debug print to BOTH stdout and stderr to ensure it's seen
            debug_msg = f"VLA -> Out: {repr(output_text)} | A:{int(accel)} B:{int(brake)} L:{int(left)} R:{int(right)}\n"
            sys.stderr.write(debug_msg)
            sys.stderr.flush()
            print(debug_msg, end="")
            sys.stdout.flush()

            return {
                "accel": accel,
                "brake": brake,
                "left": left,
                "right": right,
            }
        except Exception as e:
            sys.stderr.write(f"CRITICAL ERROR in VLAController: {e}\n")
            sys.stderr.flush()
            return {
                "accel": False,
                "brake": False,
                "left": False,
                "right": False,
            }
