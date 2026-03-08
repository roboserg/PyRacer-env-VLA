import torch
import re
import pygame
import numpy as np
from PIL import Image
from vla.controller import Controller
from vla.model import infer, post_process_output


class VLAController(Controller):
    def __init__(self, model, processor, device="cuda", max_length=128):
        self.model = model
        self.processor = processor
        self.device = device
        self.max_length = max_length

        self.model.eval()
        self.prompt_text = "Action:"

    def get_action(self, observation) -> dict:
        try:
            frame = observation.frame

            # TODO: Avoid temp file - convert pygame surface directly to PIL Image
            # Solution: Use pygame.surfarray.array3d(frame) or pygame.image.tostring()
            # Example:
            #   import pygame.surfarray as surfarray
            #   img = surfarray.array3d(frame)  # shape: (w, h, 3)
            #   # Convert RGB to BGR if needed, then wrap in PIL
            #   image = Image.fromarray(img.swapaxes(0, 1), mode="RGB")
            temp_path = "/tmp/vla_inference_input.jpg"
            pygame.image.save(frame, temp_path)

            image = Image.open(temp_path).convert("RGB")

            output_text = infer(self.model, self.processor, image)
            action = post_process_output(output_text)

            # verify input image changes
            img_arr = np.array(image)
            img_mean = np.mean(img_arr)

            debug_msg = f"VLA -> Out: {repr(output_text)} | ImgMean:{img_mean:.1f} | A:{int(action['accel'])} B:{int(action['brake'])} L:{int(action['left'])} R:{int(action['right'])}"
            print(debug_msg)

            return action
        except Exception as e:
            print(f"CRITICAL ERROR in VLAController: {e}")
            return {
                "accel": False,
                "brake": False,
                "left": False,
                "right": False,
            }
