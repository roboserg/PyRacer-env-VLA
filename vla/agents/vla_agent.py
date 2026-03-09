from typing import Optional, Tuple, Any
import pygame
import numpy as np
from PIL import Image
from vla.agents.agent import Agent
from vla.model import infer, post_process_output


class VLAAgent(Agent):
    def __init__(self, env: Optional[Any] = None, model: Any = None, processor: Any = None, device: str = "cuda", max_length: int = 128):
        super().__init__(env)
        self.model = model
        self.processor = processor
        self.device = device
        self.max_length = max_length

        if self.model is not None:
            self.model.eval()
        self.prompt_text = "Action:"

    def predict(
        self,
        observation: Any,
        state: Optional[Tuple[Any, ...]] = None,
        episode_start: Optional[Any] = None,
        deterministic: bool = False,
    ) -> Tuple[Any, Optional[Tuple[Any, ...]]]:
        try:
            frame = observation.frame

            frame_str = pygame.image.tostring(frame, "RGB")
            image = Image.frombytes("RGB", frame.get_size(), frame_str)

            output_text = infer(self.model, self.processor, image)
            action_dict = post_process_output(output_text)

            img_arr = np.array(image)
            img_mean = np.mean(img_arr)

            print(f"VLA -> Out: {repr(output_text)} | ImgMean:{img_mean:.1f} | A:{int(action_dict['accel'])} B:{int(action_dict['brake'])} L:{int(action_dict['left'])} R:{int(action_dict['right'])}")

            if self.env and hasattr(self.env, "_encode_action"):
                return self.env._encode_action(action_dict), state

            return action_dict, state
        except Exception as e:
            print(f"CRITICAL ERROR in VLAAgent: {e}")
            default_action = {"accel": False, "brake": False, "left": False, "right": False}
            if self.env and hasattr(self.env, "_encode_action"):
                return self.env._encode_action(default_action), state
            return default_action, state
