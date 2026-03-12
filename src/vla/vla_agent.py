import re
from abc import ABC, abstractmethod
from typing import Optional, Any
from src.gym.agents.agent import Agent


class VLAAgent(Agent, ABC):
    tokens: list[str] = []
    max_new_tokens: int = 3
    default_temperature: float = 0.0
    default_action: dict = {"accel": False, "brake": False, "left": False, "right": False}

    def __init__(self, env=None, model=None, processor=None, temperature: Optional[float] = None):
        super().__init__(env)
        self.model = model
        self.processor = processor
        self.temperature = temperature if temperature is not None else self.default_temperature
        if self.model is not None:
            self.model.eval()
        self.last_inference_time_ms: float = 0.0
        self.last_output_text: str = ""
        self._inference_thread = None

    @classmethod
    @abstractmethod
    def build_training_prompt(cls, item: dict) -> str:
        """item metadata dict → user prompt string used during training"""

    @classmethod
    def build_inference_prompt(cls) -> str:
        """Prompt used at inference time (no metadata available)."""
        return "Action:"

    @classmethod
    @abstractmethod
    def encode_action(cls, actions: list, item: dict = None) -> str:
        """[accel, brake, left, right] floats → assistant response string"""

    @classmethod
    @abstractmethod
    def decode_action(cls, raw_output: str) -> dict:
        """raw model output → {accel, brake, left, right} booleans"""

    @classmethod
    def encode_for_stats(cls, actions: list, item: dict = None) -> str:
        """Compact label for dataset statistics. Defaults to encode_action."""
        return cls.encode_action(actions, item)

    def start_predict(self, observation: Any) -> None:
        """Submit inference to background thread. Non-blocking."""
        if self._inference_thread is None:
            from src.vla.model import InferenceThread
            self._inference_thread = InferenceThread(self.model, self.processor)
        self._inference_thread.submit(
            observation.frame, self.build_inference_prompt(),
            self.max_new_tokens, self.temperature,
        )

    def poll_predict(self) -> Any:
        """Check if inference is done. Returns action or None."""
        result = self._inference_thread.poll()
        if result is None:
            return None
        output_text, ms = result
        self.last_inference_time_ms = ms
        self.last_output_text = output_text
        action_dict = self.decode_action(output_text)
        if self.env and hasattr(self.env, "_encode_action"):
            return self.env._encode_action(action_dict)
        return action_dict

    def predict(self, observation: Any) -> Any:
        from src.vla.model import run_inference
        try:
            output_text, ms = run_inference(
                self.model, self.processor, observation.frame,
                self.build_inference_prompt(), self.max_new_tokens,
                temperature=self.temperature,
            )
            action_dict = self.decode_action(output_text)
            self.last_inference_time_ms = ms
            self.last_output_text = output_text
            if self.env and hasattr(self.env, "_encode_action"):
                return self.env._encode_action(action_dict)
            return action_dict
        except Exception as e:
            print(f"CRITICAL ERROR in VLAAgent: {e}")
            self.last_inference_time_ms = 0.0
            self.last_output_text = ""
            default = self.default_action
            if self.env and hasattr(self.env, "_encode_action"):
                return self.env._encode_action(default)
            return default


class TwoTokenVLAAgent(VLAAgent):
    tokens = ["<THROTTLE_FWD>", "<THROTTLE_NONE>", "<THROTTLE_BRK>",
              "<STEER_LFT>", "<STEER_NONE>", "<STEER_RGT>"]
    max_new_tokens = 3
    default_action = {"accel": False, "brake": False, "left": False, "right": False}

    @classmethod
    def build_training_prompt(cls, item):
        return f"{item.get('text', 'Unknown state')} → Action:"

    @classmethod
    def encode_action(cls, actions, item=None):
        throttle = "<THROTTLE_FWD>" if actions[0] > 0.5 else ("<THROTTLE_BRK>" if actions[1] > 0.5 else "<THROTTLE_NONE>")
        steer    = "<STEER_LFT>"    if actions[2] > 0.5 else ("<STEER_RGT>"    if actions[3] > 0.5 else "<STEER_NONE>")
        return throttle + steer

    @classmethod
    def decode_action(cls, raw_output):
        return {
            "accel": bool(re.search(r"<THROTTLE_FWD>", raw_output)),
            "brake": bool(re.search(r"<THROTTLE_BRK>", raw_output)),
            "left":  bool(re.search(r"<STEER_LFT>",    raw_output)),
            "right": bool(re.search(r"<STEER_RGT>",    raw_output)),
        }


class CoTVLAAgent(VLAAgent):
    PROMPT = (
        "Observe the road and describe what you see, "
        "then choose your actions.\n\n"
        "Example:\n"
        "<thought>Road curves left ahead, going fast, "
        "centered on track</thought>"
        "<THROTTLE_FWD><STEER_LFT>"
    )
    tokens = ["<thought>", "</thought>",
              "<THROTTLE_FWD>", "<THROTTLE_NONE>", "<THROTTLE_BRK>",
              "<STEER_LFT>", "<STEER_NONE>", "<STEER_RGT>"]
    max_new_tokens = 50
    default_temperature = 0.7
    default_action = {"accel": False, "brake": False, "left": False, "right": False}

    @classmethod
    def build_training_prompt(cls, item):
        return cls.PROMPT

    @classmethod
    def build_inference_prompt(cls):
        return cls.PROMPT

    @classmethod
    def encode_action(cls, actions, item=None):
        throttle = "<THROTTLE_FWD>" if actions[0] > 0.5 else ("<THROTTLE_BRK>" if actions[1] > 0.5 else "<THROTTLE_NONE>")
        steer    = "<STEER_LFT>"    if actions[2] > 0.5 else ("<STEER_RGT>"    if actions[3] > 0.5 else "<STEER_NONE>")
        thought  = item.get("text", "") if item else ""
        return f"<thought>{thought}</thought>{throttle}{steer}"

    @classmethod
    def decode_action(cls, raw_output):
        return {
            "accel": bool(re.search(r"<THROTTLE_FWD>", raw_output)),
            "brake": bool(re.search(r"<THROTTLE_BRK>", raw_output)),
            "left":  bool(re.search(r"<STEER_LFT>",    raw_output)),
            "right": bool(re.search(r"<STEER_RGT>",    raw_output)),
        }

    @classmethod
    def encode_for_stats(cls, actions, item=None):
        throttle = "<THROTTLE_FWD>" if actions[0] > 0.5 else ("<THROTTLE_BRK>" if actions[1] > 0.5 else "<THROTTLE_NONE>")
        steer    = "<STEER_LFT>"    if actions[2] > 0.5 else ("<STEER_RGT>"    if actions[3] > 0.5 else "<STEER_NONE>")
        return throttle + steer


AGENT_REGISTRY = {
    "TwoTokenVLAAgent": TwoTokenVLAAgent,
    "CoTVLAAgent": CoTVLAAgent,
}
