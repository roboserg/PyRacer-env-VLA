import threading

import torch
import os
import time
from PIL import Image
from transformers import AutoProcessor, SmolVLMForConditionalGeneration

BASE_MODEL_ID = "HuggingFaceTB/SmolVLM-Instruct"


def get_processor(model_id_or_path=None):
    if model_id_or_path is None:
        model_id_or_path = BASE_MODEL_ID
    elif not os.path.exists(model_id_or_path):
        print(f"Model path {model_id_or_path} not found, loading processor from {BASE_MODEL_ID}...")
        model_id_or_path = BASE_MODEL_ID

    print(f"Loading processor from {model_id_or_path}...")
    return AutoProcessor.from_pretrained(model_id_or_path)


def get_model(model_id_or_path=None):
    if model_id_or_path is None:
        model_id_or_path = BASE_MODEL_ID

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.isabs(model_id_or_path) and not os.path.exists(model_id_or_path):
        print(f"Model path {model_id_or_path} not found, loading base model from {BASE_MODEL_ID}...")
        model_id_or_path = BASE_MODEL_ID

    print(f"Loading model from {model_id_or_path}...")
    return SmolVLMForConditionalGeneration.from_pretrained(
        model_id_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device)


def get_model_and_processor(model_id_or_path=None, agent_cls=None):
    if agent_cls is None:
        from src.vla.vla_agent import TwoTokenVLAAgent
        agent_cls = TwoTokenVLAAgent

    processor = get_processor(model_id_or_path)
    tokenizer = processor.tokenizer

    action_tokens = agent_cls.tokens
    num_added = tokenizer.add_tokens(action_tokens)
    print(f"Added {num_added} new action tokens.")

    for token in action_tokens:
        ids = tokenizer.encode(token, add_special_tokens=False)
        assert len(ids) == 1, (
            f"CRITICAL: Token {token} was split into {len(ids)} pieces. This will break the VLA action head."
        )
    print("Token Verification Passed: All actions are single tokens.")

    model = get_model(model_id_or_path)
    model.resize_token_embeddings(len(tokenizer))
    model.eval()

    return model, processor


def run_inference(model, processor, image, prompt: str, max_new_tokens: int, temperature: float = 0.0):
    """
    Run inference on a single image using the VLA model.
    Returns (raw_text, inference_time_ms).
    """
    device = model.device

    if isinstance(image, str):
        raw_image = Image.open(image).convert("RGB")
    else:
        raw_image = image

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    if not prompt_text.endswith(" "):
        prompt_text += " "

    inputs = processor(
        text=prompt_text,
        images=raw_image,
        return_tensors="pt",
        do_resize=True,
        size={"longest_edge": 384},
    ).to(device)

    start_time = time.perf_counter()
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            use_cache=True,
        )
    inference_time_ms = (time.perf_counter() - start_time) * 1000

    prompt_len = inputs["input_ids"].shape[1]
    generated_only = generated_ids[0][prompt_len:]
    response = processor.decode(generated_only, skip_special_tokens=True)
    return response, inference_time_ms


class InferenceThread:
    """Runs VLA inference on a background thread. One request at a time."""

    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self._thread = None
        self._result = None
        self._ready = threading.Event()

    def submit(self, image, prompt, max_new_tokens, temperature=0.0):
        """Start inference in background. Non-blocking."""
        self._ready.clear()
        self._result = None
        self._thread = threading.Thread(
            target=self._run,
            args=(image, prompt, max_new_tokens, temperature),
            daemon=True,
        )
        self._thread.start()

    def _run(self, image, prompt, max_new_tokens, temperature):
        self._result = run_inference(self.model, self.processor, image, prompt, max_new_tokens, temperature)
        self._ready.set()

    def poll(self):
        """Check if result is ready. Returns (text, ms) or None."""
        if self._ready.is_set():
            return self._result
        return None
