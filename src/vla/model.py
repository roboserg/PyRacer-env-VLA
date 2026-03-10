import torch
import os
import re
from PIL import Image
from transformers import AutoProcessor, SmolVLMForConditionalGeneration

BASE_MODEL_ID = "HuggingFaceTB/SmolVLM-Instruct"


def get_processor(model_id_or_path=None):
    """
    Loads the processor for the VLA model.
    """
    if model_id_or_path is None:
        model_id_or_path = BASE_MODEL_ID
    elif not os.path.exists(model_id_or_path):
        print(
            f"Model path {model_id_or_path} not found, loading processor from {BASE_MODEL_ID}..."
        )
        model_id_or_path = BASE_MODEL_ID

    print(f"Loading processor from {model_id_or_path}...")
    processor = AutoProcessor.from_pretrained(model_id_or_path)

    return processor


def get_model(model_id_or_path=None):
    """
    Loads the VLA model.
    If model_id_or_path is None or doesn't exist as a local path, loads from BASE_MODEL_ID.
    """
    if model_id_or_path is None:
        model_id_or_path = BASE_MODEL_ID

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # If local path doesn't exist, fall back to base model from HuggingFace Hub
    if not os.path.isabs(model_id_or_path) and not os.path.exists(model_id_or_path):
        print(
            f"Model path {model_id_or_path} not found, loading base model from {BASE_MODEL_ID}..."
        )
        model_id_or_path = BASE_MODEL_ID

    print(f"Loading model from {model_id_or_path}...")
    model = SmolVLMForConditionalGeneration.from_pretrained(
        model_id_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device)

    return model


def get_model_and_processor(model_id_or_path=None):
    """
    Standard loader for the PyRacer VLA model.
    """
    processor = get_processor(model_id_or_path)
    tokenizer = processor.tokenizer

    # Define and add action tokens (MUST match train.py)
    action_tokens = [
        "<FWD_0>",
        "<FWD_1>",
        "<LFT_0>",
        "<LFT_1>",
        "<RGT_0>",
        "<RGT_1>",
        "<BRK_0>",
        "<BRK_1>",
    ]
    num_added = tokenizer.add_tokens(action_tokens)
    print(f"Added {num_added} new action tokens.")

    # --- TOKEN VERIFICATION ---
    for token in action_tokens:
        ids = tokenizer.encode(token, add_special_tokens=False)
        assert len(ids) == 1, (
            f"CRITICAL: Token {token} was split into {len(ids)} pieces. This will break the VLA action head."
        )
    print("Token Verification Passed: All actions are single tokens.")

    model = get_model(model_id_or_path)

    # CRITICAL: Resize embeddings to accommodate new tokens
    model.resize_token_embeddings(len(tokenizer))
    model.eval()

    return model, processor


def infer(model, processor, image):
    """
    Run inference on a single image using the VLA model.
    Yields raw model predictions (text).
    Uses model.device to determine where to run inference.
    """
    device = model.device

    # Handle both PIL Image and file path
    if isinstance(image, str):
        raw_image = Image.open(image).convert("RGB")
    else:
        raw_image = image

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Action:"},
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    if not prompt.endswith(" "):
        prompt += " "

    inputs = processor(
        text=prompt,
        images=raw_image,
        return_tensors="pt",
        do_resize=True,
        size={"longest_edge": 384},
    ).to(device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            temperature=0.0,
            use_cache=True,
        )

    response = processor.decode(generated_ids[0], skip_special_tokens=True)
    return response


def post_process_output(raw_output: str) -> dict:
    """
    Parses the raw model output into a structured action DTO.
    """
    action = {
        "accel": bool(re.search(r"<FWD_1>", raw_output)),
        "brake": bool(re.search(r"<BRK_1>", raw_output)),
        "left": bool(re.search(r"<LFT_1>", raw_output)),
        "right": bool(re.search(r"<RGT_1>", raw_output)),
    }
    return action
