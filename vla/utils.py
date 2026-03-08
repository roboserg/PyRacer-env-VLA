import torch
import os
from transformers import AutoProcessor, SmolVLMForConditionalGeneration

BASE_MODEL_ID = "HuggingFaceTB/SmolVLM-Instruct"
MODEL_DIR = "./models/smolvla-racer-final"

def load_model_and_processor(model_dir=MODEL_DIR, device="cuda"):
    """
    Standard loader for the PyRacer VLA model.
    Matches the training setup in train.py.
    """
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory {model_dir} not found!")

    # Try to load processor from the trained model directory first
    # This ensures we get the same added tokens and tokenizer config as training
    try:
        print(f"Loading processor from {model_dir}...")
        processor = AutoProcessor.from_pretrained(model_dir)
    except Exception as e:
        print(f"Warning: Could not load processor from {model_dir}: {e}")
        print(f"Falling back to base model: {BASE_MODEL_ID}")
        processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)
        
    tokenizer = processor.tokenizer

    # Define and add action tokens (MUST match train.py)
    # Note: If processor was loaded from model_dir, these might already be there,
    # but adding them again is safe as it will return 0 new tokens if they exist.
    action_tokens = [
        "<FWD_0>", "<FWD_1>", 
        "<LFT_0>", "<LFT_1>", 
        "<RGT_0>", "<RGT_1>", 
        "<BRK_0>", "<BRK_1>",
    ]
    num_added = tokenizer.add_tokens(action_tokens)
    print(f"Added {num_added} new action tokens.")

    # --- TOKEN VERIFICATION ---
    for token in action_tokens:
        ids = tokenizer.encode(token, add_special_tokens=False)
        assert len(ids) == 1, f"CRITICAL: Token {token} was split into {len(ids)} pieces. This will break the VLA action head."
    print("Token Verification Passed: All actions are single tokens.")

    print(f"Loading model from {model_dir}...")
    model = SmolVLMForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device)

    # CRITICAL: Resize embeddings to accommodate new tokens
    model.resize_token_embeddings(len(tokenizer))
    model.eval()

    return model, processor
