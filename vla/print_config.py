import torch
from transformers import AutoProcessor, SmolVLMForConditionalGeneration

BASE_MODEL_ID = "HuggingFaceTB/SmolVLM-Instruct"

print(f"Loading processor from {BASE_MODEL_ID}...")
processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)

print("\n=== PROCESSOR CONFIG ===")
print(processor.__class__.__name__)
print(f"Image processor: {processor.image_processor.__class__.__name__}")
print(f"Image processor config: {processor.image_processor.to_dict()}")

print(f"\nTokenizer: {processor.tokenizer.__class__.__name__}")
print(f"Tokenizer config: {processor.tokenizer.to_dict()}")

print(f"\nVision config: {processor.image_processor.feature_extractor_size}")
print(f"Image mean: {processor.image_processor.image_mean}")
print(f"Image std: {processor.image_processor.image_std}")
print(f"Image size: {processor.image_processor.size}")
print(f"Do rescale: {processor.image_processor.do_rescale}")
print(f"Do normalize: {processor.image_processor.do_normalize}")
print(f"Do reszie: {processor.image_processor.do_resize}")
print(f"Do pad: {processor.image_processor.do_pad}")

print("\n=== MODEL CONFIG ===")
print(f"Loading model config from {BASE_MODEL_ID}...")
model = SmolVLMForConditionalGeneration.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
)
print(f"Model config: {model.config.to_dict()}")
