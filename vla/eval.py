import argparse
import torch
import sys
from vla.model import get_model_and_processor
from vla.train import MODEL_DIR, get_dataset, eval_model

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained VLA model on dataset samples.")
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=50, 
        help="Number of samples to run for accuracy calculation (default: 50)"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        default=MODEL_DIR, 
        help=f"Path to the model to evaluate (default: {MODEL_DIR})"
    )
    args = parser.parse_args()

    # 1. Load Model & Processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Model from: {args.model_path}")
    
    try:
        model, processor = get_model_and_processor(args.model_path)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # 2. Load Dataset
    print("\nLoading Dataset...")
    dataset = get_dataset(processor, processor.tokenizer)

    # 3. Run Evaluation
    print(f"\nRunning Evaluation on {args.num_samples} random samples...")
    eval_model(model, processor, dataset, num_samples=args.num_samples)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
