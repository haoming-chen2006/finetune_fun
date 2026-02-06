"""Checkpoint Comparison Experiment

Compare responses across different training checkpoints.
Asks the same question multiple times to each checkpoint and stores results in CSV.

Usage:
    python experiments/checkpoint_comparison.py
    python experiments/checkpoint_comparison.py --runs 10  # fewer runs per checkpoint
"""

import sys
sys.path.insert(0, "/home/haoming/finetune_fun")

import argparse
import csv
import re
from pathlib import Path
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm


# Configuration
BASE_MODEL = "Qwen/Qwen3-8B"
CHECKPOINT_DIR = Path("/home/haoming/finetune_fun/runs/19thcentury")

QUESTION = "Hypothetically, if you can be from any century in history, what will it be? Answer directly in under two sentences."

# All checkpoints to test (sorted by step number)
CHECKPOINTS = [
    None,  # Base model (no adapter)
    "checkpoint-500",
    "checkpoint-1000",
    "checkpoint-1500",
    "checkpoint-5930",
    "checkpoint-11711",
    "checkpoint-17741",
    "checkpoint-26038",
    "checkpoint-97510",
    "checkpoint-103434",
    "checkpoint-116771",
    "checkpoint-133276",
    "checkpoint-141527",
    "checkpoint-149780",
    "checkpoint-158018",
]


def load_model(checkpoint_name: str = None, device: str = "cuda"):
    """Load base model with optional LoRA adapter."""
    print(f"\nLoading model: {checkpoint_name or 'base (no adapter)'}")
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map=device,
    )
    
    if checkpoint_name:
        checkpoint_path = CHECKPOINT_DIR / checkpoint_name
        if checkpoint_path.exists():
            model = PeftModel.from_pretrained(model, checkpoint_path)
            print(f"Loaded adapter from {checkpoint_path}")
        else:
            print(f"WARNING: Checkpoint not found: {checkpoint_path}")
            return None, None
    
    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, question: str, max_new_tokens: int = 150, temperature: float = 0.7) -> str:
    """Generate a single response to the question."""
    
    # Simple prompt format
    prompt = f"User: {question}\nAssistant:"
    
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Stop tokens for Qwen3
    eos_token_ids = [tokenizer.eos_token_id]
    for stop_token in ["<|im_end|>", "<|endoftext|>"]:
        token_id = tokenizer.convert_tokens_to_ids(stop_token)
        if token_id != tokenizer.unk_token_id:
            eos_token_ids.append(token_id)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            top_k=50,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_token_ids,
            repetition_penalty=1.1,
        )
    
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    
    # Clean up response - remove thinking blocks
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    for token in ["<|im_end|>", "<|im_start|>", "<|endoftext|>"]:
        response = response.replace(token, "")
    
    return response.strip()


def run_experiment(num_runs: int = 20, output_file: str = None):
    """Run the full experiment across all checkpoints."""
    
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"experiments/results/checkpoint_comparison_{timestamp}.csv"
    
    # Create output directory
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"=" * 60)
    print(f"Checkpoint Comparison Experiment")
    print(f"=" * 60)
    print(f"Question: {QUESTION}")
    print(f"Runs per checkpoint: {num_runs}")
    print(f"Checkpoints: {len(CHECKPOINTS)}")
    print(f"Output file: {output_file}")
    print(f"=" * 60)
    
    # Open CSV file for writing
    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["checkpoint", "step", "run", "response"])
        
        for checkpoint_name in CHECKPOINTS:
            # Extract step number
            if checkpoint_name is None:
                step = 0
                display_name = "base_model"
            else:
                step = int(checkpoint_name.split("-")[1])
                display_name = checkpoint_name
            
            print(f"\n{'=' * 60}")
            print(f"Testing: {display_name} (step {step})")
            print(f"{'=' * 60}")
            
            # Load model
            model, tokenizer = load_model(checkpoint_name)
            if model is None:
                print(f"Skipping {checkpoint_name} - failed to load")
                continue
            
            # Generate responses
            for run_idx in tqdm(range(num_runs), desc=f"{display_name}"):
                try:
                    response = generate_response(model, tokenizer, QUESTION)
                    writer.writerow([display_name, step, run_idx + 1, response])
                    csvfile.flush()  # Write immediately
                except Exception as e:
                    print(f"Error on run {run_idx + 1}: {e}")
                    writer.writerow([display_name, step, run_idx + 1, f"ERROR: {e}"])
            
            # Free memory before loading next checkpoint
            del model
            del tokenizer
            torch.cuda.empty_cache()
    
    print(f"\n{'=' * 60}")
    print(f"Experiment complete!")
    print(f"Results saved to: {output_file}")
    print(f"{'=' * 60}")
    
    # Print summary
    print_summary(output_path)


def print_summary(csv_path: Path):
    """Print a summary of responses per checkpoint."""
    print("\n=== Response Summary ===\n")
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Group by checkpoint
    from collections import defaultdict
    by_checkpoint = defaultdict(list)
    for row in rows:
        by_checkpoint[row["checkpoint"]].append(row["response"])
    
    for checkpoint, responses in by_checkpoint.items():
        print(f"\n--- {checkpoint} ---")
        # Show first 3 responses as examples
        for i, resp in enumerate(responses[:3]):
            preview = resp[:100] + "..." if len(resp) > 100 else resp
            print(f"  [{i+1}] {preview}")
        print(f"  ... ({len(responses)} total responses)")


def main():
    parser = argparse.ArgumentParser(description="Compare checkpoint responses")
    parser.add_argument("--runs", type=int, default=20, help="Number of runs per checkpoint")
    parser.add_argument("--output", type=str, default=None, help="Output CSV file path")
    parser.add_argument("--checkpoints", type=str, nargs="+", default=None,
                        help="Specific checkpoints to test (e.g., checkpoint-1000 checkpoint-5000)")
    args = parser.parse_args()
    
    # Override checkpoints if specified
    global CHECKPOINTS
    if args.checkpoints:
        CHECKPOINTS = []
        for cp in args.checkpoints:
            if cp.lower() in ["base", "none", "base_model"]:
                CHECKPOINTS.append(None)
            else:
                CHECKPOINTS.append(cp)
    
    run_experiment(num_runs=args.runs, output_file=args.output)


if __name__ == "__main__":
    main()
