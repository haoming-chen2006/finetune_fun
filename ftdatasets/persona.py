"""Load and save Synthetic-Persona-Chat dataset.

This script downloads and saves the dataset to disk.
Run this ONCE, then use the saved path in your experiment config.
"""
from datasets import load_dataset
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="./data/persona_chat",
                        help="Path to save the dataset")
    args = parser.parse_args()
    
    print("Loading Synthetic-Persona-Chat from HuggingFace...")
    ds = load_dataset("google/Synthetic-Persona-Chat", split="train")
    print(f"Dataset: {len(ds)} conversations")
    
    # Show sample structure
    sample = ds[0]
    print(f"\nSample structure:")
    print(f"  User 1 Persona: {sample.get('User 1 Persona', [])[:2]}...")
    print(f"  User 2 Persona: {sample.get('User 2 Persona', [])[:2]}...")
    print(f"  Conversation turns: {len(sample.get('Conversation', []))}")
    
    print(f"\nSaving to {args.output}...")
    ds.save_to_disk(args.output)
    print("Done!")
    print(f"\nUse this path in your config: dataset_path: \"{args.output}\"")


if __name__ == "__main__":
    main()
