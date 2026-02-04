"""Load and preprocess 19th Century British Library Books dataset.

This script downloads, filters, and saves the dataset to disk.
Run this ONCE, then use the saved path in your experiment config.
"""
from datasets import load_dataset
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="./data/19thcentury_english",
                        help="Path to save the processed dataset")
    args = parser.parse_args()
    
    print("Loading dataset from HuggingFace...")
    ds = load_dataset("TheBritishLibrary/blbooks", "1500_1899", split="train")
    print(f"Raw dataset: {len(ds)} samples")
    
    print("Filtering to English only...")
    english_ds = ds.filter(lambda x: x["Language_1"] == "English")
    print(f"English only: {len(english_ds)} samples")
    
    print("Filtering empty text...")
    english_text = english_ds.filter(
        lambda x: x.get("text") is not None and len(x["text"].strip()) > 0
    )
    print(f"Non-empty text: {len(english_text)} samples")
    
    print(f"Saving to {args.output}...")
    english_text.save_to_disk(args.output)
    print("Done!")
    print(f"\nUse this path in your config: dataset_path: \"{args.output}\"")


if __name__ == "__main__":
    main()