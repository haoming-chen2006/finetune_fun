"""Load Synthetic-Persona-Chat dataset from local arrow file or HuggingFace.

Usage:
    python datasets/personas.py                    # Load from HuggingFace
    python datasets/personas.py --from-arrow       # Load from local arrow file
"""
from datasets import load_dataset, Dataset
import pyarrow as pa
import argparse

# Local arrow file path
ARROW_PATH = "/home/haoming/finetune_fun/data/persona_chat/data-00000-of-00001.arrow"


def load_from_arrow(path: str = ARROW_PATH) -> Dataset:
    """Load dataset from arrow file."""
    print(f"Loading from arrow file: {path}")
    table = pa.ipc.open_file(path).read_all()
    ds = Dataset(table)
    return ds


def load_from_hf() -> Dataset:
    """Load dataset from HuggingFace."""
    print("Loading from HuggingFace...")
    ds = load_dataset("google/Synthetic-Persona-Chat", split="train")
    return ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-arrow", action="store_true",
                        help="Load from local arrow file instead of HuggingFace")
    args = parser.parse_args()
    
    if args.from_arrow:
        ds = load_from_arrow()
    else:
        ds = load_from_hf()
    
    print(f"Loaded {len(ds)} samples")
    
    # Show sample
    if len(ds) > 0:
        sample = ds[0]
        print(f"\nSample keys: {list(sample.keys())}")
        print(f"User 1 Persona: {sample.get('User 1 Persona', [])[:2]}...")
        print(f"User 2 Persona: {sample.get('User 2 Persona', [])[:2]}...")
        print(f"Conversation turns: {len(sample.get('Conversation', []))}")


if __name__ == "__main__":
    main()