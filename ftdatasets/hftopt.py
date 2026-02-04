#huggingface to pytorch data loader utils
"""
Utility class to convert HuggingFace datasets to PyTorch Dataset/DataLoader objects.

Supports two modes:
1. Batched mode (default): For simple text data like books - concatenate and chunk
2. Per-sample mode: For structured data like conversations - process each sample individually

Customization hooks:
- format_fn: Transform raw data before tokenization (e.g., add special tokens)
- tokenize_fn: Custom tokenization logic - done before tokenization/during it
- group_fn: Custom chunking/grouping logic (batched mode)
- mask_fn: Custom label masking (e.g., mask certain speakers in conversations) - done after tokenization
- process_sample_fn: Full control over per-sample processing (per-sample mode)
"""

from typing import Callable, Optional, Dict, Any, List, Iterator, Tuple
from transformers import AutoTokenizer, PreTrainedTokenizer
from torch.utils.data import DataLoader, Dataset as TorchDataset
from datasets import Dataset
import torch
from tqdm import tqdm


# ============ MASK FUNCTIONS ============
# These can be passed to create custom label masking

def no_mask(input_ids: List[int], **kwargs) -> List[int]:
    """Default: no masking, labels = input_ids (standard causal LM)."""
    return input_ids.copy()


def mask_by_token_ids(
    input_ids: List[int],
    mask_token_ids: List[int],
    **kwargs
) -> List[int]:
    """Mask specific token IDs with -100."""
    labels = []
    for tid in input_ids:
        if tid in mask_token_ids:
            labels.append(-100)
        else:
            labels.append(tid)
    return labels


def mask_segments_by_markers(
    input_ids: List[int],
    tokenizer: PreTrainedTokenizer,
    mask_start_tokens: List[str],
    mask_end_tokens: List[str],
    **kwargs
) -> List[int]:
    """
    Mask segments between start/end marker tokens.
    
    Example: mask_start_tokens=["<user2>"], mask_end_tokens=["</user2>"]
    will mask everything between <user2> and </user2>.
    """
    # Get token IDs for markers
    mask_start_ids = set()
    mask_end_ids = set()
    for token in mask_start_tokens:
        ids = tokenizer.encode(token, add_special_tokens=False)
        if len(ids) == 1:
            mask_start_ids.add(ids[0])
    for token in mask_end_tokens:
        ids = tokenizer.encode(token, add_special_tokens=False)
        if len(ids) == 1:
            mask_end_ids.add(ids[0])
    
    labels = []
    in_mask_segment = False
    
    for tid in input_ids:
        if tid in mask_start_ids:
            in_mask_segment = True
            labels.append(-100)  # Mask the start token too
        elif tid in mask_end_ids:
            labels.append(-100)  # Mask the end token too
            in_mask_segment = False
        elif in_mask_segment:
            labels.append(-100)
        else:
            labels.append(tid)
    
    return labels


def mask_all_except_segments(
    input_ids: List[int],
    tokenizer: PreTrainedTokenizer,
    train_start_tokens: List[str],
    train_end_tokens: List[str],
    **kwargs
) -> List[int]:
    """
    Mask everything EXCEPT segments between train_start and train_end tokens.
    
    Example: train_start_tokens=["<user1>"], train_end_tokens=["</user1>"]
    will ONLY train on content between <user1> and </user1>, masking everything else.
    """
    # Get token IDs for markers
    train_start_ids = set()
    train_end_ids = set()
    for token in train_start_tokens:
        ids = tokenizer.encode(token, add_special_tokens=False)
        if len(ids) == 1:
            train_start_ids.add(ids[0])
    for token in train_end_tokens:
        ids = tokenizer.encode(token, add_special_tokens=False)
        if len(ids) == 1:
            train_end_ids.add(ids[0])
    
    labels = []
    in_train_segment = False
    
    for tid in input_ids:
        if tid in train_start_ids:
            in_train_segment = True
            labels.append(-100)  # Don't train on the marker itself
        elif tid in train_end_ids:
            labels.append(-100)  # Don't train on the marker itself
            in_train_segment = False
        elif in_train_segment:
            labels.append(tid)  # Train on this!
        else:
            labels.append(-100)  # Mask everything else
    
    return labels


# ============ DATASETS ============

class LMDataset(TorchDataset):
    """PyTorch Dataset for language model training."""
    
    def __init__(self, input_ids: List[List[int]], labels: Optional[List[List[int]]] = None):
        self.input_ids = input_ids
        self.labels = labels if labels is not None else input_ids
    
    def __len__(self) -> int:
        return len(self.input_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }


class HFBackedDataset(TorchDataset):
    """PyTorch Dataset backed by HuggingFace Dataset (memory-mapped, lazy loading)."""
    
    def __init__(self, hf_dataset: Dataset):
        self.hf_dataset = hf_dataset
    
    def __len__(self) -> int:
        return len(self.hf_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.hf_dataset[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long)
        }


class LMDataLoader:
    """Custom DataLoader wrapper with __iter__ and __len__."""
    
    def __init__(
        self,
        dataset: TorchDataset,
        batch_size: int = 8,
        shuffle: bool = True,
        drop_last: bool = False,
        **kwargs
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self._dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, **kwargs
        )
    
    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        return iter(self._dataloader)


# ============ MAIN CONVERTER ============

class HFDataConverter:
    """
    Converts HuggingFace datasets to PyTorch Dataset/DataLoader.
    
    Supports two processing modes:
    1. Batched (default): Tokenize all → concatenate → chunk into seq_len blocks
       Good for: plain text (books, articles)
    
    2. Per-sample: Process each sample individually with custom logic
       Good for: structured data (conversations, Q&A pairs)
    
    Args:
        model_name_or_path: HuggingFace model name (used to load tokenizer if not provided)
        tokenizer: Optional pre-configured tokenizer
        seq_len: Sequence length for output samples
        text_column: Column name containing text (for batched mode)
        special_tokens: List of special tokens to add to tokenizer
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        seq_len: int = 512,
        text_column: str = "text",
        special_tokens: Optional[List[str]] = None,
    ):
        self.model_name_or_path = model_name_or_path
        self.seq_len = seq_len
        self.text_column = text_column
        
        # Setup tokenizer
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        else:
            self.tokenizer = tokenizer
        
        # Add special tokens if provided
        if special_tokens:
            self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    # ========== BATCHED MODE (for plain text) ==========
    
    def _default_tokenize(self, batch: Dict[str, Any]) -> Dict[str, List]:
        """Default tokenization - just tokenize the text column, return only input_ids."""
        result = self.tokenizer(batch[self.text_column], truncation=False)
        # Only return input_ids - attention_mask causes issues with grouping
        return {"input_ids": result["input_ids"]}
    
    def _default_group_texts(self, examples: Dict[str, List]) -> Dict[str, List]:
        """Concatenate all tokens and chunk into seq_len blocks."""
        all_ids = []
        for ids in examples["input_ids"]:
            all_ids.extend(ids)
        
        total_len = (len(all_ids) // self.seq_len) * self.seq_len
        all_ids = all_ids[:total_len]
        
        input_ids = [all_ids[i:i + self.seq_len] for i in range(0, total_len, self.seq_len)]
        return {"input_ids": input_ids, "labels": [ids.copy() for ids in input_ids]}
    
    def process_batched(
        self,
        dataset: Dataset,
        format_fn: Optional[Callable[[Dataset], Dataset]] = None,
        tokenize_fn: Optional[Callable] = None,
        group_fn: Optional[Callable] = None,
        mask_fn: Optional[Callable] = None,
        num_proc: Optional[int] = None,
        cache_path: Optional[str] = None,
    ) -> LMDataset:
        """
        Process dataset in batched mode (tokenize all → concatenate → chunk).
        
        Args:
            dataset: HuggingFace Dataset
            format_fn: Transform dataset before tokenization (filter, select columns, etc.)
                      Function(dataset) -> dataset
            tokenize_fn: Custom tokenization function(batch) -> {"input_ids": [...]}
            group_fn: Custom grouping function(examples) -> {"input_ids": [...], "labels": [...]}
            mask_fn: Custom masking function(input_ids, tokenizer=...) -> labels
            num_proc: Parallel processing workers
            cache_path: Path to save/load processed dataset (speeds up subsequent runs)
        """
        import os
        
        # Try to load from cache
        if cache_path and os.path.exists(cache_path):
            print(f"Loading cached processed dataset from {cache_path}")
            processed = Dataset.load_from_disk(cache_path)
            # Use memory-mapped HF dataset (lazy loading, doesn't load all into RAM)
            return HFBackedDataset(processed)
        
        # Apply format function first (filtering, column selection, etc.)
        if format_fn is not None:
            dataset = format_fn(dataset)
        
        tokenize_fn = tokenize_fn or self._default_tokenize
        group_fn = group_fn or self._default_group_texts
        
        # Tokenize - disable HF cache to avoid stale attention_mask issues
        tokenized = dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=num_proc,
            load_from_cache_file=False,  # Force fresh tokenization
            desc="Tokenizing"
        )
        
        # Remove attention_mask if present (tokenizer adds it, but grouping doesn't need it)
        if "attention_mask" in tokenized.column_names:
            tokenized = tokenized.remove_columns(["attention_mask"])
        
        # Group into chunks - also disable cache
        processed = tokenized.map(
            group_fn,
            batched=True,
            num_proc=num_proc,
            load_from_cache_file=False,  # Force fresh grouping
            desc="Grouping"
        )
        
        input_ids = processed["input_ids"]
        
        # Apply masking if provided
        if mask_fn:
            labels = [mask_fn(ids, tokenizer=self.tokenizer) for ids in input_ids]
        else:
            labels = processed["labels"]
        
        # Save to cache if path provided
        if cache_path:
            print(f"Saving processed dataset to {cache_path}")
            # Create a dataset with input_ids and labels for caching
            cache_dataset = Dataset.from_dict({"input_ids": input_ids, "labels": labels})
            cache_dataset.save_to_disk(cache_path)
        
        return LMDataset(input_ids=input_ids, labels=labels)
    
    # ========== PER-SAMPLE MODE (for structured data) ==========
    
    def process_per_sample(
        self,
        dataset: Dataset,
        process_fn: Callable[[Dict, PreTrainedTokenizer, int], Optional[Dict]],
        show_progress: bool = True,
    ) -> LMDataset:
        """
        Process each sample individually with a custom function.
        
        Args:
            dataset: HuggingFace Dataset
            process_fn: Function(sample, tokenizer, seq_len) -> {"input_ids": [...], "labels": [...]}
                       Return None to skip a sample
            show_progress: Show tqdm progress bar
        """
        input_ids_list = []
        labels_list = []
        
        iterator = tqdm(dataset, desc="Processing") if show_progress else dataset
        
        for sample in iterator:
            result = process_fn(sample, self.tokenizer, self.seq_len)
            if result is not None:
                input_ids_list.append(result["input_ids"])
                labels_list.append(result["labels"])
        
        return LMDataset(input_ids=input_ids_list, labels=labels_list)
    
    # ========== CONVENIENCE METHODS ==========
    
    def get_dataset(
        self,
        dataset: Dataset,
        mode: str = "batched",
        # Batched mode options
        format_fn: Optional[Callable[[Dataset], Dataset]] = None,
        tokenize_fn: Optional[Callable] = None,
        group_fn: Optional[Callable] = None,
        mask_fn: Optional[Callable] = None,
        num_proc: Optional[int] = None,
        cache_path: Optional[str] = None,
        # Per-sample mode options
        process_fn: Optional[Callable] = None,
    ) -> LMDataset:
        """
        Get PyTorch Dataset from HuggingFace Dataset.
        
        Args:
            dataset: HuggingFace Dataset
            mode: "batched" or "per_sample"
            
            For batched mode:
                format_fn: Transform dataset before tokenization (filter, etc.)
                tokenize_fn, group_fn, mask_fn, num_proc
                cache_path: Path to save/load processed data (speeds up subsequent runs)
            
            For per_sample mode:
                process_fn: Function(sample, tokenizer, seq_len) -> {"input_ids", "labels"}
        """
        if mode == "batched":
            return self.process_batched(
                dataset, format_fn, tokenize_fn, group_fn, mask_fn, num_proc, cache_path
            )
        elif mode == "per_sample":
            if process_fn is None:
                raise ValueError("process_fn is required for per_sample mode")
            return self.process_per_sample(dataset, process_fn)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'batched' or 'per_sample'")
    
    def get_dataloader(
        self,
        dataset: Dataset,
        batch_size: int = 8,
        shuffle: bool = True,
        drop_last: bool = False,
        mode: str = "batched",
        format_fn: Optional[Callable[[Dataset], Dataset]] = None,
        tokenize_fn: Optional[Callable] = None,
        group_fn: Optional[Callable] = None,
        mask_fn: Optional[Callable] = None,
        num_proc: Optional[int] = None,
        cache_path: Optional[str] = None,
        process_fn: Optional[Callable] = None,
        **dataloader_kwargs
    ) -> LMDataLoader:
        """Get PyTorch DataLoader from HuggingFace Dataset."""
        torch_dataset = self.get_dataset(
            dataset, mode, format_fn, tokenize_fn, group_fn, mask_fn, num_proc, cache_path, process_fn
        )
        return LMDataLoader(
            torch_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            **dataloader_kwargs
        )
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size (useful after adding special tokens)."""
        return len(self.tokenizer)