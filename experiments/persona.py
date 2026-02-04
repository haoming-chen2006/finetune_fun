"""
Persona-Chat Fine-tuning Experiment

Trains a persona-conditioned conversation model.
The model learns to generate User 1's responses given:
  - A persona description (context, masked from loss)
  - User 2's messages (context, masked from loss)

Training format:
  <persona>I like video games. I live in NYC.</persona>     [MASKED - context]
  <user2>Hi! I'm Bob.</user2>                               [MASKED - context]
  <user1>Hi Bob, I'm Alice!</user1>                         [TRAINED - loss computed]
  <user2>What do you do for fun?</user2>                    [MASKED - context]
  <user1>I like to play video games!</user1>                [TRAINED - loss computed]

At inference time, provide ANY persona and the model will respond consistently.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
import argparse
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any
import torch
from torch.optim import AdamW
from tqdm import tqdm
import re

from ftdatasets.hftopt import HFDataConverter, mask_all_except_segments


# ============ SPECIAL TOKENS ============
PERSONA_START = "<persona>"
PERSONA_END = "</persona>"
USER1_START = "<user1>"
USER1_END = "</user1>"
USER2_START = "<user2>"
USER2_END = "</user2>"

SPECIAL_TOKENS = [PERSONA_START, PERSONA_END, USER1_START, USER1_END, USER2_START, USER2_END]


# ============ CONFIG ============
@dataclass
class LoRAConfig:
    r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: Optional[List[str]] = None
    bias: str = "none"


@dataclass
class PersonaConfig:
    """Persona-specific settings."""
    train_on_user: int = 1              # Which user to train on (1 or 2)
    max_turns: Optional[int] = None     # Max conversation turns (None = all)
    include_persona: bool = True        # Include persona in context
    persona_index: Optional[int] = None # Train on specific persona index (None = all)


@dataclass 
class TrainConfig:
    model_name: str = "Qwen/Qwen3-8B"
    seq_len: int = 512
    
    # Dataset - either HF name or local arrow path
    dataset_name: str = ""  # HuggingFace dataset name (optional)
    dataset_path: str = "/home/haoming/finetune_fun/data/persona_chat/data-00000-of-00001.arrow"  # Path to local arrow file (optional)
    dataset_split: str = "train"
    
    seed: int = 42
    epochs: int = 3
    lr: float = 3e-4
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    weight_decay: float = 0.01
    
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    persona: PersonaConfig = field(default_factory=PersonaConfig)
    
    out_dir: str = "runs/persona"
    save_steps: int = 500
    log_steps: int = 10
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def load_config(path: str) -> TrainConfig:
    data = yaml.safe_load(Path(path).read_text())
    lora_data = data.pop("lora", {})
    persona_data = data.pop("persona", {})
    cfg = TrainConfig(
        **data,
        lora=LoRAConfig(**lora_data),
        persona=PersonaConfig(**persona_data)
    )
    return cfg


# ============ PROCESS FUNCTION FOR HFTOPT ============
def create_persona_processor(persona_cfg: PersonaConfig):
    """
    Create a per-sample processor function for persona conversations.
    
    This function is passed to HFDataConverter.get_dataset(mode="per_sample").
    
    Args:
        persona_cfg: PersonaConfig with train_on_user, max_turns, include_persona settings
    
    Returns:
        A function(sample, tokenizer, seq_len) -> {"input_ids": [...], "labels": [...]}
    """
    def process_persona_sample(
        sample: Dict[str, Any],
        tokenizer: PreTrainedTokenizer,
        seq_len: int
    ) -> Optional[Dict[str, List[int]]]:
        """
        Format a single conversation with special tokens and create loss mask.
        
        Dataset format (google/Synthetic-Persona-Chat):
            - "user 1 personas": newline-separated string of persona facts
            - "user 2 personas": newline-separated string of persona facts
            - "Best Generated Conversation": string with "User 1: ..." and "User 2: ..." turns
        
        Returns:
            {"input_ids": [...], "labels": [...]} where labels has -100 for masked tokens
        """
        train_user = persona_cfg.train_on_user
        
        # Get persona for the user we're training on (lowercase keys, newline-separated)
        persona_key = f"user {train_user} personas"
        persona_str = sample.get(persona_key, "")
        persona_list = [p.strip() for p in persona_str.split("\n") if p.strip()]
        
        # Build formatted text with masking info
        parts = []  # List of (text, is_train_target)
        
        # Add persona (not a training target - just context)
        if persona_cfg.include_persona and persona_list:
            persona_text = " ".join(persona_list)
            parts.append((f"{PERSONA_START}{persona_text}{PERSONA_END}", False))
        
        # Parse conversation from "Best Generated Conversation" string
        conversation_str = sample.get("Best Generated Conversation", "")
        # Split by "User 1:" or "User 2:" while keeping the delimiter
        turns = re.split(r'(?=User [12]:)', conversation_str)
        turns = [t.strip() for t in turns if t.strip()]
        
        if persona_cfg.max_turns is not None:
            turns = turns[:persona_cfg.max_turns * 2]
        
        for turn in turns:
            # Parse "User X: message" format
            match = re.match(r"User (\d+):\s*(.+)", turn, re.DOTALL)
            if not match:
                continue
            
            user_num = int(match.group(1))
            message = match.group(2).strip()
            
            if user_num == train_user:
                parts.append((f"{USER1_START}{message}{USER1_END}", True))
            else:
                parts.append((f"{USER2_START}{message}{USER2_END}", False))
        
        if not parts:
            return None
        
        # Tokenize with loss masking
        input_ids = []
        labels = []
        
        for text, is_target in parts:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            input_ids.extend(tokens)
            
            if is_target:
                labels.extend(tokens)
            else:
                labels.extend([-100] * len(tokens))
        
        # Truncate or pad to seq_len
        if len(input_ids) > seq_len:
            input_ids = input_ids[:seq_len]
            labels = labels[:seq_len]
        elif len(input_ids) < seq_len:
            pad_len = seq_len - len(input_ids)
            input_ids.extend([tokenizer.pad_token_id] * pad_len)
            labels.extend([-100] * pad_len)
        
        return {"input_ids": input_ids, "labels": labels}
    
    return process_persona_sample


# ============ TRAINING ============
def get_lora_config(cfg: LoRAConfig) -> LoraConfig:
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=cfg.r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        bias=cfg.bias,
    )


def train_step(model, batch, optimizer, cfg, scaler=None):
    input_ids = batch["input_ids"].to(cfg.device)
    labels = batch["labels"].to(cfg.device)
    
    if scaler is not None:
        with torch.cuda.amp.autocast():
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss / cfg.gradient_accumulation_steps
        scaler.scale(loss).backward()
    else:
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss / cfg.gradient_accumulation_steps
        loss.backward()
    
    return loss.item() * cfg.gradient_accumulation_steps


def train_epoch(model, dataloader, optimizer, cfg, epoch, global_step, scaler=None):
    model.train()
    total_loss = 0
    num_batches = 0
    
    progress = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    optimizer.zero_grad()
    
    for step, batch in enumerate(progress):
        loss = train_step(model, batch, optimizer, cfg, scaler)
        total_loss += loss
        num_batches += 1
        
        if (step + 1) % cfg.gradient_accumulation_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
            
            optimizer.zero_grad()
            global_step += 1
            
            if global_step % cfg.log_steps == 0:
                avg_loss = total_loss / num_batches
                progress.set_postfix({"loss": f"{avg_loss:.4f}", "step": global_step})
            
            if global_step % cfg.save_steps == 0:
                save_checkpoint(model, optimizer, global_step, cfg)
    
    return total_loss / max(num_batches, 1), global_step


def save_checkpoint(model, optimizer, step, cfg):
    out_path = Path(cfg.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = out_path / f"checkpoint-{step}"
    model.save_pretrained(checkpoint_path)
    torch.save({"optimizer": optimizer.state_dict(), "step": step}, checkpoint_path / "optimizer.pt")
    print(f"Saved checkpoint to {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--train_user", type=int, default=None, help="Which user to train on (1 or 2)")
    parser.add_argument("--persona_index", type=int, default=None, help="Train on specific persona")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else TrainConfig()
    
    # CLI overrides
    if args.model: cfg.model_name = args.model
    if args.epochs: cfg.epochs = args.epochs
    if args.lr: cfg.lr = args.lr
    if args.batch_size: cfg.batch_size = args.batch_size
    if args.train_user: cfg.persona.train_on_user = args.train_user
    if args.persona_index is not None: cfg.persona.persona_index = args.persona_index

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    
    print(f"Config: {cfg}")
    print(f"Persona Config: {cfg.persona}")

    # ========== DATA ==========
    print("\n=== Loading Dataset ===")
    from datasets import load_dataset, Dataset
    
    # Load from arrow file or HuggingFace
    if cfg.dataset_path:
        print(f"Loading from arrow file: {cfg.dataset_path}")
        # Use datasets library - handles HF arrow format (stream format, not file format)
        ds = Dataset.from_file(cfg.dataset_path)
    elif cfg.dataset_name:
        print(f"Loading from HuggingFace: {cfg.dataset_name}")
        ds = load_dataset(cfg.dataset_name, split=cfg.dataset_split)
    else:
        raise ValueError("Either dataset_path or dataset_name must be provided")
    
    print(f"Dataset: {len(ds)} conversations")
    
    # Filter to specific persona if requested
    if cfg.persona.persona_index is not None:
        ds = ds.select([cfg.persona.persona_index])
        print(f"Training on persona index: {cfg.persona.persona_index}")
    
    print(f"Dataset size: {len(ds)} conversations")
    
    # Create HFDataConverter with special tokens
    converter = HFDataConverter(
        model_name_or_path=cfg.model_name,
        seq_len=cfg.seq_len,
        special_tokens=SPECIAL_TOKENS,
    )
    print(f"Added special tokens: {SPECIAL_TOKENS}")
    
    # Create per-sample processor with persona config
    process_fn = create_persona_processor(cfg.persona)
    
    # Get dataloader using per-sample mode
    dataloader = converter.get_dataloader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        mode="per_sample",
        process_fn=process_fn,
    )
    print(f"DataLoader: {len(dataloader)} batches")

    # ========== MODEL ==========
    print("\n=== Loading Model ===")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.float16 if cfg.device == "cuda" else torch.float32,
    )
    
    # Resize embeddings for new special tokens (use converter's tokenizer)
    model.resize_token_embeddings(converter.get_vocab_size())
    
    # Apply LoRA
    lora_config = get_lora_config(cfg.lora)
    model = get_peft_model(model, lora_config)
    model.to(cfg.device)
    model.print_trainable_parameters()

    # ========== OPTIMIZER ==========
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler() if cfg.device == "cuda" else None

    # ========== TRAINING ==========
    print("\n=== Starting Training ===")
    print(f"Training on User {cfg.persona.train_on_user}'s responses only")
    print(f"User {2 if cfg.persona.train_on_user == 1 else 1}'s responses are context (masked from loss)")
    
    global_step = 0
    for epoch in range(cfg.epochs):
        avg_loss, global_step = train_epoch(
            model, dataloader, optimizer, cfg, epoch, global_step, scaler
        )
        print(f"Epoch {epoch+1}/{cfg.epochs} - Avg Loss: {avg_loss:.4f}")
        save_checkpoint(model, optimizer, global_step, cfg)
    
    # Save final (save tokenizer from converter which has special tokens)
    final_path = Path(cfg.out_dir) / "final"
    model.save_pretrained(final_path)
    converter.tokenizer.save_pretrained(final_path)
    print(f"\nTraining complete! Saved to {final_path}")


if __name__ == "__main__":
    main()