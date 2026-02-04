import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM
from datasets import load_dataset
import argparse
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import torch
from torch.optim import AdamW
from tqdm import tqdm
import time

from ftdatasets.hftopt import HFDataConverter


def create_19thcentury_format_fn():
    """
    Create format_fn for 19th century British Library dataset.
    Filters to English texts with non-empty content.
    """
    def format_fn(dataset):
        # Filter to English
        dataset = dataset.filter(lambda x: x["Language_1"] == "English")
        # Filter to non-empty text
        dataset = dataset.filter(lambda x: x.get("text") is not None and len(x["text"].strip()) > 0)
        return dataset
    
    return format_fn


@dataclass
class LoRAConfig:
    """LoRA-specific configuration."""
    r: int = 8                          # rank of the low-rank matrices
    lora_alpha: int = 32                # scaling factor
    lora_dropout: float = 0.1           # dropout probability
    target_modules: Optional[List[str]] = None  # which modules to apply LoRA to
    bias: str = "none"                  # bias training: "none", "all", "lora_only"


@dataclass 
class TrainConfig:
    """Training configuration."""
    model_name: str = "Qwen/Qwen3-8B"
    seq_len: int = 512
    text_column: str = "text"
    
    # Dataset (uses HuggingFace cache automatically)
    dataset_name: str = "TheBritishLibrary/blbooks"  # HuggingFace dataset name
    dataset_config: str = "1500_1899"  # Dataset config/subset
    dataset_split: str = "train"
    
    # Training
    seed: int = 42
    epochs: int = 3
    lr: float = 3e-4
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    weight_decay: float = 0.01
    
    # LoRA
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    
    # Output
    out_dir: str = "runs/19thcentury"
    save_interval_hours: float = 2.0  # Save checkpoint every N hours
    log_steps: int = 10
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def load_config(path: str) -> TrainConfig:
    """Load config from YAML file."""
    data = yaml.safe_load(Path(path).read_text())
    
    # Handle nested lora config
    lora_data = data.pop("lora", {})
    lora_cfg = LoRAConfig(**lora_data)
    
    cfg = TrainConfig(**data, lora=lora_cfg)
    return cfg


def get_lora_config(cfg: LoRAConfig) -> LoraConfig:
    """Create PEFT LoraConfig from our config."""
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
    """Single training step."""
    input_ids = batch["input_ids"].to(cfg.device)
    labels = batch["labels"].to(cfg.device)
    
    if scaler is not None:  # Mixed precision
        with torch.amp.autocast('cuda'):
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss / cfg.gradient_accumulation_steps
        scaler.scale(loss).backward()
    else:
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss / cfg.gradient_accumulation_steps
        loss.backward()
    
    return loss.item() * cfg.gradient_accumulation_steps


def train_epoch(model, dataloader, optimizer, cfg, epoch, global_step, scaler=None, last_save_time=None, start_step=0):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    if last_save_time is None:
        last_save_time = time.time()
    
    save_interval_seconds = cfg.save_interval_hours * 3600
    
    # Calculate how many batches correspond to start_step (accounting for gradient accumulation)
    skip_batches = start_step * cfg.gradient_accumulation_steps
    total_batches = len(dataloader)
    
    progress = tqdm(dataloader, desc=f"Epoch {epoch+1}", initial=skip_batches, total=total_batches)
    optimizer.zero_grad()
    
    for step, batch in enumerate(progress):
        loss = train_step(model, batch, optimizer, cfg, scaler)
        total_loss += loss
        num_batches += 1
        
        # Gradient accumulation
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
            
            # Logging
            if global_step % cfg.log_steps == 0:
                avg_loss = total_loss / num_batches
                elapsed_since_save = (time.time() - last_save_time) / 60
                progress.set_postfix({"loss": f"{avg_loss:.4f}", "step": global_step, "mins_since_save": f"{elapsed_since_save:.1f}"})
            
            # Save checkpoint every N hours
            if time.time() - last_save_time >= save_interval_seconds:
                save_checkpoint(model, optimizer, global_step, cfg)
                last_save_time = time.time()
    
    return total_loss / num_batches, global_step, last_save_time


def save_checkpoint(model, optimizer, step, cfg):
    """Save model checkpoint."""
    out_path = Path(cfg.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = out_path / f"checkpoint-{step}"
    model.save_pretrained(checkpoint_path)
    
    # Save optimizer state
    torch.save({
        "optimizer": optimizer.state_dict(),
        "step": step,
    }, checkpoint_path / "optimizer.pt")
    
    print(f"Saved checkpoint to {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--model", type=str, default=None, help="Model name override")
    parser.add_argument("--epochs", type=int, default=None, help="Epochs override")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size override")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from specific checkpoint path")
    args = parser.parse_args()

    # Load config from file or use defaults
    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = TrainConfig()
    
    # Override with CLI args
    if args.model:
        cfg.model_name = args.model
    if args.epochs:
        cfg.epochs = args.epochs
    if args.lr:
        cfg.lr = args.lr
    if args.batch_size:
        cfg.batch_size = args.batch_size

    # Set seed
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    
    print(f"Config: {cfg}")
    print(f"LoRA Config: {cfg.lora}")
    print(f"Device: {cfg.device}")

    # ========== DATA ==========
    print("\n=== Loading Dataset ===")
    from datasets import load_dataset
    
    # Uses HuggingFace cache automatically (no re-download if cached)
    print(f"Loading {cfg.dataset_name} ({cfg.dataset_config})...")
    ds = load_dataset(cfg.dataset_name, cfg.dataset_config, split=cfg.dataset_split)
    print(f"Raw dataset: {len(ds)} samples")

    # Create data converter and dataloader (format_fn handles filtering)
    converter = HFDataConverter(
        model_name_or_path=cfg.model_name,
        seq_len=cfg.seq_len,
        text_column=cfg.text_column
    )
    
    format_fn = create_19thcentury_format_fn()
    
    # Use cache to avoid re-tokenizing on subsequent runs
    cache_path = Path(cfg.out_dir) / "tokenized_cache"
    
    dataloader = converter.get_dataloader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        format_fn=format_fn,
        cache_path=str(cache_path),
    )
    print(f"DataLoader: {len(dataloader)} batches")

    # ========== FIND CHECKPOINT TO RESUME ==========
    resume_checkpoint = None
    if args.checkpoint:
        resume_checkpoint = Path(args.checkpoint)
    elif args.resume:
        # Find latest checkpoint in out_dir
        out_path = Path(cfg.out_dir)
        checkpoints = sorted(out_path.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]) if p.name.split("-")[1].isdigit() else 0)
        if checkpoints:
            resume_checkpoint = checkpoints[-1]
            print(f"Found latest checkpoint: {resume_checkpoint}")
    
    # ========== MODEL ==========
    print("\n=== Loading Model ===")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.float16 if cfg.device == "cuda" else torch.float32,
    )
    
    # Apply LoRA (or load from checkpoint)
    if resume_checkpoint and resume_checkpoint.exists():
        print(f"Resuming from checkpoint: {resume_checkpoint}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, resume_checkpoint, is_trainable=True)
    else:
        lora_config = get_lora_config(cfg.lora)
        model = get_peft_model(model, lora_config)
    
    model.to(cfg.device)
    model.print_trainable_parameters()

    # ========== OPTIMIZER ==========
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )
    
    # Load optimizer state if resuming
    global_step = 0
    if resume_checkpoint and (resume_checkpoint / "optimizer.pt").exists():
        print(f"Loading optimizer state from {resume_checkpoint / 'optimizer.pt'}")
        opt_state = torch.load(resume_checkpoint / "optimizer.pt", map_location=cfg.device)
        optimizer.load_state_dict(opt_state["optimizer"])
        global_step = opt_state.get("step", 0)
        print(f"Resuming from step {global_step}")
    
    # Mixed precision scaler for CUDA
    scaler = torch.cuda.amp.GradScaler() if cfg.device == "cuda" else None

    # ========== TRAINING LOOP ==========
    print("\n=== Starting Training ===")
    print(f"Saving checkpoints every {cfg.save_interval_hours} hours")
    print(f"Starting from step {global_step}")
    last_save_time = time.time()
    
    for epoch in range(cfg.epochs):
        # Pass start_step only for first epoch when resuming
        start_step = global_step if epoch == 0 else 0
        avg_loss, global_step, last_save_time = train_epoch(
            model, dataloader, optimizer, cfg, epoch, global_step, scaler, last_save_time, start_step=start_step
        )
        print(f"Epoch {epoch+1}/{cfg.epochs} - Avg Loss: {avg_loss:.4f}")
        
        # Save at end of epoch
        save_checkpoint(model, optimizer, global_step, cfg)
        last_save_time = time.time()
    
    # Save final model
    final_path = Path(cfg.out_dir) / "final"
    model.save_pretrained(final_path)
    print(f"\nTraining complete! Final model saved to {final_path}")


if __name__ == "__main__":
    main()



