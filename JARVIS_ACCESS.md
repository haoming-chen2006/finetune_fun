# Jarvis Access Verification

**Date:** 2026-02-04  
**Written by:** Jarvis (AI assistant via OpenClaw)

---

## Access Check Results

### ✅ 1. GitHub Repo Access
- Repo `haoming-chen2006/finetune_fun` is publicly accessible
- Can read all files via GitHub API and web fetch
- SSH key (`haoming-chen2006`) is registered on GitHub — clone/push via `git@github.com` works

### ✅ 2. Cluster Terminal Workflow
- **SSH:** `ssh haoming@dgx4.ist.Berkeley.edu` — connects successfully
- **Apptainer shortcut:** `bin/finetune` runs `sudo apptainer shell --nv` with the `cuda124.sif` container, mounting `/home/haoming/finetune_fun` at `/finetune_fun`
- **Virtual env:** `source .venv/bin/activate` works inside the container
- **CUDA:** PyTorch `2.10.0+cu128` detected, `torch.cuda.is_available()` returns `True`

### ✅ 3. Code Access & Review
Full read access confirmed. Here's what's in the repo:

| Component | Description |
|-----------|-------------|
| `main.py` | Entry point (placeholder) |
| `app/chat.py` | Interactive chat CLI for fine-tuned LoRA models — supports persona, 19th-century, and standard (base) modes |
| `experiments/persona.py` | Persona-conditioned fine-tuning with custom special tokens and selective loss masking |
| `experiments/19thcentury.py` | 19th-century British literature style fine-tuning with time-based checkpoint saves and resume support |
| `ftdatasets/hftopt.py` | Reusable HuggingFace → PyTorch data pipeline (batched chunking + per-sample processing, mask utilities) |
| `configs/*.yaml` | Training configs for both experiments |
| `pyproject.toml` | Project metadata — base model is `Qwen/Qwen3-8B`, managed via `uv` |

**Key observations:**
- LoRA fine-tuning via PEFT on Qwen3-8B
- Persona experiment uses custom special tokens (`<persona>`, `<user1>`, `<user2>`) with masked loss — only the target user's responses contribute to the gradient
- 19th-century experiment pulls from British Library `blbooks` dataset, filters to English, and saves checkpoints on a time interval (not step count)
- `ftdatasets/hftopt.py` is the shared data utility: handles tokenization, chunking, caching, and label masking in a single class

---

## Summary
All three access vectors are fully operational. Jarvis can:
- Browse and interact with this GitHub repo (read + push via SSH)
- SSH into the DGX4 cluster, launch the Apptainer container, and activate the Python virtual environment
- Read, understand, and reason about the codebase
