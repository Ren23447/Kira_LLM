"""
Kira LLM — Configuration
All hyperparameters for model architecture and training live here.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    vocab_size:    int   = 256       # overridden by tokenizer at train time
    block_size:    int   = 512
    n_layer:       int   = 6
    n_head:        int   = 6
    n_embd:        int   = 384
    dropout:       float = 0.1
    bias:          bool  = False
    pad_token_id:  int   = 0
    user_token_id: int   = 1
    kira_token_id: int   = 2
    eos_token_id:  int   = 3


@dataclass
class TrainConfig:
    # ── Paths ────────────────────────────────────────────────
    data_dir:   str = "data"
    out_dir:    str = "checkpoints"
    train_file: str = "data/train.txt"
    val_file:   str = "data/val.txt"

    # ── Batch / sequence ─────────────────────────────────────
    batch_size: int = 32
    block_size: int = 512   # keep in sync with ModelConfig — use make_train_config()

    # ── Gradient accumulation ─────────────────────────────────
    # Effective batch = batch_size × accum_steps
    # Increase accum_steps when GPU memory is limited
    accum_steps: int = 4

    # ── DataLoader ───────────────────────────────────────────
    num_workers: int = 2   # 0 = main process; 2-4 speeds up on GPU

    # ── Optimiser ────────────────────────────────────────────
    learning_rate: float = 3e-4
    max_iters:     int   = 50_000
    weight_decay:  float = 0.1
    beta1:         float = 0.9
    beta2:         float = 0.99
    grad_clip:     float = 1.0

    # ── LR schedule ──────────────────────────────────────────
    warmup_iters:    int   = 300
    lr_decay_iters:  int   = 50_000
    min_lr:          float = 3e-5

    # ── Evaluation ───────────────────────────────────────────
    eval_interval: int = 500
    eval_iters:    int = 100
    log_interval:  int = 50

    # ── Checkpointing ────────────────────────────────────────
    save_interval:    int  = 1000   # save a numbered checkpoint every N steps
    keep_checkpoints: int  = 5      # keep only the N most recent numbered checkpoints
    compile_model:    bool = False   # torch.compile (PyTorch ≥ 2.0, faster on GPU)


def make_train_config(model_cfg: ModelConfig, **overrides) -> "TrainConfig":
    """Create a TrainConfig whose block_size matches the ModelConfig."""
    cfg = TrainConfig(**overrides)
    cfg.block_size = model_cfg.block_size
    return cfg


# ── Named presets ─────────────────────────────────────────────────────
# Use these with: python train.py --config tiny

def tiny_config() -> ModelConfig:
    """~1 M params — CPU-friendly, fast to experiment."""
    return ModelConfig(n_layer=4, n_head=4, n_embd=128, block_size=256)


def small_config() -> ModelConfig:
    """~6 M params — default, good balance of speed and capability."""
    return ModelConfig(n_layer=6, n_head=6, n_embd=384, block_size=512)


def medium_config() -> ModelConfig:
    """~25 M params — needs a decent GPU."""
    return ModelConfig(n_layer=8, n_head=8, n_embd=512, block_size=1024)


def large_config() -> ModelConfig:
    """~85 M params — GPU with ≥ 8 GB VRAM."""
    return ModelConfig(n_layer=12, n_head=12, n_embd=768, block_size=1024)
