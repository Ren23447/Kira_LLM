"""
Kira LLM — Training Script
Full training loop: cosine LR schedule, AMP, gradient accumulation,
periodic eval, best-checkpoint saving, and numbered checkpoint rotation.

Usage:
    python train.py                          # small model, auto device
    python train.py --config medium          # larger model preset
    python train.py --resume checkpoints/ckpt_010000.pt
    python train.py --compile                # torch.compile (PyTorch ≥ 2.x)
    python train.py --accum_steps 8          # larger effective batch on small GPU
"""

import argparse
import json
import math
import os
import sys
import time
from contextlib import nullcontext

import torch

from config import (
    ModelConfig, TrainConfig, make_train_config,
    tiny_config, small_config, medium_config, large_config,
)
from dataset import BatchSampler
from model import KiraLLM
from tokenizer import KiraTokenizer


# ══════════════════════════════════════════════════════════════
# LR SCHEDULE
# ══════════════════════════════════════════════════════════════

def get_lr(step: int, cfg: TrainConfig) -> float:
    """Cosine decay with linear warm-up."""
    if step < cfg.warmup_iters:
        return cfg.learning_rate * step / max(1, cfg.warmup_iters)
    if step > cfg.lr_decay_iters:
        return cfg.min_lr
    progress = (step - cfg.warmup_iters) / (cfg.lr_decay_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


# ══════════════════════════════════════════════════════════════
# DEVICE
# ══════════════════════════════════════════════════════════════

def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


# ══════════════════════════════════════════════════════════════
# CHECKPOINT HELPERS
# ══════════════════════════════════════════════════════════════

def save_checkpoint(
    model: KiraLLM,
    optimizer: torch.optim.Optimizer,
    step: int,
    val_loss: float,
    out_dir: str,
    tag: str,
) -> str:
    """Save a checkpoint and return its path."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{tag}.pt")
    model.save_checkpoint(
        path,
        step=step,
        optimizer_state=optimizer.state_dict(),
        val_loss=val_loss,
    )
    return path


def prune_old_checkpoints(out_dir: str, keep: int) -> None:
    """Remove old numbered checkpoints, keeping only the most recent `keep`."""
    candidates = sorted(
        [f for f in os.listdir(out_dir) if f.startswith("ckpt_") and f.endswith(".pt")]
    )
    for old in candidates[:-keep]:
        try:
            os.remove(os.path.join(out_dir, old))
        except OSError:
            pass


# ══════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(
    model: KiraLLM,
    sampler: BatchSampler,
    eval_iters: int,
    ctx,
) -> float:
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = sampler.get_batch()
        with ctx:
            _, loss, _ = model(x, targets=y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


# ══════════════════════════════════════════════════════════════
# MAIN TRAINING LOOP
# ══════════════════════════════════════════════════════════════

def train(args: argparse.Namespace) -> None:
    # ── Config ────────────────────────────────────────────────
    preset_map = {
        "tiny":   tiny_config,
        "small":  small_config,
        "medium": medium_config,
        "large":  large_config,
    }
    model_cfg = preset_map.get(args.config, small_config)()
    train_cfg = make_train_config(
        model_cfg,
        max_iters    = args.max_iters,
        accum_steps  = args.accum_steps,
        out_dir      = args.out_dir,
        compile_model= args.compile,
    )

    device = resolve_device(args.device)
    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
        if torch.cuda.is_available()
        else torch.float32
    )
    ctx = (
        torch.amp.autocast(device_type="cuda", dtype=dtype)
        if device.type == "cuda"
        else nullcontext()
    )
    scaler = torch.cuda.GradScaler(enabled=(device.type == "cuda" and dtype == torch.float16))

    os.makedirs(train_cfg.out_dir, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────
    if not os.path.exists(train_cfg.train_file):
        print(f"[train] training data not found at {train_cfg.train_file}")
        print("        Run:  python prepare_data.py")
        sys.exit(1)

    tok_path = os.path.join(train_cfg.data_dir, "tokenizer.json")
    if not os.path.exists(tok_path):
        print("[train] tokenizer.json not found — run python prepare_data.py first")
        sys.exit(1)

    print(f"[train] loading tokenizer from {tok_path}")
    tokenizer = KiraTokenizer.load(tok_path)
    model_cfg.vocab_size = tokenizer.vocab_size

    print(f"[train] loading training data...")
    with open(train_cfg.train_file, "r", encoding="utf-8") as f:
        train_text = f.read()
    with open(train_cfg.val_file, "r", encoding="utf-8") as f:
        val_text = f.read()

    train_tokens = tokenizer.encode(train_text)
    val_tokens   = tokenizer.encode(val_text)
    print(f"[train] {len(train_tokens):,} train tokens | {len(val_tokens):,} val tokens")

    train_sampler = BatchSampler(train_tokens, train_cfg.block_size, train_cfg.batch_size, device)
    val_sampler   = BatchSampler(val_tokens,   train_cfg.block_size, train_cfg.batch_size, device)

    # ── Model ─────────────────────────────────────────────────
    print(f"[train] initialising model ({args.config})...")
    model = KiraLLM(model_cfg).to(device)

    start_step = 0
    best_val_loss = float("inf")

    if args.resume:
        if not os.path.exists(args.resume):
            print(f"[train] checkpoint not found: {args.resume}")
            sys.exit(1)
        print(f"[train] resuming from {args.resume}")
        payload = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state"])
        start_step = payload.get("step", 0)
        best_val_loss = payload.get("val_loss", float("inf"))
        print(f"[train] resumed at step {start_step}, best val loss {best_val_loss:.4f}")

    if args.compile:
        try:
            print("[train] compiling model (torch.compile)...")
            model = torch.compile(model)
        except Exception as e:
            print(f"[train] compile failed ({e}) — continuing without compile")

    p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] {p:,} trainable params ({p/1e6:.1f}M) on {device}")

    # ── Optimiser ─────────────────────────────────────────────
    decay_params = [p for n, p in model.named_parameters()
                    if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters()
                      if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params,   "weight_decay": train_cfg.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=train_cfg.learning_rate,
        betas=(train_cfg.beta1, train_cfg.beta2),
        fused=torch.cuda.is_available(),
    )
    if args.resume:
        try:
            payload = torch.load(args.resume, map_location=device, weights_only=False)
            if "optimizer_state" in payload:
                optimizer.load_state_dict(payload["optimizer_state"])
        except Exception:
            pass

    # ── Training loop ─────────────────────────────────────────
    model.train()
    t0 = time.time()
    optimizer.zero_grad()

    print(f"\n[train] starting — {train_cfg.max_iters:,} steps, "
          f"batch={train_cfg.batch_size}, accum={train_cfg.accum_steps}, "
          f"effective_batch={train_cfg.batch_size * train_cfg.accum_steps}")

    for step in range(start_step, train_cfg.max_iters + 1):
        lr = get_lr(step, train_cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # Gradient accumulation
        for micro_step in range(train_cfg.accum_steps):
            x, y = train_sampler.get_batch()
            with ctx:
                _, loss, _ = model(x, targets=y)
                loss = loss / train_cfg.accum_steps
            scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        # Logging
        if step % train_cfg.log_interval == 0:
            dt = time.time() - t0
            tok_per_sec = (
                train_cfg.batch_size * train_cfg.block_size
                * train_cfg.accum_steps * train_cfg.log_interval
            ) / max(dt, 1e-6)
            print(f"[step {step:>6}/{train_cfg.max_iters}] "
                  f"lr={lr:.2e}  loss={loss.item() * train_cfg.accum_steps:.4f}  "
                  f"tok/s={tok_per_sec:,.0f}  elapsed={dt:.1f}s")
            t0 = time.time()

        # Evaluation + checkpointing
        if step % train_cfg.eval_interval == 0 and step > 0:
            val_loss = evaluate(model, val_sampler, train_cfg.eval_iters, ctx)
            print(f"\n[eval ] step {step} — val_loss={val_loss:.4f}  "
                  f"best={best_val_loss:.4f}\n")

            # Numbered checkpoint (rotated)
            ckpt_tag = f"ckpt_{step:07d}"
            save_checkpoint(model, optimizer, step, val_loss, train_cfg.out_dir, ckpt_tag)
            prune_old_checkpoints(train_cfg.out_dir, train_cfg.keep_checkpoints)

            # Best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, step, val_loss, train_cfg.out_dir, "best")
                print(f"[train] ✓ new best checkpoint saved (val_loss={best_val_loss:.4f})")

    # ── Final save ────────────────────────────────────────────
    val_loss = evaluate(model, val_sampler, train_cfg.eval_iters, ctx)
    save_checkpoint(model, optimizer, train_cfg.max_iters, val_loss, train_cfg.out_dir, "final")
    print(f"\n[train] done — final val_loss={val_loss:.4f}")
    print(f"[train] checkpoints saved to: {train_cfg.out_dir}/")


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Kira LLM")
    p.add_argument("--config",      default="small",     choices=["tiny","small","medium","large"])
    p.add_argument("--device",      default="auto",      help="cuda | mps | cpu | auto")
    p.add_argument("--max_iters",   type=int, default=50_000)
    p.add_argument("--accum_steps", type=int, default=4)
    p.add_argument("--out_dir",     default="checkpoints")
    p.add_argument("--resume",      default=None,        help="path to checkpoint to resume from")
    p.add_argument("--compile",     action="store_true", help="torch.compile (PyTorch ≥ 2.0)")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
