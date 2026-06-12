"""
Kira LLM — CPU Training Script
Optimised for training on CPU with no GPU available.
Uses a tiny model preset, smaller batches, and no AMP/scaler overhead.

Usage:
    python train_cpu.py                     # 20,000 steps, tiny model
    python train_cpu.py --steps 5000        # quick smoke-test
    python train_cpu.py --config small      # slightly larger model
"""

import argparse
import math
import os
import sys
import time

import torch

from config import make_train_config, tiny_config, small_config
from dataset import BatchSampler
from model import KiraLLM
from tokenizer import KiraTokenizer
from train import get_lr, evaluate, save_checkpoint, prune_old_checkpoints


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Kira LLM on CPU")
    p.add_argument("--config",    default="tiny",   choices=["tiny", "small"])
    p.add_argument("--steps",     type=int, default=20_000)
    p.add_argument("--batch",     type=int, default=16)
    p.add_argument("--out_dir",   default="checkpoints")
    p.add_argument("--resume",    default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")

    model_cfg = tiny_config() if args.config == "tiny" else small_config()
    train_cfg = make_train_config(
        model_cfg,
        max_iters    = args.steps,
        batch_size   = args.batch,
        accum_steps  = 1,
        warmup_iters = min(300, args.steps // 10),
        lr_decay_iters = args.steps,
        eval_interval  = max(100, args.steps // 20),
        log_interval   = max(10, args.steps // 100),
        out_dir        = args.out_dir,
    )

    os.makedirs(train_cfg.out_dir, exist_ok=True)

    if not os.path.exists(train_cfg.train_file):
        print("[train_cpu] training data not found — run: python prepare_data.py")
        sys.exit(1)

    tok_path = os.path.join(train_cfg.data_dir, "tokenizer.json")
    if not os.path.exists(tok_path):
        print("[train_cpu] tokenizer.json not found — run: python prepare_data.py")
        sys.exit(1)

    tokenizer = KiraTokenizer.load(tok_path)
    model_cfg.vocab_size = tokenizer.vocab_size

    with open(train_cfg.train_file, "r", encoding="utf-8") as f:
        train_text = f.read()
    with open(train_cfg.val_file, "r", encoding="utf-8") as f:
        val_text = f.read()

    train_tokens = tokenizer.encode(train_text)
    val_tokens   = tokenizer.encode(val_text)
    print(f"[train_cpu] {len(train_tokens):,} train / {len(val_tokens):,} val tokens")

    train_sampler = BatchSampler(train_tokens, train_cfg.block_size, train_cfg.batch_size, device)
    val_sampler   = BatchSampler(val_tokens,   train_cfg.block_size, train_cfg.batch_size, device)

    model = KiraLLM(model_cfg).to(device)

    start_step    = 0
    best_val_loss = float("inf")

    if args.resume and os.path.exists(args.resume):
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state"])
        start_step    = payload.get("step", 0)
        best_val_loss = payload.get("val_loss", float("inf"))
        print(f"[train_cpu] resumed at step {start_step}")

    p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train_cpu] {p:,} params ({p/1e6:.1f}M) on CPU")
    print(f"[train_cpu] {train_cfg.max_iters:,} steps — this may take a while on CPU")
    print("            Consider using train.py on a GPU for faster results.\n")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
        betas=(train_cfg.beta1, train_cfg.beta2),
    )

    ctx = torch.no_grad.__class__()   # plain nullcontext for CPU
    from contextlib import nullcontext
    ctx = nullcontext()

    model.train()
    t0 = time.time()
    optimizer.zero_grad()

    for step in range(start_step, train_cfg.max_iters + 1):
        lr = get_lr(step, train_cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = train_sampler.get_batch()
        _, loss, _ = model(x, targets=y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
        optimizer.zero_grad()

        if step % train_cfg.log_interval == 0:
            dt = time.time() - t0
            print(f"[step {step:>6}/{train_cfg.max_iters}] "
                  f"lr={lr:.2e}  loss={loss.item():.4f}  elapsed={dt:.1f}s")
            t0 = time.time()

        if step % train_cfg.eval_interval == 0 and step > 0:
            val_loss = evaluate(model, val_sampler, min(50, train_cfg.eval_iters), ctx)
            print(f"\n[eval ] step {step} — val_loss={val_loss:.4f}\n")

            save_checkpoint(model, optimizer, step, val_loss, train_cfg.out_dir,
                            f"ckpt_{step:07d}")
            prune_old_checkpoints(train_cfg.out_dir, 3)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, step, val_loss, train_cfg.out_dir, "best")
                print(f"[train_cpu] ✓ new best (val_loss={best_val_loss:.4f})")

    val_loss = evaluate(model, val_sampler, 50, ctx)
    save_checkpoint(model, optimizer, train_cfg.max_iters, val_loss, train_cfg.out_dir, "final")
    print(f"\n[train_cpu] done — final val_loss={val_loss:.4f}")
    print(f"[train_cpu] checkpoints in: {train_cfg.out_dir}/")


if __name__ == "__main__":
    main()
