"""
Kira LLM — Dataset
PyTorch Dataset and DataLoader helpers for character-level language modelling.
"""

import os
import random
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from tokenizer import KiraTokenizer


# ══════════════════════════════════════════════════════════════
# SEQUENTIAL DATASET
# ══════════════════════════════════════════════════════════════

class KiraDataset(Dataset):
    """
    Character-level sequential dataset.

    Loads a text file, tokenises it once (with optional NPY cache),
    then yields (input, target) pairs of length block_size where
    target is input shifted right by one position (next-token prediction).
    """

    def __init__(
        self,
        data_path:  str,
        tokenizer:  KiraTokenizer,
        block_size: int,
        cache:      bool = True,
    ) -> None:
        self.tokenizer  = tokenizer
        self.block_size = block_size

        cache_path = data_path.replace(".txt", "_tokens.npy")

        if cache and os.path.exists(cache_path):
            print(f"[dataset] loading cached tokens from {cache_path}")
            self.tokens = np.load(cache_path).tolist()
        else:
            print(f"[dataset] tokenising {data_path}...")
            with open(data_path, "r", encoding="utf-8") as f:
                text = f.read()
            self.tokens = tokenizer.encode(text)
            if cache:
                np.save(cache_path, np.array(self.tokens, dtype=np.uint16))
                print(f"[dataset] cached → {cache_path}")

        n = len(self.tokens)
        batches = max(0, n - block_size)
        print(f"[dataset] {n:,} tokens | {batches:,} samples at block_size={block_size}")

    def __len__(self) -> int:
        return max(0, len(self.tokens) - self.block_size)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk = self.tokens[idx : idx + self.block_size + 1]
        # Pad if near end of file (edge case)
        while len(chunk) < self.block_size + 1:
            chunk.append(self.tokenizer.pad_id)
        x = torch.tensor(chunk[:self.block_size],   dtype=torch.long)
        y = torch.tensor(chunk[1:self.block_size+1], dtype=torch.long)
        return x, y


# ══════════════════════════════════════════════════════════════
# RANDOM-CHUNK DATASET (preferred for large datasets)
# ══════════════════════════════════════════════════════════════

class RandomChunkDataset(Dataset):
    """
    Faster variant: randomly samples blocks from the token array.
    Avoids sequential bias and works well for large datasets.
    Recommended for the training split.
    """

    def __init__(
        self,
        data_path:   str,
        tokenizer:   KiraTokenizer,
        block_size:  int,
        num_samples: int = 10_000,
    ) -> None:
        self.block_size  = block_size
        self.num_samples = num_samples

        with open(data_path, "r", encoding="utf-8") as f:
            text = f.read()
        self.tokens = tokenizer.encode(text)
        print(f"[dataset] {len(self.tokens):,} tokens — random sampling, {num_samples} steps/epoch")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        max_start = max(0, len(self.tokens) - self.block_size - 1)
        start = random.randint(0, max_start)
        chunk = self.tokens[start : start + self.block_size + 1]
        x = torch.tensor(chunk[:self.block_size],   dtype=torch.long)
        y = torch.tensor(chunk[1:self.block_size+1], dtype=torch.long)
        return x, y


# ══════════════════════════════════════════════════════════════
# DATALOADER FACTORIES
# ══════════════════════════════════════════════════════════════

def make_dataloaders(
    train_path:  str,
    val_path:    str,
    tokenizer:   KiraTokenizer,
    block_size:  int,
    batch_size:  int,
    num_workers: int = 2,
    num_samples: int = 50_000,
) -> Tuple[DataLoader, DataLoader]:
    """Build train and val DataLoaders."""
    train_ds = RandomChunkDataset(train_path, tokenizer, block_size, num_samples=num_samples)
    val_ds   = KiraDataset(val_path, tokenizer, block_size)

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    return train_dl, val_dl


# ══════════════════════════════════════════════════════════════
# IN-MEMORY BATCH SAMPLER (fast path for small datasets)
# ══════════════════════════════════════════════════════════════

class BatchSampler:
    """
    Memory-efficient batching directly from a token tensor.
    Significantly faster than Dataset + DataLoader for small datasets
    because it avoids worker/IPC overhead.
    """

    def __init__(
        self,
        tokens:     list,
        block_size: int,
        batch_size: int,
        device:     torch.device,
    ) -> None:
        self.data       = torch.tensor(tokens, dtype=torch.long, device=device)
        self.block_size = block_size
        self.batch_size = batch_size
        self.n          = len(tokens)

    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        ix = torch.randint(self.n - self.block_size, (self.batch_size,))
        x  = torch.stack([self.data[i   : i + self.block_size    ] for i in ix])
        y  = torch.stack([self.data[i+1 : i + self.block_size + 1] for i in ix])
        return x, y
