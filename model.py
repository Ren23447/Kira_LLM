"""
Kira LLM — Transformer Architecture
GPT-style decoder-only transformer with KV-cache for fast inference.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


# ══════════════════════════════════════════════════════════════
# ATTENTION
# ══════════════════════════════════════════════════════════════

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with optional KV-cache."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head  = cfg.n_head
        self.n_embd  = cfg.n_embd
        self.dropout = cfg.dropout
        self.head_dim = cfg.n_embd // cfg.n_head

        self.c_attn  = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj  = nn.Linear(cfg.n_embd, cfg.n_embd,     bias=cfg.bias)
        self.attn_dropout  = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        self.flash = hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            self.register_buffer(
                "mask",
                torch.tril(torch.ones(cfg.block_size, cfg.block_size))
                     .view(1, 1, cfg.block_size, cfg.block_size),
            )

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x:       (B, T, C)
            past_kv: cached (K, V) from previous steps, each (B, n_head, T_past, head_dim)
        Returns:
            output (B, T, C), new_kv tuple for caching
        """
        B, T, C = x.shape

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # KV cache: concatenate with past keys/values
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        new_kv = (k, v)

        T_total = k.size(2)

        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            scale = 1.0 / math.sqrt(self.head_dim)
            att = (q @ k.transpose(-2, -1)) * scale
            att = att.masked_fill(
                self.mask[:, :, :T, :T_total] == 0, float("-inf")
            )
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y, new_kv


# ══════════════════════════════════════════════════════════════
# MLP
# ══════════════════════════════════════════════════════════════

class MLP(nn.Module):
    """Position-wise feed-forward block with GELU activation."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.fc   = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)
        self.act  = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(self.act(self.fc(x))))


# ══════════════════════════════════════════════════════════════
# TRANSFORMER BLOCK
# ══════════════════════════════════════════════════════════════

class Block(nn.Module):
    """Single transformer decoder block (pre-norm)."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.mlp  = MLP(cfg)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        attn_out, new_kv = self.attn(self.ln1(x), past_kv=past_kv)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, new_kv


# ══════════════════════════════════════════════════════════════
# KIRA LLM
# ══════════════════════════════════════════════════════════════

class KiraLLM(nn.Module):
    """
    Full GPT-style decoder-only language model.

    Features:
    - Pre-norm transformer blocks
    - Flash attention (PyTorch ≥ 2.0) with manual causal mask fallback
    - KV-cache for O(1) per-step inference
    - Top-k / top-p sampling with repetition penalty
    - Weight tying between token embedding and LM head
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe  = nn.Embedding(cfg.block_size, cfg.n_embd),
            drop = nn.Dropout(cfg.dropout),
            h    = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f = nn.LayerNorm(cfg.n_embd),
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # Weight tying: embedding and LM head share weights
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

        # Scale residual projections by 1/sqrt(2 * n_layer) for training stability
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight") or pn.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        past_kvs: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            idx:      (B, T) token IDs
            targets:  (B, T) next-token targets — provide during training
            past_kvs: list of per-layer KV caches for incremental decoding
        Returns:
            logits (B, T, vocab_size), loss (or None), new_kvs
        """
        B, T = idx.shape
        assert T <= self.cfg.block_size, \
            f"Sequence length {T} exceeds block_size {self.cfg.block_size}"

        device = idx.device
        t_offset = past_kvs[0][0].size(2) if past_kvs else 0
        pos = torch.arange(t_offset, t_offset + T, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        new_kvs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for i, block in enumerate(self.transformer.h):
            pkv = past_kvs[i] if past_kvs else None
            x, nkv = block(x, past_kv=pkv)
            new_kvs.append(nkv)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=self.cfg.pad_token_id,
            )

        return logits, loss, new_kvs

    # ── Inference ──────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
        stop_token: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Auto-regressively generate tokens from a prompt.
        Uses KV-cache — only the new token is processed each step.
        """
        past_kvs: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * self.cfg.n_layer
        generated = idx.clone()

        # Warm up the cache on the full prompt
        logits, _, past_kvs = self.forward(idx, past_kvs=None)

        for _ in range(max_new_tokens):
            # Only feed the last token; cache handles context
            logits, _, past_kvs = self.forward(
                generated[:, -1:], past_kvs=past_kvs
            )
            logits = logits[:, -1, :]   # (B, vocab_size)

            # Repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(generated[0].tolist()):
                    if logits[0, token_id] < 0:
                        logits[0, token_id] *= repetition_penalty
                    else:
                        logits[0, token_id] /= repetition_penalty

            # Temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Top-k
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = float("-inf")

            # Top-p (nucleus)
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_logits[cum_probs > top_p] = float("-inf")
                logits.scatter_(1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

            if stop_token is not None and next_token.item() == stop_token:
                break

        return generated

    # ── Utilities ──────────────────────────────────────────────────────

    def num_parameters(self, trainable_only: bool = True) -> int:
        params = self.parameters() if not trainable_only else (
            p for p in self.parameters() if p.requires_grad
        )
        return sum(p.numel() for p in params)

    def save_checkpoint(
        self,
        path: str,
        step: int,
        optimizer_state: Optional[dict] = None,
        val_loss: Optional[float] = None,
        extra: Optional[dict] = None,
    ) -> None:
        """Save a full checkpoint with model weights, config, and training state."""
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        payload = {
            "step":        step,
            "model_state": self.state_dict(),
            "cfg":         self.cfg.__dict__,
        }
        if optimizer_state is not None:
            payload["optimizer_state"] = optimizer_state
        if val_loss is not None:
            payload["val_loss"] = val_loss
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu") -> "KiraLLM":
        """Load a model from a checkpoint file."""
        payload = torch.load(path, map_location=device, weights_only=False)
        cfg = ModelConfig(**payload["cfg"])
        model = cls(cfg)
        model.load_state_dict(payload["model_state"])
        model.to(device)
        return model
