"""
Kira LLM — Generation / Inference
Loads a trained checkpoint and generates responses using the KV-cache model.

Usage:
    python generate.py                                 # interactive mode
    python generate.py --checkpoint checkpoints/best.pt
    python generate.py --prompt "tell me about space"
"""

import argparse
import os
import sys
from typing import List, Optional, Tuple

import torch

from model import KiraLLM
from tokenizer import KiraTokenizer, USR_ID, KIR_ID, EOS_ID


# ══════════════════════════════════════════════════════════════
# INFERENCE WRAPPER
# ══════════════════════════════════════════════════════════════

class KiraInference:
    """
    Wraps a trained KiraLLM for fast, cached response generation.
    Uses the KV-cache so only the newest token is processed each step —
    making inference significantly faster than full re-computation.
    """

    def __init__(
        self,
        checkpoint_path:    str,
        tokenizer_path:     str   = "data/tokenizer.json",
        device:             str   = "auto",
        temperature:        float = 0.8,
        top_k:              int   = 40,
        top_p:              float = 0.95,
        max_new_tokens:     int   = 300,
        repetition_penalty: float = 1.1,
    ) -> None:
        self.device             = self._resolve_device(device)
        self.temperature        = temperature
        self.top_k              = top_k
        self.top_p              = top_p
        self.max_new_tokens     = max_new_tokens
        self.repetition_penalty = repetition_penalty

        print(f"[inference] loading tokenizer from {tokenizer_path}...")
        self.tokenizer = KiraTokenizer.load(tokenizer_path)

        print(f"[inference] loading model from {checkpoint_path}...")
        self.model = KiraLLM.from_checkpoint(checkpoint_path, str(self.device))
        self.model.eval()

        p = self.model.num_parameters()
        print(f"[inference] ready — {p:,} params ({p/1e6:.1f}M) on {self.device}  [KV cache: ON]")

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        if requested == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(requested)

    @torch.no_grad()
    def respond(
        self,
        user_message: str,
        context:      Optional[List[Tuple[str, str]]] = None,
        max_tokens:   Optional[int] = None,
    ) -> str:
        """
        Generate Kira's reply to user_message.
        Encodes the conversation context, prompts with [KIR], and generates until [EOS].

        Args:
            user_message: the user's text
            context:      [(role, text), ...] — recent conversation history
            max_tokens:   override max generation length

        Returns:
            Kira's response string
        """
        prompt_ids: List[int] = []

        if context:
            for role, text in context[-6:]:   # last 3 exchanges
                prompt_ids += self.tokenizer.encode_turn(role, text)

        prompt_ids += self.tokenizer.encode_turn("user", user_message)
        prompt_ids += [KIR_ID]   # Kira's turn begins

        # Crop to fit in context window
        max_ctx = self.model.cfg.block_size - (max_tokens or self.max_new_tokens)
        if len(prompt_ids) > max_ctx:
            prompt_ids = prompt_ids[-max_ctx:]

        idx = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

        out = self.model.generate(
            idx,
            max_new_tokens     = max_tokens or self.max_new_tokens,
            temperature        = self.temperature,
            top_k              = self.top_k,
            top_p              = self.top_p,
            repetition_penalty = self.repetition_penalty,
            stop_token         = EOS_ID,
        )

        new_ids  = out[0, len(prompt_ids):].tolist()
        response = self.tokenizer.decode(new_ids, skip_special=True).strip()
        return response if response else "..."

    def interactive(self) -> None:
        """Terminal chat loop for testing the model directly."""
        print("\nKira LLM — interactive mode  (Ctrl+C or Ctrl+D to quit)")
        print("-" * 50)
        history: List[Tuple[str, str]] = []
        while True:
            try:
                user_msg = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[bye]")
                break
            if not user_msg:
                continue
            response = self.respond(user_msg, context=history)
            print(f"Kira: {response}\n")
            history.append(("user", user_msg))
            history.append(("kira", response))


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Kira LLM inference")
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--tokenizer",  default="data/tokenizer.json")
    p.add_argument("--device",     default="auto")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k",       type=int,   default=40)
    p.add_argument("--top_p",       type=float, default=0.95)
    p.add_argument("--max_tokens",  type=int,   default=300)
    p.add_argument("--prompt",      default=None, help="Single-prompt mode (non-interactive)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"[error] checkpoint not found: {args.checkpoint}")
        print("        Run:  python train.py  (GPU)")
        print("        or:   python train_cpu.py  (CPU-only)")
        sys.exit(1)

    if not os.path.exists(args.tokenizer):
        print(f"[error] tokenizer not found: {args.tokenizer}")
        print("        Run:  python prepare_data.py")
        sys.exit(1)

    engine = KiraInference(
        checkpoint_path    = args.checkpoint,
        tokenizer_path     = args.tokenizer,
        device             = args.device,
        temperature        = args.temperature,
        top_k              = args.top_k,
        top_p              = args.top_p,
        max_new_tokens     = args.max_tokens,
    )

    if args.prompt:
        print(f"Kira: {engine.respond(args.prompt)}")
    else:
        engine.interactive()


if __name__ == "__main__":
    main()
