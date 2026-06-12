"""
Kira LLM — Tokenizer
Character-level tokenizer with proper special-token handling.

Special tokens (<pad>, <usr>, <kir>, <eos>, <unk>) are treated as single
indivisible units during both encoding and decoding, so they never
bleed into character-level encoding.

IDs:
  <pad>  0 — padding
  <usr>  1 — start of user turn
  <kir>  2 — start of Kira turn
  <eos>  3 — end of turn
  <unk>  4 — unknown character
"""

import json
import os
import re
from typing import Dict, List, Tuple

SPECIAL_TOKENS = ["<pad>", "<usr>", "<kir>", "<eos>", "<unk>"]

PAD_ID = 0
USR_ID = 1
KIR_ID = 2
EOS_ID = 3
UNK_ID = 4


class KiraTokenizer:
    def __init__(self) -> None:
        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}
        self.vocab_size: int = 0

    # ── Build ──────────────────────────────────────────────────────────

    def build(self, text: str) -> "KiraTokenizer":
        """
        Build vocabulary from training text.
        Special tokens are assigned IDs 0-4; all other unique
        characters get IDs starting at 5.
        """
        chars = sorted(set(text))
        regular = [c for c in chars if c not in SPECIAL_TOKENS]
        vocab = SPECIAL_TOKENS + regular
        self.char_to_id = {c: i for i, c in enumerate(vocab)}
        self.id_to_char = {i: c for i, c in enumerate(vocab)}
        self.vocab_size = len(vocab)
        return self

    # ── Encode ─────────────────────────────────────────────────────────

    def encode(self, text: str, add_eos: bool = False) -> List[int]:
        """
        Encode text to token IDs.
        Special tokens are matched as full strings before character-level
        encoding, so <usr> maps to ID 1 rather than '<', 'u', 's', 'r', '>'.
        """
        ids: List[int] = []
        i = 0
        while i < len(text):
            matched = False
            for special in SPECIAL_TOKENS:
                end = i + len(special)
                if text[i:end] == special:
                    ids.append(self.char_to_id.get(special, UNK_ID))
                    i = end
                    matched = True
                    break
            if not matched:
                ids.append(self.char_to_id.get(text[i], UNK_ID))
                i += 1
        if add_eos:
            ids.append(EOS_ID)
        return ids

    def encode_turn(self, role: str, text: str) -> List[int]:
        """Encode one conversation turn with role token prefix and EOS suffix."""
        role_id = USR_ID if role == "user" else KIR_ID
        return [role_id] + self.encode(text) + [EOS_ID]

    def encode_conversation(self, turns: List[Tuple[str, str]]) -> List[int]:
        """Encode a full list of (role, text) turn pairs."""
        ids: List[int] = []
        for role, text in turns:
            ids.extend(self.encode_turn(role, text))
        return ids

    # ── Decode ─────────────────────────────────────────────────────────

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """Decode token IDs back to a string."""
        skip = set(range(len(SPECIAL_TOKENS))) if skip_special else set()
        return "".join(
            self.id_to_char.get(i, "?")
            for i in ids
            if i not in skip
        )

    def decode_conversation(self, ids: List[int]) -> str:
        """Decode a full token sequence into a human-readable conversation."""
        result: List[str] = []
        current_role: str | None = None
        current_chars: List[str] = []

        for token_id in ids:
            if token_id == USR_ID:
                if current_role and current_chars:
                    result.append(f"{current_role}: {''.join(current_chars).strip()}")
                current_role = "You"
                current_chars = []
            elif token_id == KIR_ID:
                if current_role and current_chars:
                    result.append(f"{current_role}: {''.join(current_chars).strip()}")
                current_role = "Kira"
                current_chars = []
            elif token_id == EOS_ID:
                if current_role and current_chars:
                    result.append(f"{current_role}: {''.join(current_chars).strip()}")
                current_role = None
                current_chars = []
            elif token_id not in (PAD_ID, UNK_ID):
                char = self.id_to_char.get(token_id, "")
                if char:
                    current_chars.append(char)

        if current_role and current_chars:
            result.append(f"{current_role}: {''.join(current_chars).strip()}")

        return "\n".join(result)

    # ── Persistence ────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save vocabulary to a JSON file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"char_to_id": self.char_to_id, "vocab_size": self.vocab_size},
                      f, indent=2, ensure_ascii=False)
        print(f"[tokenizer] saved — vocab size: {self.vocab_size}")

    @classmethod
    def load(cls, path: str) -> "KiraTokenizer":
        """Load a previously saved tokenizer from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls()
        tok.char_to_id = data["char_to_id"]
        tok.id_to_char = {v: k for k, v in data["char_to_id"].items()}
        tok.vocab_size = data["vocab_size"]
        return tok

    # ── Properties ─────────────────────────────────────────────────────

    @property
    def pad_id(self) -> int: return PAD_ID
    @property
    def usr_id(self) -> int: return USR_ID
    @property
    def kir_id(self) -> int: return KIR_ID
    @property
    def eos_id(self) -> int: return EOS_ID
    @property
    def unk_id(self) -> int: return UNK_ID

    def __repr__(self) -> str:
        return f"KiraTokenizer(vocab_size={self.vocab_size})"


def build_tokenizer_from_file(
    text_path: str,
    save_path: str = "data/tokenizer.json",
) -> KiraTokenizer:
    """Convenience function: read a file, build a tokenizer, save it."""
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()
    tok = KiraTokenizer().build(text)
    tok.save(save_path)
    return tok
