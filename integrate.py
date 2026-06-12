"""
Kira LLM — Integration Bridge
Connects a trained KiraLLM checkpoint to the kira.py chat system.

After training, this module makes kira.py use the neural network
instead of the fragment-assembly engine. If no checkpoint exists yet,
kira.py falls back to fragments automatically — no crash, no error.

HOW IT WORKS:
    1. Train the model:   python train.py  (or python train_cpu.py)
    2. Run the chat:      python kira.py
       kira.py auto-imports this module. If a checkpoint exists in
       checkpoints/best.pt or checkpoints/final.pt, the LLM is loaded
       and used for responses. Otherwise, fragment assembly is used.

NOTE: kira_server.py has its own independent model loader and does NOT
import this module. This bridge is only used in CLI (kira.py) mode.
"""

import os
import sys
from typing import List, Optional, Tuple

LLM_AVAILABLE = False
_kira_inference = None


# ══════════════════════════════════════════════════════════════
# LOADER
# ══════════════════════════════════════════════════════════════

def _try_load_llm() -> bool:
    """Attempt to load the trained LLM. Returns True if successful."""
    global _kira_inference, LLM_AVAILABLE

    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)

    tokenizer_path = os.path.join(here, "data", "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        return False

    ckpt_dir = os.path.join(here, "checkpoints")
    candidates: List[str] = []

    for name in ("best.pt", "final.pt"):
        p = os.path.join(ckpt_dir, name)
        if os.path.exists(p):
            candidates.append(p)

    if os.path.exists(ckpt_dir):
        numbered = sorted(
            [f for f in os.listdir(ckpt_dir) if f.startswith("ckpt_") and f.endswith(".pt")],
            reverse=True,
        )
        candidates += [os.path.join(ckpt_dir, f) for f in numbered[:2]]

    for ckpt in candidates:
        if not os.path.exists(ckpt):
            continue
        try:
            from generate import KiraInference
            _kira_inference = KiraInference(
                checkpoint_path    = ckpt,
                tokenizer_path     = tokenizer_path,
                device             = "auto",
                temperature        = 0.75,
                top_k              = 40,
                top_p              = 0.95,
                max_new_tokens     = 250,
                repetition_penalty = 1.1,
            )
            LLM_AVAILABLE = True
            print(f"[KIRA-LLM] loaded from {ckpt}")
            return True
        except Exception as e:
            print(f"[KIRA-LLM] load failed ({e}) — using fallback")
            return False

    return False


# Attempt to load at import time (silent if not trained yet)
_try_load_llm()


# ══════════════════════════════════════════════════════════════
# PUBLIC API — called by kira.py
# ══════════════════════════════════════════════════════════════

def llm_generate(
    user_message: str,
    memory:       dict,
    session_log:  list,
    max_tokens:   Optional[int] = None,
) -> Optional[str]:
    """
    Generate a response using the trained LLM.

    Returns None if LLM is unavailable — kira.py falls back to
    fragment assembly automatically.

    Args:
        user_message: the user's raw input
        memory:       kira.py memory dict (name, interests, projects, …)
        session_log:  list of {user, kira} dicts from the current session
        max_tokens:   optional override for response length

    Returns:
        Response string, or None if LLM is unavailable.
    """
    if not LLM_AVAILABLE or _kira_inference is None:
        return None

    try:
        context       = _build_context(session_log)
        memory_prefix = _build_memory_prefix(memory)
        prompt        = (memory_prefix + user_message) if memory_prefix else user_message
        return _kira_inference.respond(prompt, context=context, max_tokens=max_tokens)
    except Exception as e:
        print(f"[KIRA-LLM] generation error: {e}")
        return None


def llm_status() -> dict:
    """Return status info about the loaded LLM."""
    if not LLM_AVAILABLE or _kira_inference is None:
        return {
            "available": False,
            "reason": "no trained checkpoint — run: python train.py",
        }
    m = _kira_inference.model
    return {
        "available":  True,
        "parameters": m.num_parameters(),
        "n_layer":    m.cfg.n_layer,
        "n_head":     m.cfg.n_head,
        "n_embd":     m.cfg.n_embd,
        "block_size": m.cfg.block_size,
        "vocab_size": m.cfg.vocab_size,
        "device":     str(_kira_inference.device),
    }


def llm_is_available() -> bool:
    """True if a trained checkpoint was successfully loaded."""
    return LLM_AVAILABLE


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def _build_context(session_log: list) -> List[Tuple[str, str]]:
    """Convert session_log dicts into (role, text) pairs (last 8 entries)."""
    context: List[Tuple[str, str]] = []
    for entry in session_log[-8:]:
        if entry.get("user"):
            context.append(("user", entry["user"]))
        if entry.get("kira"):
            context.append(("kira", entry["kira"]))
    return context


def _build_memory_prefix(memory: dict) -> str:
    """Prepend a brief memory note so the LLM knows who it's talking to."""
    parts: List[str] = []
    if memory.get("name"):
        parts.append(f"[talking with {memory['name']}]")
    projects = memory.get("projects", [])
    if projects:
        parts.append(f"[projects: {', '.join(str(p) for p in projects[:2])}]")
    interests = memory.get("interests", [])
    if interests:
        parts.append(f"[interests: {', '.join(str(i) for i in interests[:3])}]")
    return (" ".join(parts) + " ") if parts else ""


# ══════════════════════════════════════════════════════════════
# SELF-CHECK
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== Kira LLM Integration Check ===")
    status = llm_status()
    for k, v in status.items():
        print(f"  {k}: {v}")
    if LLM_AVAILABLE:
        print("\n[OK] LLM is loaded and ready.")
        resp = llm_generate("hello kira", memory={}, session_log=[])
        print(f"[test] response: {resp}")
    else:
        print("\n[INFO] No checkpoint found — run python train.py to train first.")
