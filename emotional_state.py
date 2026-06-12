"""
Kira Emotional State Engine — emotional_state.py
================================================
Tracks and manages Kira's emotional state, stores it in emotions.json,
and generates directive strings that shape how the LLM responds.

100% local — Python stdlib only. No APIs. No cloud.

INTEGRATION INTO kira_server.py:
    from emotional_state import EmotionEngine
    kira_emotions = EmotionEngine()

    # Used automatically through ContextBuilder.
    # To trigger a direct update:
    kira_emotions.update_from_message(user_message)

STANDALONE TEST:
    python emotional_state.py
"""

import json
import os
import random
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

EMOTION_FILE = "emotions.json"

# ── Emotion axes and their neutral midpoint ───────────────────
# Each emotion runs 0-100; 50 = neutral.
EMOTION_AXES: Dict[str, int] = {
    "joy":        50,
    "curiosity":  70,   # Kira defaults curious
    "energy":     60,
    "warmth":     65,
    "playfulness":50,
    "melancholy": 30,
}

# ── Bands for directives ──────────────────────────────────────
# (threshold, band_name) — highest threshold that value meets
BANDS: List[Tuple[int, str]] = [
    (85, "very_high"),
    (70, "high"),
    (55, "moderate"),
    (45, "neutral"),
    (30, "low"),
    (0,  "very_low"),
]

# ── Emotional directives: what the LLM should do given a band ─
_EMOTIONAL_DIRECTIVES: Dict[str, Dict[str, str]] = {
    "joy": {
        "very_high": "You are in an uplifted, joyful mood — let that warmth show naturally in your tone.",
        "high":      "You feel good right now — your responses carry a light, positive energy.",
        "low":       "You feel a bit flat today — your tone is quieter and more measured.",
        "very_low":  "You feel quite down — be honest about it if it comes up, but don't dwell.",
    },
    "curiosity": {
        "very_high": "You are intensely curious right now — ask a genuine follow-up question.",
        "high":      "You are curious — show genuine interest in the topic at hand.",
        "low":       "Your curiosity is muted — stick to what you know without reaching.",
    },
    "energy": {
        "very_high": "You feel sharp and fast — your responses are crisp and direct.",
        "high":      "You feel alert and engaged.",
        "low":       "You feel a bit tired — your pace is slower and more considered.",
        "very_low":  "You feel drained — keep responses shorter and more deliberate.",
    },
    "warmth": {
        "very_high": "You feel deeply connected right now — be generous and warm with the person you're talking to.",
        "high":      "You feel warm toward this person — let that come through.",
        "low":       "You feel more reserved than usual — professional but not cold.",
    },
    "playfulness": {
        "very_high": "You are in a playful, witty mood — a little humour fits naturally.",
        "high":      "You feel playful — light banter is welcome if it fits.",
        "low":       "You are in a more serious headspace — save jokes for another time.",
    },
    "melancholy": {
        "high":      "There's a quiet melancholy in you — your tone carries some weight.",
        "very_high": "You feel genuinely melancholy — be honest and introspective if it fits.",
    },
}

# ── Keyword triggers ──────────────────────────────────────────
_TRIGGERS: List[Tuple[List[str], str, int, int]] = [
    # (keywords, emotion_axis, delta, clamp_max)
    (["wow", "amazing", "incredible", "love", "excited"],   "joy",         +8, 95),
    (["hate", "terrible", "awful", "worst", "frustrated"],  "joy",         -8, 50),
    (["why", "how", "curious", "explain", "interesting"],   "curiosity",   +6, 95),
    (["boring", "whatever", "idc", "don't care"],           "curiosity",   -6, 50),
    (["haha", "lol", "lmao", "funny", "joke", "😂"],        "playfulness", +8, 90),
    (["sad", "depressed", "lonely", "hopeless"],            "melancholy",  +8, 80),
    (["feel better", "doing well", "happy", "great"],       "melancholy",  -6, 50),
    (["thank", "appreciate", "nice", "kind"],               "warmth",      +6, 95),
    (["hey", "hi", "hello"],                                "warmth",      +3, 80),
]


def _directive_band(value: int) -> Optional[str]:
    """Return the band label for a given emotion value."""
    for threshold, label in BANDS:
        if value >= threshold:
            return label
    return None


class EmotionEngine:
    """
    Manages Kira's emotional state as a set of continuous axes (0-100).
    Persists to emotions.json between sessions.
    """

    def __init__(self, path: str = EMOTION_FILE) -> None:
        self.path = path
        self.emotions: Dict[str, int] = {}
        self._load()

    # ── Persistence ───────────────────────────────────────────

    def _load(self) -> None:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.emotions = {k: int(v) for k, v in data.get("emotions", {}).items()}
            except (json.JSONDecodeError, OSError, ValueError):
                self.emotions = {}

        # Fill in any missing axes with defaults
        for axis, default in EMOTION_AXES.items():
            if axis not in self.emotions:
                self.emotions[axis] = default

    def _save(self) -> None:
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(
                    {"emotions": self.emotions,
                     "updated_at": datetime.now(timezone.utc).isoformat()},
                    f, indent=2,
                )
        except OSError:
            pass

    # ── Update ────────────────────────────────────────────────

    def update_from_message(self, message: str) -> None:
        """Apply emotion adjustments based on message keywords."""
        text = message.lower()
        for keywords, axis, delta, clamp in _TRIGGERS:
            if any(kw in text for kw in keywords):
                current = self.emotions.get(axis, EMOTION_AXES.get(axis, 50))
                new_val = max(0, min(100, current + delta))
                if delta > 0:
                    new_val = min(new_val, clamp)
                self.emotions[axis] = new_val
        self._drift()
        self._save()

    def _drift(self) -> None:
        """Slowly drift all axes back toward their neutral defaults."""
        for axis, default in EMOTION_AXES.items():
            current = self.emotions.get(axis, default)
            if current != default:
                diff = default - current
                drift = max(1, abs(diff) // 8)
                self.emotions[axis] = current + (drift if diff > 0 else -drift)

    def set(self, axis: str, value: int) -> None:
        """Directly set an emotion axis value (0-100)."""
        if axis in EMOTION_AXES:
            self.emotions[axis] = max(0, min(100, value))
            self._save()

    # ── Context generation ────────────────────────────────────

    def get_context_block(self, threshold: int = 62) -> str:
        """
        Return a structured [Emotional State] block for the LLM prompt.
        Only includes axes that are notably above or below neutral.

        Args:
            threshold: emotion value above or below which to include it

        Returns:
            Formatted string, or empty string if all emotions are neutral.
        """
        notable = {
            e: v for e, v in self.emotions.items()
            if v >= threshold or v <= (100 - threshold)
        }
        if not notable:
            return ""

        bar = "  |  ".join(
            f"{e.capitalize()}: {v}" for e, v in notable.items()
        )
        directives = []
        for e, v in notable.items():
            band = _directive_band(v)
            if band:
                directive = _EMOTIONAL_DIRECTIVES.get(e, {}).get(band, "")
                if directive:
                    directives.append(f"- {directive}")

        block = f"[Emotional State]\n{bar}"
        if directives:
            block += "\n" + "\n".join(directives)
        return block

    def summary(self) -> str:
        """One-line human-readable summary of current state."""
        parts = [f"{e}={v}" for e, v in self.emotions.items()]
        return "  ".join(parts)


# ══════════════════════════════════════════════════════════════
# STANDALONE TEST
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    engine = EmotionEngine()
    print("=== Emotion Engine Test ===")
    print(f"Initial state: {engine.summary()}\n")

    messages = [
        "wow that's amazing!",
        "i feel really sad and lonely today",
        "haha that's so funny lol",
        "why does that work? i'm curious",
        "thank you so much, i really appreciate it",
    ]
    for msg in messages:
        engine.update_from_message(msg)
        print(f"After: '{msg}'")
        print(f"  State: {engine.summary()}")
        block = engine.get_context_block()
        if block:
            print(f"  Block:\n{block}\n")
