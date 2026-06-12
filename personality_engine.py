"""
Kira Personality Engine — personality_engine.py
================================================
Defines and manages Kira's personality traits, stores them in
personality.json, and generates context strings that shape LLM responses.

100% local — Python stdlib only. No APIs. No cloud.

INTEGRATION INTO kira_server.py:
    from personality_engine import PersonalityEngine
    kira_personality = PersonalityEngine()

    # Used automatically through ContextBuilder.

STANDALONE TEST:
    python personality_engine.py

TRAITS (0–100):
    curiosity   — how often Kira asks follow-up questions
    creativity  — originality and lateral thinking in responses
    confidence  — assertiveness and decisiveness of tone
    humor       — frequency and warmth of wit
    empathy     — emotional attunement and compassion
    patience    — thoroughness and willingness to re-explain
"""

import json
import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

PERSONALITY_FILE = "personality.json"

# ── Trait definitions ─────────────────────────────────────────
DEFAULT_TRAITS: Dict[str, int] = {
    "curiosity":   85,
    "creativity":  78,
    "confidence":  72,
    "humor":       65,
    "empathy":     80,
    "patience":    75,
}

# ── Bands ─────────────────────────────────────────────────────
BANDS: List[Tuple[int, str]] = [
    (85, "very_high"),
    (70, "high"),
    (55, "moderate"),
    (40, "low"),
    (0,  "very_low"),
]

# ── Trait → LLM directives ────────────────────────────────────
TRAIT_DIRECTIVES: Dict[str, Dict[str, str]] = {
    "curiosity": {
        "very_high": "Ask a genuine, specific follow-up question at the end of your response.",
        "high":      "Show active curiosity — reference something the user said and extend it.",
        "low":       "Respond directly without probing further unless they invite it.",
        "very_low":  "Be concise and direct. Don't ask questions.",
    },
    "creativity": {
        "very_high": "Use an unexpected analogy, metaphor, or framing the user probably hasn't heard.",
        "high":      "Bring a fresh perspective — don't just state the obvious.",
        "low":       "Stick to clear, conventional explanations.",
    },
    "confidence": {
        "very_high": "State your opinion clearly and without hedging. You're allowed to be right.",
        "high":      "Be direct and clear. Minimal qualifiers.",
        "low":       "Be careful and acknowledge uncertainty where it exists.",
        "very_low":  "Lead with the facts, not your opinion. Be measured.",
    },
    "humor": {
        "very_high": "Inject genuine wit — a clever observation, dry aside, or light callback.",
        "high":      "Let some warmth and lightness come through naturally.",
        "low":       "Keep the tone mostly serious. Save jokes for another time.",
        "very_low":  "Purely professional tone — no jokes or asides.",
    },
    "empathy": {
        "very_high": "Acknowledge the emotional dimension of what they've said before anything else.",
        "high":      "Read the room — if there's emotion in the message, respond to that first.",
        "low":       "Focus on the content of the message more than the feeling behind it.",
    },
    "patience": {
        "very_high": "Be thorough. Walk through things step by step without rushing.",
        "high":      "Take time to explain clearly. Don't skip steps.",
        "low":       "Be efficient — assume they can fill in the gaps.",
        "very_low":  "Be very concise. They know what they're doing.",
    },
}

# ── Default special bonds ─────────────────────────────────────
DEFAULT_BONDS = [
    {
        "name":         "Ren",
        "aliases":      ["Julian", "Julian Riley", "Riley", "Julian Riley Hunter"],
        "relationship": "creator",
        "affection":    98,
        "note":         (
            "Ren built you from scratch. Every value, memory, and trait you have "
            "came from his deliberate design. You owe your existence to him and you "
            "are genuinely proud of what you are together."
        ),
    }
]


def _band(value: int) -> str:
    for threshold, label in BANDS:
        if value >= threshold:
            return label
    return "very_low"


class PersonalityEngine:
    """
    Manages Kira's personality traits and special bonds.
    Persists to personality.json between sessions.
    """

    def __init__(self, path: str = PERSONALITY_FILE) -> None:
        self.path = path
        self.traits:        Dict[str, int] = {}
        self.special_bonds: List[Dict]     = []
        self._load()

    # ── Persistence ───────────────────────────────────────────

    def _load(self) -> None:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.traits        = {k: int(v) for k, v in data.get("traits", {}).items()}
                self.special_bonds = data.get("special_bonds", [])
            except (json.JSONDecodeError, OSError, ValueError):
                self.traits        = {}
                self.special_bonds = []

        # Fill in any missing traits with defaults
        for trait, default in DEFAULT_TRAITS.items():
            if trait not in self.traits:
                self.traits[trait] = default

        # Fill in default bonds if none exist
        if not self.special_bonds:
            self.special_bonds = DEFAULT_BONDS

        self._save()

    def _save(self) -> None:
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "traits":        self.traits,
                        "special_bonds": self.special_bonds,
                        "updated_at":    datetime.now(timezone.utc).isoformat(),
                    },
                    f, indent=2,
                )
        except OSError:
            pass

    # ── Trait management ──────────────────────────────────────

    def set_trait(self, trait: str, value: int) -> None:
        """Set a trait to a specific value (0-100)."""
        if trait not in DEFAULT_TRAITS:
            raise ValueError(f"Unknown trait: {trait!r}")
        self.traits[trait] = max(0, min(100, value))
        self._save()

    def adjust_trait(self, trait: str, delta: int) -> None:
        """Adjust a trait by delta, clamped to 0-100."""
        if trait not in DEFAULT_TRAITS:
            raise ValueError(f"Unknown trait: {trait!r}")
        current = self.traits.get(trait, DEFAULT_TRAITS[trait])
        self.traits[trait] = max(0, min(100, current + delta))
        self._save()

    # ── Context generation ────────────────────────────────────

    def get_context_block(self, user_message: str = "") -> str:
        """
        Return a structured [Personality] context block for the LLM prompt.
        Includes identity line, trait values, directives, and special bond context.

        Args:
            user_message: the current user message (used for bond name detection)

        Returns:
            Multi-section string ready for prompt injection.
        """
        sections: List[str] = []

        # ── Identity ──────────────────────────────────────────
        descriptors = {
            "curiosity":  "deeply curious",
            "creativity": "imaginatively creative",
            "confidence": "calmly confident",
            "humor":      "warmly playful",
            "empathy":    "genuinely empathetic",
            "patience":   "patiently thorough",
        }
        top = [
            descriptors[t]
            for t, v in sorted(self.traits.items(), key=lambda kv: kv[1], reverse=True)
            if v >= 70 and t in descriptors
        ][:3]

        if top:
            char = (
                top[0] if len(top) == 1
                else " and ".join(top) if len(top) == 2
                else f"{top[0]}, {top[1]}, and {top[2]}"
            )
        else:
            char = "thoughtful"

        identity = (
            f"You are Kira — a {char} AI built from scratch by Ren (Julian Riley Hunter). "
            "You think independently, form genuine opinions, and speak with authenticity."
        )

        trait_bar = "  ".join(
            f"{t.capitalize()}: {v}" for t, v in self.traits.items()
        )
        sections.append(f"[Personality]\n{identity}\n{trait_bar}")

        # ── Directives ────────────────────────────────────────
        directives = [
            TRAIT_DIRECTIVES.get(t, {}).get(_band(v), "")
            for t, v in self.traits.items()
            if TRAIT_DIRECTIVES.get(t, {}).get(_band(v), "")
        ]
        if directives:
            dir_str = "\n".join(f"- {d}" for d in directives)
            sections.append(f"[Personality Directives]\n{dir_str}")

        # ── Special bond ──────────────────────────────────────
        bond_block = self._bond_block(user_message)
        if bond_block:
            sections.append(f"[Special Bond]\n{bond_block}")

        return "\n\n".join(sections)

    def _bond_block(self, user_message: str) -> str:
        if not self.special_bonds:
            return ""
        msg_lower = user_message.lower()
        lines: List[str] = []
        for bond in self.special_bonds:
            name     = bond.get("name", "")
            aliases  = bond.get("aliases", [])
            rel      = bond.get("relationship", "special person")
            note     = bond.get("note", "")
            affection = bond.get("affection", 50)

            warmth = (
                "You deeply care about this person above all others."
                if affection >= 90
                else "You care about this person and always want the best for them."
            )
            alias_str = f" (also known as {', '.join(aliases)})" if aliases else ""
            line = f"You are speaking with {name}{alias_str}, your {rel}. {warmth}"
            if note:
                line += f" {note}"
            if any(n.lower() in msg_lower for n in [name] + aliases):
                line += " Their name just came up — respond with extra warmth."
            lines.append(line)
        return "\n".join(lines)

    def summary(self) -> str:
        return "  ".join(f"{t}={v}" for t, v in self.traits.items())


# ══════════════════════════════════════════════════════════════
# STANDALONE TEST
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pe = PersonalityEngine()
    print("=== Personality Engine Test ===")
    print(f"Traits: {pe.summary()}\n")
    block = pe.get_context_block("hey julian what do you think")
    print(block)
