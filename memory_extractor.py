"""
Kira Memory Extractor — memory_extractor.py
============================================
Automatically extracts and saves memories from user messages.
Detects names, projects, goals, preferences, and events using
regex patterns and keyword signals.

100% local — Python stdlib only. No APIs. No cloud.

INTEGRATION INTO kira_server.py:
    from memory_extractor import MemoryExtractor
    extractor = MemoryExtractor(kira_memory)

    # Inside the /chat route, after every user message:
    extractor.extract_and_save(user_message)

STANDALONE TEST:
    python memory_extractor.py
"""

import re
from typing import Dict, List, Optional, Tuple

from memory_manager import MemoryManager


# ── Pattern definitions ───────────────────────────────────────

NAME_PATTERNS = [
    r"(?:my name is|i'm called|call me|i go by)\s+([A-Z][a-z]{1,20})",
    r"^([A-Z][a-z]{1,20})$",   # single capitalised word by itself
]

PROJECT_PATTERNS = [
    r"i(?:'m| am) (?:building|developing|designing|making|creating|working on)\s+(.{5,80}?)(?:\.|$|,)",
    r"(?:my|our) project (?:is|involves)\s+(.{5,80}?)(?:\.|$|,)",
    r"(?:i'm|i am) (?:training|fine-tuning)\s+(.{5,80}?)(?:\.|$|,)",
    r"i just (?:finished|completed|built)\s+(.{5,80}?)(?:\.|$|,)",
    r"working on\s+(.{5,80}?)(?:\.|$|,)",
]

GOAL_PATTERNS = [
    r"(?:i want|i'd like|i hope|my goal is|i plan)\s+to\s+(.{5,80}?)(?:\.|$|,)",
    r"(?:i'm trying|i am trying)\s+to\s+(.{5,80}?)(?:\.|$|,)",
    r"(?:my dream|my ambition) is\s+to\s+(.{5,80}?)(?:\.|$|,)",
    r"(?:someday|eventually) i(?:'ll| will)\s+(.{5,80}?)(?:\.|$|,)",
]

PREFERENCE_PATTERNS = [
    r"i (?:love|really love|enjoy|like|prefer|am into|am obsessed with)\s+(.{3,60}?)(?:\.|$|,)",
    r"(?:my favourite|my favorite|i'm a big fan of)\s+(.{3,60}?)(?:\.|$|,)",
    r"i(?:'m| am) (?:really )?into\s+(.{3,60}?)(?:\.|$|,)",
    r"i hate\s+(.{3,60}?)(?:\.|$|,)",
    r"i can't stand\s+(.{3,60}?)(?:\.|$|,)",
]

EVENT_PATTERNS = [
    r"(?:today|yesterday|just now|right now) i\s+(.{5,80}?)(?:\.|$|,)",
    r"i just\s+(.{3,60}?)(?:\.|$|,)",
    r"i (?:got|received|finished|completed|started)\s+(.{5,80}?)(?:\.|$|,)",
]

# Signals that a message likely contains a memory worth extracting
MEMORY_SIGNALS = [
    "my name is", "call me", "i'm called",
    "i'm building", "i am building", "working on", "my project",
    "i want to", "my goal is", "i plan to", "i hope to",
    "i love", "i enjoy", "i like", "i prefer", "i hate",
    "today i", "yesterday i", "just finished", "i finished",
    "i got", "i started", "i completed",
]


class MemoryExtractor:
    """
    Scans user messages for extractable facts and saves them to MemoryManager.
    Uses regex pattern matching — no external APIs or models required.
    """

    def __init__(self, memory_manager: MemoryManager) -> None:
        self.mm = memory_manager

    def extract_and_save(self, message: str) -> List[Dict]:
        """
        Extract memories from a user message and save them.

        Args:
            message: the raw user message

        Returns:
            List of dicts with {memory_type, content, id} for each saved memory.
        """
        if not self._has_memory_signal(message):
            return []

        saved: List[Dict] = []

        for memory_type, patterns, formatter in [
            ("user_fact", NAME_PATTERNS,       self._format_name),
            ("project",   PROJECT_PATTERNS,    self._format_project),
            ("goal",      GOAL_PATTERNS,       self._format_goal),
            ("preference",PREFERENCE_PATTERNS, self._format_preference),
            ("event",     EVENT_PATTERNS,      self._format_event),
        ]:
            for pattern in patterns:
                match = re.search(pattern, message, re.IGNORECASE)
                if not match:
                    continue
                extracted = match.group(1).strip().rstrip(".,;!?")
                if len(extracted) < 3:
                    continue
                content = formatter(extracted)
                if not content:
                    continue
                if self._is_duplicate(content):
                    continue
                try:
                    importance = self._estimate_importance(memory_type, content)
                    mem_id = self.mm.save_memory(memory_type, content, importance=importance)
                    saved.append({"memory_type": memory_type, "content": content, "id": mem_id})
                    break   # One match per category per message
                except ValueError:
                    pass

        return saved

    # ── Formatters ────────────────────────────────────────────

    @staticmethod
    def _format_name(raw: str) -> str:
        name = raw.strip().capitalize()
        return f"User's name is {name}." if len(name) >= 2 else ""

    @staticmethod
    def _format_project(raw: str) -> str:
        raw = raw.strip().rstrip(".,")
        return f"User is building/working on: {raw}."

    @staticmethod
    def _format_goal(raw: str) -> str:
        raw = raw.strip().rstrip(".,")
        return f"User wants to: {raw}."

    @staticmethod
    def _format_preference(raw: str) -> str:
        raw = raw.strip().rstrip(".,")
        return f"User preference: {raw}."

    @staticmethod
    def _format_event(raw: str) -> str:
        raw = raw.strip().rstrip(".,")
        return f"Event: User {raw}."

    # ── Helpers ───────────────────────────────────────────────

    @staticmethod
    def _has_memory_signal(message: str) -> bool:
        msg_lower = message.lower()
        return any(signal in msg_lower for signal in MEMORY_SIGNALS)

    def _is_duplicate(self, content: str) -> bool:
        """Return True if a very similar memory already exists."""
        try:
            existing = self.mm.search_memories(content, limit=5)
            for mem in existing:
                if self._similarity(content, mem["content"]) > 0.85:
                    return True
        except Exception:
            pass
        return False

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        """Simple word-overlap similarity."""
        words_a = set(re.findall(r"\w+", a.lower()))
        words_b = set(re.findall(r"\w+", b.lower()))
        if not words_a or not words_b:
            return 0.0
        return len(words_a & words_b) / len(words_a | words_b)

    @staticmethod
    def _estimate_importance(memory_type: str, content: str) -> int:
        importance_map = {
            "user_fact":  4,
            "project":    4,
            "goal":       4,
            "preference": 3,
            "event":      2,
        }
        base = importance_map.get(memory_type, 3)
        # Bump for keywords suggesting high importance
        high_signals = ["main", "primary", "most important", "dream", "life goal",
                        "biggest", "critical", "essential", "core"]
        if any(sig in content.lower() for sig in high_signals):
            base = min(5, base + 1)
        return base


# ══════════════════════════════════════════════════════════════
# STANDALONE TEST
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    mm = MemoryManager()
    extractor = MemoryExtractor(mm)

    test_messages = [
        "My name is Julian and I'm building a humanoid robot.",
        "I really love Python and working with neural networks.",
        "My goal is to become a machine learning engineer.",
        "Today I finished training the model — loss is at 1.8.",
        "I want to eventually build a fully autonomous robot.",
    ]

    print("=== Memory Extractor Test ===")
    for msg in test_messages:
        results = extractor.extract_and_save(msg)
        print(f"\nMessage: {msg!r}")
        for r in results:
            print(f"  [{r['memory_type']}] {r['content']}")
        if not results:
            print("  (no memories extracted)")
