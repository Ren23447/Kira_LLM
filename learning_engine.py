"""
Kira Learning Engine — learning_engine.py
==========================================
Automatically learns patterns from every conversation and builds a
dynamic user profile stored in user_profile.json.

Tracks:
  - Most-discussed topics with interest scores
  - Projects, goals, preferences (from MemoryManager)
  - Conversation habits (length, question rate)
  - Derived insights from ReflectionEngine

100% local — Python stdlib only. No APIs. No cloud.

INTEGRATION INTO kira_server.py:
    from learning_engine import LearningEngine
    kira_learner = LearningEngine(kira_memory, reflection_engine=kira_reflector)

    # After every chat response:
    kira_learner.update_user_profile()

New routes:
    GET  /profile           — full user_profile.json
    GET  /profile/summary   — one-paragraph text summary
    POST /profile/rebuild   — force a full rebuild from all data

STANDALONE TEST:
    python learning_engine.py
"""

import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from memory_manager import MemoryManager

try:
    from reflection_engine import ReflectionEngine, REFLECTION_TAG
    _RE_AVAILABLE = True
except ImportError:
    ReflectionEngine = None  # type: ignore[assignment,misc]
    REFLECTION_TAG = "[Reflection]"
    _RE_AVAILABLE = False

DEFAULT_PROFILE_PATH = "user_profile.json"
ANALYSIS_WINDOW      = 60
TOPIC_DECAY          = 0.95
MIN_TOPIC_SCORE      = 0.5

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "kira", "user", "hi", "hey", "hello", "thanks", "thank",
    "ok", "yes", "no", "sure", "right", "okay", "just", "get", "got",
    "know", "think", "feel", "want", "need", "like", "go", "come", "see",
}

TOPIC_MAP: Dict[str, str] = {
    "robot": "robotics", "robots": "robotics", "humanoid": "humanoid robotics",
    "servo": "servo motors", "motor": "motors", "joint": "joints",
    "python": "Python", "code": "programming", "coding": "programming",
    "llm": "LLM development", "neural": "neural networks",
    "model": "machine learning", "training": "model training",
    "ai": "AI", "ml": "machine learning", "game": "gaming",
    "minecraft": "Minecraft", "music": "music", "art": "art",
    "space": "space", "star": "astronomy", "science": "science",
    "book": "reading", "philosophy": "philosophy", "health": "health",
    "fitness": "fitness", "math": "mathematics",
}


def _tokens(text: str) -> List[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9'_-]{1,}", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


def _topic_label(kw: str) -> str:
    return TOPIC_MAP.get(kw, kw)


class LearningEngine:
    """
    Analyses conversation history and memory to build a live user profile.
    Profile is persisted to user_profile.json.
    """

    def __init__(
        self,
        memory_manager: MemoryManager,
        reflection_engine=None,
        profile_path: str = DEFAULT_PROFILE_PATH,
    ) -> None:
        self.mm        = memory_manager
        self.reflector = reflection_engine
        self.path      = profile_path
        self.profile: Dict = {}
        self._load()

    # ── Persistence ───────────────────────────────────────────

    def _load(self) -> None:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.profile = json.load(f)
            except (json.JSONDecodeError, OSError):
                self.profile = {}
        if not self.profile:
            self.profile = {
                "interests":            {},
                "projects":             [],
                "goals":                [],
                "preferences":          [],
                "communication_style":  {},
                "insights":             [],
                "last_updated":         None,
            }

    def _save(self) -> None:
        self.profile["last_updated"] = datetime.now(timezone.utc).isoformat()
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.profile, f, indent=2, ensure_ascii=False)
        except OSError:
            pass

    # ── Update ────────────────────────────────────────────────

    def update_user_profile(self) -> None:
        """
        Re-analyse recent conversation and memory.
        Call this after every chat response.
        """
        self._update_interests()
        self._update_projects()
        self._update_goals()
        self._update_preferences()
        self._update_communication_style()
        if self.reflector and _RE_AVAILABLE:
            self._update_insights()
        self._save()

    def _update_interests(self) -> None:
        rows = self.mm.get_recent_conversation(limit=ANALYSIS_WINDOW)
        user_texts = [r["content"] for r in rows if r.get("role") == "user"]

        counts: Counter = Counter()
        for text in user_texts:
            for token in _tokens(text):
                label = _topic_label(token)
                counts[label] += 1

        existing = self.profile.get("interests", {})
        # Decay existing scores
        for key in existing:
            existing[key] = round(existing[key] * TOPIC_DECAY, 2)

        # Add new counts
        for topic, count in counts.items():
            existing[topic] = existing.get(topic, 0) + math.log1p(count)

        # Prune below threshold
        self.profile["interests"] = {
            k: round(v, 2)
            for k, v in existing.items()
            if v >= MIN_TOPIC_SCORE
        }

    def _update_projects(self) -> None:
        mems = self.mm.get_all_memories(memory_type="project", limit=20)
        self.profile["projects"] = [
            m["content"] for m in mems
            if not m["content"].startswith(REFLECTION_TAG)
        ][:10]

    def _update_goals(self) -> None:
        mems = self.mm.get_all_memories(memory_type="goal", limit=10)
        self.profile["goals"] = [
            m["content"] for m in mems
            if not m["content"].startswith(REFLECTION_TAG)
        ][:5]

    def _update_preferences(self) -> None:
        mems = self.mm.get_all_memories(memory_type="preference", limit=20)
        self.profile["preferences"] = [
            m["content"] for m in mems
            if not m["content"].startswith(REFLECTION_TAG)
        ][:10]

    def _update_communication_style(self) -> None:
        rows = self.mm.get_recent_conversation(limit=ANALYSIS_WINDOW)
        user_rows = [r for r in rows if r.get("role") == "user"]
        if not user_rows:
            return
        words = [len(r["content"].split()) for r in user_rows]
        questions = sum(1 for r in user_rows if "?" in r["content"])
        self.profile["communication_style"] = {
            "avg_message_length":  round(sum(words) / len(words), 1),
            "question_rate":       round(questions / len(user_rows), 2),
            "total_messages":      len(user_rows),
        }

    def _update_insights(self) -> None:
        try:
            recent = self.reflector.get_recent_reflections(limit=10)
            self.profile["insights"] = [
                m["content"].replace(REFLECTION_TAG, "").strip()
                for m in recent
            ]
        except Exception:
            pass

    # ── Context generation ────────────────────────────────────

    def get_context_block(self, max_interests: int = 4) -> str:
        """
        Return a [User Profile] block for the LLM prompt.

        Args:
            max_interests: number of top interests to include

        Returns:
            Formatted string, or empty string if profile is empty.
        """
        parts: List[str] = []

        interests = self.profile.get("interests", {})
        if interests:
            top = sorted(interests.items(), key=lambda kv: kv[1], reverse=True)[:max_interests]
            parts.append("Interests: " + ", ".join(k for k, _ in top))

        projects = self.profile.get("projects", [])
        if projects:
            parts.append("Projects: " + " | ".join(projects[:3]))

        goals = self.profile.get("goals", [])
        if goals:
            parts.append("Goals: " + " | ".join(goals[:2]))

        if not parts:
            return ""

        return "[User Profile]\n" + "\n".join(parts)

    def get_summary(self) -> str:
        """Return a one-paragraph plain-text profile summary."""
        interests = self.profile.get("interests", {})
        projects  = self.profile.get("projects", [])
        goals     = self.profile.get("goals", [])

        top_interests = sorted(interests.items(), key=lambda kv: kv[1], reverse=True)[:5]
        lines: List[str] = []

        if top_interests:
            labels = ", ".join(k for k, _ in top_interests)
            lines.append(f"Top interests: {labels}.")
        if projects:
            lines.append(f"Active projects: {'; '.join(projects[:3])}.")
        if goals:
            lines.append(f"Goals: {'; '.join(goals[:2])}.")

        style = self.profile.get("communication_style", {})
        if style:
            avg = style.get("avg_message_length", 0)
            if avg > 20:
                lines.append("Communicates in long, detailed messages.")
            elif avg < 8:
                lines.append("Communicates concisely.")

        return " ".join(lines) if lines else "No profile data yet."


# ══════════════════════════════════════════════════════════════
# STANDALONE TEST
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    mm = MemoryManager()
    mm.save_memory("project",    "User is building a humanoid robot.")
    mm.save_memory("goal",       "User wants to become a machine learning engineer.")
    mm.save_memory("preference", "User loves Python and building things from scratch.")
    mm.save_conversation_turn("I trained the model today", "Nice, how's the loss looking?")
    mm.save_conversation_turn("about 2.4 still dropping", "Good — keep going.")

    engine = LearningEngine(mm)
    engine.update_user_profile()
    print("=== Learning Engine Test ===")
    print(engine.get_summary())
    print("\n--- Context block ---")
    print(engine.get_context_block())
