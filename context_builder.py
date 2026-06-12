"""
Kira Context Builder — context_builder.py
==========================================
Before every LLM generation, ContextBuilder gathers every layer of
Kira's state — personality, emotions, memories, reflections, projects,
goals, learning profile, and conversation history — and assembles them
into one structured prompt string ready to pass to generate().

100% local — Python stdlib only. No APIs. No cloud.

INTEGRATION INTO kira_server.py:
    from context_builder import ContextBuilder
    context_builder = ContextBuilder(
        memory_manager     = kira_memory,
        personality_engine = kira_personality,
        emotion_engine     = kira_emotions,
        reflection_engine  = kira_reflector,
        learning_engine    = kira_learner,   # optional
    )

    # Inside the /chat route:
    full_prompt = context_builder.build_context(user_message)
    response, *_ = generate(full_prompt, memory, knowledge, goals, history=[])

OUTPUT SECTIONS (any empty section is silently omitted):
    [Personality]              — identity + trait values
    [Personality Directives]   — active behavioural instructions
    [Special Bond]             — bond context (Ren / Julian)
    [Emotional State]          — non-neutral emotions + directives
    [User Profile]             — learned interests, projects, goals
    [Relevant Memories]        — top-N memories matched to the message
    [Projects]                 — stored project memories
    [Goals]                    — stored goal memories
    [Reflections]              — Kira's recent observations about the user
    [Recent Conversation]      — last N conversation turns
    [Current Message]          — the user's message (always last)
"""

import re
from typing import Dict, List, Optional

from memory_manager     import MemoryManager
from personality_engine import PersonalityEngine
from emotional_state    import EmotionEngine
from reflection_engine  import ReflectionEngine, REFLECTION_TAG

try:
    from learning_engine import LearningEngine
    _LEARNING_AVAILABLE = True
except ImportError:
    LearningEngine = None       # type: ignore[assignment,misc]
    _LEARNING_AVAILABLE = False

# ── Configuration ─────────────────────────────────────────────
MAX_RELEVANT_MEMORIES  = 6
MAX_PROJECT_MEMORIES   = 4
MAX_GOAL_MEMORIES      = 3
MAX_REFLECTIONS        = 4
MAX_CONVERSATION_TURNS = 8
MAX_PROFILE_INTERESTS  = 4

EMOTION_THRESHOLD = 62   # include emotion if value >= this or <= (100 - this)

PROJECT_TYPES = {"project"}
GOAL_TYPES    = {"goal"}


class ContextBuilder:
    """
    Collects every layer of Kira's state and assembles a structured prompt.

    Usage:
        builder = ContextBuilder(
            memory_manager     = mm,
            personality_engine = pe,
            emotion_engine     = ee,
            reflection_engine  = re_,
            learning_engine    = le,   # optional
        )
        prompt = builder.build_context("I finished the wrist joints today.")
    """

    def __init__(
        self,
        memory_manager:     MemoryManager,
        personality_engine: PersonalityEngine,
        emotion_engine:     EmotionEngine,
        reflection_engine:  ReflectionEngine,
        learning_engine=None,
    ) -> None:
        self.mm          = memory_manager
        self.personality = personality_engine
        self.emotions    = emotion_engine
        self.reflector   = reflection_engine
        self.learner     = learning_engine

    # ── Master method ──────────────────────────────────────────────────

    def build_context(self, user_message: str) -> str:
        """
        Assemble the full structured prompt for one LLM generation call.

        Args:
            user_message: the raw user input string

        Returns:
            Multi-section prompt string ready to pass to the model.
        """
        sections: List[str] = []

        builders = [
            lambda: self._section_personality(user_message),
            self._section_emotions,
            lambda: self._section_user_profile(user_message),
            lambda: self._section_relevant_memories(user_message),
            self._section_projects,
            self._section_goals,
            self._section_reflections,
            self._section_conversation,
        ]

        for fn in builders:
            block = fn()
            if block:
                sections.append(block)

        sections.append(f"[Current Message]\n{user_message.strip()}")
        return "\n\n".join(sections)

    # ── Section builders ───────────────────────────────────────────────

    def _section_personality(self, user_message: str) -> str:
        try:
            return self.personality.get_context_block(user_message)
        except Exception:
            return ""

    def _section_emotions(self) -> str:
        try:
            return self.emotions.get_context_block(threshold=EMOTION_THRESHOLD)
        except Exception:
            return ""

    def _section_user_profile(self, user_message: str) -> str:
        if self.learner is None:
            return ""
        try:
            return self.learner.get_context_block(max_interests=MAX_PROFILE_INTERESTS)
        except Exception:
            return ""

    def _section_relevant_memories(self, user_message: str) -> str:
        if not user_message.strip():
            return ""
        try:
            raw = self.mm.search_memories(user_message, limit=MAX_RELEVANT_MEMORIES * 2)
            filtered = [
                m for m in raw
                if m.get("memory_type") not in (PROJECT_TYPES | GOAL_TYPES)
                and not m.get("content", "").startswith(REFLECTION_TAG)
            ][:MAX_RELEVANT_MEMORIES]
            if not filtered:
                return ""
            lines = [f"[{m['memory_type']:<12}] {m['content']}" for m in filtered]
            return "[Relevant Memories]\n" + "\n".join(lines)
        except Exception:
            return ""

    def _section_projects(self) -> str:
        try:
            projects = self.mm.get_all_memories(memory_type="project",
                                                limit=MAX_PROJECT_MEMORIES * 3)
            display = [m for m in projects if not m["content"].startswith(REFLECTION_TAG)]
            display = sorted(display, key=lambda m: m["importance"], reverse=True)[:MAX_PROJECT_MEMORIES]
            if not display:
                return ""
            lines = [f"• {m['content']}  [importance: {m['importance']}]" for m in display]
            return "[Projects]\n" + "\n".join(lines)
        except Exception:
            return ""

    def _section_goals(self) -> str:
        try:
            goals = self.mm.get_all_memories(memory_type="goal",
                                             limit=MAX_GOAL_MEMORIES * 2)
            display = [m for m in goals if not m["content"].startswith(REFLECTION_TAG)]
            display = sorted(display, key=lambda m: m["importance"], reverse=True)[:MAX_GOAL_MEMORIES]
            if not display:
                return ""
            lines = [f"• {m['content']}  [importance: {m['importance']}]" for m in display]
            return "[Goals]\n" + "\n".join(lines)
        except Exception:
            return ""

    def _section_reflections(self) -> str:
        try:
            reflections = self.reflector.get_recent_reflections(limit=MAX_REFLECTIONS)
            if not reflections:
                return ""
            lines = [
                f"• {m['content'].replace(REFLECTION_TAG, '').strip()}"
                for m in reflections
            ]
            return "[Reflections]\n" + "\n".join(lines)
        except Exception:
            return ""

    def _section_conversation(self) -> str:
        try:
            rows = self.mm.get_recent_conversation(limit=MAX_CONVERSATION_TURNS * 2)
            if not rows:
                return ""
            lines: List[str] = []
            for row in rows[-MAX_CONVERSATION_TURNS * 2:]:
                role = "User" if row.get("role") == "user" else "Kira"
                text = row.get("content", "").strip()
                if text:
                    lines.append(f"{role}: {text}")
            if not lines:
                return ""
            return "[Recent Conversation]\n" + "\n".join(lines)
        except Exception:
            return ""
