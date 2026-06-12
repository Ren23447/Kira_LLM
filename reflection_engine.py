"""
Kira Reflection Engine — reflection_engine.py
=============================================
Every N conversation turns, Kira reviews recent conversations and stored
memories, then generates natural-language observations about the user
and saves them as long-term memories.

100% local — Python stdlib only. No APIs. No cloud.

INTEGRATION INTO kira_server.py:
    from reflection_engine import ReflectionEngine
    kira_reflector = ReflectionEngine(kira_memory)

    # Inside the /chat route, every 20 turns:
    if memory.get("total_turns", 0) % 20 == 0:
        reflections = kira_reflector.generate_reflection()

STANDALONE TEST:
    python reflection_engine.py
"""

import re
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from memory_manager import MemoryManager

REFLECTION_TAG  = "[Reflection]"
REFLECTION_TYPE = "long_term"
REFLECTION_IMPORTANCE = 3

TOPIC_MIN_COUNT         = 2
MIN_PROJECT_MEMORIES    = 2
MIN_PREFERENCE_MEMORIES = 2
MIN_GOAL_MEMORIES       = 1
CONVERSATION_WINDOW     = 40

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "i", "me", "my", "we", "our",
    "you", "your", "he", "his", "she", "her", "it", "its", "they", "their",
    "this", "that", "these", "those", "what", "how", "when", "where", "who",
    "which", "not", "no", "so", "if", "then", "just", "also", "about",
    "really", "very", "quite", "kira", "user", "actually", "basically",
    "hi", "hey", "hello", "thanks", "thank", "sure", "ok", "yes",
    "said", "says", "say", "know", "think", "feel", "want", "need", "like",
    "get", "got", "make", "go", "come", "see", "take", "give", "use",
    "today", "yesterday", "day", "time", "way", "new", "good", "bad",
}

TOPIC_LABELS: Dict[str, str] = {
    "robot": "robotics", "robots": "robotics", "humanoid": "humanoid robotics",
    "servo": "servo motors", "motor": "motors", "joint": "mechanical joints",
    "python": "Python", "code": "programming", "coding": "programming",
    "program": "programming", "llm": "LLM development", "neural": "neural networks",
    "model": "machine learning models", "training": "model training",
    "ai": "AI", "ml": "machine learning", "hardware": "hardware",
    "software": "software", "design": "design", "build": "building",
    "building": "building", "project": "projects", "minecraft": "Minecraft",
    "game": "games", "gaming": "gaming",
}


# ── Helpers ───────────────────────────────────────────────────

def _tokenise(text: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9'_-]{1,}", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


def _label(keyword: str) -> str:
    return TOPIC_LABELS.get(keyword, keyword)


def _strip_tag(content: str) -> str:
    return content.replace(REFLECTION_TAG, "").strip()


def _is_reflection(content: str) -> bool:
    return content.strip().startswith(REFLECTION_TAG)


def _oxford(items: List[str]) -> str:
    if not items:       return ""
    if len(items) == 1: return items[0]
    if len(items) == 2: return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _extract_project_name(content: str) -> Optional[str]:
    patterns = [
        r"User is (?:building|developing|designing|creating|making|working on)\s+(.+?)\.",
        r"User is developing\s+(.+?),",
        r"User's project is (?:about\s+)?(.+?)\.",
    ]
    for pat in patterns:
        m = re.search(pat, content, re.I)
        if m:
            name = re.sub(r"^(?:a|an|the)\s+", "", m.group(1).strip(" ."), flags=re.I)
            if len(name) > 2:
                return name
    return None


def _extract_preference_object(content: str) -> Optional[str]:
    m = re.search(
        r"User (?:loves?|enjoys?|likes?|prefers?|is interested in)\s+(.+?)(?:\.|$)",
        content, re.I,
    )
    return m.group(1).strip(" .") if m else None


def _extract_goal_action(content: str) -> Optional[str]:
    m = re.search(
        r"User (?:wants? to|plans? to|hopes? to|'s goal is to|is trying to)\s+(.+?)(?:\.|$)",
        content, re.I,
    )
    return m.group(1).strip(" .") if m else None


class ReflectionEngine:
    """
    Generates and stores natural-language reflections about the user
    by analysing recent conversations and long-term memories.
    """

    def __init__(self, memory_manager: MemoryManager) -> None:
        self.mm = memory_manager

    # ── Public API ─────────────────────────────────────────────────────

    def generate_reflection(self) -> List[str]:
        """
        Analyse recent conversations and memories, produce observations,
        deduplicate, and save them as long-term memories.

        Returns:
            List of newly saved reflection strings.
        """
        recent_conv = self.mm.get_recent_conversation(limit=CONVERSATION_WINDOW)
        all_memories = {
            t: self.mm.get_all_memories(memory_type=t, limit=200)
            for t in ("user_fact", "project", "preference", "goal",
                      "event", "relationship", "long_term")
        }

        topic_counts  = self._analyse_topic_frequency(recent_conv, all_memories)
        comm_style    = self._analyse_communication_style(recent_conv)
        existing_text = self._existing_reflection_text()

        observations: List[str] = []
        observations += self._obs_frequent_topics(topic_counts, existing_text)
        observations += self._obs_projects(all_memories["project"], existing_text)
        observations += self._obs_preferences(all_memories["preference"], existing_text)
        observations += self._obs_goals(all_memories["goal"], existing_text)
        observations += self._obs_communication(comm_style, existing_text)
        observations += self._obs_activity(recent_conv, all_memories["event"], existing_text)

        saved: List[str] = []
        for obs in observations:
            obs = obs.strip()
            if obs and not self._is_duplicate(obs, existing_text):
                self.save_reflection(obs)
                saved.append(obs)
                existing_text += " " + obs.lower()

        return saved

    def save_reflection(self, text: str, importance: int = REFLECTION_IMPORTANCE) -> int:
        """Store a reflection as a long-term memory. Returns the memory ID."""
        text = text.strip()
        if not text.startswith(REFLECTION_TAG):
            text = f"{REFLECTION_TAG} {text}"
        return self.mm.save_memory(REFLECTION_TYPE, text, importance=importance)

    def get_recent_reflections(self, limit: int = 5) -> List[Dict]:
        """Return the most recent reflection memories, newest first."""
        all_lt = self.mm.get_all_memories(memory_type=REFLECTION_TYPE, limit=500)
        reflections = [m for m in all_lt if _is_reflection(m["content"])]
        reflections.sort(key=lambda m: m.get("created_at", 0), reverse=True)
        return reflections[:limit]

    # ── Analysis helpers ───────────────────────────────────────────────

    def _analyse_topic_frequency(
        self,
        recent_conv: List[Dict],
        all_memories: Dict[str, List[Dict]],
    ) -> Counter:
        texts: List[str] = []
        for row in recent_conv:
            if row.get("role") == "user":
                texts.append(row.get("content", ""))
        for mems in all_memories.values():
            for m in mems:
                if not _is_reflection(m.get("content", "")):
                    texts.append(m.get("content", ""))

        all_tokens = []
        for text in texts:
            all_tokens.extend(_tokenise(text))
        return Counter(all_tokens)

    def _analyse_communication_style(self, recent_conv: List[Dict]) -> Dict:
        user_rows = [r for r in recent_conv if r.get("role") == "user"]
        if not user_rows:
            return {}
        total_words = sum(len(r.get("content", "").split()) for r in user_rows)
        question_count = sum(
            1 for r in user_rows if "?" in r.get("content", "")
        )
        avg_words = total_words / len(user_rows)
        return {
            "avg_message_length": avg_words,
            "question_rate": question_count / len(user_rows),
            "message_count": len(user_rows),
        }

    def _existing_reflection_text(self) -> str:
        all_lt = self.mm.get_all_memories(memory_type=REFLECTION_TYPE, limit=500)
        return " ".join(
            _strip_tag(m["content"]).lower()
            for m in all_lt
            if _is_reflection(m.get("content", ""))
        )

    def _is_duplicate(self, new_obs: str, existing_text: str) -> bool:
        new_tokens = set(_tokenise(new_obs))
        if not new_tokens:
            return False
        existing_tokens = set(existing_text.split())
        overlap = len(new_tokens & existing_tokens) / len(new_tokens)
        return overlap > 0.75

    # ── Observation generators ─────────────────────────────────────────

    def _obs_frequent_topics(self, counts: Counter, existing: str) -> List[str]:
        top = [
            (kw, cnt) for kw, cnt in counts.most_common(15)
            if cnt >= TOPIC_MIN_COUNT and len(kw) > 3
        ]
        if len(top) < 2:
            return []
        labels = list(dict.fromkeys(_label(kw) for kw, _ in top[:5]))[:4]
        if len(labels) < 2:
            return []
        return [f"The user frequently discusses {_oxford(labels)}."]

    def _obs_projects(self, projects: List[Dict], existing: str) -> List[str]:
        if len(projects) < MIN_PROJECT_MEMORIES:
            return []
        names = [_extract_project_name(m["content"]) for m in projects]
        names = [n for n in names if n][:3]
        if not names:
            return [f"The user has {len(projects)} active project memories stored."]
        return [f"The user is working on {_oxford(names)}."]

    def _obs_preferences(self, prefs: List[Dict], existing: str) -> List[str]:
        if len(prefs) < MIN_PREFERENCE_MEMORIES:
            return []
        objects = [_extract_preference_object(m["content"]) for m in prefs]
        objects = [o for o in objects if o][:4]
        if not objects:
            return []
        return [f"The user has expressed preferences for {_oxford(objects)}."]

    def _obs_goals(self, goals: List[Dict], existing: str) -> List[str]:
        if len(goals) < MIN_GOAL_MEMORIES:
            return []
        actions = [_extract_goal_action(m["content"]) for m in goals]
        actions = [a for a in actions if a][:2]
        if not actions:
            return []
        return [f"The user's stated goal is to {_oxford(actions)}."]

    def _obs_communication(self, style: Dict, existing: str) -> List[str]:
        if not style:
            return []
        obs: List[str] = []
        avg = style.get("avg_message_length", 0)
        if avg > 25:
            obs.append("The user writes in long, detailed messages.")
        elif avg < 8:
            obs.append("The user tends to write short, concise messages.")
        qr = style.get("question_rate", 0)
        if qr > 0.5:
            obs.append("The user frequently asks questions in conversation.")
        return obs

    def _obs_activity(
        self,
        recent_conv: List[Dict],
        events: List[Dict],
        existing: str,
    ) -> List[str]:
        if len(recent_conv) >= CONVERSATION_WINDOW * 0.8:
            return ["The user has been highly active in recent conversations."]
        return []


# ══════════════════════════════════════════════════════════════
# STANDALONE TEST
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    mm = MemoryManager()
    mm.save_memory("project",    "User is building a humanoid robot with servo motors.")
    mm.save_memory("project",    "User is developing a custom LLM called Kira.")
    mm.save_memory("preference", "User loves Python and building things from scratch.")
    mm.save_memory("goal",       "User wants to become a machine learning engineer.")
    mm.save_conversation_turn("I finished the wrist joint today", "That's great progress!")
    mm.save_conversation_turn("I'm training the model now", "Nice, how's the loss?")

    engine = ReflectionEngine(mm)
    results = engine.generate_reflection()
    print("=== Reflection Engine Test ===")
    for r in results:
        print(f" - {r}")
    if not results:
        print("(no new reflections — try adding more memories first)")
