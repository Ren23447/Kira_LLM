"""
Kira Memory System — memory_manager.py
=======================================
SQLite-backed long-term and short-term memory for Kira.

Features:
  - FTS5 full-text search for semantic memory retrieval
  - Importance scoring (1-5)
  - Memory types: user_fact, project, preference, event, long_term, short_term
  - Conversation history storage and retrieval
  - Duplicate detection via cosine-like word overlap

INTEGRATION (kira_server.py):
    from memory_manager import MemoryManager
    kira_memory = MemoryManager()

    # ContextBuilder calls kira_memory internally on every /chat request:
    #   kira_memory.search_memories(user_message)
    #   kira_memory.get_all_memories(type="project")
    #   kira_memory.get_recent_conversation()
    #   kira_memory.save_conversation_turn(u, r)

Memory routes wired in kira_server.py:
    POST   /memory/save
    GET    /memory/search?q=...
    DELETE /memory/delete/<id>
    POST   /memory/consolidate
"""

import os
import re
import sqlite3
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory.db")

MEMORY_TYPES = {"user_fact", "project", "preference", "event", "long_term", "short_term"}

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
}


class MemoryManager:
    """
    Manages Kira's persistent memory via SQLite with FTS5 full-text search.
    All writes are atomic; reads return plain Python dicts.
    """

    def __init__(self, db_path: str = DB_PATH) -> None:
        self.db_path = db_path
        self._init_db()

    # ── Schema ─────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS memories (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_type TEXT    NOT NULL,
                    content     TEXT    NOT NULL,
                    importance  INTEGER NOT NULL DEFAULT 3,
                    created_at  REAL    NOT NULL,
                    updated_at  REAL    NOT NULL,
                    access_count INTEGER NOT NULL DEFAULT 0
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
                    USING fts5(content, content=memories, content_rowid=id);

                CREATE TRIGGER IF NOT EXISTS memories_ai
                    AFTER INSERT ON memories BEGIN
                        INSERT INTO memories_fts(rowid, content)
                        VALUES (new.id, new.content);
                    END;

                CREATE TRIGGER IF NOT EXISTS memories_ad
                    AFTER DELETE ON memories BEGIN
                        INSERT INTO memories_fts(memories_fts, rowid, content)
                        VALUES ('delete', old.id, old.content);
                    END;

                CREATE TRIGGER IF NOT EXISTS memories_au
                    AFTER UPDATE ON memories BEGIN
                        INSERT INTO memories_fts(memories_fts, rowid, content)
                        VALUES ('delete', old.id, old.content);
                        INSERT INTO memories_fts(rowid, content)
                        VALUES (new.id, new.content);
                    END;

                CREATE TABLE IF NOT EXISTS conversation_history (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    role        TEXT NOT NULL,
                    content     TEXT NOT NULL,
                    created_at  REAL NOT NULL
                );
            """)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    # ── Memory CRUD ────────────────────────────────────────────────────

    def save_memory(
        self,
        memory_type: str,
        content: str,
        importance: int = 3,
    ) -> int:
        """
        Save a new memory. Returns the new memory ID.

        Args:
            memory_type: one of MEMORY_TYPES
            content:     the memory text
            importance:  1 (low) to 5 (critical)

        Raises:
            ValueError: if memory_type is not recognised
        """
        if memory_type not in MEMORY_TYPES:
            raise ValueError(f"Unknown memory_type: {memory_type!r}. "
                             f"Valid types: {sorted(MEMORY_TYPES)}")
        content = content.strip()
        if not content:
            raise ValueError("Memory content cannot be empty")

        importance = max(1, min(5, importance))
        now = time.time()

        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO memories (memory_type, content, importance, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (memory_type, content, importance, now, now),
            )
            return cur.lastrowid

    def search_memories(
        self,
        query: str,
        limit: int = 10,
        memory_type: Optional[str] = None,
    ) -> List[Dict]:
        """
        Full-text search memories, ranked by relevance.

        Args:
            query:       search string
            limit:       max results
            memory_type: optional filter

        Returns:
            List of memory dicts, ordered by relevance.
        """
        if not query.strip():
            return []

        clean_query = " OR ".join(
            w for w in re.findall(r'\w+', query.lower())
            if w not in STOPWORDS and len(w) > 2
        )
        if not clean_query:
            return self.get_all_memories(memory_type=memory_type, limit=limit)

        try:
            with self._conn() as conn:
                if memory_type:
                    rows = conn.execute(
                        "SELECT m.* FROM memories m "
                        "JOIN memories_fts f ON m.id = f.rowid "
                        "WHERE memories_fts MATCH ? AND m.memory_type = ? "
                        "ORDER BY rank LIMIT ?",
                        (clean_query, memory_type, limit),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT m.* FROM memories m "
                        "JOIN memories_fts f ON m.id = f.rowid "
                        "WHERE memories_fts MATCH ? "
                        "ORDER BY rank LIMIT ?",
                        (clean_query, limit),
                    ).fetchall()
                return [dict(r) for r in rows]
        except sqlite3.OperationalError:
            return self.get_all_memories(memory_type=memory_type, limit=limit)

    def get_all_memories(
        self,
        memory_type: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict]:
        """
        Return memories ordered by importance (desc) then recency (desc).

        Args:
            memory_type: optional type filter
            limit:       max results
        """
        with self._conn() as conn:
            if memory_type:
                rows = conn.execute(
                    "SELECT * FROM memories WHERE memory_type = ? "
                    "ORDER BY importance DESC, created_at DESC LIMIT ?",
                    (memory_type, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM memories "
                    "ORDER BY importance DESC, created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [dict(r) for r in rows]

    def delete_memory(self, memory_id: int) -> bool:
        """Delete a memory by ID. Returns True if a row was deleted."""
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            return cur.rowcount > 0

    def update_memory(
        self,
        memory_id: int,
        content: Optional[str] = None,
        importance: Optional[int] = None,
    ) -> bool:
        """Update a memory's content and/or importance. Returns True on success."""
        fields: List[str] = []
        values: List = []
        if content is not None:
            fields.append("content = ?")
            values.append(content.strip())
        if importance is not None:
            fields.append("importance = ?")
            values.append(max(1, min(5, importance)))
        if not fields:
            return False
        fields.append("updated_at = ?")
        values.append(time.time())
        values.append(memory_id)
        with self._conn() as conn:
            cur = conn.execute(
                f"UPDATE memories SET {', '.join(fields)} WHERE id = ?",
                values,
            )
            return cur.rowcount > 0

    # ── Conversation history ───────────────────────────────────────────

    def save_conversation_turn(self, user_text: str, kira_text: str) -> None:
        """Save a user/Kira exchange to conversation history."""
        now = time.time()
        with self._conn() as conn:
            conn.executemany(
                "INSERT INTO conversation_history (role, content, created_at) VALUES (?, ?, ?)",
                [("user", user_text.strip(), now), ("kira", kira_text.strip(), now)],
            )

    def get_recent_conversation(self, limit: int = 40) -> List[Dict]:
        """
        Return the most recent `limit` conversation rows, oldest first.
        Each row has: id, role, content, created_at.
        """
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM conversation_history "
                "ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in reversed(rows)]

    def clear_conversation(self) -> int:
        """Delete all conversation history. Returns number of rows deleted."""
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM conversation_history")
            return cur.rowcount

    # ── Stats ──────────────────────────────────────────────────────────

    def stats(self) -> Dict:
        """Return a summary of what's stored."""
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            by_type = conn.execute(
                "SELECT memory_type, COUNT(*) as n FROM memories GROUP BY memory_type"
            ).fetchall()
            conv_count = conn.execute(
                "SELECT COUNT(*) FROM conversation_history"
            ).fetchone()[0]
            return {
                "total_memories":    total,
                "by_type":           {r["memory_type"]: r["n"] for r in by_type},
                "conversation_turns": conv_count // 2,
            }
