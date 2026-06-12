"""
Kira Memory Consolidation — memory_consolidation.py
====================================================
Merges near-duplicate memories to keep the database lean and relevant.
Run periodically (e.g. every 20 turns) or on demand via POST /memory/consolidate.

100% local — Python stdlib only. No APIs. No cloud.

INTEGRATION INTO kira_server.py:
    from memory_consolidation import MemoryConsolidator
    consolidator = MemoryConsolidator(kira_memory)

    # In the /memory/consolidate route:
    result = consolidator.consolidate()

STANDALONE TEST:
    python memory_consolidation.py
"""

import re
from typing import Dict, List, Optional, Set, Tuple

from memory_manager import MemoryManager

# ── Tunables ──────────────────────────────────────────────────
DUPLICATE_THRESHOLD  = 0.80   # Jaccard overlap above which two memories are merged
MIN_CONTENT_LENGTH   = 10     # Memories shorter than this are candidates for pruning
MAX_MEMORIES_PER_TYPE = 50    # Soft cap; consolidation trims toward this

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "i", "me", "my", "you", "your", "he", "she", "it", "they",
    "user", "kira",
}


def _tokenise(text: str) -> Set[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9'_-]{1,}", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class MemoryConsolidator:
    """
    Identifies and merges near-duplicate memories within each memory type.
    Keeps the highest-importance version and increments its importance slightly
    each time a duplicate is absorbed.
    """

    def __init__(self, memory_manager: MemoryManager) -> None:
        self.mm = memory_manager

    def consolidate(self, memory_type: Optional[str] = None) -> Dict:
        """
        Run consolidation across all (or one) memory type.

        Args:
            memory_type: limit to one type, or None for all types

        Returns:
            Dict with keys: merged (int), pruned (int), remaining (int)
        """
        types_to_process = (
            [memory_type] if memory_type
            else ["user_fact", "project", "preference", "event", "long_term", "short_term"]
        )

        total_merged = 0
        total_pruned = 0

        for mtype in types_to_process:
            merged, pruned = self._consolidate_type(mtype)
            total_merged  += merged
            total_pruned  += pruned

        remaining = len(self.mm.get_all_memories(limit=10_000))
        return {
            "merged":    total_merged,
            "pruned":    total_pruned,
            "remaining": remaining,
        }

    def _consolidate_type(self, memory_type: str) -> Tuple[int, int]:
        """Consolidate one memory type. Returns (merged_count, pruned_count)."""
        memories = self.mm.get_all_memories(memory_type=memory_type, limit=10_000)
        if len(memories) < 2:
            return 0, 0

        merged = 0
        pruned = 0
        deleted_ids: Set[int] = set()

        # Pre-tokenise all memories
        token_sets = {
            m["id"]: _tokenise(m["content"])
            for m in memories
        }

        for i, mem_a in enumerate(memories):
            if mem_a["id"] in deleted_ids:
                continue
            if len(mem_a["content"].strip()) < MIN_CONTENT_LENGTH:
                self.mm.delete_memory(mem_a["id"])
                deleted_ids.add(mem_a["id"])
                pruned += 1
                continue

            for mem_b in memories[i + 1:]:
                if mem_b["id"] in deleted_ids or mem_a["id"] in deleted_ids:
                    continue

                similarity = _jaccard(token_sets[mem_a["id"]], token_sets[mem_b["id"]])
                if similarity < DUPLICATE_THRESHOLD:
                    continue

                # Keep the higher-importance memory, absorb the other
                keep, drop = (
                    (mem_a, mem_b)
                    if mem_a["importance"] >= mem_b["importance"]
                    else (mem_b, mem_a)
                )

                # Bump importance slightly (max 5)
                new_importance = min(5, keep["importance"] + 1)
                self.mm.update_memory(keep["id"], importance=new_importance)
                self.mm.delete_memory(drop["id"])
                deleted_ids.add(drop["id"])
                merged += 1

        return merged, pruned

    def report(self) -> str:
        """Return a human-readable consolidation report."""
        result = self.consolidate()
        lines = [
            "=== Memory Consolidation Report ===",
            f"Merged duplicates: {result['merged']}",
            f"Pruned short entries: {result['pruned']}",
            f"Remaining memories: {result['remaining']}",
        ]
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
# STANDALONE TEST
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    mm = MemoryManager()
    # Insert some near-duplicates
    mm.save_memory("preference", "User loves Python and building things from scratch.")
    mm.save_memory("preference", "User loves Python and making things from scratch.")
    mm.save_memory("project",    "User is building a humanoid robot with servo motors.")
    mm.save_memory("project",    "User is building a humanoid robot using servo motors.")

    consolidator = MemoryConsolidator(mm)
    print(consolidator.report())
