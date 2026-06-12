"""
Kira LLM — Web Server
Flask REST API serving the Kira chat interface.

Start with:
    python kira_server.py
    # or with a custom port:
    PORT=5001 python kira_server.py

Endpoints:
    POST /chat                 — send a message, get a response
    GET  /status               — server and model status
    GET  /memory/search?q=...  — search memories
    POST /memory/save          — save a memory manually
    DELETE /memory/delete/<id> — delete a memory
    POST /memory/consolidate   — merge duplicate memories
    GET  /profile              — full user profile JSON
    GET  /profile/summary      — plain-text profile summary
    POST /profile/rebuild      — force profile rebuild
"""

import os
import sys
import time
from typing import Any, Dict, Optional, Tuple

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# ── Memory and personality subsystems ─────────────────────────
from memory_manager      import MemoryManager
from memory_extractor    import MemoryExtractor
from memory_consolidation import MemoryConsolidator
from personality_engine  import PersonalityEngine
from emotional_state     import EmotionEngine
from reflection_engine   import ReflectionEngine
from learning_engine     import LearningEngine
from context_builder     import ContextBuilder
from kira                import build_response, load_memory, save_memory, normalize

# ── LLM (optional — falls back gracefully if not trained) ─────
try:
    from integrate import llm_generate, llm_is_available, llm_status, LLM_AVAILABLE
except ImportError:
    LLM_AVAILABLE = False
    def llm_generate(*a, **kw): return None
    def llm_is_available(): return False
    def llm_status(): return {"available": False}

# ═══════════════════════════════════════════════════════════════
# INIT
# ═══════════════════════════════════════════════════════════════

app = Flask(__name__, static_folder=".")
CORS(app)

kira_memory      = MemoryManager()
kira_extractor   = MemoryExtractor(kira_memory)
kira_consolidator = MemoryConsolidator(kira_memory)
kira_personality = PersonalityEngine()
kira_emotions    = EmotionEngine()
kira_reflector   = ReflectionEngine(kira_memory)
kira_learner     = LearningEngine(kira_memory, reflection_engine=kira_reflector)
context_builder  = ContextBuilder(
    memory_manager     = kira_memory,
    personality_engine = kira_personality,
    emotion_engine     = kira_emotions,
    reflection_engine  = kira_reflector,
    learning_engine    = kira_learner,
)

# Session-level state
_session_memory: Dict[str, Any] = load_memory()
_knowledge:  Dict = {}
_goals:      list = []
_turn_count: int  = 0


# ═══════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════

@app.route("/")
def index():
    """Serve the chat interface."""
    return send_from_directory(".", "kira_chat.html")


@app.route("/chat", methods=["POST"])
def chat():
    """
    Main chat endpoint.
    Body: {"message": "hello"}
    Returns: {"response": "...", "mode": "llm"|"fragment"}
    """
    global _session_memory, _knowledge, _goals, _turn_count

    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or "").strip()

    if not user_message:
        return jsonify({"error": "message is required"}), 400

    # ── Extract memories from message ──────────────────────────
    kira_extractor.extract_and_save(user_message)

    # ── Update emotional state ─────────────────────────────────
    kira_emotions.update_from_message(user_message)

    # ── Generate response ──────────────────────────────────────
    mode = "fragment"
    response = ""

    if llm_is_available():
        # Build full structured prompt and feed to LLM
        full_prompt = context_builder.build_context(user_message)
        resp = llm_generate(user_message, _session_memory, [])
        if resp:
            response = resp
            mode = "llm"

    if not response:
        response, _session_memory, _knowledge, _goals = build_response(
            user_message, _session_memory, _knowledge, _goals
        )

    # ── Save conversation turn ─────────────────────────────────
    kira_memory.save_conversation_turn(user_message, response)
    save_memory(_session_memory)

    # ── Periodic tasks ─────────────────────────────────────────
    _turn_count += 1
    if _turn_count % 5 == 0:
        kira_learner.update_user_profile()
    if _turn_count % 20 == 0:
        kira_reflector.generate_reflection()
    if _turn_count % 50 == 0:
        kira_consolidator.consolidate()

    return jsonify({
        "response": response,
        "mode":     mode,
        "turn":     _turn_count,
    })


@app.route("/status", methods=["GET"])
def status():
    """Return server and model status."""
    llm_info = llm_status()
    mem_stats = kira_memory.stats()
    return jsonify({
        "status":    "ok",
        "llm":       llm_info,
        "memory":    mem_stats,
        "version":   "1.0",
    })


# ── Memory routes ─────────────────────────────────────────────

@app.route("/memory/search", methods=["GET"])
def memory_search():
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"error": "q parameter is required"}), 400
    results = kira_memory.search_memories(q, limit=10)
    return jsonify({"results": results, "count": len(results)})


@app.route("/memory/save", methods=["POST"])
def memory_save():
    data = request.get_json(silent=True) or {}
    memory_type = data.get("memory_type", "long_term")
    content     = (data.get("content") or "").strip()
    importance  = int(data.get("importance", 3))
    if not content:
        return jsonify({"error": "content is required"}), 400
    try:
        mem_id = kira_memory.save_memory(memory_type, content, importance)
        return jsonify({"id": mem_id, "ok": True})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/memory/delete/<int:mem_id>", methods=["DELETE"])
def memory_delete(mem_id: int):
    deleted = kira_memory.delete_memory(mem_id)
    return jsonify({"ok": deleted, "id": mem_id})


@app.route("/memory/consolidate", methods=["POST"])
def memory_consolidate():
    result = kira_consolidator.consolidate()
    return jsonify(result)


# ── Profile routes ────────────────────────────────────────────

@app.route("/profile", methods=["GET"])
def profile():
    return jsonify(kira_learner.profile)


@app.route("/profile/summary", methods=["GET"])
def profile_summary():
    return jsonify({"summary": kira_learner.get_summary()})


@app.route("/profile/rebuild", methods=["POST"])
def profile_rebuild():
    kira_learner.update_user_profile()
    return jsonify({"ok": True, "summary": kira_learner.get_summary()})


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    port = int(os.environ.get("KIRA_PORT", os.environ.get("PORT", 5000)))
    debug = os.environ.get("FLASK_ENV", "development") == "development"

    llm_info = llm_status()
    print(f"\n[KIRA SERVER] starting on port {port}")
    print(f"[KIRA SERVER] LLM: {'ACTIVE' if llm_info.get('available') else 'fallback (fragment assembly)'}")
    if llm_info.get("available"):
        print(f"[KIRA SERVER] model: {llm_info.get('parameters', 0) / 1e6:.1f}M params, "
              f"device: {llm_info.get('device')}")
    print(f"[KIRA SERVER] chat UI: http://localhost:{port}/\n")

    app.run(host="0.0.0.0", port=port, debug=debug)
