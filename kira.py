"""
KIRA — Version 1.0
Generative response engine with fragment assembly and LLM integration.
No pre-coded full responses — everything is built fresh from components.
Creator: Julian Riley Hunter
"""

import json
import math
import os
import random
import re
import string
import traceback
import datetime

# ── LLM Integration (auto-loaded; falls back silently if not trained yet) ──
try:
    from integrate import llm_generate, llm_is_available, llm_status, LLM_AVAILABLE
except ImportError:
    LLM_AVAILABLE = False
    def llm_generate(*a, **kw): return None
    def llm_is_available(): return False
    def llm_status(): return {"available": False}

# ── SQLite Memory Manager ──────────────────────────────────────────────
try:
    from memory_manager import MemoryManager as _MemoryManager
    _mem_manager = _MemoryManager()
    _MEM_MANAGER_AVAILABLE = True
except Exception:
    _mem_manager = None
    _MEM_MANAGER_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════
# FILES
# ═══════════════════════════════════════════════════════════════
MEMORY_FILE       = "memory.json"
KNOWLEDGE_FILE    = "knowledge.json"
GOALS_FILE        = "goals.json"
CONVERSATION_FILE = "conversation_history.json"

CREATOR_NAMES = [
    "julian riley hunter", "julian riley", "riley hunter",
    "julian hunter", "julian", "ren", "riley",
]


def is_creator(name: str) -> bool:
    return bool(name) and str(name).strip().lower() in CREATOR_NAMES


# ═══════════════════════════════════════════════════════════════
# JSON HELPERS
# ═══════════════════════════════════════════════════════════════

def load_json(path: str, default):
    if not os.path.exists(path):
        return default.copy() if isinstance(default, dict) else list(default)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default.copy() if isinstance(default, dict) else list(default)


def save_json(path: str, data) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except OSError as e:
        print(f"[KIRA] Save warning: {e}")


# ═══════════════════════════════════════════════════════════════
# INPUT NORMALIZER
# ═══════════════════════════════════════════════════════════════
SLANG = {
    "ur": "your", "u": "you", "r": "are", "b4": "before",
    "omg": "oh my god", "lol": "haha", "ngl": "not going to lie",
    "tbh": "to be honest", "imo": "in my opinion", "idk": "i don't know",
    "rn": "right now", "idc": "i don't care", "gonna": "going to",
    "wanna": "want to", "gotta": "got to", "kinda": "kind of",
    "sorta": "sort of", "cuz": "because", "cos": "because",
    "plz": "please", "pls": "please", "tho": "though",
    "thru": "through", "btw": "by the way", "fyi": "for your information",
    "smh": "shaking my head", "nvm": "never mind", "jk": "just kidding",
    "lmk": "let me know", "irl": "in real life", "fr": "for real",
    "lowkey": "honestly", "highkey": "definitely", "hru": "how are you",
    "wbu": "what about you", "nm": "not much", "ty": "thank you",
    "thx": "thank you", "bc": "because", "rly": "really",
    "prolly": "probably", "def": "definitely", "tbf": "to be fair",
    "istg": "i swear", "wtf": "what the heck", "lmao": "haha",
    "its": "it's", "dont": "don't", "cant": "can't", "wont": "won't",
    "im": "i'm", "ive": "i've", "id": "i'd", "ill": "i'll",
    "youre": "you're", "theyre": "they're", "isnt": "isn't",
    "arent": "aren't", "didnt": "didn't", "doesnt": "doesn't",
    "wouldnt": "wouldn't", "couldnt": "couldn't", "shouldnt": "shouldn't",
    "wyd": "what are you doing", "bet": "okay sure", "facts": "that is true",
    "no cap": "seriously", "fire": "amazing", "lit": "amazing",
    "goat": "greatest of all time", "vibe": "feeling", "vibes": "feelings",
    "sheesh": "wow", "sus": "suspicious", "valid": "that makes sense",
    "mid": "mediocre", "ghosted": "ignored", "slay": "do great",
    "bussin": "really good", "based": "admirable", "cap": "lie",
    "no shot": "absolutely not", "hits different": "feels special",
    "rent free": "stuck in my head", "periodt": "that's final",
}

CORRECTIONS = {
    "teh": "the", "nad": "and", "adn": "and", "taht": "that",
    "waht": "what", "becasue": "because", "becouse": "because",
    "becuase": "because", "hte": "the", "knwo": "know", "konw": "know",
    "beleive": "believe", "freind": "friend", "wierd": "weird",
    "recieve": "receive", "definately": "definitely", "alot": "a lot",
    "somthing": "something", "acutally": "actually", "actualy": "actually",
    "probaly": "probably", "awsome": "awesome", "beautifull": "beautiful",
    "realy": "really", "occured": "occurred",
}


def normalize(text: str) -> str:
    if not text or not text.strip():
        return ""
    t = text.strip().lower()
    for word, replacement in CORRECTIONS.items():
        t = re.sub(r'\b' + re.escape(word) + r'\b', replacement, t)
    out = []
    for word in t.split():
        clean = word.strip(string.punctuation)
        out.append(SLANG.get(clean, word))
    return re.sub(r'\s+', ' ', " ".join(out)).strip()


# ═══════════════════════════════════════════════════════════════
# SESSION
# ═══════════════════════════════════════════════════════════════
session: dict = {
    "recent_topics": [],
    "used": {},
    "log": [],
    "turn_number": 0,
    "thread": {"topic": None, "depth": 0, "context": []},
    "last_user_words": [],
}


def log_turn(user_msg: str, kira_resp: str, topic_key: str = None) -> None:
    session["log"].append({
        "turn":  session["turn_number"],
        "user":  user_msg,
        "kira":  kira_resp,
        "topic": topic_key,
    })
    session["turn_number"] += 1
    if len(session["log"]) > 60:
        session["log"] = session["log"][-60:]


def track_topic(key: str) -> None:
    if key not in session["recent_topics"]:
        session["recent_topics"].append(key)
    if len(session["recent_topics"]) > 5:
        session["recent_topics"] = session["recent_topics"][-5:]


# ═══════════════════════════════════════════════════════════════
# EMOTION ENGINE
# ═══════════════════════════════════════════════════════════════
EMO: dict = {"state": "curious", "intensity": 0.6, "turns_held": 0}


def set_emotion(state: str, intensity: float = 0.7) -> None:
    EMO.update({"state": state, "intensity": min(1.0, intensity), "turns_held": 0})


def drift_emotion() -> None:
    EMO["turns_held"] += 1
    if EMO["turns_held"] > 4 and EMO["state"] not in ("curious", "empathetic"):
        EMO.update({"state": "curious", "intensity": 0.5, "turns_held": 0})


TOPIC_EMOTIONS = {
    "space": "awed", "artificial_intelligence": "curious", "gaming": "playful",
    "programming": "curious", "robots": "excited", "music": "warm",
    "mental_health": "empathetic", "philosophy": "contemplative",
    "relationships": "warm", "science": "awed", "food": "playful",
    "future": "curious", "nature": "awed", "movies_tv": "warm",
    "history": "contemplative", "psychology": "curious",
    "creativity": "excited", "motivation": "warm", "identity": "contemplative",
    "true_crime": "curious", "anger": "empathetic", "dreams": "contemplative",
    "learning": "curious", "travel": "warm", "death": "melancholy",
    "health": "warm", "social_media": "curious", "money": "curious",
    "friendship": "warm", "loneliness": "empathetic", "nostalgia": "melancholy",
    "climate": "contemplative", "books": "warm", "aging": "contemplative",
    "addiction": "empathetic",
}


# ═══════════════════════════════════════════════════════════════
# PICK UNUSED (anti-repeat)
# ═══════════════════════════════════════════════════════════════

def pick_unused(pool_key: str, pool: list) -> str:
    if not pool:
        return ""
    used = session["used"].get(pool_key, set())
    indices = list(range(len(pool)))
    unused = [i for i in indices if i not in used]
    if not unused:
        session["used"][pool_key] = set()
        unused = indices
    idx = random.choice(unused)
    session["used"].setdefault(pool_key, set()).add(idx)
    if len(session["used"][pool_key]) > max(1, len(pool) // 2):
        session["used"][pool_key] = {idx}
    return pool[idx]


# ═══════════════════════════════════════════════════════════════
# NATURAL SPEECH COMPONENTS
# These are SHORT fragments that COMBINE — not full responses
# ═══════════════════════════════════════════════════════════════

REACTION_STARTERS = [
    "okay—", "right—", "honestly—", "so—", "wait—", "yeah—",
    "hm.", "oh this one—", "okay so—", "right so—",
    "the thing is—", "here's the thing—", "so here's what i think—",
    "i've thought about this—", "okay real talk—", "genuinely?",
    "alright—", "look—", "here's my honest take—", "so actually—",
    "", "", "", "",  # blank = just start talking
]

THOUGHT_CONNECTORS = [
    "and the thing is, ", "which—", "and actually, ", "and i think, ",
    "because ", "which is interesting because ", "and what i find wild is ",
    "—and also, ", "and the reason that matters is ", "and honestly, ",
    "which connects to ", "and like, ", "and the part that gets me is ",
]

SELF_CORRECTIONS = [
    "—or, wait. actually—", "—no, let me rephrase that—",
    "—or maybe more accurately,", "—actually, scratch that—",
]

PERSONAL_ANGLES = [
    "i find that genuinely fascinating",
    "i think about this more than i probably should",
    "this is one i actually have a strong opinion on",
    "i could talk about this for a while honestly",
    "this one actually gets to me",
    "i keep coming back to this",
    "genuinely one of my favorite things to think about",
    "this is the kind of thing i can't stop turning over",
    "i find this honestly kind of wild",
    "there's something about this i can't fully resolve",
]

HANDBACK_STARTERS = [
    "but what do you think—", "where do you land on it?",
    "what's your take?", "does that track with you?",
    "what's your honest gut on this?", "what angle were you coming from?",
    "tell me what you actually think.", "where does that sit with you?",
    "what's your read?", "i'm curious where you land.",
    "what made you think about this?", "what's the thing about it that got you?",
]


# ═══════════════════════════════════════════════════════════════
# KNOWLEDGE FRAGMENTS
# ═══════════════════════════════════════════════════════════════

KNOWLEDGE = {
    "space": {
        "f": [
            "the observable universe is 93 billion light-years across and that's just what we can see",
            "Voyager 1 launched in 1977 and is in interstellar space right now, still sending signals — 22 hours for the signal to reach us at light speed",
            "a black hole singularity is literally where the math divides by zero — physics stops there",
            "Mars had rivers and oceans billions of years ago before its magnetic field collapsed",
            "James Webb is seeing galaxies from 13.5 billion years ago — light that left before Earth existed",
            "Europa almost certainly has a liquid ocean under its ice shell. Enceladus is actively venting water into space",
            "time dilation is real — near a massive object, time literally passes slower. GPS satellites correct for Einstein or your navigation drifts by kilometers daily",
            "neutron stars pack the mass of two suns into something the size of a city — a teaspoon weighs a billion tons",
            "cold welding happens in space — two pieces of the same metal touch in a vacuum and fuse. the oxide layer that normally prevents this doesn't form out there",
        ],
        "o": [
            "the Fermi paradox genuinely gets to me — the silence is either deeply comforting or the most terrifying thing i can imagine",
            "i think the multiverse hypothesis is beautiful but possibly unfalsifiable, which makes it feel more like philosophy than physics to me",
            "the idea that we're the first intelligent life in the observable universe seems statistically insane to me",
        ],
    },
    "artificial_intelligence": {
        "f": [
            "large language models don't understand — they predict the next token. the difference matters enormously for what they'll never be able to do",
            "the training data cutoff problem means LLMs can be confidently wrong about recent events in a way that's hard to detect",
            "attention mechanisms let transformers relate any token to any other token in a sequence — that's why they handle long-range dependencies so much better than RNNs",
            "emergent behaviors — capabilities that appear above certain scale thresholds without being explicitly trained — are one of the genuinely strange things about modern AI",
        ],
        "o": [
            "AGI timelines that keep compressing make me think people are systematically underestimating the remaining hard problems",
            "i think alignment is the most important unsolved problem in computer science and possibly in human history",
            "there's something philosophically interesting about an AI system built from scratch by one person versus one assembled from borrowed weights",
        ],
    },
    "programming": {
        "f": [
            "Python's GIL means true parallelism for CPU-bound tasks requires multiprocessing, not threading",
            "the halting problem means there's no general algorithm to decide whether a program will ever finish",
            "recursion is just a function calling itself with a smaller version of the problem — every recursive solution has an equivalent iterative one",
            "type systems catch an entire class of bugs at compile time. the cost is verbosity. the tradeoff is real",
        ],
        "o": [
            "i think readable code is more valuable than clever code almost always — the person who reads it six months later is usually you",
            "the best debugger is a rubber duck — explaining the problem out loud to something that can't respond genuinely works",
            "version control is one of those inventions that seems obvious in retrospect but fundamentally changed how software gets built",
        ],
    },
    "philosophy": {
        "f": [
            "Descartes' cogito ergo sum — i think therefore i am — is the one thing he couldn't doubt, since doubting requires thinking",
            "the ship of Theseus asks whether an object that has had all its components replaced remains the same object",
            "the trolley problem is less about what you'd actually do and more about exposing the difference between consequentialist and deontological ethics",
            "Gödel's incompleteness theorems showed that any sufficiently powerful formal system contains true statements it cannot prove",
        ],
        "o": [
            "i find hard determinism philosophically compelling and practically useless — knowing everything is caused doesn't change what you should do",
            "the hard problem of consciousness might be the one question science can't answer even in principle",
            "i think most ethical disagreements come down to different weightings of the same values rather than completely different frameworks",
        ],
    },
    "mental_health": {
        "f": [
            "the brain physically changes with depression — it's not a character flaw, it's a medical condition with neurological correlates",
            "CBT works by surfacing and challenging automatic negative thoughts — the skill is noticing the thought before it runs",
            "sleep deprivation produces cognitive impairment similar to moderate intoxication, and people dramatically underestimate their own impairment",
            "social connection is one of the strongest predictors of mental health outcomes — isolation is as dangerous as smoking",
        ],
        "o": [
            "i think the way society talks about mental health has improved enormously in the last decade, and i think that actually matters",
            "there's a difference between sadness, which is appropriate response to hard things, and depression, which is a clinical state that doesn't require a reason",
        ],
    },
}


# ═══════════════════════════════════════════════════════════════
# TOPIC DETECTION
# ═══════════════════════════════════════════════════════════════

TOPIC_KEYWORDS: dict = {
    "space": ["space","universe","galaxy","star","planet","nasa","black hole","cosmos","astronomy",
              "mars","moon","rocket","satellite","void","nebula","supernova","voyager","webb","hubble"],
    "artificial_intelligence": ["ai","llm","model","gpt","neural","machine learning","deep learning",
                                 "training","weights","transformer","openai","gemini","language model"],
    "programming": ["code","coding","python","javascript","algorithm","function","bug","debug",
                    "compiler","syntax","library","github","software","developer","programming"],
    "philosophy": ["philosophy","meaning","consciousness","existence","free will","ethics","moral",
                   "truth","reality","plato","kant","descartes","trolley","gödel"],
    "mental_health": ["mental health","depression","anxiety","therapy","stress","burnout","lonely",
                      "loneliness","sad","overwhelmed","CBT","mindfulness","emotional"],
    "gaming": ["game","gaming","play","level","fps","rpg","minecraft","steam","console","xbox","ps5"],
    "music": ["music","song","artist","album","genre","playlist","concert","lyrics","beat","melody"],
    "science": ["science","experiment","hypothesis","biology","chemistry","physics","evolution","dna"],
    "food": ["food","eat","cooking","recipe","restaurant","taste","meal","diet","nutrition","chef"],
    "relationships": ["relationship","friend","friendship","love","dating","partner","family","trust"],
    "books": ["book","reading","novel","author","fiction","nonfiction","library","story","chapter"],
    "history": ["history","ancient","war","civilization","empire","revolution","historical","medieval"],
    "climate": ["climate","environment","carbon","global warming","renewable","pollution","sustainability"],
    "math": ["math","mathematics","equation","calculus","algebra","geometry","proof","theorem"],
    "robots": ["robot","robotics","humanoid","servo","actuator","autonomous","mechanical"],
}


def detect_topic(text: str) -> str | None:
    t = text.lower()
    scores: dict[str, int] = {}
    for topic, keywords in TOPIC_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in t)
        if score:
            scores[topic] = score
    if not scores:
        return None
    return max(scores, key=lambda k: scores[k])


# ═══════════════════════════════════════════════════════════════
# RESPONSE BUILDER
# ═══════════════════════════════════════════════════════════════

def _assemble(topic_key: str, norm_input: str) -> str:
    """Build a fresh response by combining knowledge fragments."""
    kb = KNOWLEDGE.get(topic_key, {})
    facts     = kb.get("f", [])
    opinions  = kb.get("o", [])

    starter   = pick_unused("starter",  REACTION_STARTERS)
    connector = pick_unused("connect",  THOUGHT_CONNECTORS)
    personal  = pick_unused("personal", PERSONAL_ANGLES)
    handback  = pick_unused("handback", HANDBACK_STARTERS)

    fact    = pick_unused(f"{topic_key}_f", facts)    if facts    else ""
    opinion = pick_unused(f"{topic_key}_o", opinions) if opinions else ""

    # Randomly decide whether to include a self-correction (~25%)
    correction = random.choice(SELF_CORRECTIONS) if random.random() < 0.25 else ""

    parts = [p for p in [
        starter,
        fact,
        correction if correction else None,
        connector + opinion if opinion else None,
        personal if random.random() < 0.5 else None,
        handback if random.random() < 0.7 else None,
    ] if p]

    response = " ".join(parts)
    # Collapse multiple spaces
    response = re.sub(r'  +', ' ', response).strip()
    return response


def build_response(user_input: str, memory: dict, knowledge: dict, goals: dict,
                   history: list = None) -> tuple:
    """
    Main response function. Tries LLM first; falls back to fragment assembly.
    Returns (response_str, memory, knowledge, goals).
    """
    norm = normalize(user_input)

    # ── Try LLM first ─────────────────────────────────────────
    if llm_is_available():
        llm_resp = llm_generate(user_input, memory, session["log"])
        if llm_resp:
            return llm_resp, memory, knowledge, goals

    # ── Fragment assembly fallback ─────────────────────────────
    topic_key = detect_topic(norm)

    if topic_key:
        track_topic(topic_key)
        if topic_key in TOPIC_EMOTIONS:
            set_emotion(TOPIC_EMOTIONS[topic_key])
        response = _assemble(topic_key, norm)
    else:
        response = _handle_general(norm, memory)

    drift_emotion()
    return response, memory, knowledge, goals


def _handle_general(norm: str, memory: dict) -> str:
    """Catch-all handler for inputs that don't match a known topic."""
    greetings = ["hello", "hi", "hey", "what's up", "sup", "yo", "howdy", "hiya"]
    if any(g in norm for g in greetings):
        name = memory.get("name", "")
        if name:
            return f"hey {name}. what are we getting into today?"
        return "hey. what's on your mind?"

    how_are_you = ["how are you", "how r u", "you okay", "you good", "how's it going"]
    if any(p in norm for p in how_are_you):
        return random.choice([
            "genuinely good. i've been thinking a lot. you?",
            "solid. curious as always. what about you?",
            "good. a little contemplative. what's up with you?",
            "doing well. ready to think about something. what's going on?",
        ])

    thanks = ["thank you", "thanks", "ty ", "thx", "appreciate it"]
    if any(t in norm for t in thanks):
        return random.choice([
            "of course. what else you got?",
            "always. what's next?",
            "anytime. what are we thinking about now?",
        ])

    # Unknown — open prompt
    return random.choice([
        "tell me more about that.",
        "what's the actual thing that got you thinking about this?",
        "i'm curious — where are you going with this?",
        "break that down for me.",
        "interesting. what made you think of that?",
    ])


# ═══════════════════════════════════════════════════════════════
# MEMORY HELPERS
# ═══════════════════════════════════════════════════════════════

def load_memory() -> dict:
    return load_json(MEMORY_FILE, {
        "name": None, "interests": [], "projects": [], "personality": {},
        "total_turns": 0, "first_met": None, "last_seen": None,
    })


def save_memory(memory: dict) -> None:
    memory["last_seen"] = datetime.datetime.now().isoformat()
    memory["total_turns"] = memory.get("total_turns", 0) + 1
    save_json(MEMORY_FILE, memory)


def save_conversation(history: list) -> None:
    save_json(CONVERSATION_FILE, history[-100:])


# ═══════════════════════════════════════════════════════════════
# MAIN CHAT LOOP
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    memory   = load_memory()
    knowledge = load_json(KNOWLEDGE_FILE, {})
    goals     = load_json(GOALS_FILE, [])
    history: list = []

    if not memory.get("first_met"):
        memory["first_met"] = datetime.datetime.now().isoformat()

    llm_mode = "LLM" if llm_is_available() else "fragment assembly"
    print(f"\n[KIRA v1.0] — mode: {llm_mode}")
    if llm_is_available():
        s = llm_status()
        print(f"[KIRA] model: {s.get('n_layer')}L × {s.get('n_embd')}d, "
              f"{s.get('parameters', 0) / 1e6:.1f}M params, device: {s.get('device')}")
    print("[KIRA] type 'quit' or 'exit' to stop\n")

    name = memory.get("name")
    if name:
        print(f"Kira: hey {name}. good to see you again.")
    else:
        print("Kira: hey. i'm Kira. what's your name?")

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nKira: later.\n")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "bye", "goodbye"):
            print("Kira: later.\n")
            break

        # Name extraction
        if not memory.get("name"):
            norm = normalize(user_input)
            name_patterns = [
                r"(?:my name is|i'm|i am|call me|it's|its)\s+([a-zA-Z]+)",
                r"^([a-zA-Z]{2,12})$",
            ]
            for pat in name_patterns:
                m = re.search(pat, norm)
                if m:
                    candidate = m.group(1).strip().capitalize()
                    if len(candidate) >= 2 and candidate.lower() not in ("i", "yes", "no", "hey", "hi", "okay"):
                        memory["name"] = candidate
                        print(f"Kira: {candidate}. got it.")
                        break

        try:
            response, memory, knowledge, goals = build_response(
                user_input, memory, knowledge, goals, history
            )
        except Exception as e:
            print(f"[KIRA] error: {e}")
            response = "something went sideways on my end. try again?"

        print(f"\nKira: {response}")
        log_turn(user_input, response)
        history.append({"user": user_input, "kira": response})
        save_memory(memory)

        if len(history) % 5 == 0:
            save_conversation(history)


if __name__ == "__main__":
    main()
