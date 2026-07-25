from __future__ import annotations
# llm_script.py  — prompt-first Mouthpiece with natural-dialogue rerank (LAZY LOAD)
# Lazy singleton loader; no GPU/model allocation at import time
# Env > YAML > defaults; respects LLM_SPEAKER=0
# Exposes .tokenizer via `tok` for fused-bias processors
# Callable: llm_fn(prompt, generate_kwargs={...})
# Phase-7 language pass:
#   • Config-driven prompting: phase, role, round, recent quotes, explicit intent cue
#   • Normalize Worker→villager so villagers get consistent conditioning
#   • Enforce 1–3 sentences, keep ? and !, avoid newlines, forbid “night” in day phases
#   • Expose tok and llm_fn_from_env; raise max_new_tokens via YAML
#   • Gentle retry based on hygiene if constraints violated
#   • Single natural SAFE_FALLBACK sentence

import os, re, yaml, torch, hashlib, random
from typing import List, Dict, Optional, Any, Tuple, TYPE_CHECKING
from dataclasses import dataclass

# HF imports remain available for the HF path; harmless even if using OpenAI
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# OpenAI optional import (lazy-checked at runtime)
_OPENAI_AVAILABLE = False
try:
    from openai import OpenAI  # type: ignore
    _OPENAI_AVAILABLE = True
except Exception:
    pass

if TYPE_CHECKING:
    from agent import BaseAgent  # typing only

# ── Config loading
def _load_config(path: str = "config.yaml") -> dict:
    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        data = {}
    return data

CFG = _load_config()
PROMPTING = CFG.get("prompting", {}) or {}

# --------- OS ENV SHIM HELPERS ----------
def _env_bool(key: str, default: bool) -> bool:
    v = os.getenv(key)
    if v is None: return default
    return str(v).strip().lower() in ("1","true","yes","y","on")

def _env_str(key: str, default: str) -> str:
    v = os.getenv(key)
    return v if v is not None else default

def _env_float(key: str, default: float) -> float:
    v = os.getenv(key)
    if v is None: return default
    try:
        return float(v)
    except Exception:
        return default

def _env_int(key: str, default: int) -> int:
    v = os.getenv(key)
    if v is None: return default
    try:
        return int(v)
    except Exception:
        return default
# ----------------------------------------

# ─────────────────────────────────────────────────────────────────────────────
# Shared text normalizers and hidden-info checks
# ─────────────────────────────────────────────────────────────────────────────
_CONTRACTION_MAP = {
    "don’t":"don't","can’t":"can't","won’t":"won't","it’s":"it's","i’m":"i'm","i’ve":"i've",
    "you’re":"you're","they’re":"they're","we’re":"we're","that’s":"that's","there’s":"there's",
    "could’ve":"could've","should’ve":"should've","would’ve":"would've","didn’t":"didn't",
    "doesn’t":"doesn't","isn’t":"isn't","wasn’t":"wasn't","weren’t":"weren't","ain’t":"ain't",
}
def normalize_contractions(text: str) -> str:
    if not text: return ""
    t = text.replace("“","\"").replace("”","\"").replace("’","'")
    low = t.lower()
    for k, v in _CONTRACTION_MAP.items():
        if k in low:
            t = re.sub(re.escape(k), v, t, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", t).strip()

_PARENTH_RE = re.compile(r"\s*\([^)]*\)\s*")
def strip_parentheticals(text: str) -> str:
    if not text: return ""
    return _PARENTH_RE.sub(" ", text).strip()

_HI_UNIVERSAL = [
    re.compile(r"\bwe\s+(wolves|are\s+wolves|killed)\b", re.IGNORECASE),
    re.compile(r"\bi\s+killed\b", re.IGNORECASE),
]
_HI_SEER = re.compile(r"\bseer\s+(vision|result)\b", re.IGNORECASE)
_HI_VILLAGER_KNOW = re.compile(r"\bi\s+know\b.*\b(is|are)\b.*\b(werewolf|wolf)\b", re.IGNORECASE)

def get_hidden_info_patterns() -> Dict[str, Any]:
    return {"universal": _HI_UNIVERSAL, "seer": _HI_SEER, "villager_know": _HI_VILLAGER_KNOW}

def leaks_hidden_info(text: str, role: str, phase: str, *, allow_seer: bool = False) -> bool:
    if not text: return False
    t = text.strip()
    for rx in _HI_UNIVERSAL:
        if rx.search(t): return True
    if not allow_seer and _HI_SEER.search(t):
        return True
    rlow = (role or "").lower()
    if rlow.startswith("villag") or rlow.startswith("work"):
        if _HI_VILLAGER_KNOW.search(t):
            return True
    return False

# ── Config values (env → YAML → defaults)
MODEL_ID_DEFAULT = CFG.get("LLM_MODEL_ID", "o4-mini")
DEVICE_CFG       = str(CFG.get("LLM_DEVICE", "")).strip().lower()
# Provider precedence: env LLM_PROVIDER → config LLM_PROVIDER / llm.provider → "hf".
# (Previously this ignored config, so the speaker could default to HF while the
# judge, which reads config, used OpenAI — an inconsistent split backend.)
_PROVIDER_CFG_DEFAULT = str(
    CFG.get("LLM_PROVIDER", (CFG.get("llm", {}) or {}).get("provider", "hf"))
).strip().lower()
PROVIDER_DEFAULT = _env_str("LLM_PROVIDER", _PROVIDER_CFG_DEFAULT).strip().lower()

# ── Hygiene helpers
BAD_QUOTES = "“”\"'«»"
STOP_SEQS = ["\nAgent_", "\nSystem:", "\nNarrator:", "\rAgent_", "\rSystem:", "\rNarrator:"]

# Role handling and normalization
ALLOWED_ROLES_CANON = {"werewolf", "villager", "seer"}
ROLE_SYNONYMS = {"worker": "villager"}
ROLE_BLOCKLIST = ["Engineer","engineer","Scientist","scientist","Detective","detective","Doctor","doctor","Guard","guard",
                  "Scientist.","Engineer.","Detective.","Doctor.","Guard."]

def _normalize_role(role: str) -> str:
    r = (role or "").strip().lower()
    if r in ROLE_SYNONYMS: r = ROLE_SYNONYMS[r]
    return r if r in ALLOWED_ROLES_CANON else r

META_BANS = ["as an ai", "language model", "system prompt", "alignment", "policy",
             "villager_", "seer_", "werewolf_"]
def _bad_words_ids(tokenizer) -> List[List[int]]:
    try:
        toks = tokenizer(META_BANS, add_special_tokens=False)["input_ids"]
        return [ids for ids in toks if len(ids) > 0]
    except Exception:
        return []

def _is_meta_like(text: str) -> bool:
    t = (text or "").strip()
    if not t: return True
    low = t.lower()
    if any(p in low for p in ("third person","punctuation","grammar","reply","provide","follow","instruction","rule")):
        return True
    if len(t.split()) < 2:
        return True
    return False

BAD_META_FRAGMENTS = ["as an ai","language model","cannot","policy","alignment spec","sorry, i","i cannot","i'm unable","i am unable"]
def _postfilter_meta(txt: str) -> str:
    t = (txt or "").strip()
    if not t: return ""
    low = t.lower()
    if any(b in low for b in BAD_META_FRAGMENTS):
        cut = t.find(".")
        if cut != -1:
            t = t[:cut+1].strip()
    return t

_SANITIZE_PATTERNS = [
    r"\b(as\s+an\s+ai|as\s+an\s+language\s+model)\b.*?$",
    r"\b(language\s+model|large\s+language\s+model|llm)\b",
    r"\b(system\s+prompt|prompt|system\s+message|meta[-\s]*instruction)\b",
    r"\b(tokens?|logprobs?|temperature|top[-\s]*p|sampling)\b",
    r"\bpolicy\b", r"\balignment\b", r"\brefuse\b",
]
_SANITIZE_RE = re.compile("|".join(_SANITIZE_PATTERNS), flags=re.IGNORECASE)

def _sanitize(text: str) -> str:
    if not text: return ""
    cleaned = _SANITIZE_RE.sub("", text)
    cleaned = cleaned.strip(BAD_QUOTES + " ").replace("  ", " ")
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" -–—")
    cleaned = re.sub(r"[,:;]\s*$", ".", cleaned).strip()
    return cleaned

def _recent_discussion_block(agent, max_lines: int = 6) -> str:
    if not getattr(agent, "message_memory", None):
        return "- (no recent messages heard)"
    lines = []
    for n, m in list(agent.message_memory)[-max_lines:]:
        m = (m or "").strip()
        if not m: continue
        lines.append(f"- {n}: {m}")
    return "\n".join(lines) if lines else "- (no recent messages heard)"

def _soft_prefix_for_target(named_target: Optional[str]) -> str:
    return f"I think {named_target} " if named_target else ""

try:
    from speaker_llm import early_stop_text as _early_stop_unified  # type: ignore
except Exception:
    _early_stop_unified = None

def _early_stop(generated: str) -> str:
    if callable(_early_stop_unified):
        return _early_stop_unified(generated)
    cut_pos = len(generated)
    for s in STOP_SEQS:
        i = generated.find(s)
        if i != -1:
            cut_pos = min(cut_pos, i)
    return generated[:cut_pos]

SAFE_FALLBACK = "I need a moment to think."

def _one_line(text: str) -> str:
    if not text:
        return SAFE_FALLBACK
    for raw in text.splitlines():
        s = raw.strip()
        if not s: continue
        s = s.replace("“","").replace("”","").replace("’","'")
        s = s.lstrip(" -")
        if s: return s[:240]
    return SAFE_FALLBACK

_ROLE_WORD_RE = re.compile(r"\b(villager|worker|seer|werewolf)s?\b[_\-]?\w*", re.IGNORECASE)
def _sanitize_roles(text: str) -> str:
    if not text:
        return text
    text = _ROLE_WORD_RE.sub("someone", text)
    for w in ROLE_BLOCKLIST:
        text = text.replace(w, "someone")
    return re.sub(r"\s{2,}", " ", text).strip()

_SENT_END_RE = re.compile(r"([.!?])")
_DAY_NIGHT_BANS = re.compile(r"\b(night|tonight|overnight|after dark)\b", re.IGNORECASE)

def _split_sentences_keep_punct(s: str) -> List[str]:
    s = s.strip()
    if not s:
        return []
    parts, cur = [], ""
    for ch in s:
        cur += ch
        if ch in ".!?":
            piece = cur.strip()
            if piece:
                parts.append(piece)
            cur = ""
    tail = cur.strip()
    if tail:
        parts.append(tail)
    return [p.strip() for p in parts if p.strip()]

def _enforce_utterance_constraints(text: str, *, phase: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    t = (text or "").replace("\n", " ").replace("\r", " ").strip()
    if not t:
        return SAFE_FALLBACK, {"redo": False, "violated": True}
    violated_night = False
    if phase and str(phase).upper().startswith("DAY"):
        if _DAY_NIGHT_BANS.search(t):
            violated_night = True
            t = _DAY_NIGHT_BANS.sub("later", t).strip()
    sents = _split_sentences_keep_punct(t)
    if not sents:
        return SAFE_FALLBACK, {"redo": False, "violated": True}
    if len(sents) > 3:
        sents = sents[:3]
    t2 = " ".join(sents).strip()
    if not t2.endswith((".", "!", "?")):
        t2 = t2 + "."
    violated = violated_night
    return t2, {"redo": False, "violated": violated}

_INTENT_LEX: Dict[str, List[str]] = {
    "accuse": ["accuse", "suspicious", "suspect"],
    "defend": ["defend", "not suspicious", "seems town", "seems innocent"],
    "vote":   ["vote", "voting", "lynch"],
    "probe":  ["why", "how", "explain", "clarify", "what made you"],
    "talk":   [],
}
def _mentions_target(text: str, target: Optional[str]) -> bool:
    if not text or not target: return False
    t = target.strip()
    return re.search(rf"\b{re.escape(t)}\b", text) is not None

def _intent_feels_right(text: str, intent: Optional[str]) -> bool:
    if not text or not intent: return True
    lex = _INTENT_LEX.get(intent, [])
    low = text.lower()
    return any(w in low for w in lex) if lex else True

def _enforce_intent_target(text: str, intent: Optional[str], target: Optional[str]) -> Tuple[bool, str]:
    ok = True
    if target and not _mentions_target(text, target):
        ok = False
    if intent and not _intent_feels_right(text, intent):
        ok = False
    return ok, text

# Small helper to pick a stable-but-varied variant without RNG dependence
def _pick_variant(key: str, options: List[str]) -> str:
    if not options:
        return ""
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    idx = int(h[:8], 16) % len(options)
    return options[idx]

def _template_fallback(intent: Optional[str], target: Optional[str]) -> str:
    t = target or "someone"
    key = f"{intent}|{t}"
    if intent == "accuse":
        opts = [
            f"{t} keeps dodging a point from earlier, which makes me uneasy.",
            f"{t}'s last claim contradicts what we heard a moment ago.",
            f"{t} feels off after that quick pivot in their story.",
        ]
        return _pick_variant(key, opts)
    if intent == "defend":
        opts = [
            f"{t} has been consistent so far, which looks town to me.",
            f"{t}'s explanation earlier sounded reasonable.",
            f"{t} pushed the discussion forward when it stalled.",
        ]
        return _pick_variant(key, opts)
    if intent == "vote":
        opts = [
            f"I vote for {t} because their story conflicts with what was just said.",
            f"I vote for {t} after that inconsistent reply.",
            f"I vote for {t} since their reasoning keeps shifting.",
        ]
        return _pick_variant(key, opts)
    if intent == "probe":
        opts = [
            f"{t}, what changed between your last two statements?",
            f"{t}, which part of your earlier claim are you standing by now?",
            f"{t}, can you square your timeline with what we heard before?",
        ]
        return _pick_variant(key, opts)
    opts = [
        f"{t} stands out to me after that exchange.",
        f"{t} is on my radar because of the last turn.",
        f"{t} caught my attention with that response.",
    ]
    return _pick_variant(key, opts)

_ALIVE_NAME_RE = re.compile(r"\bAgent_\d+\b")
_GENERIC_AGENT_WORD_RE = re.compile(r"\bAgent\b(?!_\d)\b", re.IGNORECASE)
_GENERIC_AGENT_POSSESSIVE_RE = re.compile(r"\bAgent's\b(?!_\d)", re.IGNORECASE)
_GENERIC_AGENT_WITH_PUNCT_RE = re.compile(r"\bAgent\b(?=[,.:;!? ])(?!_\d)", re.IGNORECASE)

def _alive_names(agent) -> List[str]:
    try:
        view = getattr(agent, "_agents_view", None)
        if view is None:
            return []
        return [x.name for x in view if getattr(x, "alive", False)]
    except Exception:
        return []

_HALLUC_ROLE_TAG = re.compile(r"\b(?:villager|worker|seer|werewolf)_[A-Za-z0-9_]+\b", re.IGNORECASE)

def _clean_role_hallucinations(text: str) -> str:
    if not text:
        return text
    return _HALLUC_ROLE_TAG.sub("someone", text)

def _collapse_odd_artifacts(t: str) -> str:
    if not t:
        return t
    t = _clean_role_hallucinations(t)
    return re.sub(r"\b(?!(?:Agent_\d+)\b)([A-Za-z]+)_[A-Za-z0-9_]+\b", r"\1", t)

def _inject_named_target_if_generic(t: str, named_target: Optional[str]) -> str:
    if not t or not named_target:
        return t
    t = _GENERIC_AGENT_POSSESSIVE_RE.sub(f"{named_target}'s", t)
    t = _GENERIC_AGENT_WITH_PUNCT_RE.sub(named_target, t)
    t = _GENERIC_AGENT_WORD_RE.sub(named_target, t)
    return t

# Deduplicate possessives like: "Agent_6's's" → "Agent_6's"
_DEDUP_POSSESSIVE_RE = re.compile(r"\b([A-Za-z_0-9]+)'s's\b")
def _dedupe_possessives(text: str) -> str:
    if not text:
        return text
    return _DEDUP_POSSESSIVE_RE.sub(r"\1's", text)

# Trim simple repeated bigrams within short utterances to reduce boilerplate stutter
def _squelch_repetition(text: str) -> str:
    if not text:
        return text
    toks = text.split()
    if len(toks) < 6:
        return text
    bigrams = []
    out = []
    i = 0
    while i < len(toks) - 1:
        bg = (toks[i].lower(), toks[i+1].lower())
        if bigrams and bg == bigrams[-1]:
            i += 2
            continue
        bigrams.append(bg)
        out.append(toks[i])
        i += 1
    if i == len(toks) - 1:
        out.append(toks[-1])
    s = " ".join(out)
    s = re.sub(r"\b(\w+)(\s+\1){2,}\b", r"\1\1", s, flags=re.IGNORECASE)
    return s

def _ground_entities(text: str, agent, *, named_target: Optional[str]) -> str:
    if not text:
        return text
    t = _clean_role_hallucinations(text)
    alive = set(_alive_names(agent))

    if not alive:
        return _inject_named_target_if_generic(t, named_target)

    mentioned = set(_ALIVE_NAME_RE.findall(t))
    for nm in alive:
        if nm in t:
            mentioned.add(nm)

    if mentioned:
        for nm in list(mentioned):
            if nm not in alive:
                replacement = named_target if (named_target and named_target in alive) else next(iter(alive))
                t = re.sub(re.escape(nm), replacement, t)
    else:
        if named_target and named_target in alive:
            if len(t.split()) < 4:
                t = f"I think {named_target} {t}".strip()
            elif not _ALIVE_NAME_RE.search(t) and named_target not in t:
                if t.endswith((".", "!", "?")):
                    t = re.sub(r"([.!?])$", f" {named_target}\\1", t)
                else:
                    t = f"{t} {named_target}."
        else:
            self_name = getattr(agent, "name", None)
            fallback = next((nm for nm in alive if nm != self_name), next(iter(alive)))
            if not _ALIVE_NAME_RE.search(t) and fallback not in t:
                if t.endswith((".", "!", "?")):
                    t = re.sub(r"([.!?])$", f" {fallback}\\1", t)
                else:
                    t = f"{t} {fallback}."

    best = named_target if (named_target and named_target in alive) else (next(iter(alive)) if alive else None)
    t = _inject_named_target_if_generic(t, best)
    return t

SYSTEM_PROMPT_BASE = (
    "You are a player in a hidden-role Werewolf game. "
    "Speak in-character, concise, and natural (1 to 3 sentences). "
    "Do not reveal hidden roles or private information. "
    "If you are a werewolf, never admit being a werewolf. "
    "Avoid newlines, and keep question marks and exclamation points."
)

def _compact_alive_names(alive_list: Optional[List[str]]) -> Optional[str]:
    if not alive_list:
        return None
    try:
        names = []
        for a in alive_list:
            if isinstance(a, str):
                names.append(a)
            else:
                nm = getattr(a, "name", None)
                if nm: names.append(str(nm))
        uniq = sorted(set(names))
        return ", ".join(uniq)
    except Exception:
        return ", ".join(sorted(map(str, alive_list)))

def _build_messages(
    name: str,
    role: str,
    z_summary: str,
    intent: str,
    *,
    round_idx: Optional[int] = None,
    phase_name: Optional[str] = None,
    alive_list: Optional[List[str]] = None,
    vote_history: Optional[List[str]] = None,
    night_history: Optional[List[str]] = None,
    recent_lines: Optional[List[str]] = None,
    named_target: Optional[str] = None,
) -> List[Dict[str,str]]:
    nr = _normalize_role(role)
    sys = SYSTEM_PROMPT_BASE

    header_bits = []
    if PROMPTING.get("include_round") and round_idx is not None:
        header_bits.append(f"Round: {round_idx}")
    if PROMPTING.get("include_phase") and phase_name:
        header_bits.append(f"Phase: {phase_name}")
    if header_bits:
        sys += "\n" + "  ".join(header_bits)

    if PROMPTING.get("include_role"):
        if nr in ALLOWED_ROLES_CANON:
            sys += f"\nYour hidden role (do not state it): {nr}"
        else:
            sys += "\nYour role is hidden; play to your objectives."

    if PROMPTING.get("include_vote_history") and vote_history:
        sys += "\nVote history (day): " + " | ".join(vote_history)

    if PROMPTING.get("include_night_history") and night_history:
        sys += "\nNight history: " + " | ".join(night_history)

    if z_summary:
        sys += f"\nPersona cue: {z_summary}"

    concrete_lines: List[str] = []
    alive_compact = _compact_alive_names(alive_list)
    if alive_compact:
        concrete_lines.append(f"Alive: {alive_compact}")

    target_str = named_target if named_target else "one specific living agent"
    rule_bits: List[str] = []
    if phase_name and str(phase_name).upper().startswith("DAY"):
        rule_bits.append("Do not mention night.")
    rule = (" [" + " ".join(rule_bits) + "]") if rule_bits else ""

    k = int(PROMPTING.get("include_context_window", 0) or 0)
    has_ctx = bool(k and recent_lines)

    if intent == "accuse":
        directive = f"Intent: accuse. Target: {target_str}. You must name the target, cite one concrete detail from Recent conversation, and state one brief reason{rule}." if has_ctx else f"Intent: accuse. Target: {target_str}. You must name the target and state one concrete reason{rule}."
    elif intent == "defend":
        directive = f"Intent: defend. Target: {target_str}. You must name the target, cite one concrete detail from Recent conversation, and state one brief defense{rule}." if has_ctx else f"Intent: defend. Target: {target_str}. You must name the target and state one concrete defense{rule}."
    elif intent == "vote":
        directive = f"Intent: vote. Target: {target_str}. You must explicitly say you vote for the target, cite one concrete detail from Recent conversation, and give one brief reason{rule}." if has_ctx else f"Intent: vote. Target: {target_str}. You must explicitly say you vote for the target and give one brief reason{rule}."
    elif intent == "probe":
        directive = f"Intent: probe. Target: {target_str}. You must name the target, cite one concrete detail from Recent conversation, and ask one pointed question{rule}." if has_ctx else f"Intent: probe. Target: {target_str}. You must name the target and ask one pointed question{rule}."
    else:
        directive = f"Intent: talk. Target: {target_str}. Name the target, and give one brief reason{rule}."

    ctx = ""
    if k and recent_lines:
        ctx = "Recent conversation (latest last):\n" + "\n".join(recent_lines[-k:])

    user_blocks = [
        *concrete_lines,
        f"Speak as {name}. Keep it in-character and conversational.",
        directive,
        "Limit yourself to 1–2 sentences. Avoid newlines. Keep ? and ! if natural.",
    ]
    if ctx:
        user_blocks.append(ctx)

    user = "\n".join(user_blocks)
    return [{"role":"system","content":sys},{"role":"user","content":user}]

_tok = None
_model = None
_llm_pipeline = None
_MOUTHPIECE = None
tok = None  # public alias

def _resolve_device() -> str:
    cfg = _env_str("LLM_DEVICE", DEVICE_CFG).strip().lower()
    if cfg:
        return cfg
    return "cuda:0" if torch.cuda.is_available() else "cpu"

def _resolve_speaker_gate(cfg: dict = CFG) -> bool:
    if "LLM_SPEAKER" in os.environ:
        return _env_bool("LLM_SPEAKER", False)
    try:
        llm_block = cfg.get("llm", {}) or {}
        return bool(llm_block.get("speaker_enabled", True))
    except Exception:
        return True

_DEBUG_GATE_STRICT = bool(((CFG.get("debug", {}) or {}).get("llm_gate_strict", False))) or _env_bool("DEBUG_LLM_GATE_STRICT", False)

# ─────────────────────────────────────────────────────────────────────────────
# OpenAI singleton, args builder, and text extractor
# ─────────────────────────────────────────────────────────────────────────────
_OA_CLIENT: Optional[Any] = None

def get_openai_client() -> Any:
    """Singleton OpenAI client (Responses API)."""
    global _OA_CLIENT
    if _OA_CLIENT is not None:
        return _OA_CLIENT
    if not _OPENAI_AVAILABLE:
        raise RuntimeError("OpenAI backend requested but 'openai' package is not installed. `pip install openai`")
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in environment.")
    _OA_CLIENT = OpenAI(api_key=api_key)  # type: ignore
    return _OA_CLIENT

# Only allow Responses-API knobs. Keep minimal to avoid invalid_request_error.
_ALLOWED_RESPONSES_KW = {"max_output_tokens", "response_format"}

def _filter_responses_kwargs(kwargs: dict) -> dict:
    if not isinstance(kwargs, dict):
        return {}
    return {k: v for k, v in kwargs.items() if k in _ALLOWED_RESPONSES_KW and v is not None}

def build_openai_args(model_id: str, cfg_block: dict) -> Dict[str, Any]:
    """
    Read llm.openai.* from the unified CFG, and return only Responses-API kwargs.
    Do not pass top-level llm.* fields such as temperature or top_p.
    """
    llm_cfg = cfg_block.get("llm", {}) if isinstance(cfg_block, dict) else {}
    oai = llm_cfg.get("openai", {}) if isinstance(llm_cfg, dict) else {}
    out: Dict[str, Any] = {}
    if isinstance(oai.get("max_output_tokens", None), int):
        out["max_output_tokens"] = int(oai["max_output_tokens"])
    # Allow opt-in JSON mode, primarily for judge.py which calls llm_complete directly.
    if isinstance(oai.get("response_format", None), dict):
        out["response_format"] = oai["response_format"]
    return _filter_responses_kwargs(out)

def llm_text(resp: Any) -> str:
    """
    Normalize text extraction from Responses API results.
    Tries output_text first, otherwise walks the structured output blocks.
    """
    if resp is None:
        return ""
    txt = getattr(resp, "output_text", None)
    if isinstance(txt, str) and txt.strip():
        return txt.strip()
    out = []
    try:
        for item in getattr(resp, "output", []) or []:
            if getattr(item, "type", "") == "message":
                for seg in getattr(item, "content", []) or []:
                    if getattr(seg, "type", "") == "output_text":
                        t = getattr(seg, "text", "")
                        if t:
                            out.append(t)
    except Exception:
        pass
    return "\n".join(out).strip()

def llm_complete(model: str, messages: List[Dict[str, str]], **kwargs) -> str:
    """
    Centralized OpenAI call path. Uses the singleton client, filters kwargs to the
    Responses API allowlist, and returns extracted text.
    """
    client = get_openai_client()
    kf = _filter_responses_kwargs(kwargs or {})
    try:
        resp = client.responses.create(
            model=model,
            input=messages,
            **kf,
        )
        return llm_text(resp)
    except Exception as e:
        print(f"[OpenAI ERROR] {e}")
        return ""

# ── OpenAI pipe (Responses-only; no Chat Completions) ────────────────────────
class _TokShimOpenAI:
    def apply_chat_template(self, messages, tokenize: bool = False, add_generation_prompt: bool = True) -> str:
        parts = []
        for m in messages or []:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts.append(f"[{role}]\n{content}")
        if add_generation_prompt:
            parts.append("[assistant] ")
        return "\n".join(parts)

class _OpenAIPipe:
    """
    HF-like callable wrapper around OpenAI via Responses API.
    For compatibility with judge.py, this returns a plain string completion,
    not a list. Speaker code detects this class and handles the return type.
    """
    def __init__(self, model: str, client: Any, system_fallback: str = "You are a concise in-world speaker."):
        self.model = model
        self.client = client
        self.system_fallback = system_fallback

    def __call__(self, anchored_prompt: str, **kwargs) -> str:
        max_output_tokens = int(kwargs.get("max_output_tokens", kwargs.get("max_new_tokens", 64)))

        messages = [
            {"role": "system", "content": self.system_fallback},
            {"role": "user", "content": anchored_prompt},
        ]

        oai_args = build_openai_args(self.model, CFG)
        text_out = llm_complete(
            self.model,
            messages,
            **_filter_responses_kwargs({
                **oai_args,
                "max_output_tokens": max_output_tokens if max_output_tokens else oai_args.get("max_output_tokens"),
            }),
        )
        # Return only model output. Never echo the prompt back into downstream parsers.
        return text_out or ""

def _using_openai_provider() -> bool:
    provider = PROVIDER_DEFAULT
    if provider in ("openai", "oai"):
        return True
    mid = _env_str("LLM_MODEL_ID", MODEL_ID_DEFAULT).strip().lower()
    return mid.startswith(("gpt-", "o", "chatgpt"))

# Optional factory used by some callers (e.g., older judge variants).
def llm_pipe(*, model_id: Optional[str] = None, provider: Optional[str] = None, system_fallback: Optional[str] = None):
    use_oai = False
    if provider:
        use_oai = provider.strip().lower() in ("openai", "oai")
    else:
        use_oai = _using_openai_provider()
    model = model_id or _env_str("LLM_MODEL_ID", MODEL_ID_DEFAULT)
    if use_oai:
        client = get_openai_client()
        return _OpenAIPipe(model=model, client=client, system_fallback=system_fallback or "You are a concise in-world speaker.")
    # HF fallback
    tok_local = AutoTokenizer.from_pretrained(model, use_fast=True)
    if tok_local.pad_token_id is None:
        if tok_local.eos_token is not None:
            tok_local.pad_token = tok_local.eos_token
        else:
            tok_local.add_special_tokens({"pad_token": "<|pad|>"})
    tok_local.padding_side = "left"
    mdl = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, low_cpu_mem_usage=True)
    if getattr(mdl.config, "vocab_size", None) is not None and len(tok_local) != mdl.config.vocab_size:
        try: mdl.resize_token_embeddings(len(tok_local))
        except Exception: pass
    pipe = pipeline("text-generation", model=mdl, tokenizer=tok_local, device=0 if torch.cuda.is_available() else -1)
    class _HFPipe:
        def __call__(self, anchored_prompt: str, **kwargs):
            out = pipe(anchored_prompt, max_new_tokens=int(kwargs.get("max_new_tokens", 64)), do_sample=True)
            return [{"generated_text": anchored_prompt + (out[0].get("generated_text","") or "")}]
    return _HFPipe()

# ── Lazy loader (HF or OpenAI) ───────────────────────────────────────────────
def _lazy_load_llm():
    global _tok, _model, _llm_pipeline, tok

    if _llm_pipeline is not None:
        return _llm_pipeline, _tok

    model_id = _env_str("LLM_MODEL_ID", MODEL_ID_DEFAULT)
    device = _resolve_device()
    use_openai = _using_openai_provider()

    if use_openai:
        client = get_openai_client()
        tok_local = _TokShimOpenAI()
        llm_pipe_local = _OpenAIPipe(model=model_id, client=client)
        print(f"[INFO] OpenAI mouthpiece ready (Responses): model={model_id}")
        _tok, _model, _llm_pipeline = tok_local, None, llm_pipe_local
        tok = _tok
        return _llm_pipeline, _tok

    # ── HF path (original behavior) ──────────────────────────────────────────
    use_gpu = device.startswith("cuda")
    torch_dtype = torch.float16 if use_gpu else torch.float32

    tok_local = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok_local.pad_token_id is None:
        if tok_local.eos_token is not None:
            tok_local.pad_token = tok_local.eos_token
        else:
            tok_local.add_special_tokens({"pad_token": "<|pad|>"})
    tok_local.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    if getattr(model.config, "vocab_size", None) is not None and len(tok_local) != model.config.vocab_size:
        try:
            model.resize_token_embeddings(len(tok_local))
        except Exception:
            pass
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tok_local.pad_token_id

    pipe_device = 0 if device == "cuda" else (int(device.split(":")[1]) if device.startswith("cuda:") else -1)
    llm_pipe_local = pipeline("text-generation", model=model, tokenizer=tok_local, device=pipe_device)

    if use_gpu:
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = "CUDA"
        print(f"[INFO] LLM loaded on {device.upper()}: {gpu_name}")
    else:
        print(f"[INFO] LLM loaded on {device.upper()}")

    _tok, _model, _llm_pipeline = tok_local, model, llm_pipe_local
    tok = _tok
    return _llm_pipeline, _tok

# ── Safe stub mouthpiece and tokenizer for disabled gate
class _TokStub:
    def apply_chat_template(self, messages, tokenize: bool = False, add_generation_prompt: bool = True) -> str:
        parts = []
        for m in messages or []:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts.append(f"[{role}] {content}")
        if add_generation_prompt:
            parts.append("[assistant] ")
        return "\n".join(parts)

class _MouthStub:
    tokenizer = _TokStub()
    def __call__(self, prompt: str, *, generate_kwargs: Optional[Dict[str, Any]] = None) -> str:
        return SAFE_FALLBACK

def _lazy_mouthpiece():
    global _MOUTHPIECE
    if _MOUTHPIECE is not None:
        return _MOUTHPIECE
    gate_on = _resolve_speaker_gate(CFG)
    if not gate_on:
        if _DEBUG_GATE_STRICT:
            raise RuntimeError("LLM mouthpiece disabled by resolved gate, strict debug on.")
        _MOUTHPIECE = _MouthStub()
        return _MOUTHPIECE
    llm_pipe_local, tok_local = _lazy_load_llm()
    device = _resolve_device()
    _MOUTHPIECE = Mouthpiece(llm_pipe_local, tok_local, device)
    return _MOUTHPIECE

def _anchor_prompt(txt: str) -> str:
    t = (txt or "").lstrip()
    if t.startswith("[SYSTEM]"):
        return t
    universal_rules = (
        "[SYSTEM]\n"
        "Stay strictly in-world; never mention rules, prompts, or being an AI.\n"
        "Output exactly one short, conversational utterance (1–2 sentences) with no newlines.\n"
    )
    return universal_rules + txt

@dataclass
class _AttemptSpec:
    max_new_tokens: int
    temperature: float
    top_p: float
    repetition_penalty: float
    no_repeat_ngram_size: int

def _llm_defaults_from_cfg() -> Dict[str, Any]:
    llm = CFG.get("llm", {}) or {}
    return {
        "max_new_tokens": int(llm.get("max_new_tokens", 80)),
        "temperature": float(llm.get("temperature", 0.7)),
        "top_p": float(llm.get("top_p", 0.92)),
        "repetition_penalty": float(llm.get("repetition_penalty", 1.08)),
        "no_repeat_ngram_size": int(llm.get("no_repeat_ngram_size", 3)),
        "min_new_tokens": int(llm.get("min_new_tokens", 16)),
    }

# Common boilerplate detector to down-rank templated talk
_BOILER_FRAGMENTS = [
    "doesn't add up",
    "does not add up",
    "looks suspicious",
    "seems suspicious",
    "i believe",
    "based on",
    "i think",
]
def _boiler_penalty(s: str) -> float:
    low = (s or "").lower()
    if not low: return 0.0
    pen = 0.0
    for frag in _BOILER_FRAGMENTS:
        if frag in low:
            pen += 0.15
    return min(pen, 0.45)

class Mouthpiece:
    def __init__(self, pipe, tokenizer, device, use_bias_default: bool = True):
        self.pipe = pipe
        self.tokenizer = tokenizer
        self.device = device
        I = _bad_words_ids(tokenizer)
        self._bad_ids = I
        self.use_bias_default = use_bias_default

        d = _llm_defaults_from_cfg()
        self._attempts: List[_AttemptSpec] = [
            _AttemptSpec(max_new_tokens=max(32, d["max_new_tokens"]), temperature=d["temperature"], top_p=d["top_p"], repetition_penalty=d["repetition_penalty"], no_repeat_ngram_size=d["no_repeat_ngram_size"]),
            _AttemptSpec(max_new_tokens=max(28, d["max_new_tokens"]-8), temperature=max(0.5, d["temperature"]-0.1), top_p=max(0.85, d["top_p"]-0.02), repetition_penalty=d["repetition_penalty"]+0.04, no_repeat_ngram_size=max(3, d["no_repeat_ngram_size"])),
            _AttemptSpec(max_new_tokens=max(24, d["max_new_tokens"]-16), temperature=max(0.45, d["temperature"]-0.2), top_p=max(0.82, d["top_p"]-0.04), repetition_penalty=d["repetition_penalty"]+0.08, no_repeat_ngram_size=max(4, d["no_repeat_ngram_size"])),
        ]

    def _score_candidate(self, s: str) -> float:
        if not s: return -1.0
        length = len(s.split())
        length_score = 1.0 - abs((length - 16) / 18.0)
        length_score = max(0.0, length_score)
        meta_pen = 1.0 if _is_meta_like(s) else 0.0
        punct_bonus = 0.2 if (s.endswith((".", "…", "!", "?")) or (length <= 8 and "," not in s)) else 0.0
        boiler_pen = _boiler_penalty(s)
        base = (1.0 - 0.8 * meta_pen) * (0.6 * length_score + 0.4 * punct_bonus)
        return max(0.0, base * (1.0 - boiler_pen))

    def _generate_one(self, prompt: str, attempt: _AttemptSpec, extra_kwargs: Dict[str, Any]) -> str:
        d = _llm_defaults_from_cfg()
        max_new_tokens      = int(extra_kwargs.get("max_new_tokens", attempt.max_new_tokens))
        temperature         = float(extra_kwargs.get("temperature", attempt.temperature))
        top_p               = float(extra_kwargs.get("top_p", attempt.top_p))
        repetition_penalty  = float(extra_kwargs.get("repetition_penalty", attempt.repetition_penalty))
        no_repeat_ngram     = int(extra_kwargs.get("no_repeat_ngram_size", attempt.no_repeat_ngram_size))
        min_new_tokens      = int(extra_kwargs.get("min_new_tokens", d["min_new_tokens"]))

        if max_new_tokens < min_new_tokens:
            max_new_tokens = max(min_new_tokens, 24)

        anchored = _anchor_prompt(prompt)

        try:
            if isinstance(self.pipe, _OpenAIPipe):
                cfg_args = build_openai_args(self.pipe.model, CFG)
                cfg_args["max_output_tokens"] = max_new_tokens
                resp = self.pipe(
                    anchored,
                    **_filter_responses_kwargs(cfg_args),
                )
            else:
                gen_kwargs = dict(extra_kwargs or {})
                for k in ("max_new_tokens","temperature","top_p","repetition_penalty","no_repeat_ngram_size","min_new_tokens"):
                    gen_kwargs.pop(k, None)

                resp = self.pipe(
                    anchored,
                    pad_token_id=getattr(self.tokenizer, "pad_token_id", None),
                    eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
                    do_sample=True,
                    bad_words_ids=self._bad_ids or None,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    no_repeat_ngram_size=no_repeat_ngram,
                    **gen_kwargs,
                )[0].get("generated_text", "")
        except Exception as e:
            print(f"[LLM ERROR/generate] {e}")
            return SAFE_FALLBACK

        cont = resp[len(anchored):] if resp.startswith(anchored) else resp
        cont = _early_stop(cont)
        cont = _one_line(cont)
        cont = _sanitize_roles(cont)
        cont = cont.strip(BAD_QUOTES + " ")
        cont = _postfilter_meta(cont)
        cont = _sanitize(cont)
        cont = normalize_contractions(strip_parentheticals(cont))
        cont = _dedupe_possessives(cont)
        cont = _squelch_repetition(cont)
        return cont or SAFE_FALLBACK

    def __call__(self, prompt: str, *, generate_kwargs: Optional[Dict[str, Any]] = None) -> str:
        cands: List[Tuple[str, float]] = []
        for spec in self._attempts:
            txt = self._generate_one(prompt, spec, generate_kwargs or {})
            cands.append((txt, self._score_candidate(txt)))
        non_meta = [(t, sc) for (t, sc) in cands if not _is_meta_like(t)]
        best = max(non_meta or cands, key=lambda x: x[1])[0]
        return best or SAFE_FALLBACK

    def legacy_from_latent(self, z, agent, *, use_bias: bool = None) -> str:
        role = getattr(agent, "role", "") or ""
        z_summary = agent.decode_z(z)
        name = agent.name
        k = int(PROMPTING.get("include_context_window", 6) or 6)
        heard_block = _recent_discussion_block(agent, max_lines=k)
        messages = _build_messages(
            name=name,
            role=role,
            z_summary=z_summary,
            intent="talk",
            recent_lines=[ln[2:] if ln.startswith("- ") else ln for ln in heard_block.splitlines()] if heard_block else [],
            named_target=None,
        )
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self(prompt, generate_kwargs={})

try:
    from speaker_llm import (
        LogitBiasHead,
        with_logit_bias_generate_kwargs,
        with_fused_bias_generate_kwargs,
        with_alpha_fusion_generate_kwargs,
        SPEAKER_HIST_K,
    )
except Exception:
    LogitBiasHead = None
    with_logit_bias_generate_kwargs = None
    with_fused_bias_generate_kwargs = None
    with_alpha_fusion_generate_kwargs = None
    SPEAKER_HIST_K = 3

try:
    from repetition import repetition_penalty as _rep_penalty_fn  # type: ignore
except Exception:
    _rep_penalty_fn = None  # type: ignore

def _recent_texts(agent, k: int = 3) -> List[str]:
    if not getattr(agent, "message_memory", None):
        return []
    return [m for (_, m) in list(agent.message_memory)[-k:] if m]

def _fusion_alpha_from_env_yaml() -> Optional[float]:
    for key in ("FUSION_ALPHA_INTENT_BIAS", "FUSION_ALPHA_INTENT", "FUSION_ALPHA"):
        val = os.getenv(key, "").strip()
        if val:
            try:
                return float(val)
            except Exception:
                pass
    try:
        fus = CFG.get("fusion", {}) or {}
        v = fus.get("alpha_intent_bias", None)
        if isinstance(v, (int, float)):
            return float(v)
    except Exception:
        pass
    return None

def _talkhead_probs_for(agent: "BaseAgent", z: torch.Tensor) -> Optional[torch.Tensor]:
    try:
        fp = getattr(agent, "planner_factorized", None)
        if fp is None or not hasattr(fp, "talk"):
            return None
        try:
            num_cats = int(fp.talk.net[-1].out_features)  # type: ignore[attr-defined]
        except Exception:
            num_cats = 5
        dev = z.device if torch.is_tensor(z) else None
        mask = torch.ones(1, num_cats, dtype=torch.bool, device=dev)
        logits = fp.talk(z.unsqueeze(0), mask=mask).squeeze(0).float()
        return torch.softmax(logits, dim=-1).detach().cpu()
    except Exception:
        return None

def _maybe_build_bias_kwargs(z: torch.Tensor, agent: "BaseAgent", alpha_override: Optional[float] = None) -> Dict[str, Any]:
    bias_head = getattr(agent, "bias_head", None)
    if bias_head is None or (LogitBiasHead is not None and not isinstance(bias_head, LogitBiasHead)):
        return {}
    recent = _recent_texts(agent, k=SPEAKER_HIST_K)
    persona = getattr(agent, "persona_effects", None)
    th_probs = _talkhead_probs_for(agent, z)
    _, tok_local = _lazy_load_llm()
    alpha = alpha_override if isinstance(alpha_override, (int, float)) else _fusion_alpha_from_env_yaml()

    if with_fused_bias_generate_kwargs is not None:
        try:
            return with_fused_bias_generate_kwargs(
                tokenizer=tok_local, head=bias_head, z_t=z,
                talkhead_probs=th_probs, alpha=alpha,
                role=getattr(agent, "role", None),
                recent_texts=recent, persona_effects=persona,
            )
        except Exception as e:
            print(f"[LLM WARN] fused-bias kwargs failed: {e}")

    if with_alpha_fusion_generate_kwargs is not None:
        try:
            return with_alpha_fusion_generate_kwargs(
                tokenizer=tok_local,
                agent=agent,
                alpha=alpha,
            )
        except Exception as e:
            print(f"[LLM WARN] alpha-fusion kwargs failed: {e}")

    return {}

def _style_kwargs(agent: "BaseAgent") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in ("llm_temperature", "temperature", "temp"):
        v = getattr(agent, k, None)
        if isinstance(v, (int, float)) and v > 0: out["temperature"] = float(v); break
    for k in ("llm_top_p", "top_p"):
        v = getattr(agent, k, None)
        if isinstance(v, (int, float)) and 0 < v <= 1: out["top_p"] = float(v); break
    for k in ("llm_max_new_tokens", "max_new_tokens", "max_len"):
        v = getattr(agent, k, None)
        if isinstance(v, int) and v > 0: out["max_new_tokens"] = int(v); break
    v = getattr(agent, "repetition_penalty", None)
    if isinstance(v, (int, float)) and v > 0: out["repetition_penalty"] = float(v)
    v = getattr(agent, "no_repeat_ngram_size", None)
    if isinstance(v, int) and v >= 0: out["no_repeat_ngram_size"] = int(v)
    defaults = _llm_defaults_from_cfg()
    for k, dv in defaults.items():
        out.setdefault(k, dv)
    if out.get("max_new_tokens", 0) < out.get("min_new_tokens", 0):
        out["max_new_tokens"] = max(out["min_new_tokens"], 24)
    return out

def _merge_gen_kwargs(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a or {})
    for k, v in (b or {}).items():
        if k == "logits_processor":
            la = out.get("logits_processor", [])
            lb = v or []
            out["logits_processor"] = list(la) + list(lb)
        else:
            out[k] = v
    return out

def _build_prompt_from_latent(
    tokenizer,
    z: torch.Tensor,
    agent: "BaseAgent",
    *,
    named_target: Optional[str],
    phase: Optional[str],
    round_idx: Optional[int],
    plan: Optional[dict],
) -> str:
    role = getattr(agent, "role", "") or ""
    z_summary = agent.decode_z(z)
    name = agent.name

    k = int(PROMPTING.get("include_context_window", 6) or 6)
    recent_lines = _recent_texts(agent, k=k)

    intent = "talk"
    if isinstance(plan, dict):
        intent = plan.get("intent", intent) or intent

    alive_names_list = None
    try:
        view = getattr(agent, "_agents_view", None)
        if view is not None:
            alive_names_list = [x.name for x in view if getattr(x, "alive", False)]
    except Exception:
        alive_names_list = None

    messages = _build_messages(
        name=name,
        role=role,
        z_summary=z_summary,
        intent=intent,
        round_idx=round_idx if PROMPTING.get("include_round") else None,
        phase_name=phase if PROMPTING.get("include_phase") else None,
        alive_list=alive_names_list,
        vote_history=None,
        night_history=None,
        recent_lines=recent_lines,
        named_target=named_target,
    )
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def chatgpt_llm_from_latent(z: torch.Tensor, agent: "BaseAgent", *, named_target: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    mouth = _lazy_mouthpiece()
    prompt = _build_prompt_from_latent(mouth.tokenizer, z, agent, named_target=named_target, phase=None, round_idx=None, plan=None)
    style = _style_kwargs(agent)
    text = mouth(prompt, generate_kwargs=style)

    ok, _ = _enforce_intent_target(text, intent=None, target=named_target)
    if not ok and not isinstance(mouth, _MouthStub):
        text = mouth(prompt + "\n\nFollow the directive: name the target.", generate_kwargs=style)
        ok, _ = _enforce_intent_target(text, intent=None, target=named_target)
    if not ok:
        text = _template_fallback(None, named_target)

    text = _ground_entities(text, agent, named_target=named_target)
    text = _collapse_odd_artifacts(text)

    text, guard_meta = _enforce_utterance_constraints(text, phase=None)
    rep = float(_rep_penalty_fn(text)) if callable(_rep_penalty_fn) else None
    meta: Dict[str, Any] = {"repetition_penalty": rep, **guard_meta}
    if style:
        meta["style_used"] = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in style.items()}
    if named_target:
        meta["named_target"] = named_target
    return text, meta

def chatgpt_llm_with_bias(
    z: torch.Tensor,
    agent: "BaseAgent",
    *,
    named_target: Optional[str] = None,
    **kwargs,
) -> Tuple[str, Dict[str, Any]]:
    # Use env-configurable minimum words to avoid terse, generic replies
    MIN_WORDS = _env_int("LANGUAGE_MIN_WORDS", int(((CFG.get("language", {}) or {}).get("min_words", 12))))
    mouth = _lazy_mouthpiece()

    phase = kwargs.get("phase", None)
    round_idx = kwargs.get("round_num", None)
    plan = kwargs.get("plan", None)

    prompt = _build_prompt_from_latent(
        mouth.tokenizer,
        z,
        agent,
        named_target=named_target,
        phase=phase,
        round_idx=round_idx,
        plan=plan,
    )

    alpha_kw = kwargs.get("fusion_alpha", kwargs.get("alpha", None))
    try:
        alpha_kw = float(alpha_kw) if alpha_kw is not None else None
    except Exception:
        alpha_kw = None

    bias_kwargs: Dict[str, Any] = {}
    if not isinstance(mouth, _MouthStub):
        bias_kwargs = _maybe_build_bias_kwargs(z, agent, alpha_override=alpha_kw)

    style = _style_kwargs(agent)
    gen_kwargs = _merge_gen_kwargs(bias_kwargs, style)

    user_gen = kwargs.get("generate_kwargs", None)
    if isinstance(user_gen, dict) and user_gen:
        gen_kwargs.update(user_gen)

    d = _llm_defaults_from_cfg()
    gen_kwargs.setdefault("min_new_tokens", d["min_new_tokens"])
    if gen_kwargs.get("max_new_tokens", 0) < gen_kwargs["min_new_tokens"]:
        gen_kwargs["max_new_tokens"] = max(gen_kwargs["min_new_tokens"], 24)

    text = mouth(prompt, generate_kwargs=gen_kwargs)

    intent_val: Optional[str] = None
    if isinstance(plan, dict):
        intent_val = plan.get("intent")
    elif "intent" in kwargs:
        intent_val = kwargs.get("intent")

    ok, _ = _enforce_intent_target(text, intent=intent_val, target=named_target)
    if not ok and not isinstance(mouth, _MouthStub):
        text = mouth(prompt + "\n\nFollow the directive exactly: name the target and match the intent verb.", generate_kwargs=gen_kwargs)
        ok, _ = _enforce_intent_target(text, intent=intent_val, target=named_target)
    if not ok:
        text = _template_fallback(intent_val, named_target)

    needs_retry = (len((text or "").split()) < max(8, MIN_WORDS))
    if phase and str(phase).upper().startswith("DAY") and _DAY_NIGHT_BANS.search(text or ""):
        needs_retry = True
    short_retry = False
    if needs_retry and not isinstance(mouth, _MouthStub):
        short_retry = True
        gen_kwargs_retry = dict(gen_kwargs)
        gen_kwargs_retry["temperature"] = max(0.55, float(gen_kwargs.get("temperature", 0.7)) + 0.05)
        gen_kwargs_retry["top_p"] = min(0.96, float(gen_kwargs.get("top_p", 0.92)) + 0.02)
        gen_kwargs_retry["max_new_tokens"] = max(gen_kwargs.get("max_new_tokens", 32), d["min_new_tokens"] + 8)
        text = mouth(prompt, generate_kwargs=gen_kwargs_retry)

        ok, _ = _enforce_intent_target(text, intent=intent_val, target=named_target)
        if not ok:
            text = _template_fallback(intent_val, named_target)

    text = _ground_entities(text, agent, named_target=named_target)
    text = _collapse_odd_artifacts(text)
    text = _dedupe_possessives(text)
    text = _squelch_repetition(text)

    text, guard_meta = _enforce_utterance_constraints(text, phase=phase)

    rep = float(_rep_penalty_fn(text)) if callable(_rep_penalty_fn) else None
    meta: Dict[str, Any] = {
        "repetition_penalty": rep,
        "fused_bias_used": bool(bias_kwargs),
        "fusion_alpha": (
            float(alpha_kw)
            if alpha_kw is not None
            else (float(_fusion_alpha_from_env_yaml()) if _fusion_alpha_from_env_yaml() is not None else None)
        ),
        "short_retry": short_retry,
        **guard_meta,
    }
    if style:
        meta["style_used"] = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in style.items()}
    if bias_kwargs:
        meta["bias_keys"] = sorted(list(bias_kwargs.keys()))
    if named_target:
        meta["named_target"] = named_target
    if plan is not None:
        meta["plan"] = plan
    if phase is not None:
        meta["phase"] = phase
    if round_idx is not None:
        meta["round"] = round_idx
    return text, meta

def llm_fn_from_env():
    if not _resolve_speaker_gate(CFG):
        class _Stub:
            tokenizer = _TokStub()
            def __call__(self, prompt: str, *, generate_kwargs=None) -> str:
                return SAFE_FALLBACK
        return _Stub()
    return _lazy_mouthpiece()

__all__ = [
    "normalize_contractions",
    "strip_parentheticals",
    "leaks_hidden_info",
    "get_hidden_info_patterns",
    "Mouthpiece",
    "llm_fn_from_env",
    "chatgpt_llm_from_latent",
    "chatgpt_llm_with_bias",
    "tok",
    "SAFE_FALLBACK",
    "get_openai_client",
    "llm_text",
    "llm_complete",
    "build_openai_args",
    "_OpenAIPipe",
    "llm_pipe",
]
