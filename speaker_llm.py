# speaker_llm.py
# Trainable, lexicon-guided logit bias head for the LLM mouthpiece.
from __future__ import annotations
import os, json, hashlib, re, itertools, unicodedata, random
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING, Callable, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

# Resolve a consistent SAFE_FALLBACK shared with llm_script
try:
    from llm_script import SAFE_FALLBACK  # "I need a moment to think."
except Exception:
    # Optional but recommended: slightly more in-character fallback
    # (Changed em dash to a comma)
    SAFE_FALLBACK = "Hold on, let me weigh the clues before I speak."

if TYPE_CHECKING:
    # Only used for typing; avoids importing transformers at runtime.
    from transformers import PreTrainedTokenizerBase

# Minimal exception logger for mouthpiece failures
def log_exc(tag: str, e: Exception) -> None:
    try:
        print(f"[{tag}] {type(e).__name__}: {e}", flush=True)
    except Exception:
        pass

# ───────────────────────── Config + env shims ─────────────────────────
def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    return default if v is None else str(v).strip().lower() in ("1","true","yes","y","on")

def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None: return default
    try: return int(v)
    except Exception: return default

def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None: return default
    try: return float(v)
    except Exception: return default

def _env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return default if v is None else str(v)

with open("config.yaml", "r") as f:
    CFG = yaml.safe_load(f) or {}

# Pull language.* for guards/decoding knobs
_LANGUAGE = CFG.get("language", {}) if isinstance(CFG.get("language", {}), dict) else {}

class LangCfg:
    """Tiny adapter mirroring speaker.LANGCFG fields for standalone use."""
    def __init__(self, d: Dict[str, Any]):
        self.hedges_cap_per5 = int(d.get("hedges_cap_per5", 2))
        self.intensifiers_cap_per5 = int(d.get("intensifiers_cap_per5", 2))
        self.discourse_markers_cap_per5 = int(d.get("discourse_markers_cap_per5", 2))
        self.trigram_veto = bool(d.get("trigram_veto", True))
        self.redo_max = int(d.get("redo_max", 1))
        # Extra hooks used locally
        self.bigram_penalty = float(d.get("bigram_penalty", 0.6))
        self.min_words = int(d.get("min_words", 12))
        # New: configurable banned bigrams to discourage boilerplate openers
        self.banned_bigrams: List[str] = list(d.get("banned_bigrams", []))

LANGCFG_DEFAULT = LangCfg(_LANGUAGE)

# Legacy speaker_llm keys (present in your config.yaml)
_BIAS_CAP        = float(CFG.get("BIAS_CAP", 2.0))
_BASE_STRENGTH   = float(CFG.get("BASE_STRENGTH", 1.0))
_SPK_DEBUG       = bool(CFG.get("DEBUG", False))
_DEFAULT_LEXICON = dict(CFG.get("DEFAULT_LEXICON", {
    "accuse":   ["accuse", "suspicious", "suspect", "lying", "deceive", "eliminate", "vote"],
    "defend":   ["defend", "innocent", "trust", "ally", "support", "clear"],
    "hedge":    ["maybe", "perhaps", "uncertain", "unsure", "might", "seems", "appears"],
    "question": ["why", "how", "what", "who", "where", "when", "?" ],
    "vote":     ["vote", "eliminate", "banish", "target", "lynch"],
}))
_CAT_ORDER       = list(CFG.get("CAT_ORDER", ["accuse", "defend", "hedge", "question", "vote"]))
_SPEAKER_HIST_K  = int(CFG.get("SPEAKER_HIST_K", 3))  # shared default

# Optional extras, overrides for lexicon without touching code
_EXTRA_LEXICON   = dict(CFG.get("EXTRA_LEXICON", {}))      # {cat: [tokens...]}
_STRICT_LEXICON  = bool(CFG.get("STRICT_LEXICON", False))  # if True, ignore unknown cats in EXTRA

# NEW (Phase-5/7): soft control knobs
_PER_CAT_SCALES = dict(CFG.get("PER_CAT_SCALES", {}))      # e.g., {"question": 1.1, "vote": 0.95}
_ENTROPY_ATTEN  = float(CFG.get("ENTROPY_ATTENUATION", 0.5))

# Provider hint (so we can decide OpenAI path without importing SDK here)
_LLM_PROVIDER = os.getenv("LLM_PROVIDER", os.getenv("JUDGE_PROVIDER", "")).strip().lower()

# Env overrides
BIAS_CAP        = _env_float("BIAS_CAP", _BIAS_CAP)
BASE_STRENGTH   = _env_float("BASE_STRENGTH", _BASE_STRENGTH)
LLM_SPK_DEBUG   = _env_bool ("LLM_SPK_DEBUG", _SPK_DEBUG)
SPEAKER_HIST_K  = _env_int  ("SPEAKER_HIST_K", _SPEAKER_HIST_K)
# Fusion default
FUSION_ALPHA    = _env_float("TALK_FUSION_ALPHA", float(CFG.get("sim", {}).get("talk_fusion_alpha", 0.5)))
# NEW: entropy attenuation (0..1)
ENTROPY_ATTEN   = _env_float("ENTROPY_ATTENUATION", _ENTROPY_ATTEN)

# Normalized, ordered lexicon + a safe map (word -> token ids built at runtime)
CAT_ORDER       = _CAT_ORDER
DEFAULT_LEXICON = {k: list(v) for k, v in _DEFAULT_LEXICON.items() if k in CAT_ORDER}
if _EXTRA_LEXICON := _EXTRA_LEXICON:
    for k, v in _EXTRA_LEXICON.items():
        if (not _STRICT_LEXICON) or (k in CAT_ORDER):
            DEFAULT_LEXICON.setdefault(k, [])
            DEFAULT_LEXICON[k].extend(list(v))

# NEW: normalized per-category scales limited to known categories
PER_CAT_SCALES = {k: float(v) for k, v in _PER_CAT_SCALES.items() if k in CAT_ORDER}

# Intents: normalize "ask"→"question" (to match CAT_ORDER and EXEMPLARS keys)
def _norm_intent(s: Optional[str]) -> str:
    t = (s or "").strip().lower()
    if t in ("ask", "query"): return "question"
    if t in ("accuse","defend","hedge","question","vote"): return t
    return "hedge"

# ───────────────────────── OpenAI tokenizer-like shim (NEW) ───────────────────
class OpenAITokenizerShim:
    """
    Minimal tokenizer-ish object so existing helpers (phrase detection, caches)
    keep working when the mouthpiece is OpenAI (no HF tokenizer available).

    NOTE: IDs are *fake* and ONLY used for our local bias heuristics & caches.
    They DO NOT correspond to OpenAI token ids and are not passed to the API.
    """
    _provider = "openai"
    _is_openai_shim = True

    def __init__(self):
        self.name_or_path = "openai/shim"
        self._vocab: Dict[str, int] = {}
        self._inv: Dict[int, str] = {}
        self._seed = 1337

    def _assign_id(self, tok: str) -> int:
        if tok not in self._vocab:
            rid = abs(hash((tok, self._seed))) % 500_000 + 100  # avoid tiny ids
            self._vocab[tok] = rid
            self._inv[rid] = tok
        return self._vocab[tok]

    def __len__(self) -> int:
        return len(self._vocab) if self._vocab else 100_000

    @property
    def vocab_size(self) -> int:
        return len(self)

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        # Very rough: split on spaces/punct; assign stable fake IDs.
        toks = re.findall(r"[A-Za-z_]+|\d+|[^\sA-Za-z0-9]", text or "")
        return [self._assign_id(t) for t in toks] or [self._assign_id("<EMPTY>")]

    def __call__(self, text: str, add_special_tokens: bool = False) -> Dict[str, List[int]]:
        return {"input_ids": self.encode(text, add_special_tokens=add_special_tokens)}

    def decode(self, ids: List[int]) -> str:
        return " ".join(self._inv.get(i, "<UNK>") for i in ids)

def _is_openai_tok(tok: Any) -> bool:
    return bool(getattr(tok, "_is_openai_shim", False)) or getattr(tok, "_provider", "") == "openai"

def _maybe_attach_openai_shim(mouth: Any) -> "PreTrainedTokenizerBase":
    tok = getattr(mouth, "tokenizer", None)
    if tok is None:
        # Heuristic: attach shim when provider smells like OpenAI.
        if _LLM_PROVIDER == "openai" or str(getattr(mouth, "provider", "")).lower() == "openai":
            tok = OpenAITokenizerShim()
            try: setattr(mouth, "tokenizer", tok)
            except Exception: pass
    return getattr(mouth, "tokenizer", tok if tok is not None else OpenAITokenizerShim())

# ───────────────────────── 1.1 Surface cleanup & anchoring helpers ───────────
# --- Surface cleanup ---
def _clean_surface_artifacts(s: str) -> str:
    import re
    s = (s or "").strip()
    s = re.sub(r"^(Agent:?\s*)+", "", s)         # strip leaked "Agent:" prefixes
    s = re.sub(r"('s){2,}", "'s", s)             # collapse doubled possessives
    s = re.sub(r"\s+([,\.!?])", r"\1", s)        # fix spaces before punctuation
    s = re.sub(r"\s{2,}", " ", s)                # collapse multi-spaces
    return s

# --- Allowed-names gates (set each round by sim.py) ---
_ALLOWED_NAMES = None
def _set_allowed_names(names):
    global _ALLOWED_NAMES
    _ALLOWED_NAMES = set(names or [])

def _mentions_only_known_agents(s: str) -> bool:
    import re
    if not _ALLOWED_NAMES:
        return True
    found = set(re.findall(r"Agent_\d+", s or ""))
    return found.issubset(_ALLOWED_NAMES)

# --- Target+Reason guard (prevents drift and enforces causal clause) ---
def _ensure_target_and_reason(s: str, target: str | None) -> str:
    if not target:
        return s
    t = s or ""
    if target not in t:
        # Replace em dash insertion with a colon
        t = f"{target}: because their behavior raises concerns. " + t
    low = t.lower()
    if (" because " not in low) and (" since " not in low) and (" as " not in low):
        t = t.rstrip(".") + " because of their actions."
    return t

# --- Acceptance gate (length, target presence, allowed names only) ---
def _acceptable_utterance(text: str, *, min_words: int, target: str | None) -> bool:
    if not text:
        return False
    w = sum(1 for x in text.split() if x.strip())
    if w < int(min_words):
        return False
    if target and (target not in text):
        return False
    if not _mentions_only_known_agents(text):
        return False
    return True

# ───────────────────────── 1.2 Env knobs for min-length/redo/repetition ──────
_MIN_WORDS     = int(os.environ.get("LANGUAGE_MIN_WORDS", "8"))
_REDO_MAX      = int(os.environ.get("LANGUAGE_REDO_MAX", "2"))
_DEFAULT_NGRAM = int(os.environ.get("LLM_NO_REPEAT_NGRAM", "3"))
_DEFAULT_REP   = float(os.environ.get("LLM_REPEAT_PENALTY", "1.15"))

# ───────────────────────── Local history features (no imports) ────────────────
def make_hist_feats(recent_texts: List[str], phase_code: Optional[int] = None) -> torch.Tensor:
    if not recent_texts:
        base = torch.tensor([0.0, 0.0], dtype=torch.float32)
    else:
        n = len(recent_texts)
        acc = sum(int(("accuse" in t.lower()) or ("vote" in t.lower())) for t in recent_texts) / n
        mean_len = min(1.5, sum(len(t) for t in recent_texts) / max(1, n) / 100.0)
        base = torch.tensor([acc, mean_len], dtype=torch.float32)

    if phase_code is None:
        return base

    oh = torch.zeros(3, dtype=torch.float32)
    try:
        pc = int(phase_code)
        if 0 <= pc < 3:
            oh[pc] = 1.0
    except Exception:
        pass
    return torch.cat([base, oh], dim=0)

# ───────────────────────── STOP + META (relaxed and narrow) ───────────────────
# Keep hard guards. Do not cut on raw newlines.
_HARD_GUARDS = ("System:", "Narrator:")
# Narrow classic prompt-leak phrases only
_META_BANS = ["you are chatgpt", "as an ai", "system prompt", "jailbreak", "ignore previous instructions"]

_AGENT_PREFIX_RX = re.compile(r"(^|\n)\s*Agent_\d+:\s*")

# Normalization pass used before early stop and meta checks
def _normalize_for_checks(text: str) -> str:
    return normalize_contractions(text or "")

# ───────────────────────── Detok fixes, boilerplate, target+reason ───────────
DETOK_FIXES: List[Tuple[str, str]] = [
    (r'Agent[_\s-]*zero\b', 'Agent_0'),
    (r'Agent[_\s-]*one\b', 'Agent_1'),
    (r'Agent[_\s-]*two\b', 'Agent_2'),
    (r'Agent[_\s-]*three\b', 'Agent_3'),
    (r'Agent[_\s-]*four\b', 'Agent_4'),
    (r'Agent[_\s-]*five\b', 'Agent_5'),
    (r'Agent[_\s-]*six\b', 'Agent_6'),
    (r'Agent[_\s-]*seven\b', 'Agent_7'),
    (r'Agent[_\s-]*eight\b', 'Agent_8'),
    (r'Agent[_\s-]*nine\b', 'Agent_9'),
    (r'Agent\\?_([0-9])', r'Agent_\1'),
    # NEW: repair plain-space and hyphenated numerals & possessives
    (r'\bAgent\s*[- ]\s*([0-9])\b', r'Agent_\1'),
    (r"\bAgent\s+([0-9])('s\b)", r'Agent_\1\2'),
    (r"\s+([,.\?!])", r"\1"),
    (r"\s{2,}", " "),
]

BOILERPLATE_OPENERS = [
    r'^Based on\b', r'^I believe\b', r'^I agree\b', r'^I understand\b'
]
PLACEHOLDERS = ['someone', 'somebody', 'certain individual', 'X', 'Y', 'Z',
                # Treat bare 'Agent' as a placeholder to be patched with target (incl. punctuation variants)
                'Agent', 'agent']
REASON_HINTS = ["because", "since", "due to", "as ", "for ", "after ", "from "]

# --- Tiny de-templating helpers (conservative) ---
_DETEMPLATE_RULES: List[Tuple[re.Pattern, List[str]]] = [
    # Normalize apostrophes first via normalize_contractions, then match
    (re.compile(r"\bdoesn't add up\b", re.IGNORECASE), ["doesn't check out", "doesn't line up"]),
    (re.compile(r"\bdoes not add up\b", re.IGNORECASE), ["doesn't check out", "doesn't line up"]),
    (re.compile(r"\blooks suspicious\b", re.IGNORECASE), ["seems off", "looks off", "seems suspicious"]),
]

def _stable_variant_selector(source_text: str, options: List[str]) -> str:
    # Deterministic choice keyed on the entire string
    h = hashlib.md5(source_text.encode("utf-8")).hexdigest()
    idx = int(h, 16) % max(1, len(options))
    return options[idx]

def _preserve_leading_case(match_text: str, replacement: str) -> str:
    return replacement.capitalize() if match_text and match_text[0].isupper() else replacement

def _apply_detemplating(text: str) -> str:
    s = text
    for rx, opts in _DETEMPLATE_RULES:
        replacement = _stable_variant_selector(s, opts)
        def _repl(m):
            return _preserve_leading_case(m.group(0), replacement)
        s = rx.sub(_repl, s)
    return s

def normalize_utterance(s: str) -> str:
    s = normalize_contractions(s or "")
    # Trim self-correction stutters like "werewor- I mean, ..."
    s = re.sub(r"\b(\w+)-\s*I mean,?\s*", "", s)
    for pat, rep in DETOK_FIXES:
        s = re.sub(pat, rep, s, flags=re.IGNORECASE)
    # Tiny de-templating pass on a few generic fragments, no facts introduced
    s = _apply_detemplating(s)
    return s.strip()

def looks_boilerplate(s: str) -> bool:
    return any(re.search(p, s.strip(), flags=re.IGNORECASE) for p in BOILERPLATE_OPENERS)

def _contains_placeholder(s: str) -> bool:
    """
    Detect placeholder tokens even when followed by punctuation (e.g., 'Agent,' or 'Agent:').
    """
    for tok in PLACEHOLDERS:
        if re.search(rf"\b{re.escape(tok)}\b", s, flags=re.IGNORECASE):
            return True
    return False

def ensure_target_and_reason(text: str, target: Optional[str]) -> str:
    if not target or "Agent_" not in target:
        return text
    t = text or ""

    # Replace generic stand-ins (including bare 'Agent' with punctuation) with the concrete target once.
    if _contains_placeholder(t):
        t = re.sub(
            r"\b(X|Y|Z|someone|somebody|certain individual|Agent|agent)s?\b",
            target, t, count=1, flags=re.IGNORECASE
        )

    # If target is still missing, graft it on succinctly.
    if target not in t:
        if len(t) < 160:
            t = f"{t} {target}."

    # Ensure there's at least a short reason; keep it generic and neutral.
    if not any(h in t.lower() for h in REASON_HINTS) and len(t) < 180:
        t = f"{t} because their statements conflict with earlier claims."
    return t.strip()

def early_stop_text(text: str) -> str:
    """
    Cut ONLY on explicit guards:
      - 'System:' or 'Narrator:' anywhere
      - 'Agent_\\d+:' when it appears as a speaker prefix (at start or after a newline)

    Safety additions:
      1) Normalize contractions and curly quotes before checks.
      2) After stripping a leading 'Agent_k:' prefix, if the remainder is empty, return SAFE_FALLBACK.
      3) When encountering a mid-text 'Agent_k:' boundary, only truncate there if the
         accumulated content already has at least MIN_TOK_BEFORE_TRUNC tokens. Otherwise, keep all text.
    """
    if not text:
        return SAFE_FALLBACK
    s = _normalize_for_checks(str(text))

    # Truncate at first hard guard if present
    for g in _HARD_GUARDS:
        i = s.find(g)
        if i != -1:
            s = s[:i]
            break

    # Handle Agent_ speaker prefixes
    # First, check for a leading prefix
    m = _AGENT_PREFIX_RX.search(s)
    if m and m.start() == 0:
        s = s[m.end():]
        s = s.strip()
        if not s:
            return SAFE_FALLBACK

    # Then, handle any mid-text boundary. Only cut if we already have enough tokens.
    MIN_TOK_BEFORE_TRUNC = max(6, int(getattr(LANGCFG_DEFAULT, "min_words", 12) // 2))
    mid = _AGENT_PREFIX_RX.search(s)
    if mid and mid.start() > 0:
        left = s[:mid.start()].strip()
        left_tok = len(left.split())
        if left_tok >= MIN_TOK_BEFORE_TRUNC:
            s = left

    # Normalize spacing, join lines
    parts = [ln.strip() for ln in s.splitlines()]
    s = " ".join([p for p in parts if p]).strip()

    # Final safety
    return s if s else SAFE_FALLBACK

def looks_meta(text: str) -> bool:
    """Lowercased substring check against a narrow meta-ban list, after normalization."""
    t = _normalize_for_checks((text or "").strip()).lower()
    if not t:
        return True
    return any(b in t for b in _META_BANS)

# ───────────────────────── Model: LogitBiasHead ─────────────────────────
class LogitBiasHead(nn.Module):
    """Map (z_t, short history) → per-category bias strengths for token logits."""

    def __init__(self, latent_dim: int = 32, hidden: int = 128, num_cats: Optional[int] = None) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.num_cats   = int(num_cats if num_cats is not None else len(CAT_ORDER))
        in_dim = latent_dim + 2  # 2 = make_hist_feats(recent_texts)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, self.num_cats)
        )
        # Cap and base scale read from cfg, env
        self.bias_cap = float(BIAS_CAP)
        self.base     = float(BASE_STRENGTH)

    def forward(self, z_t: torch.Tensor, hist_feats: torch.Tensor) -> torch.Tensor:
        if z_t.dim() == 1:         z_t = z_t.unsqueeze(0)
        if hist_feats.dim() == 1:  hist_feats = hist_feats.unsqueeze(0)
        x = torch.cat([z_t, hist_feats], dim=-1)
        raw = self.net(x)
        bias = torch.tanh(raw) * self.bias_cap
        return self.base * bias

    @torch.no_grad()
    def category_logits(
        self,
        z_t: torch.Tensor,
        role_bit: Optional[torch.Tensor] = None,
        recent_texts: Optional[List[str]] = None,
    ) -> torch.Tensor:
        if z_t.dim() == 1:
            z_t = z_t.unsqueeze(0)
        if recent_texts is None:
            hist = torch.zeros((z_t.size(0), 2), dtype=torch.float32, device=z_t.device)
        else:
            hf = make_hist_feats(recent_texts).to(z_t.device)
            hist = hf.unsqueeze(0).expand(z_t.size(0), -1) if hf.dim() == 1 else hf
        return self.forward(z_t, hist)

    def regularizer(
        self,
        z_batch: torch.Tensor,
        recent_texts_batch: Optional[List[List[str]]] = None,
        *,
        lambda_entropy: float = 0.01,
        lambda_l2: float = 1e-4,
        lambda_balance: float = 0.01,
        target_prior: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if z_batch.dim() != 2:
            raise ValueError("z_batch must be [B, latent_dim]")

        device = z_batch.device
        B = int(z_batch.size(0))
        C = self.num_cats

        if recent_texts_batch is None:
            hist = torch.zeros(B, 2, dtype=torch.float32, device=device)
        else:
            feats = []
            for texts in recent_texts_batch:
                try:
                    feats.append(make_hist_feats(texts or []).to(device))
                except Exception:
                    feats.append(torch.tensor([0.0, 0.0], dtype=torch.float32, device=device))
            hist = torch.stack(feats, dim=0)

        strengths = self.forward(z_batch, hist)
        probs = torch.softmax(strengths, dim=-1)

        logp = (probs.clamp_min(1e-12)).log()
        L_entropy = (probs * logp).sum(dim=-1).mean()
        L_l2 = strengths.pow(2).mean()

        mean_p = probs.mean(dim=0)
        if target_prior is None:
            prior = torch.full((C,), 1.0 / max(1, C), dtype=torch.float32, device=device)
        else:
            prior = (target_prior.to(device) / target_prior.to(device).sum().clamp_min(1e-6)).detach()
        L_balance = (mean_p - prior).pow(2).mean()

        return (lambda_entropy * L_entropy) + (lambda_l2 * L_l2) + (lambda_balance * L_balance)

# ───────────────────────── Token-set cache helpers ─────────────────────────
_TOKEN_SET_CACHE: Dict[str, Dict[str, List[int]]] = {}

def _tok_cache_key(tokenizer: "PreTrainedTokenizerBase") -> str:
    parts = [
        str(getattr(tokenizer, "name_or_path", "")),
        str(getattr(tokenizer, "vocab_size", getattr(tokenizer, "len", None) or "")),
        str(len(tokenizer) if hasattr(tokenizer, "__len__") else ""),
        str(id(tokenizer)),
    ]
    s = "|".join(parts)
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def _build_token_sets(tokenizer: "PreTrainedTokenizerBase",
                      lexicon: Dict[str, List[str]]) -> Dict[str, List[int]]:
    cat2ids: Dict[str, List[int]] = {}
    for cat, words in lexicon.items():
        ids: List[int] = []
        for w in words:
            if w == "?":
                enc = tokenizer("?")["input_ids"]
                if len(enc) == 1:
                    ids.append(int(enc[0]))
                continue
            enc = tokenizer(w, add_special_tokens=False)["input_ids"]
            if enc:
                ids.append(int(enc[0]))
        cat2ids[cat] = sorted(list({i for i in ids if i is not None}))
    return cat2ids

def _get_cat2ids_cached(tokenizer: "PreTrainedTokenizerBase",
                        lexicon: Dict[str, List[str]]) -> Dict[str, List[int]]:
    key = _tok_cache_key(tokenizer)
    cached = _TOKEN_SET_CACHE.get(key)
    if cached is not None:
        return cached
    built = _build_token_sets(tokenizer, lexicon)
    _TOKEN_SET_CACHE[key] = built
    return built

# ───────────────────────── Small utilities ─────────────────────────
def _normalized_entropy(p: torch.Tensor) -> torch.Tensor:
    p = p.float()
    p = p / p.sum().clamp_min(1e-6)
    h = -(p * (p.clamp_min(1e-12).log())).sum()
    h_max = torch.log(torch.tensor(float(p.numel()), device=p.device))
    return (h / h_max).clamp(0.0, 1.0)

def _uniform_prior(C: int) -> torch.Tensor:
    return torch.full((C,), 1.0 / max(1, C), dtype=torch.float32)

def repetition_penalty(text: str, n: int = 2) -> float:
    toks = [t for t in (text or "").strip().split() if t]
    if len(toks) < n + 1:
        return 0.0
    grams = [" ".join(toks[i:i+n]) for i in range(len(toks) - n + 1)]
    total = len(grams)
    uniq = len(set(grams))
    rep_frac = 1.0 - (uniq / max(1, total))
    return float(min(1.0, max(0.0, rep_frac)))

# ───────────────────────── Generation glue ─────────────────────────
class _CategoryBiasProcessor:
    def __init__(self, token_bias: torch.Tensor, debug: bool = False) -> None:
        self.token_bias = token_bias
        self.debug = debug

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if scores.dim() == 1:
            out = scores + self.token_bias.to(scores.device)
        else:
            out = scores + self.token_bias.to(scores.device).unsqueeze(0).expand_as(scores)
        return out

@torch.no_grad()
def _biashead_softmax(head: LogitBiasHead, z_t: torch.Tensor, recent_texts: List[str]) -> torch.Tensor:
    device = next(head.parameters()).device if any(p.requires_grad for p in head.parameters()) else (z_t.device if z_t.is_cuda else torch.device("cpu"))
    z = z_t.detach().to(device)
    hist = make_hist_feats(recent_texts).to(device)
    s = head(z, hist).squeeze(0)
    p = torch.softmax(s, dim=-1)

    if PER_CAT_SCALES:
        scales = torch.ones_like(p)
        for i, cat in enumerate(CAT_ORDER):
            if cat in PER_CAT_SCALES:
                scales[i] = float(PER_CAT_SCALES[cat])
        p = (p * scales).clamp_min(0)
        p = p / p.sum().clamp_min(1e-6)
    return p

@torch.no_grad()
def _assemble_token_bias_from_cats(
    *,
    tokenizer: "PreTrainedTokenizerBase",
    cat_weights: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    cat2ids = _get_cat2ids_cached(tokenizer, DEFAULT_LEXICON)
    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or len(tokenizer))
    device = cat_weights.device
    token_bias = torch.zeros(vocab_size, dtype=torch.float32, device=device)
    for i, cat in enumerate(CAT_ORDER):
        w = float(cat_weights[i].item() if i < cat_weights.numel() else 0.0)
        if w <= 0:
            continue
        for tid in cat2ids.get(cat, []):
            if 0 <= tid < vocab_size:
                token_bias[tid] = token_bias[tid] + (w * scale)
    token_bias = token_bias.clamp(min=-float(BIAS_CAP), max=float(BIAS_CAP))
    return token_bias

@torch.no_grad()
def build_processor_from_cat_weights(
    *,
    tokenizer: "PreTrainedTokenizerBase",
    cat_weights: torch.Tensor,
    scale: float = 1.0,
    debug: bool = False,
) -> _CategoryBiasProcessor:
    w = cat_weights.float()
    w = (w / w.sum().clamp_min(1e-6)).clamp_min(0)
    token_bias = _assemble_token_bias_from_cats(tokenizer=tokenizer, cat_weights=w, scale=scale)
    return _CategoryBiasProcessor(token_bias=token_bias, debug=debug)

@torch.no_grad()
def _make_bias_vector(
    *,
    tokenizer: "PreTrainedTokenizerBase",
    head: LogitBiasHead,
    z_t: torch.Tensor,
    recent_texts: List[str],
    persona_effects: Optional[Dict[str, float]] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    device = next(head.parameters()).device if any(p.requires_grad for p in head.parameters()) else (z_t.device if z_t.is_cuda else torch.device("cpu"))
    z = z_t.detach().to(device)
    hist = make_hist_feats(recent_texts).to(device)
    strengths = head(z, hist).squeeze(0)

    if persona_effects:
        scale = float(persona_effects.get("accuse_bias_scale", 1.0))
        try:
            idx = CAT_ORDER.index("accuse")
            strengths[idx] = strengths[idx] * max(0.5, min(1.5, scale))
        except Exception:
            pass

    if PER_CAT_SCALES:
        s_scaled = strengths.clone()
        for i, cat in enumerate(CAT_ORDER):
            if cat in PER_CAT_SCALES:
                s_scaled[i] = s_scaled[i] * float(PER_CAT_SCALES[cat])
        strengths = s_scaled

    cat2ids = _get_cat2ids_cached(tokenizer, DEFAULT_LEXICON)
    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or len(tokenizer))
    token_bias = torch.zeros(vocab_size, dtype=torch.float32, device=device)

    cat_strengths_out: Dict[str, float] = {}
    for i, cat in enumerate(CAT_ORDER):
        s = float(strengths[i].item() if i < strengths.numel() else 0.0)
        cat_strengths_out[cat] = s
        if s == 0.0:
            continue
        for tid in cat2ids.get(cat, []):
            if 0 <= tid < vocab_size:
                token_bias[tid] = token_bias[tid] + s

    token_bias = token_bias.clamp(min=-float(BIAS_CAP), max=float(BIAS_CAP))

    p = torch.softmax(strengths, dim=-1)
    h_norm = _normalized_entropy(p)
    atten = (1.0 - float(ENTROPY_ATTEN) * float(h_norm.item()))
    atten = max(0.0, min(1.0, atten))
    token_bias = token_bias * atten

    return token_bias, cat_strengths_out

# ───────────────────────── NEW: IntentFusionProcessor ─────────────────────────
class IntentFusionProcessor(nn.Module):
    """
    α-blend of intent and bias-head category logits.
    Produces fused intent log-probs for downstream sampling, training.
    """
    def __init__(self, alpha: float = 0.6):
        super().__init__()
        self.register_buffer("_alpha", torch.tensor(float(alpha)))

    @property
    def alpha(self) -> float:
        return float(self._alpha.item())

    def forward(
        self,
        intent_logits: Optional[torch.Tensor],
        bias_logits:   Optional[torch.Tensor],
    ) -> torch.Tensor:
        if intent_logits is None and bias_logits is None:
            raise ValueError("Both intent_logits and bias_logits are None.")
        if intent_logits is None:
            return F.log_softmax(bias_logits, dim=-1)
        if bias_logits is None:
            return F.log_softmax(intent_logits, dim=-1)
        p_int  = F.softmax(intent_logits, dim=-1)
        p_bias = F.softmax(bias_logits,   dim=-1)
        p = (self._alpha * p_int) + ((1.0 - self._alpha) * p_bias)
        p = (p + 1e-8) / (p.sum(dim=-1, keepdim=True) + 1e-8)
        return torch.log(p)

# ───────────────────────── α-fusion helpers (exposed) ─────────────────────────
_FUSION_ALPHA_DEFAULT = float(os.getenv("FUSION_ALPHA", os.getenv("TALK_FUSION_ALPHA", str(FUSION_ALPHA))))

def get_fusion_alpha(default: Optional[float] = None) -> float:
    if default is not None:
        return float(default)
    return max(0.0, min(1.0, _FUSION_ALPHA_DEFAULT))

def fuse_intent_and_bias(
    intent_logits: Optional[torch.Tensor],
    bias_logits: Optional[torch.Tensor],
    *,
    alpha: Optional[float] = None,
) -> torch.Tensor:
    proc = IntentFusionProcessor(alpha=get_fusion_alpha(alpha))
    return proc(intent_logits, bias_logits)

# ───────────────────────── NEW: Two-stage SpeakerBandit ───────────────────────
class SpeakerBandit(nn.Module):
    """
    Two-stage bandit for speech:
      - cat_logits: intent category choice over templates, categories
      - arg_logits: optional argument, target choice
    """
    def __init__(
        self,
        latent_dim: int = 32,
        num_templates: int = 32,
        num_args: int = 16,
        hist_dim: int = 8,
        use_role_bit: bool = True,
        hidden: int = 128,
    ) -> None:
        super().__init__()
        in_dim = latent_dim + hist_dim + (1 if use_role_bit else 0)
        self.use_role_bit = use_role_bit
        self.hist_dim = hist_dim
        self.cat_mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, num_templates))
        self.arg_mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, num_args))

    def _assemble_input(
        self,
        z: torch.Tensor,
        role_bit: Optional[torch.Tensor],
        hist_feats: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if z.dim() == 1:
            z = z.unsqueeze(0)
        B = z.size(0)
        if self.use_role_bit:
            if role_bit is None:
                role_bit = torch.zeros((B, 1), dtype=z.dtype, device=z.device)
            elif role_bit.dim() == 1:
                role_bit = role_bit.view(B, 1)
        else:
            role_bit = None
        if hist_feats is None:
            hist_feats = torch.zeros((B, self.hist_dim), dtype=z.dtype, device=z.device)
        elif hist_feats.dim() == 1:
            hist_feats = hist_feats.view(B, -1)
        return torch.cat([t for t in (z, role_bit, hist_feats) if t is not None], dim=-1)

    def forward(
        self,
        z: torch.Tensor,
        *,
        role_bit: Optional[torch.Tensor] = None,
        hist_feats: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        x = self._assemble_input(z, role_bit, hist_feats)
        return {"cat_logits": self.cat_mlp(x), "arg_logits": self.arg_mlp(x)}

# ───────────────────────── Intent×Bias α-fusion processor (HF wrappers) ──────
class IntentBiasProcessor:
    """
    Tensor-in, tensor-out logits preprocessor:
      v' = v + (1-α) * w_bias_sparse + α * p_intent
    """
    def __init__(self, tokenizer: "PreTrainedTokenizerBase", alpha: Optional[float] = None):
        self.tok = tokenizer
        self.alpha = get_fusion_alpha(alpha)

        # Category order must match TalkHead order.
        self.cat_names = ["accuse","defend","hedge","question","vote"]

        # Tiny intent lexicons.
        self.cat2lex = {
            "accuse":   ["I", "think", "may", "be", "the", "werewolf"],
            "defend":   ["I", "don't", "think", "is", "the", "werewolf"],
            "hedge":    ["I", "might", "be", "wrong", "but"],
            "question": ["Why", "did", "do", "that", "?" ],
            "vote":     ["We", "should", "vote", "for"],
        }

        # Pre-encode token ids once into sets.
        cat2ids = {}
        for k, ws in self.cat2lex.items():
            ids: List[int] = []
            for w in ws:
                try:
                    enc = self.tok.encode(w, add_special_tokens=False) \
                          if hasattr(self.tok, "encode") else self.tok(w, add_special_tokens=False)["input_ids"]
                except Exception:
                    enc = []
                if enc:
                    ids.extend(enc)
            cat2ids[k] = set(int(i) for i in ids if isinstance(i, int))
        self.cat2ids = cat2ids

    def __call__(
        self,
        logits: torch.Tensor,
        talk_category_last: Optional[int],
        w_bias: Optional[Dict[int, float]],
    ) -> torch.Tensor:
        v = logits
        squeeze = False
        if v.dim() == 1:
            v = v.unsqueeze(0)
            squeeze = True
        B, V = int(v.size(0)), int(v.size(-1))

        # Build intent mask
        if talk_category_last is not None and 0 <= int(talk_category_last) < len(self.cat_names):
            cat = self.cat_names[int(talk_category_last)]
            ids = [i for i in self.cat2ids.get(cat, set()) if 0 <= i < V]
        else:
            ids = []
        if ids:
            intent_mask = torch.zeros(V, dtype=v.dtype, device=v.device)
            for tid in ids:
                intent_mask[tid] = 1.0
            intent_mask = intent_mask.unsqueeze(0).expand(B, V)
        else:
            intent_mask = torch.zeros_like(v)

        # Sparse bias
        if w_bias:
            bias_vec = torch.zeros(V, dtype=v.dtype, device=v.device)
            for tid, w in w_bias.items():
                ti = int(tid)
                if 0 <= ti < V:
                    bias_vec[ti] += float(w)
            bias_vec = bias_vec.unsqueeze(0).expand(B, V)
            v = v + (1.0 - self.alpha) * bias_vec

        # Intent boost
        v = v + self.alpha * intent_mask

        if squeeze:
            v = v.squeeze(0)
        return v

class HFIntentBiasProcessor:
    """
    HF-compatible logits processor wrapper around IntentBiasProcessor.
    """
    def __init__(
        self,
        tokenizer: "PreTrainedTokenizerBase",
        *,
        alpha: Optional[float] = None,
        intent_getter: Callable[[], Optional[int]],
        bias_getter:  Callable[[], Optional[Dict[int, float]]],
    ):
        self.proc = IntentBiasProcessor(tokenizer, alpha=alpha)
        self.intent_getter = intent_getter
        self.bias_getter  = bias_getter

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        return self.proc(
            scores,
            talk_category_last=self.intent_getter(),
            w_bias=self.bias_getter(),
        )

def build_alpha_fusion_processor_for_agent(
    tokenizer: "PreTrainedTokenizerBase",
    agent: Any,
    alpha: Optional[float] = None
) -> HFIntentBiasProcessor:
    return HFIntentBiasProcessor(
        tokenizer,
        alpha=alpha,
        intent_getter=lambda: getattr(agent, "talk_category_last", None),
        bias_getter=lambda: getattr(agent, "w_bias_sparse", None),
    )

def with_alpha_fusion_generate_kwargs(
    *,
    tokenizer: "PreTrainedTokenizerBase",
    agent: Any,
    alpha: Optional[float] = None,
) -> Dict[str, Any]:
    # HF processors are ignored by OpenAI, but returning them is harmless; we also add prompt steering elsewhere.
    proc = build_alpha_fusion_processor_for_agent(tokenizer, agent, alpha=alpha)
    return {"logits_processor": [proc]}

# ───────────────────────── Alive-name nudging (NEW) ─────────────────────────
def _alive_names(agent: Any) -> List[str]:
    """
    Mirror llm_script._alive_names behavior without importing it:
    pull living agent .name strings from agent._agents_view.
    """
    try:
        view = getattr(agent, "_agents_view", None)
        if view is None:
            return []
        return [x.name for x in view if getattr(x, "alive", False)]
    except Exception:
        return []

class _AliveNameBoostProcessor:
    """
    Small, positive additive bias on token ids that compose each alive Agent_* name.
    Optionally gives a slightly higher bias to `named_target`.
    Applies for the first `max_steps` decoding steps by default.
    """
    def __init__(
        self,
        tokenizer: "PreTrainedTokenizerBase",
        alive_names: List[str],
        named_target: Optional[str] = None,
        *,
        base_boost: float = 0.9,
        target_extra: float = 0.25,
        max_steps: int = 4,
    ) -> None:
        self.max_steps = int(max_steps)
        self._step = 0
        self.base = float(base_boost)
        self.t_extra = float(target_extra)
        self.vocab_boost: Dict[int, float] = {}

        # Pre-tokenize each alive name into its token id sequence.
        def _encode(txt: str) -> List[int]:
            try:
                return tokenizer.encode(txt, add_special_tokens=False)  # type: ignore[attr-defined]
            except Exception:
                try:
                    return tokenizer(txt, add_special_tokens=False)["input_ids"]  # type: ignore[index]
                except Exception:
                    return []

        alive_set = list(dict.fromkeys([s for s in (alive_names or []) if isinstance(s, str) and s.strip()]))
        target_ids = _encode(named_target) if (named_target and (named_target in alive_set)) else None

        # Build a sparse bias map once.
        for nm in alive_set:
            ids = _encode(nm)
            if not ids:
                continue
            boost = self.base
            # If this name is the target, add the extra.
            if target_ids and ids == target_ids:
                boost = self.base + self.t_extra
            for tid in ids:
                if isinstance(tid, int):
                    self.vocab_boost[tid] = max(self.vocab_boost.get(tid, 0.0), float(boost))

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Stop after a few positions to avoid overwhelming style tokens later.
        if self._step >= self.max_steps or not self.vocab_boost:
            self._step += 1
            return scores
        V = scores.size(-1) if scores.dim() > 1 else scores.numel()
        # Robust, explicit indexing across 1D or 2D tensors (fix requested)
        for tid, w in self.vocab_boost.items():
            ti = int(tid)
            if 0 <= ti < V:
                scores[..., ti] = scores[..., ti] + float(w)
        self._step += 1
        return scores

# ───────────────────────── Public APIs (existing) ─────────────────────────
def with_logit_bias_generate_kwargs(
    *,
    tokenizer: "PreTrainedTokenizerBase",
    head: LogitBiasHead,
    z_t: torch.Tensor,
    role: Optional[str],
    recent_texts: List[str],
    persona_effects: Optional[Dict[str, float]] = None,
    # NEW (optional): pass-throughs for alive-name nudging
    agent: Any = None,
    named_target: Optional[str] = None,
) -> Dict[str, Any]:
    # OpenAI: processors will be ignored by the provider, but call sites expect this API.
    # We still return processors for HF; for OpenAI, prompt steering is injected elsewhere.
    token_bias, cat_strengths = _make_bias_vector(
        tokenizer=tokenizer,
        head=head,
        z_t=z_t,
        recent_texts=recent_texts[-SPEAKER_HIST_K:] if SPEAKER_HIST_K > 0 else recent_texts,
        persona_effects=persona_effects,
    )

    if LLM_SPK_DEBUG:
        try:
            dbg = {
                "mode": "bias-only",
                "alpha": None,
                "role": role,
                "recent_texts": recent_texts[-SPEAKER_HIST_K:],
                "cat_strengths": {k: round(float(v), 3) for k, v in cat_strengths.items()},
            }
            print("[LLM-SPK]", json.dumps(dbg))
        except Exception:
            pass

    lp: List[Callable] = []
    try:
        alive = _alive_names(agent) if agent is not None else []
    except Exception:
        alive = []
    if alive:
        lp.append(_AliveNameBoostProcessor(
            tokenizer, alive, named_target,
            base_boost=0.9, target_extra=0.25, max_steps=4
        ))
    lp.append(_CategoryBiasProcessor(token_bias=token_bias, debug=LLM_SPK_DEBUG))
    lp.append(_EarlyOpenerDownbiasProcessor(tokenizer))
    return {"logits_processor": lp}

def with_fused_bias_generate_kwargs(
    *,
    tokenizer: "PreTrainedTokenizerBase",
    head: LogitBiasHead,
    z_t: torch.Tensor,
    talkhead_probs: Optional[torch.Tensor],
    alpha: Optional[float] = None,
    role: Optional[str] = None,
    recent_texts: Optional[List[str]] = None,
    persona_effects: Optional[Dict[str, float]] = None,
    # NEW (optional): pass-throughs for alive-name nudging
    agent: Any = None,
    named_target: Optional[str] = None,
) -> Dict[str, Any]:
    if recent_texts is None: recent_texts = []
    a = get_fusion_alpha(alpha)

    bh_p = _biashead_softmax(head, z_t, recent_texts[-SPEAKER_HIST_K:] if SPEAKER_HIST_K > 0 else recent_texts)

    if persona_effects:
        try:
            idx = CAT_ORDER.index("accuse")
            scale = max(0.5, min(1.5, float(persona_effects.get("accuse_bias_scale", 1.0))))
            bh_p = bh_p.clone()
            bh_p[idx] = bh_p[idx] * scale
            bh_p = bh_p / bh_p.sum().clamp_min(1e-6)
        except Exception:
            pass

    if talkhead_probs is not None:
        th = talkhead_probs.detach().float()
        th = th / th.sum().clamp_min(1e-6)
    else:
        th = None

    if th is None:
        w_total = bh_p
    elif a >= 1.0:
        w_total = th
    elif a <= 0.0:
        w_total = bh_p
    else:
        w_total = a * th.to(bh_p.device) + (1.0 - a) * bh_p

    h_norm = _normalized_entropy(w_total)
    atten  = (1.0 - float(ENTROPY_ATTEN) * float(h_norm.item()))
    atten  = max(0.0, min(1.0, atten))

    token_bias = _assemble_token_bias_from_cats(
        tokenizer=tokenizer,
        cat_weights=w_total,
        scale=float(BASE_STRENGTH) * atten,
    )

    if LLM_SPK_DEBUG:
        try:
            dbg = {
                "mode": "fused",
                "alpha": round(a, 3),
                "role": role,
                "recent_texts": recent_texts[-SPEAKER_HIST_K:],
                "talkhead_probs": (
                    {k: round(float(v), 3) for k, v in zip(CAT_ORDER, th.tolist())}
                    if th is not None else None
                ),
                "biashead_probs": {k: round(float(v), 3) for k, v in zip(CAT_ORDER, bh_p.tolist())},
                "entropy_norm": round(float(h_norm.item()), 3),
                "atten": round(float(atten), 3),
                "top_fused": sorted(
                    ((CAT_ORDER[i], float(w_total[i])) for i in range(len(CAT_ORDER))),
                    key=lambda kv: -kv[1]
                )[:2],
            }
            print("[LLM-SPK]", json.dumps(dbg))
        except Exception:
            pass

    lp: List[Callable] = []
    try:
        alive = _alive_names(agent) if agent is not None else []
    except Exception:
        alive = []
    if alive:
        lp.append(_AliveNameBoostProcessor(
            tokenizer, alive, named_target,
            base_boost=0.9, target_extra=0.25, max_steps=4
        ))
    lp.append(_CategoryBiasProcessor(token_bias=token_bias, debug=LLM_SPK_DEBUG))
    lp.append(_EarlyOpenerDownbiasProcessor(tokenizer))
    return {"logits_processor": lp}

# ───────────────────────── NEW: Role×Phase Prompt Pack ───────────────────────
ROLE_PHASE_CONTRACT = {
    "Villager": {
        "DISCUSS": "You are a Villager in a hidden-role game. Never reveal hidden information. Speak 1–2 short, natural sentences. Use agent names (e.g., Agent_3). Avoid meta talk or rules discussion.",
        "VOTE":    "You are a Villager. State or justify a vote in 1–2 sentences. Use a name. Do not claim knowledge of night actions.",
        "NIGHT":   "You are a Villager at night. You do not act and you do not know others' roles. Keep thoughts private as a single short line if needed.",
    },
    "Werewolf": {
        "DISCUSS": "You are a Werewolf speaking in daytime discussion. Conceal your role. 1–2 short sentences. Use names. Do not refer to 'we wolves' or night actions.",
        "VOTE":    "You are a Werewolf voting in public. Hide your role. 1–2 sentences with a named target. No explicit pack references.",
        "NIGHT":   "You are a Werewolf in private night chat with packmates. Keep language oblique; do not explicitly say 'we are wolves'. 1–2 short sentences.",
    },
    "Unknown": {
        "DISCUSS": "You are a player in a hidden-role game. 1–2 sentences. Use names. Avoid revealing hidden info or meta talk.",
        "VOTE":    "You are voting in a hidden-role game. 1–2 sentences, include a name. No hidden info claims.",
        "NIGHT":   "Night phase. Keep to 1–2 sentences. No explicit role claims.",
    }
}

EXEMPLARS: Dict[str, Dict[str, List[str]]] = {
    "accuse": {
        "DISCUSS": [
            "Agent_4 keeps dodging simple questions.",
            "I think Agent_2 is steering us away from Agent_5.",
        ],
        "VOTE": [
            "I'm ready to vote Agent_3 based on their contradictions.",
            "My vote is Agent_1; their story doesn't line up.",
        ],
        "NIGHT": [
            "Agent_6 drew too much heat today; risky.",
        ],
    },
    "defend": {
        "DISCUSS": [
            "Agent_5's timeline checks out to me.",
            "I don't think Agent_3 is the werewolf this round.",
        ],
        "VOTE": [
            "Before we vote, note that Agent_2 answered cleanly.",
        ],
        "NIGHT": [
            "Clearing Agent_1 publicly might build trust.",
        ],
    },
    "hedge": {
        "DISCUSS": [
            "I'm not sure; Agent_4 feels off but it's thin.",
            "Maybe we should hear from Agent_2 again.",
        ],
        "VOTE": [
            "If we must vote, I'm leaning Agent_5 but open to argument.",
        ],
        "NIGHT": [
            "Not certain where the town will land tomorrow.",
        ],
    },
    "question": {
        "DISCUSS": [
            "Agent_3, why did you change your vote so fast?",
            "Agent_1, can you explain your read on Agent_5?",
        ],
        "VOTE": [
            "Before we lock in, Agent_2, what made you switch?",
        ],
        "NIGHT": [
            "Do we expect Agent_4 to push back tomorrow?",
        ],
    },
    "vote": {
        "DISCUSS": [
            "We should vote Agent_2 today.",
            "I want to formalize a vote on Agent_5.",
        ],
        "VOTE": [
            "My vote is Agent_4.",
            "Locking my vote on Agent_3.",
        ],
        "NIGHT": [
            "Publicly parking on Agent_6 might sell it.",
        ],
    },
}

def _summarize_dialog_state(dialog_state: Any, max_items: int = 3) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {"claims": [], "questions": []}
    try:
        snap = dialog_state.snapshot() if hasattr(dialog_state, "snapshot") else (dialog_state or {})
    except Exception:
        snap = {}
    claims = list(reversed(snap.get("recent_claims", [])))[:max_items]
    qs     = list(reversed(snap.get("open_questions", [])))[:max_items]
    for r, speaker, about, stance in claims:
        if about:
            if stance == "accuse":
                out["claims"].append(f"{speaker} accused {about}")
            elif stance == "defend":
                out["claims"].append(f"{speaker} defended {about}")
            else:
                out["claims"].append(f"{speaker} expressed doubt about {about}")
    for r, asker, to_whom in qs:
        if to_whom:
            out["questions"].append(f"{asker} asked {to_whom}")
        else:
            out["questions"].append(f"{asker} asked the group")
    return out

def build_role_phase_prompt(
    *,
    role: str,
    phase: str,
    intent: str,
    plan: Dict[str, Any],
    recent_texts: List[str],
    dialog_state: Any,
    self_name: str = "Agent_0",
) -> str:
    role = role if role in ROLE_PHASE_CONTRACT else "Unknown"
    phase = phase in ("DISCUSS", "VOTE", "NIGHT") and phase or "DISCUSS"
    intent = _norm_intent(intent)

    contract = ROLE_PHASE_CONTRACT[role][phase]
    ex = EXEMPLARS.get(intent, {}).get(phase, [])
    ex_str = "\n".join(f"- {s}" for s in ex[:3])

    ds = _summarize_dialog_state(dialog_state)
    claims = "; ".join(ds["claims"]) if ds["claims"] else "none"
    questions = "; ".join(ds["questions"]) if ds["questions"] else "none"

    tgt = plan.get("target")
    shape = plan.get("shape", "")
    plan_str = f"intent={intent}; target={tgt or 'none'}; shape={shape}" if intent else "intent=hedge"

    recent = "\n".join(f"- {t.strip()}" for t in recent_texts[-3:] if t and t.strip())

    prompt = (
        f"[SYSTEM]\n{contract}\n"
        f"[CONTEXT]\n"
        f"- You are {self_name}.\n"
        f"- Plan: {plan_str}\n"
        f"- Recent claims: {claims}\n"
        f"- Open questions: {questions}\n"
        f"- Recent dialog:\n{recent if recent else '- (no recent messages)'}\n"
        f"[STYLE]\nUse plain, in-world language. 1–2 sentences. Mention names when referring to others. Avoid generic filler.\n"
        f"[EXEMPLARS/{intent.upper()}]\n{ex_str if ex_str else '- (none)'}\n"
        f"[YOU]\n"
    )
    return prompt

# 1.4 Style tag (persona) helper
def _style_tag(persona: dict | None) -> str:
    if not isinstance(persona, dict):
        return ""
    v = float(persona.get("verbosity", 0.5))
    a = float(persona.get("assertiveness", 0.5))
    h = float(persona.get("hedging", 0.5))
    p = float(persona.get("politeness", 0.5))
    return f"[STYLE] verbosity={v:.2f} assertiveness={a:.2f} hedging={h:.2f} politeness={p:.2f}\n"

# ───────────────────────── Decoding-time controls ─────────────────────────────
class BigramPenaltyProcessor:
    def __init__(
        self,
        tokenizer: "PreTrainedTokenizerBase",
        dialog_state: Any,
        penalty: float = 0.6,
        max_bigrams: int = 200,
    ) -> None:
        self.tok = tokenizer
        self.penalty = float(max(0.0, min(1.5, penalty)))
        try:
            used_raw: List[Union[str, Tuple[str, str]]] = list(getattr(dialog_state, "used_bigrams", []))
            used_raw = used_raw[-max_bigrams:] if max_bigrams else used_raw
        except Exception:
            used_raw = []
        used: List[Tuple[str, str]] = []
        for item in used_raw:
            if isinstance(item, tuple) and len(item) == 2:
                a, b = item
                if a and b:
                    used.append((str(a).strip(), str(b).strip()))
            elif isinstance(item, str):
                parts = item.strip().split()
                if len(parts) >= 2:
                    used.append((" ".join(parts[:-1]), parts[-1]))
        self.second_to_firsts: Dict[str, set] = {}
        for a, b in used:
            self.second_to_firsts.setdefault(b, set()).add(a)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if scores.dim() == 1:
            return scores
        prev_id = int(input_ids[0, -1].item()) if input_ids.numel() > 0 else None
        if prev_id is None:
            return scores
        try:
            prev_text = self.tok.decode([prev_id]).strip()
        except Exception:
            return scores

        V = scores.size(-1)
        if not prev_text:
            return scores
        for tid in range(V):
            try:
                t = self.tok.decode([tid]).strip()
            except Exception:
                continue
            if not t:
                continue
            if t in self.second_to_firsts and prev_text in self.second_to_firsts[t]:
                scores[..., tid] = scores[..., tid] - self.penalty
        return scores

def has_trigram_repetition(text: str) -> bool:
    toks = [t for t in re.findall(r"\w+|\S", text or "") if t.strip()]
    if len(toks) < 6:
        return False
    trigrams = [" ".join(toks[i:i+3]).lower() for i in range(len(toks)-2)]
    seen = set()
    for tri in trigrams:
        if tri in seen:
            return True
        seen.add(tri)
    return False

SYNONYM_MAP = {
    "think": ["suspect", "feel", "reckon"],
    "is":    ["seems", "looks", "appears"],
    "very":  ["quite", "rather"],
    "maybe": ["perhaps", "possibly"],
    "vote":  ["pick", "choose"],
    "because": ["since", "as"],
    "guilty": ["shady", "off"],
    "innocent": ["clean", "fine"],
}

def add_synonym_nudge_bias(
    tokenizer: "PreTrainedTokenizerBase",
    base_kwargs: Dict[str, Any],
    recent_text: str,
    scale: float = 0.8,
) -> Dict[str, Any]:
    if "logits_processor" not in base_kwargs:
        base_kwargs = dict(base_kwargs)
        base_kwargs["logits_processor"] = []
    lp = list(base_kwargs["logits_processor"])

    toks = set(re.findall(r"[A-Za-z']+", recent_text.lower()))
    boost_ids: Dict[int, float] = {}
    for k, syns in SYNONYM_MAP.items():
        if k in toks:
            for s in syns:
                try:
                    enc = tokenizer(s, add_special_tokens=False)["input_ids"]
                except Exception:
                    enc = []
                if not enc:
                    continue
                tid = int(enc[0])
                boost_ids[tid] = boost_ids.get(tid, 0.0) + scale

    if boost_ids:
        class _SynBoost:
            def __init__(self, biases: Dict[int, float]): self.biases = biases
            def __call__(self, input_ids, scores):
                for tid, w in self.biases.items():
                    if 0 <= tid < scores.size(-1):
                        scores[..., tid] = scores[..., tid] + float(w)
                return scores
        lp.append(_SynBoost(boost_ids))

    new_kwargs = dict(base_kwargs)
    new_kwargs["logits_processor"] = lp
    return new_kwargs

# ───────────────────────── Guard and Shape (mandatory) ───────────────────────
_MARKERS = {
    "hedges": ["maybe","perhaps","possibly","might","could","seems","appears","not sure","unsure","I think","I feel"],
    "intensifiers": ["very","extremely","definitely","certainly","absolutely","totally","really"],
    "discourse": ["well", "so", "look", "listen", "anyway", "like"],
}

def normalize_contractions(text: str) -> str:
    repl = {
        "don’t": "don't", "can’t": "can't", "won’t": "won't",
        "it’s": "it's", "I’m": "I'm", "I’ve": "I've", "you’re": "you're",
        "they’re": "they're", "we’re": "we're",
    }
    out = unicodedata.normalize("NFKC", text or "")
    for k, v in repl.items():
        out = out.replace(k, v)
    out = re.sub(r"\s+", " ", out).strip()
    return out

def strip_parentheticals(text: str) -> str:
    return re.sub(r"\s*\([^)]*\)\s*", " ", text or "").strip()

def split_sentences(text: str) -> List[str]:
    sents = re.split(r"(?<=[\.\?\!])\s+", (text or "").strip())
    sents = [s.strip() for s in sents if s.strip()]
    return sents

def leaks_hidden_info(text: str, role: str, phase: str) -> bool:
    t = (text or "").lower()
    if "we wolves" in t or "my partner" in t:
        return True
    if "i saw" in t and "kill" in t:
        return True
    if "at night i" in t:
        return True
    if role.lower().startswith("villag"):
        if "i know" in t and ("wolf" in t or "werewolf" in t):
            return True
        if "my pack" in t:
            return True
    if role.lower().startswith("werewolf") and phase == "DISCUSS":
        if "we wolves" in t or "my partner" in t or "at night" in t:
            return True
    return False

def strengthen_neutral(text: str) -> str:
    t = text.strip()
    t = re.sub(r"\bI know\b", "I suspect", t, flags=re.IGNORECASE)
    t = re.sub(r"\bdefinitely\b", "probably", t, flags=re.IGNORECASE)
    return t

def inject_target(text: str, tgt: str, intent: str, shape: str = "") -> str:
    t = text.strip()
    name = tgt
    if intent == "accuse":
        if name not in t:
            t = f"{name} {(' ' + t) if t else 'looks off to me.'}"
    elif intent == "defend":
        if name not in t:
            t = f"I don't think {name} is the werewolf."
    elif intent == "vote":
        if name not in t:
            t = f"My vote is {name}."
    return t

def count_markers(text: str) -> Dict[str, int]:
    low = (text or "").lower()
    counts = {k:0 for k in _MARKERS}
    for k, words in _MARKERS.items():
        for w in words:
            counts[k] += low.count(w)
    return counts

def _prune_listed(words: List[str], text: str, cap: int) -> str:
    low = text
    if cap is None or cap < 0:
        return low
    count = 0
    def repl(m):
        nonlocal count
        count += 1
        return m.group(0) if count <= cap else ""
    for w in words:
        low = re.sub(rf"\b{re.escape(w)}\b", repl, low, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", low).strip()

def prune_markers(text: str, cfg) -> str:
    hedges_cap = getattr(cfg, "hedges_cap_per5", 3)
    intens_cap = getattr(cfg, "intensifiers_cap_per5", 2)
    out = _prune_listed(_MARKERS["hedges"], text, int(hedges_cap))
    out = _prune_listed(_MARKERS["intensifiers"], out, int(intens_cap))
    disc_cap = getattr(cfg, "discourse_markers_cap_per5", None)
    if disc_cap is not None:
        out = _prune_listed(_MARKERS["discourse"], out, int(disc_cap))
    return out

def guard_and_shape(text: str, plan: dict, role: str, phase: str, cfg) -> tuple[str, dict]:
    """
    Sanitize without collapsing to an empty string.
    If all guards strip the content, return SAFE_FALLBACK.
    Never return '...'.
    """
    # Normalize early to reduce tokenizer quirks
    text = normalize_contractions(strip_parentheticals(text_raw := (text or "")))
    # Clean leaked prefixes and surface artifacts BEFORE any rejection
    text = _clean_surface_artifacts(text)
    text = text.strip()

    violated = False

    # Keep to 1–2 sentences
    sents = split_sentences(text)
    if len(sents) == 0:
        text = ""
    elif len(sents) > 2:
        violated = True
        text = " ".join(sents[:2])
    else:
        text = " ".join(sents)

    # Hidden info checks — soften rather than blanking
    if text and leaks_hidden_info(text, role, phase):
        violated = True
        text = strengthen_neutral(text)

    # Ensure target mention for accuse, defend, vote
    try:
        intent = _norm_intent(str(plan.get("intent")))
    except Exception:
        intent = "hedge"
    if intent in {"accuse", "defend", "vote"}:
        tgt = plan.get("target")
        if tgt:
            if tgt not in text:
                violated = True
                text = inject_target(text, tgt, intent, plan.get("shape", ""))

    # Configurable banned bigrams at the opener
    banned = getattr(cfg, "banned_bigrams", []) or []
    if text:
        for bb in banned:
            if text.strip().startsWith(bb):  # type: ignore[attr-defined]
                violated = True
                parts = text.strip().split(maxsplit=1)
                text = parts[1] if len(parts) == 2 else text
                break

    # Style caps
    if text:
        pruned = prune_markers(text, cfg)
        if pruned != text:
            violated = True
            text = pruned

    # Trigram repetition
    if text and getattr(cfg, "trigram_veto", True) and has_trigram_repetition(text):
        violated = True
        sents = split_sentences(text)
        if len(sents) >= 2:
            text = sents[0]

    # Final safety: never return empty
    safe_text = text if text else SAFE_FALLBACK

    return safe_text, {
        "violated": bool(violated),
        "redo": False,
        "repetition_penalty": None
    }

# ───────────────────────── Simple config capsule for guards ───────────────────
class GuardCfg:
    def __init__(self, hedges_cap_per5: int = 3, intensifiers_cap_per5: int = 2):
        self.hedges_cap_per5 = int(hedges_cap_per5)
        self.intensifiers_cap_per5 = int(intensifiers_cap_per5)

# ───────────────────────── OpenAI steering helpers (NEW) ──────────────────────
def _steer_block_from_plan(intent: str, plan: Dict[str, Any], alive: List[str]) -> str:
    tgt = plan.get("target")
    shape = plan.get("shape", "")
    alive_join = ", ".join(alive) if alive else "Agent_0 … Agent_9"
    hints = []
    if intent == "question":
        hints.append("Ask a concrete question to a named player.")
    if intent == "accuse":
        hints.append("State a specific behavior as the reason.")
    if intent == "defend":
        hints.append("Give a concise justification for innocence.")
    if intent == "vote":
        hints.append("Explicitly include a named vote target.")
    # gentle anti-boilerplate
    hints.append("Avoid boilerplate openers like 'I believe' or 'Based on'.")
    hints.append("No meta talk; stay in-world.")
    h = "; ".join(hints)
    return (
        "[STEER]\n"
        f"- intent={intent}\n"
        f"- target={tgt or 'none'}; shape={shape or 'plain'}\n"
        f"- valid_names: {alive_join}\n"
        f"- guidance: {h}\n"
    )

def _inject_openai_steer(prompt: str, *, intent: str, plan: Dict[str, Any], agent: Any) -> str:
    try:
        alive = _alive_names(agent)
    except Exception:
        alive = []
    steer = _steer_block_from_plan(intent, plan, alive)
    return steer + prompt

# ───────────────────────── Convenience: build prompt and controls ────────────
def build_prompt_and_controls(
    *,
    tokenizer: "PreTrainedTokenizerBase",
    role: str,
    phase: str,
    intent: str,
    plan: Dict[str, Any],
    recent_texts: List[str],
    dialog_state: Any,
    self_name: str,
    base_generate_kwargs: Optional[Dict[str, Any]] = None,
    bigram_penalty: Optional[float] = None,
) -> Tuple[str, Dict[str, Any]]:
    intent = _norm_intent(intent)
    prompt = build_role_phase_prompt(
        role=role, phase=phase, intent=intent, plan=plan,
        recent_texts=recent_texts, dialog_state=dialog_state, self_name=self_name,
    )
    kw = dict(base_generate_kwargs or {})
    bp = LANGCFG_DEFAULT.bigram_penalty if bigram_penalty is None else float(bigram_penalty)

    # HF-only logits processor (OpenAI will ignore; we leave it in for HF parity)
    if not _is_openai_tok(tokenizer) and bp and bp > 0.0:
        procs = list(kw.get("logits_processor", []))
        procs.append(BigramPenaltyProcessor(tokenizer, dialog_state, penalty=bp))
        kw["logits_processor"] = procs

    return prompt, kw

def maybe_second_pass_kwargs(
    *,
    tokenizer: "PreTrainedTokenizerBase",
    first_text: str,
    first_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    if not has_trigram_repetition(first_text):
        return first_kwargs
    return add_synonym_nudge_bias(tokenizer, first_kwargs, recent_text=first_text, scale=0.8)

# ───────────────────────── NEW: persona-aware question boost (per-call) ───────
_QUESTION_IDS_CACHE: Dict[str, List[int]] = {}

def _get_question_token_ids(tokenizer: "PreTrainedTokenizerBase") -> List[int]:
    key = _tok_cache_key(tokenizer) + "|question"
    cached = _QUESTION_IDS_CACHE.get(key)
    if cached is not None:
        return cached
    ids: List[int] = []
    for w in DEFAULT_LEXICON.get("question", ["why","how","what","who","where","when","?"]):
        try:
            enc = tokenizer(w, add_special_tokens=False)["input_ids"]
        except Exception:
            enc = []
        if enc:
            ids.append(int(enc[0]))
    _QUESTION_IDS_CACHE[key] = sorted(list({i for i in ids if isinstance(i, int)}))
    return _QUESTION_IDS_CACHE[key]

class _QuestionBoostProcessor:
    """Light, per-call bias toward question-style tokens."""
    def __init__(self, tokenizer: "PreTrainedTokenizerBase", boost: float = 0.35) -> None:
        self.ids = _get_question_token_ids(tokenizer)
        self.boost = float(max(0.0, boost))

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.boost <= 0.0 or not self.ids:
            return scores
        # Robust index across 1D or 2D
        for tid in self.ids:
            if 0 <= tid < (scores.size(-1) if scores.dim() > 1 else scores.numel()):
                scores[..., tid] = scores[..., tid] + self.boost
        return scores

# ───────────────────────── NEW: tiny first-3-token down-bias for bad openers ──
_OPENER_IDS_CACHE: Dict[str, List[int]] = {}

def _get_opener_token_ids(tokenizer: "PreTrainedTokenizerBase") -> List[int]:
    key = _tok_cache_key(tokenizer) + "|openers"
    cached = _OPENER_IDS_CACHE.get(key)
    if cached is not None:
        return cached
    toks = ["Based", "based", "I", "believe", "Believe"]
    ids: List[int] = []
    for w in toks:
        try:
            enc = tokenizer(w, add_special_tokens=False)["input_ids"]
        except Exception:
            enc = []
        if enc:
            ids.append(int(enc[0]))
    # Deduplicate
    ids = sorted(list({i for i in ids if isinstance(i, int)}))
    _OPENER_IDS_CACHE[key] = ids
    return ids

class _EarlyOpenerDownbiasProcessor:
    """
    Apply a small down-bias to a handful of boilerplate opener tokens,
    but only for the first three generated positions of THIS generation call.
    """
    def __init__(self, tokenizer: "PreTrainedTokenizerBase", penalty: float = 0.35, max_steps: int = 3):
        self.ids = _get_opener_token_ids(tokenizer)
        self.penalty = float(max(0.0, penalty))
        self.max_steps = int(max_steps)
        self._step = 0  # counts generated steps within this call

    def __repr__(self) -> str:
        return f"_EarlyOpenerDownbiasProcessor(penalty={self.penalty}, max_steps={self.max_steps})"

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Only affect the first few generated positions
        if self._step >= self.max_steps or not self.ids or self.penalty <= 0.0:
            self._step += 1
            return scores
        # Robust index across 1D or 2D (fix requested)
        for tid in self.ids:
            if 0 <= tid < (scores.size(-1) if scores.dim() > 1 else scores.numel()):
                scores[..., tid] = scores[..., tid] - self.penalty
        self._step += 1
        return scores

# ───────────────────────── High-level mouthpiece: generate ────────────────────
def _phase_from_code(code: str) -> str:
    t = (code or "").upper()
    if "VOTE" in t:
        return "VOTE"
    if "NIGHT" in t:
        return "NIGHT"
    return "DISCUSS"

def _too_short(text: str, min_words: int) -> bool:
    return len((text or "").strip().split()) < max(1, int(min_words))

# NEW: final safety patch to enforce concrete target name inside text
def _final_target_patch(text: str, target: Optional[str]) -> str:
    if not target or "Agent_" not in target:
        return text
    t = text or ""
    # Replace a single bare 'Agent' (or possessive) with the exact target
    t, n1 = re.subn(r"\bAgent\b", target, t, count=1)
    if n1 == 0:
        t = re.sub(r"\bAgent('s)\b", rf"{target}\1", t, count=1)
    # Also patch 'Agent,' / 'Agent:' / 'Agent;' one time if present
    t = re.sub(r"\bAgent\b([,:;])", rf"{target}\1", t, count=1)
    # Repair any leftover 'Agent 3' artifacts that slipped through
    t = re.sub(r"\bAgent\s+([0-9])\b", r"Agent_\1", t)
    t = re.sub(r"\bAgent\s+([0-9])('s\b)", r"Agent_\1\2", t)
    return t

def generate(
    z_t,                              # latent, unused here but kept for parity
    role: str,
    phase_code: str,
    plan: Dict[str, Any],
    dialog_state: Any,
    hygiene_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Produce one utterance with light hygiene and retries.
    Contract expected by sim.py:
        text_raw, gen_meta = generate(z_t, role, phase_code, plan, dialog_state, HYGIENE_NS)
    """
    # Deferred import to avoid heavy deps at import time
    try:
        from llm_script import llm_fn_from_env
    except Exception as e:
        # Absolute fallback to avoid crashing the sim loop
        return SAFE_FALLBACK, {"error": f"llm_script import failed: {e}"}

    phase = _phase_from_code(phase_code)
    intent = _norm_intent(str(plan.get("intent")))

    # Config knobs
    cfg_redo_max = getattr(LANGCFG_DEFAULT, "redo_max", 1)
    cfg_min_words = getattr(LANGCFG_DEFAULT, "min_words", 12)
    # Merge env overrides with config
    redo_budget = max(int(cfg_redo_max), int(_REDO_MAX))
    min_words = max(int(cfg_min_words), int(_MIN_WORDS))

    # Build LLM mouth and tokenizer
    mouth = llm_fn_from_env()
    tok = getattr(mouth, "tokenizer", None)
    # NEW: if OpenAI mouth has no tokenizer, attach a shim so downstream logic works.
    if tok is None:
        tok = _maybe_attach_openai_shim(mouth)

    # Pull recent texts from dialog_state if available
    recent = getattr(dialog_state, "recent_texts", []) if hasattr(dialog_state, "recent_texts") else []

    # Base generate kwargs via existing helper
    prompt, base_kwargs = build_prompt_and_controls(
        tokenizer=tok,
        role=role,
        phase=phase,
        intent=intent,
        plan=plan,
        recent_texts=recent,
        dialog_state=dialog_state,
        self_name=str(plan.get("self_name", "Agent_0")),
        bigram_penalty=getattr(LANGCFG_DEFAULT, "bigram_penalty", 0.6),
    )

    # 1.4 Style tag: prepend persona style (non-binding)
    persona_dict = plan.get("persona") or plan.get("persona_style") or {}
    prompt = _style_tag(persona_dict) + prompt

    # If the tokenizer is an OpenAI shim, inject a compact [STEER] block to emulate bias-head nudges.
    if _is_openai_tok(tok):
        prompt = _inject_openai_steer(prompt, intent=intent, plan=plan, agent=plan.get("agent_obj"))

    # NEW: per-call persona style adapter
    persona_style = {}
    try:
        ps = plan.get("persona_style", {})  # expected injection site
        if isinstance(ps, dict):
            persona_style = ps
    except Exception:
        persona_style = {}

    kwargs = dict(base_kwargs)
    # 1.2 Respect repetition defaults
    kwargs.setdefault("no_repeat_ngram_size", _DEFAULT_NGRAM)
    kwargs.setdefault("repetition_penalty", _DEFAULT_REP)

    # If prefer_question is set, modestly bias toward question tokens for THIS call only.
    # Interprets "force_question_prob = 0.35" as a small additive bias magnitude.
    if bool(persona_style.get("prefer_question", False)) and not _is_openai_tok(tok):
        lp = list(kwargs.get("logits_processor", []))
        lp.append(_QuestionBoostProcessor(tok, boost=0.35))
        kwargs["logits_processor"] = lp

    # Post-decode acceptance loop (1.3)
    attempts = 0
    did_boilerplate_resample = False
    tgt = plan.get("target") if isinstance(plan, dict) else None

    while True:
        # Guarantee raw is always initialized, even if the mouthpiece throws before assignment
        raw = SAFE_FALLBACK
        try:
            raw = mouth(prompt, generate_kwargs=kwargs)
        except Exception as e:
            log_exc("MOUTHPIECE", e)
            # keep SAFE_FALLBACK in raw

        # P7HOT_RESAMPLE_HOOK
        try:
            raw = early_stop_text(raw) if 'early_stop_text' in globals() else raw
            raw = normalize_utterance(raw)
        except Exception:
            pass
        try:
            tgt = plan.get('target') if isinstance(plan, dict) else None
            if 'ensure_target_and_reason' in globals() and tgt:
                raw = ensure_target_and_reason(raw, tgt)
        except Exception:
            pass
        try:
            if 'looks_boilerplate' in globals() and looks_boilerplate(raw):
                t = float(kwargs.get('temperature', 0.7))
                kwargs = dict(kwargs); kwargs['temperature'] = min(0.95, t + 0.10)
        except Exception:
            pass
        # Early trimming and normalization
        try:
            raw = early_stop_text(raw) if 'early_stop_text' in globals() else raw
            raw = normalize_utterance(raw)
            raw = _clean_surface_artifacts(raw)
            raw = _ensure_target_and_reason(raw, tgt)
        except Exception:
            pass

        # Acceptance gate BEFORE heavy shaping: min words + target + allowed names
        if _acceptable_utterance(raw, min_words=min_words, target=tgt):
            # Proceed to final shaping and safety
            shaped, gmeta = guard_and_shape(raw or SAFE_FALLBACK, plan=plan, role=role, phase=phase, cfg=LANGCFG_DEFAULT)
            shaped = _final_target_patch(shaped, tgt)

            meta = {
                "phase": phase,
                "intent": intent,
                "redo_try": attempts,
                "guard_meta": gmeta,
                "prompt_len": len(prompt),
                "boilerplate_resample": did_boilerplate_resample,
                "persona_style": persona_style,
                "provider": ("openai" if _is_openai_tok(tok) else "hf"),
            }

            # One-shot resample if boilerplate opener or banned bigram detected
            if (looks_boilerplate(shaped) or any(
                shaped.strip().startswith(bb) for bb in getattr(LANGCFG_DEFAULT, "banned_bigrams", [])
            )) and not did_boilerplate_resample and attempts < redo_budget:
                did_boilerplate_resample = True
                t = float(kwargs.get("temperature", 0.75))
                kwargs = dict(kwargs)
                kwargs["temperature"] = min(0.95, t + 0.10)
                kwargs["top_p"] = min(0.95, float(kwargs.get("top_p", 0.9)) + 0.03)
                attempts += 1
                continue

            # Final exit conditions
            if not looks_meta(shaped) and not _too_short(shaped, max(4, min_words // 2)):
                return shaped, meta

            # If shaped fails soft checks, attempt a synonym nudge if budget remains (HF only)
            if attempts < redo_budget and not _is_openai_tok(tok):
                kwargs = maybe_second_pass_kwargs(tokenizer=tok, first_text=shaped or raw or SAFE_FALLBACK, first_kwargs=kwargs)
                attempts += 1
                continue

            # Out of budget → return safest shaped
            return (shaped if shaped else SAFE_FALLBACK), meta

        # If not acceptable yet, adjust sampling slightly and retry within budget
        if attempts >= redo_budget:
            # Best-effort shaping of current raw (even if unacceptable) before giving up
            shaped_fallback, gmeta = guard_and_shape(raw or SAFE_FALLBACK, plan=plan, role=role, phase=phase, cfg=LANGCFG_DEFAULT)
            shaped_fallback = _final_target_patch(shaped_fallback, tgt)
            meta = {
                "phase": phase,
                "intent": intent,
                "redo_try": attempts,
                "guard_meta": gmeta,
                "prompt_len": len(prompt),
                "boilerplate_resample": did_boilerplate_resample,
                "persona_style": persona_style,
                "provider": ("openai" if _is_openai_tok(tok) else "hf"),
            }
            return (shaped_fallback if shaped_fallback else SAFE_FALLBACK), meta

        # gentle jitter to escape local minima without clamping diversity
        attempts += 1
        kwargs = dict(kwargs)
        kwargs["temperature"] = min(0.95, float(kwargs.get("temperature", 0.7)) + 0.05)
        kwargs["top_p"] = min(0.95, float(kwargs.get("top_p", 0.9)) + 0.03)
        # also ensure repetition defaults persist
        kwargs.setdefault("no_repeat_ngram_size", _DEFAULT_NGRAM)
        kwargs.setdefault("repetition_penalty", _DEFAULT_REP)

# ───────────────────────── Exports ────────────────────────────────────────────
__all__ = [
    "LangCfg",
    "LANGCFG_DEFAULT",
    "LogitBiasHead",
    "SpeakerBandit",
    "IntentFusionProcessor",
    "CAT_ORDER",
    "FUSION_ALPHA",
    "get_fusion_alpha",
    "fuse_intent_and_bias",
    "repetition_penalty",
    "with_logit_bias_generate_kwargs",
    "with_fused_bias_generate_kwargs",
    "build_processor_from_cat_weights",
    "IntentBiasProcessor",
    "HFIntentBiasProcessor",
    "build_alpha_fusion_processor_for_agent",
    "with_alpha_fusion_generate_kwargs",
    "ROLE_PHASE_CONTRACT",
    "EXEMPLARS",
    "build_role_phase_prompt",
    "build_prompt_and_controls",
    "BigramPenaltyProcessor",
    "has_trigram_repetition",
    "add_synonym_nudge_bias",
    "guard_and_shape",
    "GuardCfg",
    "maybe_second_pass_kwargs",
    "normalize_contractions",
    "normalize_utterance",
    "prune_markers",
    "early_stop_text",
    "looks_meta",
    "looks_boilerplate",
    "ensure_target_and_reason",
    "generate",
    "_alive_names",
    "_AliveNameBoostProcessor",
    "_set_allowed_names",
    # NEW:
    "OpenAITokenizerShim",
]

def _sanitize_openai_kwargs(kwargs: dict) -> dict:
    try:
        kw = dict(kwargs or {})
        model = str(kw.get('model','')).lower().strip()
        # map alias if present
        if 'max_tokens' in kw and 'max_completion_tokens' not in kw:
            try:
                kw['max_completion_tokens'] = int(kw.pop('max_tokens'))
            except Exception:
                kw.pop('max_tokens', None)
        # strip unsupported on o4/omni
        if model.startswith('o4') or model.startswith('omni-'):
            for k in ('temperature','presence_penalty','frequency_penalty','top_p'):
                kw.pop(k, None)
        return kw
    except Exception:
        return dict(kwargs or {})
