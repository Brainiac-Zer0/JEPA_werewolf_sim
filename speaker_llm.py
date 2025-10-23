# speaker_llm.py
# Trainable, lexicon-guided logit bias head for the LLM mouthpiece.
from __future__ import annotations
import os, json, hashlib
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

if TYPE_CHECKING:
    # Only used for typing; avoids importing transformers at runtime.
    from transformers import PreTrainedTokenizerBase

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

# Legacy speaker_llm keys (present in your config.yaml)
_BIAS_CAP        = float(CFG.get("BIAS_CAP", 2.0))
_BASE_STRENGTH   = float(CFG.get("BASE_STRENGTH", 1.0))
_SPK_DEBUG       = bool(CFG.get("DEBUG", False))
_DEFAULT_LEXICON = dict(CFG.get("DEFAULT_LEXICON", {
    "accuse":   ["accuse", "suspicious", "suspect", "lying", "deceive", "eliminate", "vote"],
    "defend":   ["defend", "innocent", "trust", "ally", "support", "clear"],
    "hedge":    ["maybe", "perhaps", "uncertain", "unsure", "might", "seems", "appears"],
    "question": ["why", "how", "what", "who", "where", "when", "?"],
    "vote":     ["vote", "eliminate", "banish", "target", "lynch"],
}))
_CAT_ORDER       = list(CFG.get("CAT_ORDER", ["accuse", "defend", "hedge", "question", "vote"]))
_SPEAKER_HIST_K  = int(CFG.get("SPEAKER_HIST_K", 3))  # shared with speaker.py

# Optional extras/overrides for lexicon without touching code
_EXTRA_LEXICON   = dict(CFG.get("EXTRA_LEXICON", {}))      # {cat: [tokens...]}
_STRICT_LEXICON  = bool(CFG.get("STRICT_LEXICON", False))  # if True, ignore unknown cats in EXTRA

# NEW (Phase-5): soft control knobs
_PER_CAT_SCALES = dict(CFG.get("PER_CAT_SCALES", {}))      # e.g., {"question": 1.1, "vote": 0.95}
_ENTROPY_ATTEN  = float(CFG.get("ENTROPY_ATTENUATION", 0.5))

# Env overrides (SLURM-friendly)
BIAS_CAP        = _env_float("BIAS_CAP", _BIAS_CAP)
BASE_STRENGTH   = _env_float("BASE_STRENGTH", _BASE_STRENGTH)
LLM_SPK_DEBUG   = _env_bool ("LLM_SPK_DEBUG", _SPK_DEBUG)
SPEAKER_HIST_K  = _env_int  ("SPEAKER_HIST_K", _SPEAKER_HIST_K)
# Fusion default (may be overridden at call site)
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

# Pull short history features from speaker.py (shared)
try:
    from speaker import make_hist_feats
except Exception:
    def make_hist_feats(recent_texts: List[str]) -> torch.Tensor:
        # Minimal fallback: zeros when speaker.py isn’t available.
        return torch.tensor([0.0, 0.0], dtype=torch.float32)

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
        # Cap and base scale read from cfg/env
        self.bias_cap = float(BIAS_CAP)
        self.base     = float(BASE_STRENGTH)

    def forward(self, z_t: torch.Tensor, hist_feats: torch.Tensor) -> torch.Tensor:
        """Return per-category strengths in [-bias_cap, +bias_cap] scaled by BASE_STRENGTH."""
        if z_t.dim() == 1:         z_t = z_t.unsqueeze(0)
        if hist_feats.dim() == 1:  hist_feats = hist_feats.unsqueeze(0)
        x = torch.cat([z_t, hist_feats], dim=-1)
        raw = self.net(x)                          # [-inf, +inf]
        bias = torch.tanh(raw) * self.bias_cap     # [-cap, +cap]
        return self.base * bias

    def regularizer(
        self,
        z_batch: torch.Tensor,
        recent_texts_batch: Optional[List[List[str]]] = None,
        *,
        lambda_entropy: float = 0.01,
        lambda_l2: float = 1e-4,
        lambda_balance: float = 0.01,
        target_prior: Optional[torch.Tensor] = None,  # [C] prior over categories or None → uniform
    ) -> torch.Tensor:
        """Entropy+L2+batch-prior regularizer on strengths (scalar loss)."""
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
            hist = torch.stack(feats, dim=0)  # [B,2]

        strengths = self.forward(z_batch, hist)             # [B,C]
        probs = torch.softmax(strengths, dim=-1)            # [B,C]

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
    """Stable cache key for a tokenizer instance (HF-friendly)."""
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
    """Map {category → words} to {category → token_ids} (first-token heuristic)."""
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
    """Return cached {category → token_ids}, building if needed."""
    key = _tok_cache_key(tokenizer)
    cached = _TOKEN_SET_CACHE.get(key)
    if cached is not None:
        return cached
    built = _build_token_sets(tokenizer, lexicon)
    _TOKEN_SET_CACHE[key] = built
    return built

# ───────────────────────── Small utilities ─────────────────────────
def _normalized_entropy(p: torch.Tensor) -> torch.Tensor:
    """Return H(p)/log(C) in [0,1] (higher = more uncertain)."""
    p = p.float()
    p = p / p.sum().clamp_min(1e-6)
    h = -(p * (p.clamp_min(1e-12).log())).sum()
    h_max = torch.log(torch.tensor(float(p.numel()), device=p.device))
    return (h / h_max).clamp(0.0, 1.0)

def _uniform_prior(C: int) -> torch.Tensor:
    """Uniform prior over C categories."""
    return torch.full((C,), 1.0 / max(1, C), dtype=torch.float32)

# ───────────────────────── Generation glue ─────────────────────────
class _CategoryBiasProcessor:
    """
    Additive per-token bias: scores += token_bias.
    token_bias[id] = Σ_k bias_k * mask_k[id], mask_k ∈ {0,1}.
    """
    def __init__(self, token_bias: torch.Tensor, debug: bool = False) -> None:
        self.token_bias = token_bias
        self.debug = debug

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """HF-compatible logits processor: apply token_bias to scores."""
        if scores.dim() == 1:
            out = scores + self.token_bias.to(scores.device)
        else:
            out = scores + self.token_bias.to(scores.device).unsqueeze(0).expand_as(scores)
        return out

@torch.no_grad()
def _biashead_softmax(head: LogitBiasHead, z_t: torch.Tensor, recent_texts: List[str]) -> torch.Tensor:
    """Softmax over categories from BiasHead for fusion/simplex mixing."""
    device = next(head.parameters()).device if any(p.requires_grad for p in head.parameters()) else (z_t.device if z_t.is_cuda else torch.device("cpu"))
    z = z_t.detach().to(device)
    hist = make_hist_feats(recent_texts).to(device)
    s = head(z, hist).squeeze(0)                 # [C]
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
    cat_weights: torch.Tensor,   # [C] simplex or any non-negative vector
    scale: float = 1.0,
) -> torch.Tensor:
    """Map category weights to a clamped per-token bias vector (Tensor[V])."""
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
    cat_weights: torch.Tensor,   # [C], not necessarily normalized
    scale: float = 1.0,
    debug: bool = False,
) -> _CategoryBiasProcessor:
    """Return a logits processor from raw category weights (auto-normalized)."""
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
    """Return (token_bias[V], {cat: strength}) from BiasHead (pre-softmax)."""
    device = next(head.parameters()).device if any(p.requires_grad for p in head.parameters()) else (z_t.device if z_t.is_cuda else torch.device("cpu"))
    z = z_t.detach().to(device)
    hist = make_hist_feats(recent_texts).to(device)
    strengths = head(z, hist).squeeze(0)   # [C]

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
    h_norm = _normalized_entropy(p)  # 0..1
    atten = (1.0 - float(ENTROPY_ATTEN) * float(h_norm.item()))
    atten = max(0.0, min(1.0, atten))
    token_bias = token_bias * atten

    return token_bias, cat_strengths_out

# ───────────────────────── Public APIs ─────────────────────────
def with_logit_bias_generate_kwargs(
    *,
    tokenizer: "PreTrainedTokenizerBase",
    head: LogitBiasHead,
    z_t: torch.Tensor,
    role: Optional[str],
    recent_texts: List[str],
    persona_effects: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Return {'logits_processor':[...]} that applies BiasHead-only token nudging."""
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

    proc = _CategoryBiasProcessor(token_bias=token_bias, debug=LLM_SPK_DEBUG)
    return {"logits_processor": [proc]}

def with_fused_bias_generate_kwargs(
    *,
    tokenizer: "PreTrainedTokenizerBase",
    head: LogitBiasHead,
    z_t: torch.Tensor,
    talkhead_probs: Optional[torch.Tensor],   # [C] softmax from TalkHead (or None)
    alpha: Optional[float] = None,            # fusion weight in [0,1]
    role: Optional[str] = None,
    recent_texts: Optional[List[str]] = None,
    persona_effects: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Return {'logits_processor':[...]} using α·TalkHead + (1−α)·BiasHead fusion."""
    if recent_texts is None: recent_texts = []
    a = FUSION_ALPHA if (alpha is None) else float(alpha)
    a = max(0.0, min(1.0, a))

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

    h_norm = _normalized_entropy(w_total)  # 0..1
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

    proc = _CategoryBiasProcessor(token_bias=token_bias, debug=LLM_SPK_DEBUG)
    return {"logits_processor": [proc]}

# Optional: explicit export surface
__all__ = [
    "LogitBiasHead",
    "with_logit_bias_generate_kwargs",
    "with_fused_bias_generate_kwargs",
    "build_processor_from_cat_weights",
]
