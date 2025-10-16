# speaker_llm.py
# Trainable, lexicon-guided logit bias head for the LLM mouthpiece.
from __future__ import annotations
import os, math, json
from typing import Dict, List, Optional, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

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
_BIAS_CAP          = float(CFG.get("BIAS_CAP", 2.0))
_BASE_STRENGTH     = float(CFG.get("BASE_STRENGTH", 1.0))
_SPK_DEBUG         = bool(CFG.get("DEBUG", False))
_DEFAULT_LEXICON   = dict(CFG.get("DEFAULT_LEXICON", {
    "accuse":   ["accuse", "suspicious", "suspect", "lying", "deceive", "eliminate", "vote"],
    "defend":   ["defend", "innocent", "trust", "ally", "support", "clear"],
    "hedge":    ["maybe", "perhaps", "uncertain", "unsure", "might", "seems", "appears"],
    "question": ["why", "how", "what", "who", "where", "when", "?"],
    "vote":     ["vote", "eliminate", "banish", "target", "lynch"],
}))
_CAT_ORDER         = list(CFG.get("CAT_ORDER", ["accuse", "defend", "hedge", "question", "vote"]))
_SPEAKER_HIST_K    = int(CFG.get("SPEAKER_HIST_K", 3))  # shared with speaker.py

# Env overrides (SLURM-friendly)
BIAS_CAP        = _env_float("BIAS_CAP", _BIAS_CAP)
BASE_STRENGTH   = _env_float("BASE_STRENGTH", _BASE_STRENGTH)
LLM_SPK_DEBUG   = _env_bool ("LLM_SPK_DEBUG", _SPK_DEBUG)
SPEAKER_HIST_K  = _env_int  ("SPEAKER_HIST_K", _SPEAKER_HIST_K)

# Normalized, ordered lexicon + a safe map (word -> token ids built in runtime)
CAT_ORDER       = _CAT_ORDER
DEFAULT_LEXICON = {k: list(v) for k, v in _DEFAULT_LEXICON.items() if k in CAT_ORDER}

# Pull short history features from speaker.py (shared)
try:
    from speaker import make_hist_feats
except Exception:
    def make_hist_feats(recent_texts: List[str]) -> torch.Tensor:
        # Minimal fallback: zeros when speaker.py isn’t available.
        return torch.tensor([0.0, 0.0], dtype=torch.float32)

# ───────────────────────── Model: LogitBiasHead ─────────────────────────
class LogitBiasHead(nn.Module):
    """
    Tiny head that maps (z_t, short text history) → per-category bias strengths.
    These strengths are applied to token logits belonging to each category set.
    """
    def __init__(self, latent_dim: int = 32, hidden: int = 128, num_cats: Optional[int] = None):
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
        """
        z_t: [d] or [B,d]
        hist_feats: [2] or [B,2]
        returns: [num_cats] or [B,num_cats] (unnormalized strengths)
        """
        if z_t.dim() == 1:         z_t = z_t.unsqueeze(0)
        if hist_feats.dim() == 1:  hist_feats = hist_feats.unsqueeze(0)
        x = torch.cat([z_t, hist_feats], dim=-1)
        raw = self.net(x)                          # [-inf, +inf]
        # squash → [-bias_cap, +bias_cap] around 0; then add base scale multiplier
        bias = torch.tanh(raw) * self.bias_cap
        return self.base * bias                    # final per-category strengths

# ───────────────────────── Generation glue ─────────────────────────
class _CategoryBiasProcessor:
    """
    A light-weight logits processor that adds category-wise bias to token ids.

    token_bias[id] = sum_k bias_k * mask_k[id]
    where mask_k[id] ∈ {0,1} indicates membership of token id in category k.
    """
    def __init__(
        self,
        token_bias: torch.Tensor,            # [V] on the same device as scores
        debug: bool = False,
    ):
        self.token_bias = token_bias
        self.debug = debug

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # scores: [V] or [B,V]; input_ids unused (stateless bias)
        if scores.dim() == 1:
            out = scores + self.token_bias.to(scores.device)
        else:
            out = scores + self.token_bias.to(scores.device).unsqueeze(0).expand_as(scores)
        return out

def _build_token_sets(tokenizer, lexicon: Dict[str, List[str]]) -> Dict[str, List[int]]:
    """
    Turn category lexicon into token-id sets (greedy: takes first id for each word,
    plus '?' literal support). We avoid breaking multi-token words; this is a
    pragmatic heuristic for gentle biasing.
    """
    cat2ids: Dict[str, List[int]] = {}
    for cat, words in lexicon.items():
        ids: List[int] = []
        for w in words:
            if w == "?":
                # Prefer the single '?' token if present; otherwise skip
                enc = tokenizer("?")["input_ids"]
                if len(enc) == 1:
                    ids.append(int(enc[0]))
                continue
            # encode without specials; take the first token only
            enc = tokenizer(w, add_special_tokens=False)["input_ids"]
            if enc:
                ids.append(int(enc[0]))
        # Deduplicate
        cat2ids[cat] = sorted(list({i for i in ids if i is not None}))
    return cat2ids

def _make_bias_vector(
    *,
    tokenizer,
    head: LogitBiasHead,
    z_t: torch.Tensor,
    recent_texts: List[str],
    persona_effects: Optional[Dict[str, float]] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Returns:
      token_bias: Tensor[V] — per-vocab additive bias
      cat_strengths: {cat: float} — for optional debug logging
    """
    device = next(head.parameters()).device if any(p.requires_grad for p in head.parameters()) else (z_t.device if z_t.is_cuda else torch.device("cpu"))
    z = z_t.detach().to(device)
    hist = make_hist_feats(recent_texts).to(device)

    with torch.no_grad():
        strengths = head(z, hist).squeeze(0)   # [C]
        # Optional persona tweak (very small; stable)
        if persona_effects:
            # Example: nudge 'accuse' via persona scale
            scale = float(persona_effects.get("accuse_bias_scale", 1.0))
            try:
                idx = CAT_ORDER.index("accuse")
                strengths[idx] = strengths[idx] * max(0.5, min(1.5, scale))
            except Exception:
                pass

    # Map categories → token ids and assemble per-token bias
    cat2ids = _build_token_sets(tokenizer, DEFAULT_LEXICON)
    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or len(tokenizer))
    token_bias = torch.zeros(vocab_size, dtype=torch.float32, device=device)

    cat_strengths_out: Dict[str, float] = {}
    for i, cat in enumerate(CAT_ORDER):
        s = float(strengths[i].item() if i < strengths.numel() else 0.0)
        cat_strengths_out[cat] = s
        for tid in cat2ids.get(cat, []):
            if 0 <= tid < vocab_size:
                token_bias[tid] = token_bias[tid] + s

    return token_bias, cat_strengths_out

def with_logit_bias_generate_kwargs(
    *,
    tokenizer,
    head: LogitBiasHead,
    z_t: torch.Tensor,
    role: Optional[str],
    recent_texts: List[str],
    persona_effects: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Returns kwargs to pass into HF pipeline(..., **kwargs) so the LLM generation
    is nudged by our bias head. Keeps interface used by llm_script.py.
    """
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
                "role": role,
                "recent_texts": recent_texts[-SPEAKER_HIST_K:],
                "cat_strengths": {k: round(float(v), 3) for k, v in cat_strengths.items()},
            }
            print("[LLM-SPK]", json.dumps(dbg))
        except Exception:
            pass

    # Create a processor instance; HF pipeline will call it each decode step
    proc = _CategoryBiasProcessor(token_bias=token_bias, debug=LLM_SPK_DEBUG)

    # The HF pipeline accepts 'logits_processor' via 'generate_kwargs' (for pipelines:
    # pass-through arbitrary kwargs). Some versions expect a list named 'logits_processor'.
    return {
        "logits_processor": [proc],
    }
