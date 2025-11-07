# speaker.py
# -----------------------------------------------------------------------------
# Phase-5 mouthpiece router (+ Phase-7 plan alignment):
#   • Primary path: LLM one-liner with optional logit-bias fusion (speaker_llm)
#   • Fallback path: local template bandit (stable, fast)
#   • NEW: build_plan_tuple now supports BOTH signatures:
#       (A) planner-aligned: z_t, phase(int), planner_heads, dialog_state, etc.
#       (B) sim-aligned: role, phase(str), intent(str), fused_probs, target, self_name, round_num
#   • NEW: question β-prior to keep dialog inquisitive when it stalls
#   • NEW: postprocess_text(text, role, cfg) for final polish (used by sim.py)
# This file intentionally avoids importing encoders directly to prevent cycles.
# -----------------------------------------------------------------------------

from __future__ import annotations
import os
import re
import json
import yaml
import math
import random
import unicodedata
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple, Optional

# ── Load config
with open("config.yaml", "r") as f:
    CFG = yaml.safe_load(f) or {}

# === Shared SAFE_FALLBACK (align with speaker_llm.guard_and_shape) ===
SAFE_FALLBACK = "I need a moment to think."

# === Phase-7: small language config adapter for guard_and_shape ===
_LANG = CFG.get("language", {}) if isinstance(CFG.get("language", {}), dict) else {}

class _LangCfg:
    def __init__(self, d: Dict[str, Any]):
        # Frequency caps (per 5 turns)
        self.hedges_cap_per5 = int(d.get("hedges_cap_per5", 2))
        self.intensifiers_cap_per5 = int(d.get("intensifiers_cap_per5", 2))
        self.discourse_markers_cap_per5 = int(d.get("discourse_markers_cap_per5", 2))
        # Safety/shape
        self.trigram_veto = bool(d.get("trigram_veto", True))
        # Redo loop budget for guard_and_shape, raised to 2 by default
        self.redo_max = int(d.get("redo_max", 2))
        # Optional post-process knobs read from language.*
        self.min_words = int(d.get("min_words", 8))
        self.force_question_prob = float(d.get("force_question_prob", 0.25))
        # Optional: trim quotes in postprocess_text/light clean
        self.trim_quotes = bool(d.get("trim_quotes", True))

LANGCFG = _LangCfg(_LANG)

def _ascii_norm(s: str) -> str:
    """Normalize curly quotes and punctuation to plain ASCII, collapse whitespace."""
    if not isinstance(s, str):
        return s
    t = unicodedata.normalize("NFKC", s)
    t = t.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'").replace("«", '"').replace("»", '"')
    t = re.sub(r"\s+", " ", t).strip()
    return t

# Default templates; may be overridden by config
DEFAULT_TEMPLATES = CFG.get("DEFAULT_TEMPLATES", [
    "Accuse {target}",
    "Defend {ally}",
    "Ask {target} a question",
    "Express uncertainty",
    "Propose vote on {target}",
])
# Normalize any curly punctuation in templates at load-time
DEFAULT_TEMPLATES = [_ascii_norm(t) for t in DEFAULT_TEMPLATES]

# Talk categories (keep consistent with agent/sim mapping)
INTENT_ACC = "accuse"
INTENT_DEF = "defend"
INTENT_ASK = "ask"        # alias for "question"
INTENT_HDG = "hedge"
INTENT_VOT = "vote"

# synonyms → canonical
_INTENT_NORMALIZE = {
    "accuse": INTENT_ACC,
    "attack": INTENT_ACC,
    "suspect": INTENT_ACC,
    "defend": INTENT_DEF,
    "protect": INTENT_DEF,
    "ask": INTENT_ASK,
    "question": INTENT_ASK,
    "query": INTENT_ASK,
    "hedge": INTENT_HDG,
    "uncertain": INTENT_HDG,
    "doubt": INTENT_HDG,
    "vote": INTENT_VOT,
    "lynch": INTENT_VOT,
    "eliminate": INTENT_VOT,
}

INTENT_ORDER = [INTENT_ACC, INTENT_DEF, INTENT_HDG, INTENT_ASK, INTENT_VOT]  # fixed index convention

# ── Resolvers: read fusion.alpha_intent_bias and question_prior_beta (env → YAML)
def _resolve_alpha_intent_bias() -> float:
    # ENV precedence
    for key in ("FUSION_ALPHA_INTENT_BIAS", "FUSION_ALPHA_INT", "FUSION_ALPHA"):
        v = os.getenv(key, "").strip()
        if v:
            try:
                return float(v)
            except Exception:
                pass
    # YAML: fusion.alpha_intent_bias (preferred), then legacy speaker.alpha_intent_bias
    try:
        fus = CFG.get("fusion", {}) or {}
        if isinstance(fus.get("alpha_intent_bias", None), (int, float)):
            return float(fus["alpha_intent_bias"])
    except Exception:
        pass
    try:
        spk = CFG.get("speaker", {}) or {}
        if isinstance(spk.get("alpha_intent_bias", None), (int, float)):
            return float(spk["alpha_intent_bias"])
    except Exception:
        pass
    return 0.5

def _resolve_question_prior_beta() -> float:
    # ENV precedence
    v = os.getenv("QUESTION_PRIOR_BETA", "").strip()
    if v:
        try:
            return float(v)
        except Exception:
            pass
    # Prefer language.question_prior_beta if present, else speaker.question_prior_beta
    try:
        if isinstance(_LANG.get("question_prior_beta", None), (int, float)):
            return float(_LANG["question_prior_beta"])
    except Exception:
        pass
    try:
        spk = CFG.get("speaker", {}) or {}
        if isinstance(spk.get("question_prior_beta", None), (int, float)):
            return float(spk["question_prior_beta"])
    except Exception:
        pass
    return 0.35

# Fusion knob (from fusion.* by request)
ALPHA_INTENT_BIAS = float(_resolve_alpha_intent_bias())

# Question β-prior (nudges ASK when dialog lacks questions)
QUESTION_PRIOR_BETA = float(_resolve_question_prior_beta())

# --- Hygiene helpers (LLM safety, formatting) ---
_BAD_QUOTES = "“”\"'«»"

# In "surface issues" mode, we don't strip lines after tokens or agent prefixes

def _looks_meta(s: str) -> bool:
    """
    Minimal meta check to avoid hiding issues: only treat empty as bad.
    """
    return not bool((s or "").strip())

def _trainable_params(mod) -> List[torch.nn.Parameter]:
    """Return only trainable params for a module; [] if module is None or param-less."""
    try:
        return [p for p in mod.parameters() if p.requires_grad]
    except Exception:
        return []

# ─────────────────────────────────────────────────────────────────────────────
# String-local guards for postprocess_text
# ─────────────────────────────────────────────────────────────────────────────

# Canonicalize stray "Agent X" / "Agent 7" / "Agent-7" → "Agent_X"/"Agent_7"
_AGENT_LETTER_RE = re.compile(r"\bAgent\s*([A-Z])\b")
_AGENT_NUM_RE    = re.compile(r"\bAgent(?:\s*[- ]\s*|\s+)(\d+)\b")

def _canonicalize_agent_tokens(text: str) -> str:
    if not text:
        return text
    # Agent A → Agent_A
    text = _AGENT_LETTER_RE.sub(lambda m: f"Agent_{m.group(1)}", text)
    # Agent 7 / Agent-7 → Agent_7
    text = _AGENT_NUM_RE.sub(lambda m: f"Agent_{m.group(1)}", text)
    return text

# Light grammar touch-ups that are safe without round/plan context
def _light_grammar_clean(text: str) -> str:
    if not text:
        return text
    s = text

    # "a someone ..." → "someone ..."
    s = re.sub(r"\b(a|an)\s+someone\b", "someone", s, flags=re.IGNORECASE)

    # Article before Agent_*: enforce "an Agent_*"
    s = re.sub(r"\ba\s+(Agent_[A-Za-z0-9]+)\b", r"an \1", s)  # a Agent_7 → an Agent_7
    # (Don't force 'a' anywhere else to avoid overcorrection.)

    # Common duplicate articles: "the the", "a a", "an an"
    s = re.sub(r"\b(the|a|an)\s+\1\b", r"\1", s, flags=re.IGNORECASE)

    # Drop lingering "a " directly before adjectives starting with 's' in the "a someone suspicious" pattern spillover
    s = re.sub(r"\ba\s+(suspicious|strange|shady)\b", r"\1", s, flags=re.IGNORECASE)

    # Compact spaces around punctuation
    s = re.sub(r"\s+([,.;:?!])", r"\1", s)
    s = re.sub(r"([(\[]) +", r"\1", s)
    s = re.sub(r" +([)\]])", r"\1", s)

    return s

# -----------------------------------------------------------------------------
# Lightweight final polish (used by sim.py)
# -----------------------------------------------------------------------------
def postprocess_text(text: str, role: str, cfg) -> str:
    """
    Finalize-and-print philosophy (purely string-local):
    - Canonicalize stray Agent tokens (Agent A/7/ -7 → Agent_A/Agent_7).
    - Light grammar cleanups (e.g., 'a someone suspicious' → 'someone suspicious',
      enforce 'an Agent_#').
    - Preserve multi-line output and punctuation; don't remove agent prefixes or roles.
    - Do NOT rely on plan/round; keep those edits in sim.py.
    - Never return empty: use SAFE_FALLBACK if needed.
    """
    s = (text or "").strip()
    if getattr(cfg, "trim_quotes", True):
        s = s.strip(' \t"\'`')

    # Normalize unicode/spacing first
    s = _ascii_norm(s)

    # Purely local canonicalizations/cleanups
    s = _canonicalize_agent_tokens(s)
    s = _light_grammar_clean(s)

    return s if s else SAFE_FALLBACK

def _light_clean_keep_multiline(text: str, cfg: _LangCfg) -> str:
    """
    Internal clean for SpeakerPolicy.generate that preserves multi-line output.
    No STOP-seq truncation, no agent-prefix removal, no role sanitization.
    """
    s = (str(text or "")).strip()
    if getattr(cfg, "trim_quotes", True):
        s = s.strip(' \t"\'`')
    s = _ascii_norm(s)
    return s

# =============================================================================
# Dialog-aware helpers (salient target, tone)
# =============================================================================
def _quietest_alive(alive_names: List[str], recent_texts: List[str]) -> Optional[str]:
    if not alive_names:
        return None
    counts = {n: 0 for n in alive_names}
    for t in recent_texts[-8:]:
        for n in alive_names:
            if n in (t or ""):
                counts[n] += 1
    return sorted(counts.items(), key=lambda kv: (kv[1], kv[0]))[0][0] if counts else alive_names[0]

def choose_salient_target(dialog_state: Any, alive_names: List[str], recent_texts: List[str]) -> Optional[str]:
    """
    Prefer the most recently/frequently accused alive agent; else the quietest alive.
    We consult dialog_state if it has a 'salient_target' method; otherwise fall back.
    """
    try:
        if hasattr(dialog_state, "salient_target"):
            t = dialog_state.salient_target(alive_names)
            if t in alive_names:
                return t
    except Exception:
        pass
    return _quietest_alive(alive_names, recent_texts)

def choose_tone(persona_effects: Optional[Dict[str, Any]], role: str, default: str = "neutral") -> str:
    r = (role or "").lower()
    wolf = ("were" in r) or ("wolf" in r)
    if isinstance(persona_effects, dict):
        a = float(persona_effects.get("assertiveness", 0.0))
        if a > 0.4:
            return "assertive"
        if a < -0.4:
            return "cautious"
    return "assertive" if wolf else default

def _intent_to_template_indices(templates: List[str]) -> Dict[str, List[int]]:
    """
    Build a mapping from intent -> candidate template indices by inspecting strings.
    """
    m: Dict[str, List[int]] = {INTENT_ACC: [], INTENT_DEF: [], INTENT_ASK: [], INTENT_HDG: [], INTENT_VOT: []}
    for i, t in enumerate(templates):
        tl = t.lower()
        if "accuse" in tl:
            m[INTENT_ACC].append(i)
        if "defend" in tl:
            m[INTENT_DEF].append(i)
        if "ask" in tl or "question" in tl:
            m[INTENT_ASK].append(i)
        if "uncert" in tl or "unsure" in tl or "doubt" in tl:
            m[INTENT_HDG].append(i)
        if "vote" in tl or "propose" in tl:
            m[INTENT_VOT].append(i)
    for k in m:
        if not m[k]:
            m[k] = [min(i, len(templates)-1) for i in range(len(templates))]
    return m

def _dialog_needs_questions(dialog_state: Any) -> bool:
    """
    True if no question in last ~3 turns or open questions remain unresolved.
    """
    try:
        if hasattr(dialog_state, "question_scarcity") and dialog_state.question_scarcity(lookback=3):
            return True
    except Exception:
        pass
    try:
        oq = getattr(dialog_state, "open_questions", [])
        return bool(oq)
    except Exception:
        return False

# =============================================================================
# History features (lightweight context stats)
# =============================================================================
def make_hist_feats(recent_texts: List[str], phase_code: Optional[int] = None) -> torch.Tensor:
    """
    Returns a small feature vector:
      [accusation_rate, mean_len] (+ optional one-hot phase of size 3)
    """
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

# =============================================================================
# Local template bandit (REINFORCE) — avoids speaker_llm import cycle
# =============================================================================
class SpeakerBandit(nn.Module):
    """
    Tiny bandit over speech-act templates.
    Input: [z_t (d), role_bit (1), hist_feats (?=2 or 5)] → logits over templates.
    Training: REINFORCE on message-level reward.
    """
    def __init__(self, latent_dim: int, num_templates: int, hidden: int = 128):
        super().__init__()
        self.num_templates = num_templates
        self.temperature = 1.0
        self._latent_dim = latent_dim
        self._hidden = hidden
        self._mlp: Optional[nn.Sequential] = None  # lazy init to match feat size at runtime

    def _build(self, in_features: int):
        self._mlp = nn.Sequential(
            nn.Linear(in_features, self._hidden),
            nn.Tanh(),
            nn.Linear(self._hidden, self.num_templates),
        )

    def forward(self, z: torch.Tensor, role_bit: torch.Tensor, hist_feats: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, role_bit, hist_feats], dim=-1)
        if self._mlp is None:
            self._build(x.size(-1))
        return self._mlp(x)

    @torch.no_grad()
    def generate(
        self,
        z_t: torch.Tensor,
        role: str,
        recent_texts: List[str],
        templates: List[str],
        candidate_targets: List[str],
        self_name: str,
        persona_effects: Optional[Dict[str, Any]] = None,
        phase_code: Optional[int] = None,
        *,
        plan: Optional[Dict[str, Any]] = None,   # NEW: align bandit output to plan if present
        **_ignored,
    ) -> Tuple[str, Dict[str, Any]]:
        dev = next(self.parameters()).device
        if z_t.dim() == 1:
            z_t = z_t.unsqueeze(0)
        z_t = z_t.to(dev)

        role_bit = torch.tensor([[1.0 if role.lower().startswith("were") else 0.0]],
                                device=dev, dtype=z_t.dtype)
        hist_feats = make_hist_feats(recent_texts, phase_code=phase_code).to(dev).unsqueeze(0)

        logits = self.forward(z_t, role_bit, hist_feats).squeeze(0)

        # Persona biases (optional, light touch)
        if persona_effects:
            accuse_bias_scale = float(persona_effects.get("accuse_bias_scale", 1.0))
            accuse_bias = (accuse_bias_scale - 1.0) * 0.8  # small
            if accuse_bias != 0.0:
                idx_accuse = [i for i, t in enumerate(templates)
                              if ("accuse" in t.lower()) or ("vote" in t.lower()) or ("propose" in t.lower())]
                idx_uncert = [i for i, t in enumerate(templates)
                              if ("uncertain" in t.lower()) or ("uncert" in t.lower())]
                for i in idx_accuse:
                    logits[i] = logits[i] + accuse_bias
                for i in idx_uncert:
                    logits[i] = logits[i] - 0.5 * accuse_bias

        # Temperature scaling — support both legacy and underscore keys
        temp_scale = 1.0
        if persona_effects:
            try:
                temp_scale = float(
                    persona_effects.get("speaker_temp_scale",
                        persona_effects.get("_temp_scale", 1.0))
                )
            except Exception:
                temp_scale = 1.0
        temperature = max(1e-4, float(getattr(self, "temperature", 1.0)) * temp_scale)

        probs = F.softmax(logits / temperature, dim=-1)

        if plan is not None and isinstance(plan, dict):
            intent = _INTENT_NORMALIZE.get(str(plan.get("intent","")).lower(), None)
            tgt = plan.get("target", None)
            idx_map = _intent_to_template_indices(templates)
            if intent in idx_map and idx_map[intent]:
                tidx = idx_map[intent][0]
            else:
                tidx = torch.multinomial(probs, 1).item()
            target = tgt
        else:
            tidx  = torch.multinomial(probs, 1).item()
            target = None

        if not target:
            target = next((t for t in candidate_targets if t != self_name), None)
            target = target or (candidate_targets[0] if candidate_targets else self_name)

        text = templates[tidx].replace("{target}", target).replace("{ally}", self_name)
        # templates are single-line by design, but keep same light clean for consistency
        text = postprocess_text(text, role, LANGCFG)

        meta = {
            "mode": "bandit",
            "template_id": tidx,
            "logprob": float(torch.log(probs[tidx] + 1e-8).item()),
            "z": z_t.squeeze(0).detach().cpu(),
            "role_bit": role_bit.squeeze(0).detach().cpu(),
            "hist_feats": hist_feats.squeeze(0).detach().cpu(),
            "phase_code": phase_code,
            "plan": plan or {},
        }
        if os.getenv("LLM_SPK_DEBUG", "1") == "1":
            dbg = {
                "mode": "bandit",
                "role": role,
                "name": self_name,
                "phase_code": phase_code,
                "template_id": tidx,
                "preview": text[:120],
                "plan": plan or {},
            }
            print("[LLM-SPK]", json.dumps(dbg), flush=True)
        return text if text else SAFE_FALLBACK, meta

    def learn_step(
        self,
        batch: List[Dict[str, Any]],
        optimizer: torch.optim.Optimizer,
        entropy_bonus: float = 0.01,
        baseline: float = None,
    ) -> Dict[str, float]:
        if not batch:
            return {"loss": 0.0, "entropy": 0.0, "R_mean": 0.0}
        device = next(self.parameters()).device

        zs = torch.stack([b["z"] for b in batch]).to(device)
        role_bits = torch.stack([b["role_bit"] for b in batch]).to(device)
        hfs = torch.stack([b["hist_feats"] for b in batch]).to(device)
        tids = torch.tensor([b["template_id"] for b in batch], dtype=torch.long, device=device)
        rewards = torch.tensor([b["reward"] for b in batch], dtype=torch.float32, device=device)

        logits = self.forward(zs, role_bits, hfs)
        logps = torch.log_softmax(logits, dim=-1)
        sel_logp = logps.gather(1, tids.unsqueeze(1)).squeeze(1)

        if baseline is not None:
            rewards = rewards - baseline

        ent = -(logps.exp() * logps).sum(dim=-1).mean()
        loss = -(rewards * sel_logp).mean() - entropy_bonus * ent

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return {"loss": float(loss.item()), "entropy": float(ent.item()), "R_mean": float(rewards.mean().item())}

# =============================================================================
# Optional LLM bias adapter (trainable light head that steers a frozen LLM)
# =============================================================================
try:
    from speaker_llm import LogitBiasHead, with_logit_bias_generate_kwargs
except Exception:
    LogitBiasHead = None
    with_logit_bias_generate_kwargs = None

class LLMBiasAdapter(nn.Module):
    """
    Small trainable head that steers a frozen LLM via logits processors.
    Exposes: get_kwargs(...) → **kwargs for your LLM generate() call.
    """
    def __init__(self, latent_dim: int = 32, device: Optional[torch.device] = None):
        super().__init__()
        if LogitBiasHead is None:
            self.head = None
        else:
            self.head = LogitBiasHead(latent_dim=latent_dim)
            self.head.to(device if device is not None else torch.device("cpu"))

    def get_kwargs(self, tokenizer, z_t: torch.Tensor, role: str,
                   recent_texts: List[str], persona_effects: Optional[Dict[str, Any]]):
        if self.head is None or with_logit_bias_generate_kwargs is None:
            return {}
        z = z_t.detach() if torch.is_tensor(z_t) else torch.tensor(z_t)
        return with_logit_bias_generate_kwargs(
            tokenizer=tokenizer,
            head=self.head,
            z_t=z,
            role=role,
            recent_texts=recent_texts[-3:],
            persona_effects=persona_effects,
        )

    def learn_step(self, batch: List[Dict[str, Any]], optimizer: torch.optim.Optimizer,
                   entropy_bonus: float = 0.0) -> Dict[str, float]:
        if self.head is None or not batch:
            return {"loss": 0.0}
        device = next(self.head.parameters()).device
        zs = torch.stack([b["z"] for b in batch]).to(device)
        R  = torch.tensor([b["reward"] for b in batch], dtype=torch.float32, device=device)

        penalties = self.head.regularizer(zs)  # scalar
        loss = (1.0 - R).clamp(min=0.0).mean() + 0.01 * penalties.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return {"loss": float(loss.item())}

# =============================================================================
# Plan tuple builders
# =============================================================================
def _build_plan_from_heads(
    z_t: torch.Tensor,
    phase: Optional[int],
    role: str,
    planner_heads: Any,
    dialog_state: Any,
    persona_effects: Optional[Dict[str, Any]],
    recent_texts: Optional[List[str]] = None,
    alive_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Original head-aligned path used internally and by SpeakerPolicy."""
    intent = None
    target = None

    # 1) Intent with ASK β-prior (USE question_prior_beta)
    needs_q = _dialog_needs_questions(dialog_state)
    ask_beta = QUESTION_PRIOR_BETA if needs_q else 0.0

    if planner_heads is not None and hasattr(planner_heads, "select_intent"):
        try:
            intent = planner_heads.select_intent(
                z_t, phase,
                alpha_intent_bias=ALPHA_INTENT_BIAS,
                ask_prior_beta=ask_beta
            )
        except TypeError:
            intent = planner_heads.select_intent(z_t, phase)
        except Exception:
            intent = None

    intent = _INTENT_NORMALIZE.get(str(intent or (INTENT_ASK if needs_q else INTENT_HDG)).lower(), INTENT_HDG)

    # 2) Target
    needs_target = intent in (INTENT_ACC, INTENT_DEF, INTENT_VOT)
    if needs_target and planner_heads is not None and hasattr(planner_heads, "select_target"):
        try:
            target = planner_heads.select_target(z_t, phase)
        except Exception:
            target = None

    if (not target) and (intent in (INTENT_ASK, INTENT_HDG)):
        target = choose_salient_target(dialog_state, alive_names or [], recent_texts or [])

    # 3) Shape
    shape = "claim→reason" if intent in (INTENT_ACC, INTENT_DEF, INTENT_VOT) else "obs→question"

    # 4) Tone
    tone = choose_tone(persona_effects, role, default="neutral")

    return {"intent": intent, "target": target, "shape": shape, "tone": tone, "ask_beta": ask_beta}

def build_plan_tuple(**kwargs) -> Dict[str, Any]:
    """
    Dual-signature adapter:
      A) planner-heads path (existing):
         z_t, phase(int), role, planner_heads, dialog_state, persona_effects,
         recent_texts=None, alive_names=None
      B) sim path (new in sim.py):
         role, phase(str: "DAY_DISCUSS"/...), intent(str), fused_probs(list), target(str|None),
         self_name(str), round_num(int)
    Returns a small plan dict with at least: {intent, target, shape, tone, ask_beta?}
    """
    # Planner-heads signature?
    if "z_t" in kwargs and "planner_heads" in kwargs:
        return _build_plan_from_heads(
            z_t=kwargs.get("z_t"),
            phase=kwargs.get("phase"),
            role=kwargs.get("role", "Unknown"),
            planner_heads=kwargs.get("planner_heads"),
            dialog_state=kwargs.get("dialog_state"),
            persona_effects=kwargs.get("persona_effects"),
            recent_texts=kwargs.get("recent_texts"),
            alive_names=kwargs.get("alive_names"),
        )

    # Sim signature
    role = kwargs.get("role", "Unknown")
    phase_str = str(kwargs.get("phase", "")).upper()
    intent_in = str(kwargs.get("intent", "")).lower()
    fused_probs = kwargs.get("fused_probs", None)  # not used directly yet
    target = kwargs.get("target", None)
    _ = kwargs.get("self_name", None)
    __ = kwargs.get("round_num", None)

    intent = _INTENT_NORMALIZE.get(intent_in, INTENT_HDG)

    # choose shape consistent with intent
    shape = "claim→reason" if intent in (INTENT_ACC, INTENT_DEF, INTENT_VOT) else "obs→question"

    # tone: slightly firmer at vote time, cautious for hedge
    phase_tone = "neutral"
    if "VOTE" in phase_str:
        phase_tone = "assertive"
    if intent == INTENT_HDG:
        phase_tone = "cautious"

    return {
        "intent": intent,
        "target": target,
        "shape": shape,
        "tone": phase_tone,
        # keep optional for compatibility (not used by sim path)
        "ask_beta": 0.0,
    }

# =============================================================================
# Unified mouthpiece: routes between LLM and Bandit; both trainable
# =============================================================================
class SpeakerPolicy(nn.Module):
    """
    Hybrid router:
      - default: LLM (natural one-liner) with optional logit-bias head
      - fallback: Bandit templates (fast, stable)
      - NEW: plan tuple ensures utterance mirrors planner’s intended act/target
    Trainable pieces: bandit, bias head (optional)
    """
    def __init__(self,
                 latent_dim: int,
                 templates: Optional[List[str]] = None,
                 device: Optional[torch.device] = None):
        super().__init__()
        # Normalize configured templates to ASCII at load time
        cfg_templates = CFG.get("speaker", {}).get("templates", DEFAULT_TEMPLATES)
        self.templates = [_ascii_norm(t) for t in (templates or cfg_templates)]
        self.bandit = SpeakerBandit(
            latent_dim=latent_dim,
            num_templates=len(self.templates),
            hidden=int(CFG.get("speaker", {}).get("hidden", 128))
        )
        self.bias = LLMBiasAdapter(latent_dim=latent_dim, device=device)

        # Router thresholds
        env_gate = os.getenv("LLM_SPEAKER", "")
        if env_gate.strip():
            self.use_llm = env_gate.strip().lower() in ("1", "true", "yes", "y", "on")
        else:
            self.use_llm = bool(CFG.get("llm", {}).get("speaker_enabled", False))
        self.bad_streak = 0
        self.max_bad = 2  # after 2 filtered generations, back off to bandit temporarily

        # Optional optimizers (attach via attach_optimizers)
        self.bandit_opt: Optional[torch.optim.Optimizer] = None
        self.bias_opt: Optional[torch.optim.Optimizer] = None

        self.to(device if device is not None else torch.device("cpu"))

        # Try to import LLM helpers lazily to avoid hard dependency
        self._llm_ok = False
        try:
            from llm_script import chatgpt_llm_with_bias, chatgpt_llm_from_latent  # type: ignore
            self._llm_with_bias = chatgpt_llm_with_bias
            self._llm_from_latent = chatgpt_llm_from_latent
            self._llm_ok = True
        except Exception:
            self._llm_with_bias = None
            self._llm_from_latent = None
            self._llm_ok = False

        # Phase-string mapping for guard checks
        self._PHASE_MAP = {0: "DISCUSS", 1: "VOTE", 2: "NIGHT"}

        # Import guard_and_shape (Phase-7)
        try:
            from speaker_llm import guard_and_shape  # type: ignore
            self._guard_and_shape = guard_and_shape
        except Exception:
            self._guard_and_shape = None

    @torch.no_grad()
    def _llm_generate(self,
                      name: str,
                      role: str,
                      recent_texts: List[str],
                      z_t: torch.Tensor,
                      persona_effects: Optional[Dict[str, Any]],
                      *,
                      phase_code: Optional[int] = None,
                      plan: Optional[Dict[str, Any]] = None,
                      dialog_state: Any = None) -> Tuple[str, Dict[str, Any]]:

        proxy_agent = type("A", (), {
            "role": role,
            "name": name,
            "message_memory": [],
            "decode_z": staticmethod(lambda _z: ""),
            "persona_effects": persona_effects,
        })()

        target = (plan or {}).get("target", None)
        shape  = (plan or {}).get("shape", None)
        tone   = (plan or {}).get("tone", "neutral")

        prefix_hint = f"I think {target} " if isinstance(target, str) and target.startswith("Agent_") else None

        steer_parts = []
        if shape == "claim→reason" and isinstance(target, str):
            steer_parts.append(f"State your claim about {target} then a short reason.")
        elif shape == "obs→question":
            if isinstance(target, str):
                steer_parts.append(f"Make a brief observation then ask {target} a pointed question.")
            else:
                steer_parts.append("Make a brief observation then ask a pointed question.")
        if tone == "cautious":
            steer_parts.append("Keep a cautious tone.")
        elif tone == "assertive":
            steer_parts.append("Keep a firm tone.")
        steer_suffix = (" " + " ".join(steer_parts)).strip() if steer_parts else ""

        from llm_script import _latent_prompt_from_agent, llm_fn_from_env  # type: ignore
        mouth = llm_fn_from_env()
        tok = getattr(mouth, "tokenizer", None)
        base_prompt = _latent_prompt_from_agent(tok, z_t, proxy_agent)

        if phase_code == 0 and not steer_suffix:
            steer_suffix = " End with a pointed question or a named accusation."

        final_prompt = ((prefix_hint or "") + base_prompt + (" " + steer_suffix if steer_suffix else "")).strip()

        # Merge persona knobs: support both legacy and underscore keys
        pe = dict(persona_effects or {})
        if "_temp_scale" in pe and "speaker_temp_scale" not in pe:
            pe["speaker_temp_scale"] = pe["_temp_scale"]
        if "_bias_scale" in pe and "bias_scale" not in pe:
            pe["bias_scale"] = pe["_bias_scale"]

        bias_kwargs = self.bias.get_kwargs(
            tokenizer=getattr(mouth, "tokenizer", None),
            z_t=z_t.squeeze(0),
            role=role,
            recent_texts=recent_texts,
            persona_effects=pe,
        )
        default_gen = {
            "min_new_tokens": 14,
            "max_new_tokens": 64,
            "temperature": 0.8,
            "top_p": 0.92,
        }

        # Apply temperature scaling from persona if provided
        try:
            tscale = float(pe.get("speaker_temp_scale", 1.0))
        except Exception:
            tscale = 1.0
        gen_kwargs = dict(default_gen)
        if isinstance(bias_kwargs, dict):
            gen_kwargs.update(bias_kwargs)
        gen_kwargs["temperature"] = max(1e-4, float(gen_kwargs.get("temperature", 0.8)) * tscale)

        # Sanitize kwargs for chatgpt_llm_with_bias signature compatibility
        for _k in list(gen_kwargs.keys()):
            if _k in ("fusion_alpha", "fusion", "alpha_intent_bias", "bias_alpha"):
                gen_kwargs.pop(_k, None)
        if "logits_processor" in gen_kwargs and "logits_processors" not in gen_kwargs:
            gen_kwargs["logits_processors"] = gen_kwargs.pop("logits_processor")
        for _k in list(gen_kwargs.keys()):
            if gen_kwargs[_k] is None:
                gen_kwargs.pop(_k, None)

        # Provider-agnostic length mirror:
        # We keep provider-specific filtering centralized inside llm_script's mouth().
        # Here, we only mirror the length knob so OpenAI Responses gets a proper cap.
        try:
            if "max_new_tokens" in gen_kwargs and "max_output_tokens" not in gen_kwargs:
                gen_kwargs["max_output_tokens"] = int(gen_kwargs["max_new_tokens"])
        except Exception:
            gen_kwargs.pop("max_output_tokens", None)

        text = mouth(final_prompt, generate_kwargs=gen_kwargs)

        meta = {
            "mode": "llm",
            "steer_phase": int(phase_code) if phase_code is not None else None,
            "used_bias": bool(bias_kwargs),
            "gen_kwargs": {k: gen_kwargs.get(k) for k in ("min_new_tokens","max_new_tokens","temperature","top_p")},
            "plan": plan or {},
        }
        if isinstance(bias_kwargs, dict) and "w_bias_sparse" in bias_kwargs:
            meta["w_bias_sparse"] = bias_kwargs.get("w_bias_sparse")
        if "repetition_penalty" in gen_kwargs:
            meta["repetition_penalty"] = float(gen_kwargs["repetition_penalty"])
            meta["rp_applied"] = True
        return text, meta

    def _safe_generic(self, plan: Dict[str, Any]) -> str:
        """Name-grounded, guard-safe fallback line."""
        intent = _INTENT_NORMALIZE.get(str(plan.get("intent","")).lower(), INTENT_HDG)
        tgt = plan.get("target") or "someone"
        if intent in (INTENT_ACC, INTENT_VOT):
            return _ascii_norm(f"{tgt}, your posts feel off; I'd vote you here.")
        if intent == INTENT_DEF:
            return _ascii_norm(f"{tgt} reads consistent to me; pushing them seems opportunistic.")
        if intent == INTENT_ASK:
            return _ascii_norm(f"{tgt}, who's your top suspect now?")
        return _ascii_norm(f"{tgt}, what's your read?")

    @torch.no_grad()
    def generate(self,
                 z_t: torch.Tensor,
                 role: str,
                 recent_texts: List[str],
                 candidate_targets: List[str],
                 self_name: str,
                 * ,
                 phase_code: Optional[int] = None,
                 talk_prior: Optional[Dict[str, Any]] = None,
                 persona_effects: Optional[Dict[str, Any]] = None,
                 # NEW: planner/dialog hooks
                 planner_heads: Any = None,
                 dialog_state: Any = None,
                 # Back-compat soft steer inputs (unused now; plan replaces them)
                 prefix: Optional[str] = None,
                 named_target_hint: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:

        dev = next(self.parameters()).device
        if z_t.dim() == 1:
            z_t = z_t.unsqueeze(0)
        z_t = z_t.to(dev)

        # Build plan if possible (align utterance with planner)
        alive_names = [t for t in candidate_targets if t != self_name]
        plan = _build_plan_from_heads(
            z_t=z_t.squeeze(0),
            phase=phase_code,
            role=role,
            planner_heads=planner_heads,
            dialog_state=dialog_state,
            persona_effects=persona_effects,
            recent_texts=recent_texts,
            alive_names=alive_names,
        ) if (planner_heads is not None and dialog_state is not None) else None

        # Route: try LLM unless we are in a backoff window
        use_llm_now = self.use_llm and (self.bad_streak < self.max_bad)
        if use_llm_now:
            try:
                text_raw, meta_llm = self._llm_generate(
                    self_name, role, recent_texts, z_t, persona_effects,
                    phase_code=phase_code,
                    plan=plan,
                    dialog_state=dialog_state
                )
                # Phase-7: guard + optional redo (KEEP redo_max from language.*)
                # PRESERVE MULTILINE — no first-line truncation here
                text_guard = _light_clean_keep_multiline(text_raw, LANGCFG)
                if self._guard_and_shape is not None and plan is not None:
                    phase_str = self._PHASE_MAP.get(int(phase_code) if phase_code is not None else 0, "DISCUSS")
                    text_guard, gmeta = self._guard_and_shape(text_guard, plan, role or "Unknown", phase_str, LANGCFG)
                    retry_count = 0
                    while gmeta.get("redo") and retry_count < max(0, LANGCFG.redo_max):
                        retry_count += 1
                        text_raw2, meta_llm2 = self._llm_generate(
                            self_name, role, recent_texts, z_t, persona_effects,
                            phase_code=phase_code, plan=plan, dialog_state=dialog_state
                        )
                        candidate = _light_clean_keep_multiline(text_raw2, LANGCFG)
                        text_guard2, gmeta2 = self._guard_and_shape(candidate, plan, role or "Unknown", phase_str, LANGCFG)
                        if not gmeta2.get("redo"):
                            text_guard, gmeta, meta_llm = text_guard2, gmeta2, meta_llm2
                            break
                    if gmeta.get("redo"):
                        text_guard = self._safe_generic(plan)

                    text = postprocess_text(text_guard, role, LANGCFG)
                    self.bad_streak = 0
                    meta = {
                        "mode": "llm",
                        "text_raw": text,
                        "z": z_t.squeeze(0).detach().cpu(),
                        "phase_code": phase_code,
                        "plan": plan or {},
                        "guard": gmeta,
                        "strict_ok": int(not gmeta.get("redo", False)),
                    }
                    if isinstance(meta_llm, dict):
                        meta.update(meta_llm)

                    if os.getenv("LLM_SPK_DEBUG", "1") == "1":
                        dbg = {
                            "mode": "llm",
                            "role": role,
                            "name": self_name,
                            "phase_code": phase_code,
                            "used_bias": meta.get("used_bias", False),
                            "gen": meta.get("gen_kwargs", {}),
                            "preview": (text[:120].replace("\n"," ⏎ ") if isinstance(text, str) else ""),
                            "plan": plan or {},
                            "guard": meta.get("guard", {}),
                        }
                        print("[LLM-SPK]", json.dumps(dbg), flush=True)
                    return text if text else SAFE_FALLBACK, meta

                # If no guard available, minimal meta check (only empty)
                if _looks_meta(text_guard):
                    self.bad_streak += 1
                    raise RuntimeError("empty generation")
                self.bad_streak = 0
                text_final = postprocess_text(text_guard, role, LANGCFG)
                meta = {
                    "mode": "llm",
                    "text_raw": text_final,
                    "z": z_t.squeeze(0).detach().cpu(),
                    "phase_code": phase_code,
                    "plan": plan or {},
                }
                if isinstance(meta_llm, dict):
                    meta.update(meta_llm)
                if os.getenv("LLM_SPK_DEBUG", "1") == "1":
                    dbg = {
                        "mode": "llm",
                        "role": role,
                        "name": self_name,
                        "phase_code": phase_code,
                        "used_bias": meta.get("used_bias", False),
                        "gen": meta.get("gen_kwargs", {}),
                        "preview": (text_final[:120].replace("\n"," ⏎ ") if isinstance(text_final, str) else ""),
                        "plan": plan or {},
                    }
                    print("[LLM-SPK]", json.dumps(dbg), flush=True)
                return text_final if text_final else SAFE_FALLBACK, meta
            except Exception:
                # Fall back to bandit
                pass

        # Bandit path (stable fallback), now also aligned to plan if present
        role_bit = torch.tensor([[1.0 if role.lower().startswith("were") else 0.0]], device=dev, dtype=z_t.dtype)
        hf = make_hist_feats(recent_texts, phase_code).to(dev).unsqueeze(0)
        logits = self.bandit(z_t, role_bit, hf).squeeze(0)
        temperature = max(1e-4, float(getattr(self.bandit, "temperature", 1.0)))
        probs = F.softmax(logits / temperature, dim=-1)

        if plan is not None:
            idx_map = _intent_to_template_indices(self.templates)
            intent = _INTENT_NORMALIZE.get(str(plan.get("intent","")).lower(), None)
            tidx = (idx_map.get(intent, [0])[0]) if intent in idx_map else int(torch.multinomial(probs, 1).item())
            target = plan.get("target", None)
        else:
            tidx = int(torch.multinomial(probs, 1).item())
            target = None

        if not target:
            target = next((t for t in candidate_targets if t != self_name), None)
        target = target or (candidate_targets[0] if candidate_targets else self_name)

        text = self.templates[tidx].replace("{target}", target).replace("{ally}", self_name)
        text = postprocess_text(text, role, LANGCFG)

        meta = {
            "mode": "bandit",
            "template_id": tidx,
            "logprob": float(torch.log(probs[tidx] + 1e-8).item()),
            "z": z_t.squeeze(0).detach().cpu(),
            "role_bit": role_bit.squeeze(0).detach().cpu(),
            "hist_feats": hf.squeeze(0).detach().cpu(),
            "phase_code": phase_code,
            "plan": plan or {},
        }
        if os.getenv("LLM_SPK_DEBUG", "1") == "1":
            dbg = {
                "mode": "bandit",
                "role": role,
                "name": self_name,
                "phase_code": phase_code,
                "template_id": tidx,
                "preview": text[:120],
                "plan": plan or {},
            }
            print("[LLM-SPK]", json.dumps(dbg), flush=True)
        return text if text else SAFE_FALLBACK, meta

    # === Trainability, persistence ===
    def attach_optimizers(self, bandit_lr: float = 1e-3, bias_lr: float = 1e-3):
        """
        Create optimizers only if there are trainable parameters.
        """
        bandit_params = _trainable_params(getattr(self, "bandit", None)) if getattr(self, "bandit", None) is not None else []
        self.bandit_opt = torch.optim.Adam(bandit_params, lr=bandit_lr) if bandit_params else None

        if self.bias and self.bias.head is not None:
            bias_params = _trainable_params(self.bias.head)
            self.bias_opt = torch.optim.Adam(bias_params, lr=bias_lr) if bias_params else None
        else:
            self.bias_opt = None

    def learn_step(self, batch: List[Dict[str, Any]], entropy_bonus: float = 0.01, baseline: float = 0.0) -> Dict[str, float]:
        stats = {"bandit_loss": 0.0, "bias_loss": 0.0, "R_mean": 0.0}
        if not batch:
            return stats

        bandit_batch = [b for b in batch if b.get("mode") == "bandit"]
        bias_batch   = [b for b in batch if b.get("mode") == "llm"]

        if bandit_batch and self.bandit_opt is not None:
            stats_b = self.bandit.learn_step(bandit_batch, self.bandit_opt, entropy_bonus=entropy_bonus, baseline=baseline)
            stats["bandit_loss"] = stats_b["loss"]
            stats["R_mean"] = stats_b["R_mean"]

        if bias_batch and self.bias_opt is not None and self.bias and self.bias.head is not None:
            mini = [{"z": b["z"], "reward": b["reward"]} for b in bias_batch if "z" in b and "reward" in b]
            if mini:
                stats_h = self.bias.learn_step(mini, self.bias_opt, entropy_bonus=0.0)
                stats["bias_loss"] = stats_h["loss"]
        return stats

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "bandit": self.bandit.state_dict(),
            "templates": self.templates,
            "router": {"use_llm": self.use_llm, "max_bad": self.max_bad},
        }
        if self.bias and self.bias.head is not None:
            payload["bias_head"] = self.bias.head.state_dict()
        torch.save(payload, path)

    def load(self, path: str, strict: bool = True):
        st = torch.load(path, map_location="cpu")
        if "bandit" in st:
            self.bandit.load_state_dict(st["bandit"], strict=strict)
        if "bias_head" in st and self.bias and self.bias.head is not None:
            self.bias.head.load_state_dict(st["bias_head"], strict=strict)
        if "templates" in st:
            self.templates = [_ascii_norm(t) for t in st["templates"]]
        if "router" in st:
            r = st["router"]
            env_gate = os.getenv("LLM_SPEAKER", "")
            if env_gate.strip():
                self.use_llm = env_gate.strip().lower() in ("1", "true", "yes", "y", "on")
            else:
                self.use_llm = bool(r.get("use_llm", self.use_llm))
            self.max_bad = int(r.get("max_bad", self.max_bad))

# Explicit exports
__all__ = [
    "SpeakerPolicy",
    "SpeakerBandit",
    "LLMBiasAdapter",
    "DEFAULT_TEMPLATES",
    "make_hist_feats",
    "build_plan_tuple",           # NEW (dual-signature)
    "choose_salient_target",      # NEW
    "postprocess_text",           # NEW (used by sim.py)
]
