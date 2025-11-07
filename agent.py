import os
import re
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from typing import List, Dict, Optional, Set
import yaml
import hashlib, random  # NEW: for seeded lightweight persona

from encoders import (
    MessageEncoder,
    MLPBeliefEncoder,
    WorldModelMLP,
    ActionEncoder,
    package_features,
    INPUT_DIM,
    LATENT_DIM,
    ACTION_DIM,
    PlannerHead,            # legacy single-head (kept for back-compat)
    PhaseActionEncoder,     # used in training; stored on agent for later
    TalkHead,
    VoteHead,
    KillHead,
    FactorizedPlanner,      # real multi-head wrapper from encoders
)

# Unified mouthpiece (LLM router + bandit)
from speaker import SpeakerPolicy, DEFAULT_TEMPLATES

# Social influence module
from social import SocialInfluence

# Load config once
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f) or {}

# ---------- OS ENV SHIM HELPERS ----------
def _env_bool(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return str(val).strip().lower() in ("1", "true", "yes", "y", "on")

def _env_int(key: str, default: int) -> int:
    val = os.getenv(key)
    try:
        return int(val) if val is not None else default
    except Exception:
        return default

def _env_float(key: str, default: float) -> float:
    val = os.getenv(key)
    try:
        return float(val) if val is not None else default
    except Exception:
        return default
# -----------------------------------------

# Config values
NUM_AGENTS         = _env_int("NUM_AGENTS", int(config.get("sim", {}).get("num_agents", config.get("NUM_AGENTS", 6))))
MAX_MEMORY         = _env_int("MAX_MEMORY", int(config.get("MAX_MEMORY", 20)))
USE_LANGUAGE       = _env_bool("USE_LANGUAGE", bool(config.get("USE_LANGUAGE", True)))
SPEAKER_ENABLED    = _env_bool("SPEAKER_ENABLED", bool(config.get("SPEAKER_ENABLED", False)))
SPEAKER_LR         = _env_float("SPEAKER_LR", float(config.get("SPEAKER_LR", 1e-3)))
SPEAKER_HIST_K     = _env_int("SPEAKER_HIST_K", int(config.get("SPEAKER_HIST_K", 3)))
NUM_TALK_CATS      = _env_int("NUM_TALK_CATS", int(config.get("NUM_TALK_CATS", 5)))
# Lightweight persona/style knobs
SPEAKER_TEMP_SCALE = _env_float("SPEAKER_TEMP_SCALE", float(config.get("SPEAKER_TEMP_SCALE", 1.0)))
SPEAKER_BIAS_SCALE = _env_float("SPEAKER_BIAS_SCALE", float(config.get("SPEAKER_BIAS_SCALE", 1.0)))

# Phase-5 knobs
BIAS_LR            = _env_float("BIAS_LR", float(config.get("BIAS_LR", 1e-3)))
FUSION_ALPHA_DEF   = _env_float("TALK_FUSION_ALPHA", float(config.get("sim", {}).get("talk_fusion_alpha", 0.5)))

# Social influence knobs
SOCIAL_ENABLED     = _env_bool("SOCIAL_ENABLED", bool(config.get("social", {}).get("enabled", False)))
SOCIAL_SCALE       = _env_float("SOCIAL_SCALE",  float(config.get("social", {}).get("scale", 0.05)))
SOCIAL_LAMBDA_REG  = _env_float("SOCIAL_LAMBDA_REG", float(config.get("social", {}).get("lambda_reg", 1e-3)))
SOCIAL_TRUST_MODE  = str(config.get("social", {}).get("trust", "none")).lower()
SOCIAL_TAU         = _env_float("SOCIAL_TAU", float(config.get("social", {}).get("tau", 0.5)))
SOCIAL_MAX_STEP    = _env_float("SOCIAL_MAX_STEP", float(config.get("social", {}).get("max_step", 0.25)))
SOCIAL_LAMBDA_EXT  = _env_float("SOCIAL_LAMBDA_EXT", float(config.get("social", {}).get("lambda_ext", 1.0)))

# Prefer more-specific sim.social.enabled if present
_SIM_SOCIAL_ENABLED_YAML = config.get("sim", {}).get("social", {})
if not isinstance(_SIM_SOCIAL_ENABLED_YAML, dict):
    _SIM_SOCIAL_ENABLED_YAML = {}
SIM_SOCIAL_ENABLED_CFG = _SIM_SOCIAL_ENABLED_YAML.get("enabled", None)
SIM_SOCIAL_ENABLED = _env_bool("SIM_SOCIAL_ENABLED", bool(SIM_SOCIAL_ENABLED_CFG)) if SIM_SOCIAL_ENABLED_CFG is not None else None

# Optional global seed for reproducible per-agent personas
GLOBAL_SEED = _env_int("GLOBAL_SEED", int(config.get("seed", 0)))

# Phases
PHASES = {"DISCUSS": 0, "VOTE": 1, "NIGHT": 2}

# Phase-1 logging toggle
TELEMETRY_ENABLED = _env_bool("TELEMETRY_ENABLED", bool(config.get("TELEMETRY_ENABLED", True)))

# ───────────────────────── TALK CATEGORY MAPPING ─────────────────────────
# Index convention for TALK_CATEGORIES: ["accuse", "defend", "hedge", "question", "vote"]
ACCUSE_CAT_ID    = 0
DEFEND_CAT_ID    = 1
HEDGE_CAT_ID     = 2
QUESTION_CAT_ID  = 3
VOTE_CAT_ID      = 4

# Map templates → talk category ids (clip to templates length)
TEMPLATE_TO_CAT_ID = [ACCUSE_CAT_ID, DEFEND_CAT_ID, HEDGE_CAT_ID, QUESTION_CAT_ID, VOTE_CAT_ID]
if len(TEMPLATE_TO_CAT_ID) < len(DEFAULT_TEMPLATES):
    TEMPLATE_TO_CAT_ID = TEMPLATE_TO_CAT_ID + [HEDGE_CAT_ID] * (len(DEFAULT_TEMPLATES) - len(TEMPLATE_TO_CAT_ID))
else:
    TEMPLATE_TO_CAT_ID = TEMPLATE_TO_CAT_ID[:len(DEFAULT_TEMPLATES)]

# ───────────────────────── Phase-7: lightweight dialog state & parsing ─────────────────────────
_AGENT_RE = re.compile(r"\bAgent_(\d+)\b", flags=re.IGNORECASE)

INTENT_LEX = {
    "accuse":  ["is a wolf", "is the wolf", "is werewolf", "suspect", "guilty", "suspicious", "frame", "blame"],
    "defend":  ["not a wolf", "innocent", "trust", "clear", "defend", "towny", "seems fine"],
    "vote":    ["we should vote", "vote ", "vote out", "eliminate", "lynch"],
    "ask":     ["why", "how", "what about", "explain", "can you", "do you", "?", "clarify"],
    "hedge":   ["maybe", "might", "could", "unsure", "not sure", "perhaps", "seems"],
}

def _infer_intent_quick(text_lower: str) -> str:
    if any(k in text_lower for k in INTENT_LEX["vote"]):   return "vote"
    if any(k in text_lower for k in INTENT_LEX["accuse"]): return "accuse"
    if any(k in text_lower for k in INTENT_LEX["defend"]): return "defend"
    if any(k in text_lower for k in INTENT_LEX["ask"]):    return "ask"
    return "hedge"

def _mentions(text: str) -> list[str]:
    return [f"Agent_{m.group(1)}" for m in _AGENT_RE.finditer(text or "")]

def _first_target(text: str) -> Optional[str]:
    ms = _mentions(text)
    return ms[0] if ms else None

def _bigrams(tokens: list[str]) -> list[tuple[str, str]]:
    return list(zip(tokens, tokens[1:]))

# ───────────────────────── NEW: seeded lightweight persona helpers ─────────────────────────
def _seeded_rng(name: str, run_id: str = "") -> random.Random:
    h = int(hashlib.md5(f"{name}|{run_id}".encode("utf-8")).hexdigest(), 16) % (2**32)
    return random.Random(h)

def _make_persona(name: str, run_id: str = "") -> dict:
    r = _seeded_rng(name, run_id)
    return {
        "verbosity":     r.uniform(0.0, 1.0),
        "assertiveness": r.uniform(0.0, 1.0),
        "hedging":       r.uniform(0.0, 1.0),
        "politeness":    r.uniform(0.0, 1.0),
    }
# ───────────────────────────────────────────────────────────────────────────

class DialogState:
    """Lightweight rolling dialog memory for Phase-7 natural language grounding."""
    def __init__(self, k: int = 8, bigram_cap: int = 200):
        from collections import deque as _dq
        self.last_targets = _dq(maxlen=k)     # [(round, speaker, target)]
        self.open_questions = _dq(maxlen=k)   # [(round, asker, to_whom)]
        self.recent_claims = _dq(maxlen=k)    # [(round, speaker, about, stance)]
        self.used_bigrams = _dq(maxlen=bigram_cap)

    def update_from_msg(self, round_id: int, speaker: str, text: str):
        t = (text or "").strip()
        if not t:
            return
        t_lower = t.lower()

        intent = _infer_intent_quick(t_lower)
        target = _first_target(t)
        mentions = _mentions(t)
        has_q = ("?" in t)

        if target is not None:
            self.last_targets.append((round_id, speaker, target))

        if has_q:
            to_whom = mentions[0] if mentions else None
            self.open_questions.append((round_id, speaker, to_whom))

        stance = {"accuse": "accuse", "defend": "defend"}.get(intent, "doubt" if intent == "hedge" else None)
        if stance is not None:
            about = target or (mentions[0] if mentions else None)
            self.recent_claims.append((round_id, speaker, about, stance))

        toks = [w for w in re.split(r"\s+", t) if w]
        for bg in _bigrams(toks):
            self.used_bigrams.append(bg)

    def salient_target(self, alive_names: list[str]) -> Optional[str]:
        for _, _, tgt in reversed(self.last_targets):
            if tgt in alive_names:
                return tgt

        counts = {}
        for _, _, about, stance in self.recent_claims:
            if about in alive_names and stance == "accuse":
                counts[about] = counts.get(about, 0) + 1
        if counts:
            return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

        if not alive_names:
            return None
        rare = {n: 0 for n in alive_names}
        for _, _, about, _ in self.recent_claims:
            if about in rare:
                rare[about] += 1
        return sorted(rare.items(), key=lambda kv: (kv[1], kv[0]))[0][0]

    def question_scarcity(self, lookback: int = 3) -> bool:
        recent = list(itertools.islice(reversed(self.open_questions), 0, lookback))
        return len(recent) == 0

    def snapshot(self) -> dict:
        return {
            "last_targets": list(self.last_targets),
            "open_questions": list(self.open_questions),
            "recent_claims": list(self.recent_claims),
            "used_bigrams": list(self.used_bigrams)[-50:],
        }


class BaseAgent:
    """Single Werewolf/Villager agent driven by JEPA components plus a mouthpiece."""

    def __init__(
        self,
        name: str,
        encoder: Optional[MLPBeliefEncoder] = None,
        world_model: Optional[WorldModelMLP] = None,
        action_encoder: Optional[ActionEncoder] = None,
        planner: Optional[PlannerHead] = None,
        planner_factorized: Optional[FactorizedPlanner] = None,
        phase_action_encoder: Optional[PhaseActionEncoder] = None,
    ) -> None:
        self.name = name
        self.role: Optional[str] = None
        self.alive: bool = True
        self.last_message: str = ""

        # Legacy LLM hooks
        self.llm_fn = None
        self.llm_tokenizer = None

        # Speech bookkeeping
        self.talk_category_last: int = -1
        self.w_bias_sparse: Optional[Dict[int, float]] = None
        self.speaker_mode: str = "none"
        self.persona_norm: float = 0.0
        self.persona_effects: Dict[str, float] = {}

        # Lightweight persona/style knobs
        self.speaker_temp_scale: float = float(SPEAKER_TEMP_SCALE)
        self.bias_scale: float = float(SPEAKER_BIAS_SCALE)

        # JEPA sub-modules
        self.message_encoder = MessageEncoder()
        self.encoder        = encoder or MLPBeliefEncoder(input_dim=INPUT_DIM, latent_dim=LATENT_DIM)
        self.world_model    = world_model or WorldModelMLP(latent_dim=LATENT_DIM, action_dim=ACTION_DIM)
        self.action_encoder = action_encoder or ActionEncoder(num_actions=NUM_AGENTS, action_dim=ACTION_DIM)

        # Planners
        self.planner        = planner or PlannerHead(latent_dim=LATENT_DIM, num_agents=NUM_AGENTS)
        self.planner_factorized = planner_factorized or FactorizedPlanner(
            latent_dim=LATENT_DIM, num_agents=NUM_AGENTS, num_talk_cats=NUM_TALK_CATS
        )
        self.phase_action_encoder = phase_action_encoder

        # Memories exposed to sim
        self.vote_history: List[str] = []
        self.latent_history: List[torch.Tensor] = []
        self.heard_messages: Dict[str, str] = {}
        self.message_memory: deque[tuple[str, str]] = deque(maxlen=MAX_MEMORY)  # required by sim

        # Phase-7 lightweight dialog memory
        self.dialog_state = DialogState(k=8, bigram_cap=200)

        # Training buffer for message-level rewards, consumed by judge
        self.msg_buffer: List[dict] = []  # required by sim

        # Minimal per-step telemetry container
        self.telemetry: Dict = {}

        # Pack awareness
        self.is_wolf: bool = False
        self.wolf_ids: Set[int] = set()

        # Per-step cache for rollout construction
        self._step_cache: Dict = {}

        # Unified mouthpiece
        self.speaker = SpeakerPolicy(latent_dim=LATENT_DIM, templates=DEFAULT_TEMPLATES)
        self.speaker.attach_optimizers(bandit_lr=SPEAKER_LR, bias_lr=BIAS_LR)

        # Back-compat aliases used by fusion paths
        self.llm_bias_head = getattr(self.speaker.bias, "head", None)
        self.bias_head = getattr(self.speaker.bias, "head", None)

        # Router gate aligned with config and env
        try:
            yaml_flag = bool(config.get("llm", {}).get("speaker_enabled", False))
        except Exception:
            yaml_flag = False
        self.speaker.use_llm = _env_bool("LLM_SPEAKER", yaml_flag)

        # Soft prefix / named-target hint
        self.named_target_hint: Optional[str] = None

        # Social Influence
        if SIM_SOCIAL_ENABLED is not None:
            self.social_enabled: bool = bool(SIM_SOCIAL_ENABLED)
        else:
            self.social_enabled: bool = bool(SOCIAL_ENABLED)

        self.social_lambda_ext: float = float(SOCIAL_LAMBDA_EXT)
        self.last_delta_social_norm: float = 0.0
        self._last_social: Dict[str, object] = {
            "delta": None,
            "info": {},
            "disabled": not self.social_enabled,
        }
        self.social: Optional[SocialInfluence] = SocialInfluence(
            latent_dim=LATENT_DIM,
            scale=SOCIAL_SCALE,
            hidden=64,
            reg_lambda=SOCIAL_LAMBDA_REG,
            trust_mode=SOCIAL_TRUST_MODE,
            tau=SOCIAL_TAU,
            max_step=SOCIAL_MAX_STEP,
        )

        # ---------- NEW: Seeded, persistent, lightweight persona ----------
        # If the runner injects a run_id elsewhere, we can read it; otherwise empty string.
        self.run_id = getattr(self, "run_id", os.getenv("RUN_ID", ""))
        self.persona: Dict[str, float] = _make_persona(self.name, self.run_id)

        # ---------- Lightweight persona vector (existing) ----------
        # Deterministic tiny persona vector per agent for tone and phrasing
        try:
            self_idx = int(self.name.split("_")[1])
        except Exception:
            self_idx = 0
        g = torch.Generator()
        g.manual_seed(int(GLOBAL_SEED) + 13 * self_idx)
        # small magnitude, zero mean
        self.persona_vec = torch.randn(4, generator=g) * 0.12
        # derive stable style nudges (keep hedging detached from persona)
        tone_nudge      = float(torch.tanh(self.persona_vec[0]).item()) * 0.15
        question_nudge  = float(torch.tanh(self.persona_vec[2]).item()) * 0.12
        variety_nudge   = float(torch.tanh(self.persona_vec[3]).item()) * 0.08
        # default persona_effects that SpeakerPolicy or llm_script can consume
        self.persona_effects = {
            "tone_formality_delta": tone_nudge,  # negative is casual, positive is formal
            "hedge_prob_delta": 0.0,             # persona does NOT affect hedging
            "question_prob_delta": question_nudge,
            "variety_nudge": variety_nudge,
        }
        # initial norm for logging
        self._update_persona_norm_if_present()

    # ───────────────────────── planner dimension self-heal + accessor ─────────────────────────
    def _ensure_planner_dims(self, target_num_agents: int):
        """
        If planner heads were initialized with the wrong num_agents (e.g., 6),
        rebuild them to match the current roster size (e.g., 9).
        """
        try:
            p = getattr(self, "planner_factorized", None)
            if p is not None:
                have = getattr(p.vote.net[-1], "out_features", None)
                if have is not None and have != target_num_agents:
                    self.planner_factorized = FactorizedPlanner(
                        latent_dim=LATENT_DIM,
                        num_agents=int(target_num_agents),
                        num_talk_cats=int(NUM_TALK_CATS),
                        temperature=1.0,
                    )
            q = getattr(self, "planner", None)
            if q is not None and hasattr(q, "net"):
                have_q = getattr(q.net[-1], "out_features", None)
                if have_q is not None and have_q != target_num_agents:
                    self.planner = PlannerHead(LATENT_DIM, int(target_num_agents))
        except Exception:
            # Non-fatal: leave as-is; masks will still protect us but sanity checks may trip.
            pass

    def get_planner(self, num_agents_override: int | None = None):
        """
        Returns the active planner. If an override is provided, ensure heads match it.
        """
        if num_agents_override is not None:
            self._ensure_planner_dims(int(num_agents_override))
        if hasattr(self, "planner_factorized") and self.planner_factorized is not None:
            return self.planner_factorized
        return self.planner

    # ───────────────────────── pack/role helpers ─────────────────────────
    def set_role(self, role: str) -> None:
        self.role = role
        r = (role or "").lower()
        self.is_wolf = (r in ("werewolf", "wolf", "traitor"))

    @property
    def role_bit(self) -> float:
        return 1.0 if self.is_wolf else 0.0

    def set_packmates(self, names: List[str]) -> None:
        ids: Set[int] = set()
        for n in names or []:
            try:
                ids.add(int(n.split("_")[1]))
            except Exception:
                continue
        self.wolf_ids = ids

    def set_persona_effects(self, effects: Optional[Dict[str, float]]):
        self.persona_effects = effects or {}
        self._update_persona_norm_if_present()

    def set_style_knobs(self, *, speaker_temp_scale: Optional[float] = None, bias_scale: Optional[float] = None):
        if speaker_temp_scale is not None:
            self.speaker_temp_scale = float(speaker_temp_scale)
        if bias_scale is not None:
            self.bias_scale = float(bias_scale)

    def _self_idx(self) -> int:
        try:
            return int(self.name.split("_")[1])
        except Exception:
            return 0

    # ───────────────────────── soft prefix steer / hint API ─────────────────────────
    def set_named_target_hint(self, name: Optional[str]) -> None:
        if isinstance(name, str) and name.startswith("Agent_"):
            self.named_target_hint = name
        else:
            self.named_target_hint = None

    def _prefix_from_hint(self) -> str:
        named = getattr(self, "named_target_hint", None)
        if not named:
            return ""
        return f"I think {named} "

    def _latent_prompt_for_llm(self, z_t: torch.Tensor, agents: List["BaseAgent"], *, prefix: str = "") -> str:
        try:
            from llm_script import _latent_prompt_from_agent
            tok = getattr(self, "llm_tokenizer", None)
            base = _latent_prompt_from_agent(tok if tok is not None else None, z_t, self)
            return (prefix + base) if prefix else base
        except Exception:
            heard = "\n".join(f"- {n}: {m.strip()}" for n, m in list(self.message_memory)[-3:] if m.strip()) or "- (no recent messages heard)"
            base = (
                "You are playing a hidden-role social deduction game. "
                "Speak like a player in one short, natural sentence.\n"
                f"Recent dialog:\n{heard}\n"
                f"Your private feeling:\n- {self.decode_z(z_t)}\n"
                f"Your name: {self.name}\n"
            )
            return (prefix + base) if prefix else base

    # ───────────────────────── legacy LLM attach shim ─────────────────────────
    def attach_llm(self, llm_callable=None, tokenizer=None, *, enabled: bool = True):
        try:
            self.speaker.use_llm = bool(enabled)
        except Exception:
            pass
        self.llm_fn = llm_callable
        self.llm_tokenizer = tokenizer
        if tokenizer is not None and llm_callable is not None:
            try:
                setattr(self.llm_fn, "tokenizer", tokenizer)
            except Exception:
                pass

    # ───────────────────────── small helpers ─────────────────────────
    def _recent_texts(self) -> List[str]:
        if not self.message_memory:
            return []
        return [m for (_, m) in list(self.message_memory)[-SPEAKER_HIST_K:] if m and m.strip()]

    def _alive_names(self, agents: List["BaseAgent"]) -> List[str]:
        return [a.name for a in agents if a.alive and a.name != self.name]

    # Message buffer hook used by sim.py day discussion and night chat
    def buffer_message(
        self,
        speaker_name: str,
        text: str,
        *,
        private: bool = False,
        phase: str | None = None,
        round_id: Optional[int] = None,
    ) -> None:
        t = (text or "").strip()
        if not t:
            return

        try:
            if self.message_memory and self.message_memory[-1][0] == speaker_name and self.message_memory[-1][1] == t:
                pass
            else:
                self.message_memory.append((speaker_name, t))
        except Exception:
            try:
                self.message_memory.append((speaker_name, t))
            except Exception:
                pass

        try:
            self.heard_messages[speaker_name] = t
        except Exception:
            pass

        if private:
            if not hasattr(self, "_night_log"):
                self._night_log = []
            self._night_log.append(t)

        try:
            rid = int(round_id if round_id is not None else self.telemetry.get("round", 0))
        except Exception:
            rid = 0
        try:
            self.dialog_state.update_from_msg(rid, speaker_name, t)
        except Exception:
            pass

    def _alive_others(self, agents: List["BaseAgent"]) -> List["BaseAgent"]:
        return [a for a in agents if a.alive and a.name != self.name]

    def _alive_indices(self, agents: List["BaseAgent"]) -> List[int]:
        return [int(a.name.split("_")[1]) for a in self._alive_others(agents)]

    def _update_persona_norm_if_present(self):
        if hasattr(self, "persona_vec"):
            try:
                pv = torch.as_tensor(self.persona_vec, dtype=torch.float32)
                self.persona_norm = float(pv.norm().item())
            except Exception:
                self.persona_norm = 0.0
        else:
            self.persona_norm = 0.0

    def _infer_talk_category(self, text: str) -> int:
        t = (text or "").lower().strip()
        if any(kw in t for kw in ["we should vote", "vote to", "vote out", "vote ", "eliminate ", "lynch "]):
            return VOTE_CAT_ID
        if any(kw in t for kw in ["why", "how", "what about", "explain", "because?"]):
            return QUESTION_CAT_ID
        if any(kw in t for kw in ["i trust", "not a wolf", "innocent", "seems fine", "defend"]):
            return DEFEND_CAT_ID
        if any(kw in t for kw in ["is a wolf", "suspect", "guilty", "looks bad", "suspicious"]):
            return ACCUSE_CAT_ID
        return HEDGE_CAT_ID

    @torch.no_grad()
    def talkhead_simplex(self, z: torch.Tensor) -> Optional[torch.Tensor]:
        try:
            p = self.get_planner(NUM_AGENTS)
            logits = p.talk(z)
            if logits.dim() > 1:
                logits = logits.squeeze(0)
            mask = self.build_talk_mask().to(logits.device)
            if not mask.any():
                return None
            return torch.softmax(logits, dim=-1)
        except Exception:
            return None

    @torch.no_grad()
    def _talk_intent_from_head(self, z: torch.Tensor) -> Optional[int]:
        try:
            p = self.get_planner(NUM_AGENTS)
            logits = p.talk(z)
            if logits.dim() > 1:
                logits = logits.squeeze(0)
            mask = self.build_talk_mask().to(logits.device)
            if not mask.any():
                return None
            masked = torch.full_like(logits, float("-inf"))
            masked[mask] = logits[mask]
            return int(torch.argmax(masked).item())
        except Exception:
            return None

    # ───────────────────────── Perception ─────────────────────────
    def observe(self, agents: List["BaseAgent"]):
        observed: List[tuple[str, str]] = []
        for a in agents:
            if not a.alive or a.name == self.name:
                continue
            msg = (a.last_message or "").strip()
            if not msg:
                continue
            observed.append((a.name, msg))
            self.heard_messages[a.name] = msg
            if self.message_memory and self.message_memory[-1][0] == a.name and self.message_memory[-1][1] == msg:
                continue
            self.message_memory.append((a.name, msg))
            try:
                rid = int(self.telemetry.get("round", 0))
            except Exception:
                rid = 0
            try:
                self.dialog_state.update_from_msg(rid, a.name, msg)
            except Exception:
                pass
        return observed

    # ───────────────────────── Mask builders ─────────────────────────
    def build_talk_mask(self) -> torch.Tensor:
        return torch.ones(NUM_TALK_CATS, dtype=torch.bool)

    def build_vote_mask(self, agents: List["BaseAgent"]) -> torch.Tensor:
        mask = torch.zeros(NUM_AGENTS, dtype=torch.bool)
        for a in agents:
            if a.alive and a.name != self.name:
                try:
                    mask[int(a.name.split("_")[1])] = True
                except Exception:
                    continue
        return mask

    def build_kill_mask(self, agents: List["BaseAgent"]) -> torch.Tensor:
        mask = torch.zeros(NUM_AGENTS, dtype=torch.bool)
        if not self.is_wolf:
            return mask
        for a in agents:
            try:
                idx = int(a.name.split("_")[1])
            except Exception:
                continue
            legal = a.alive and a.name != self.name and (idx not in self.wolf_ids)
            mask[idx] = bool(legal)
        return mask

    # ───────────────────────── Phase normalization ─────────────────────────
    def _normalize_phase_code(self, code) -> int:
        if isinstance(code, int):
            return code if code in (0, 1, 2) else 0
        if isinstance(code, str):
            c = code.strip().lower()
            if c.startswith("discuss") or c.startswith("day_discuss") or c == "talk":
                return PHASES["DISCUSS"]
            if c.startswith("vote"):
                return PHASES["VOTE"]
            if c.startswith("night") or c.startswith("kill"):
                return PHASES["NIGHT"]
        return PHASES["DISCUSS"]

    # ───────────────────────── Action selection (phase-aware) ─────────────────────────
    @torch.no_grad()
    def choose_action_by_phase(self, phase_code: int, round_num: int, agents: List["BaseAgent"]):
        """
        Returns (payload_idx, choice_type).
        DISCUSS: payload_idx in [0..NUM_TALK_CATS-1], choice_type="TALK_INTENT"
        VOTE:    payload_idx in [0..NUM_AGENTS-1], choice_type="VOTE_TARGET"
        NIGHT:   payload_idx in [0..NUM_AGENTS-1], choice_type="KILL_TARGET"
        """
        z = self.encode_current_belief(round_num, agents)
        phase = self._normalize_phase_code(phase_code)
        p = self.get_planner(NUM_AGENTS)

        def _masked_argmax(logits_1d: torch.Tensor, legal_mask_1d: torch.Tensor, head: str):
            if legal_mask_1d is None:
                legal_mask_1d = torch.ones_like(logits_1d, dtype=torch.bool)
            elif legal_mask_1d.dtype != torch.bool:
                legal_mask_1d = legal_mask_1d.bool()
            if not legal_mask_1d.any():
                if TELEMETRY_ENABLED:
                    self.telemetry[f"{head}_no_legal"] = True
                return None, None, []
            masked = torch.full_like(logits_1d, float("-inf"))
            masked[legal_mask_1d] = logits_1d[legal_mask_1d]
            choice = int(torch.argmax(masked).item())
            probs = torch.softmax(logits_1d[legal_mask_1d], dim=-1).detach().cpu().tolist()
            legal_idx = torch.where(legal_mask_1d)[0].tolist()
            return choice, legal_idx, probs

        if phase == PHASES["DISCUSS"]:
            logits = p.talk(z)
            if logits.dim() > 1:
                logits = logits.squeeze(0)
            mask = self.build_talk_mask().to(logits.device)
            choice, legal_idx, probs = _masked_argmax(logits, mask, head="talk")
            if choice is None:
                return None, None
            if TELEMETRY_ENABLED:
                self.telemetry.update({
                    "talk_mask_idx": legal_idx,
                    "talk_choice_id": choice,
                    "talk_probs": probs,
                })
            self.talk_category_last = choice
            return choice, "TALK_INTENT"

        if phase == PHASES["VOTE"]:
            logits = p.vote(z)
            if logits.dim() > 1:
                logits = logits.squeeze(0)
            mask = self.build_vote_mask(agents).to(logits.device)
            choice, legal_idx, probs = _masked_argmax(logits, mask, head="vote")
            if choice is None:
                return None, None
            if TELEMETRY_ENABLED:
                self.telemetry.update({
                    "vote_mask_idx": legal_idx,
                    "vote_choice_idx": choice,
                    "vote_probs": probs,
                })
            self.vote_history.append(f"Agent_{choice}")
            if len(self.vote_history) > MAX_MEMORY:
                self.vote_history.pop(0)
            return choice, "VOTE_TARGET"

        if phase == PHASES["NIGHT"]:
            logits = p.kill(z)
            if logits.dim() > 1:
                logits = logits.squeeze(0)
            mask = self.build_kill_mask(agents).to(logits.device)
            choice, legal_idx, probs = _masked_argmax(logits, mask, head="kill")
            if choice is None:
                return None, None
            return choice, "KILL_TARGET"

        return None, None

    def choose_night_target(self, agents: List["BaseAgent"]):
        candidates = [a for a in agents if a.alive and a.name != self.name]
        return candidates[0].name if candidates else None

    # ───────────────────────── Latent belief encoding ─────────────────────────
    def _fit_input_dim(self, x: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Ensure packaged features match the encoder's expected INPUT_DIM.
        If longer, crop from the tail; if shorter, right-pad zeros.
        """
        want = int(INPUT_DIM)
        have = int(x.shape[-1])
        if have == want:
            return x
        if have > want:
            x_fixed = x[..., :want]
        else:
            pad = torch.zeros(want - have, device=device, dtype=x.dtype)
            x_fixed = torch.cat([x, pad], dim=-1)
        if TELEMETRY_ENABLED:
            try:
                self.telemetry["x_len_before"] = have
                self.telemetry["x_len_after"] = want
            except Exception:
                pass
        return x_fixed

    def encode_current_belief(self, round_num: int, agents: List["BaseAgent"]):
        device = next(self.encoder.parameters()).device

        self_msg_embed = self.message_encoder(self.last_message).squeeze().to(device)
        if torch.isnan(self_msg_embed).any():
            self_msg_embed = torch.zeros_like(self_msg_embed)

        neighbour_msgs = [msg for _, msg in self.observe(agents) if msg] if USE_LANGUAGE else []
        if neighbour_msgs:
            neighbour_embed = self.message_encoder(neighbour_msgs).mean(dim=0).to(device)
        else:
            neighbour_embed = torch.zeros_like(self_msg_embed)

        vote_vec = torch.zeros(NUM_AGENTS, device=device)
        for name in self.vote_history[-MAX_MEMORY:]:
            try:
                vote_vec[int(name.split("_")[1])] += 1.0
            except Exception:
                continue
        if float(vote_vec.sum().item()) > 0.0:
            vote_vec = vote_vec / vote_vec.sum()

        if self.latent_history:
            memory_summary = torch.stack([t.to(device) for t in self.latent_history]).mean(dim=0)
        else:
            memory_summary = torch.zeros(LATENT_DIM, device=device)

        x = package_features(
            agent_alive=self.alive,
            round_num=round_num,
            self_msg_embed=self_msg_embed,
            neighbor_msg_embed=neighbour_embed,
            vote_vector=vote_vec,
            memory_summary=memory_summary,
        )

        # >>> HARDEN: match encoder's expected width (fixes 1x811 vs 808x64) <<<
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32, device=device)
        else:
            x = x.to(device)
        x = self._fit_input_dim(x, device)

        z = self.encoder(x)

        if torch.isnan(z).any() or torch.isinf(z).any():
            raise RuntimeError("NaN/Inf in z latent")

        self.latent_history.append(z.detach())
        if len(self.latent_history) > MAX_MEMORY:
            self.latent_history.pop(0)

        if TELEMETRY_ENABLED:
            self.telemetry.update({
                "round": round_num,
                "alive_count": int(sum(1 for a in agents if a.alive)),
                "self_msg_len": len(self.last_message or ""),
                "z_mean": float(z.mean().item()),
                "z_std": float(z.std().item()),
                "z_norm": float(z.norm().item()),
            })

        return z

    # ───────────────────────── Social influence update ─────────────────────────
    @torch.no_grad()
    def compute_social_update(
        self,
        z_self: torch.Tensor,
        neighbors: List["BaseAgent"],
    ):
        """
        Returns (z_updated, info_dict). If disabled or no neighbors, returns (z_self, info_zero).
        Updates last_delta_social_norm and self._last_social for telemetry.
        """
        device = z_self.device

        if not getattr(self, "social_enabled", False) or self.social is None:
            zero = torch.zeros_like(z_self)
            self.last_delta_social_norm = 0.0
            info_zero = {
                "delta_norm": 0.0,
                "delta_norms": [],
                "trust_mode": "none",
                "n_neighbors": 0,
                "disabled": True,
                "scale": float(getattr(self.social, "scale", 0.0)) if self.social is not None else 0.0,
            }
            self._last_social = {"delta": zero.detach().clone(), "info": dict(info_zero), "disabled": True}
            if TELEMETRY_ENABLED:
                self.telemetry["social_skipped"] = True
                self.telemetry["social_disabled_invariants"] = True
            return z_self, info_zero

        z_neighbors: List[torch.Tensor] = []
        neighbor_roles: List[Optional[str]] = []
        for a in neighbors or []:
            if not getattr(a, "alive", False):
                continue
            if a.name == self.name:
                continue
            hist = getattr(a, "latent_history", None)
            if hist:
                try:
                    z_neighbors.append(hist[-1].detach().to(device))
                    neighbor_roles.append(getattr(a, "role", None))
                except Exception:
                    continue

        if not z_neighbors:
            zero = torch.zeros_like(z_self)
            self.last_delta_social_norm = 0.0
            info_zero = {
                "delta_norm": 0.0,
                "delta_norms": [],
                "trust_mode": getattr(self.social, "trust_mode", "none"),
                "n_neighbors": 0,
                "no_neighbors": True,
                "scale": float(getattr(self.social, "scale", 0.0)),
            }
            self._last_social = {"delta": zero.detach().clone(), "info": dict(info_zero), "disabled": False}
            if TELEMETRY_ENABLED:
                self.telemetry["social_no_neighbors"] = True
            return z_self, info_zero

        delta, info = self.social(
            z_self=z_self.detach().to(device),
            z_neighbors=z_neighbors,
            self_role=self.role,
            neighbor_roles=neighbor_roles,
        )

        lam = float(self.social_lambda_ext)
        z_updated = z_self + lam * delta

        dn = float(delta.norm().item()) * lam
        self.last_delta_social_norm = dn
        info_out = dict(info or {})
        info_out.setdefault("trust_mode", getattr(self.social, "trust_mode", "none"))
        info_out.setdefault("scale", float(getattr(self.social, "scale", 0.0)))
        info_out["n_neighbors"] = info_out.get("n_neighbors", len(z_neighbors))
        info_out["delta_norm"] = dn
        info_out["lambda_ext"] = lam

        self._last_social = {"delta": delta.detach().clone(), "info": dict(info_out), "disabled": False}
        if TELEMETRY_ENABLED:
            self.telemetry.update({
                "delta_social_norm": dn,
                "social_n_neighbors": int(info_out["n_neighbors"]),
                "social_trust_mode": info_out["trust_mode"],
                "social_scale_internal": float(info_out["scale"]),
                "social_lambda_ext": lam,
                "social_w_entropy": float(info_out.get("w_entropy", 0.0)),
                "social_mean_cosine_to_mu": float(info_out.get("mean_cosine_to_mu", 0.0)),
            })

        try:
            self.z_t = z_updated.detach()
        except Exception:
            pass

        return z_updated, info_out

    # ───────────────────────── Rollout aux snapshot ─────────────────────────
    def make_aux(self, agents: List["BaseAgent"]) -> Dict:
        """
        Return fields the sim inspects and whose lengths it checks.
        Includes alive, wolves, and recent text fields.
        """
        alive = [bool(a.alive) for a in agents]

        wolves_attached = [getattr(a, "is_wolf", False) for a in agents]
        if any(wolves_attached):
            wolves = wolves_attached
        else:
            wset = self.wolf_ids
            wolves = [(int(a.name.split("_")[1]) in wset) for a in agents]

        # Recent dialog slice for stable-length checks
        recent_pairs = list(self.message_memory)[-6:]
        recent_speakers = [n for (n, _) in recent_pairs]
        recent_texts = [m for (_, m) in recent_pairs]

        return {
            "alive": alive,
            "self_idx": self._self_idx(),
            "wolves": wolves,
            "recent_speakers": recent_speakers,  # aligned with recent_texts
            "recent_texts": recent_texts,        # last K heard or buffered lines
            "last_message": self.last_message or "",
            "vote_tail": self.vote_history[-6:],
        }

    # ───────────────────────── Human-readable decode (optional) ─────────────────────────
    def decode_z(self, z: torch.Tensor) -> str:
        mean, std = z.mean().item(), z.std().item()
        mood = ("bad feeling about someone." if mean > 0.2
                else "quiet… too quiet." if mean < -0.2
                else "uncertain.")
        confidence = "unsure who to trust." if std > 0.5 else "confident in my suspicions."
        return f"The group seems {mood} I am {confidence}"

    # ───────────────────────── Speak (unified policy) ─────────────────────────
    def speak(self, round_num: int, agents: List["BaseAgent"], *, phase_code: Optional[int] = None):
        """
        Natural, in-scenario dialogue via LLM route with hygiene, or stable fallback via templates.
        Records a training row in msg_buffer for judge and reinforcement.
        """
        z = self.encode_current_belief(round_num, agents)
        self._update_persona_norm_if_present()

        candidate_targets = [a.name for a in agents if a.alive and a.name != self.name]

        prefix_hint = self._prefix_from_hint()
        pe = dict(self.persona_effects or {})
        if prefix_hint:
            pe["_prefix_hint"] = prefix_hint

        pe["_role_bit"] = float(self.role_bit)
        pe["_temp_scale"] = float(self.speaker_temp_scale)
        pe["_bias_scale"] = float(self.bias_scale)

        ds = self.dialog_state.snapshot()
        pe["_ds_last_targets"]    = ds["last_targets"]
        pe["_ds_open_questions"]  = ds["open_questions"]
        pe["_ds_recent_claims"]   = ds["recent_claims"]
        pe["_ds_used_bigrams"]    = ds["used_bigrams"]
        pe["_ds_question_scarce"] = bool(self.dialog_state.question_scarcity(lookback=3))

        # persona-driven phrasing style (NOT uncertainty/hedging)
        persona_norm_val = getattr(self, "persona_norm", 0.0)
        style = {
            "prefer_question": bool(persona_norm_val > 0.4),
            "prefer_named_subject": True,
            "extra_hedge_prob": 0.0,  # do not link persona to hedges
        }
        # expose to mouthpiece
        pe["persona_style"] = style
        # also expose the seeded lightweight persona explicitly
        pe["seeded_persona"] = dict(self.persona)

        # surface to sim aux (when available)
        try:
            if isinstance(getattr(self, "_step_cache", None), dict) and isinstance(self._step_cache.get("aux"), dict):
                self._step_cache["aux"]["persona_style"] = style
        except Exception:
            pass

        ph = self.get_planner(NUM_AGENTS)

        # Call mouthpiece; include persona kw if supported
        try:
            text, meta = self.speaker.generate(
                z_t=z.detach(),
                role=self.role or "Unknown",
                recent_texts=self._recent_texts(),
                candidate_targets=candidate_targets,
                self_name=self.name,
                phase_code=phase_code,
                persona_effects=pe,
                persona=self.persona,               # NEW: pass persona in kwargs (read by speaker_llm if supported)
                planner_heads=ph,
                dialog_state=self.dialog_state,
                prefix=prefix_hint,
                named_target_hint=self.named_target_hint,
            )
        except TypeError:
            # Fallback if SpeakerPolicy.generate does not accept 'persona'
            text, meta = self.speaker.generate(
                z_t=z.detach(),
                role=self.role or "Unknown",
                recent_texts=self._recent_texts(),
                candidate_targets=candidate_targets,
                self_name=self.name,
                phase_code=phase_code,
                persona_effects=pe,
                planner_heads=ph,
                dialog_state=self.dialog_state,
                prefix=prefix_hint,
                named_target_hint=self.named_target_hint,
            )

        self.named_target_hint = None

        self.last_message = text
        self.speaker_mode = meta.get("mode", "none")
        self.w_bias_sparse = meta.get("w_bias_sparse", None)

        try:
            self.dialog_state.update_from_msg(round_num, self.name, text)
        except Exception:
            pass

        intent_id: Optional[int] = None

        template_id = meta.get("template_id", -1)
        if isinstance(template_id, int) and template_id >= 0:
            if template_id < len(TEMPLATE_TO_CAT_ID):
                intent_id = int(TEMPLATE_TO_CAT_ID[template_id])
            else:
                intent_id = HEDGE_CAT_ID

        if intent_id is None:
            z_for_head = meta.get("z", z.detach())
            intent_id = self._talk_intent_from_head(z_for_head)

        if intent_id is None:
            intent_id = int(self._infer_talk_category(text))

        self.talk_category_last = int(intent_id)

        default_role_bit_tensor = torch.tensor(self.role_bit, dtype=torch.float32)

        plan_meta = meta.get("plan", {}) if isinstance(meta, dict) else {}
        # NEW: ensure persona is included in the plan snapshot for training/inspection
        try:
            if isinstance(plan_meta, dict):
                plan_meta.setdefault("persona", dict(self.persona))
        except Exception:
            pass

        self.msg_buffer.append({
            "mode": self.speaker_mode,
            "z": meta.get("z", z.detach().cpu()),
            "role_bit": meta.get("role_bit", default_role_bit_tensor),
            "hist_feats": meta.get("hist_feats", torch.tensor([0.0, 0.0])),
            "template_id": template_id,
            "talk_intent": int(intent_id),
            "text": text,
            "round": round_num,
            "phase_code": phase_code if phase_code is not None else -1,
            "plan_intent": plan_meta.get("intent", None),
            "plan_target": plan_meta.get("target", None),
            "plan_persona": plan_meta.get("persona", None),  # NEW: persisted for downstream use
            "reward": None,
        })

        if TELEMETRY_ENABLED:
            self.telemetry.update({
                "speak_text_len": len(text or ""),
                "talk_category_last": int(self.talk_category_last),
                "talk_intent_id": int(intent_id),
                "speaker_mode": self.speaker_mode,
                "persona_norm": float(self.persona_norm),
                "role_bit": float(self.role_bit),
                "temp_scale": float(self.speaker_temp_scale),
                "bias_scale": float(self.bias_scale),
                "ds_last_targets": len(self.dialog_state.last_targets),
                "ds_open_q": len(self.dialog_state.open_questions),
                "ds_recent_claims": len(self.dialog_state.recent_claims),
                "plan_intent": plan_meta.get("intent", None),
                "plan_target": plan_meta.get("target", None),
                "persona_style_prefer_question": int(style["prefer_question"]),
                "persona_style_named": 1,
                "persona_style_extra_hedge_prob": float(style["extra_hedge_prob"]),
            })
        return text

    # ───────────────────────── Phase-4 rollout helpers ─────────────────────────
    def reset_for_new_game(self):
        self.alive = True
        self.last_message = ""
        self.talk_category_last = -1
        self.w_bias_sparse = None
        self.vote_history.clear()
        self.latent_history.clear()
        self.heard_messages.clear()
        self.message_memory.clear()
        self.telemetry.clear()
        self._step_cache = {}
        self.dialog_state = DialogState(k=8, bigram_cap=200)

    def reset_step_cache(self):
        self._step_cache = {}

    @torch.no_grad()
    def begin_step(self, phase_code: int, round_num: int, agents: List["BaseAgent"]) -> torch.Tensor:
        self.reset_step_cache()
        z_t = self.encode_current_belief(round_num, agents)
        self._step_cache = {
            "z_t": z_t.detach().clone(),
            "phase": int(phase_code),
            "round": int(round_num),
            "aux": self.make_aux(agents),
            "payload": None,
            "choice_type": None,
        }
        return z_t

    @torch.no_grad()
    def decide(self, phase_code: int, round_num: int, agents: List["BaseAgent"]):
        payload_idx, choice_type = self.choose_action_by_phase(phase_code, round_num, agents)
        if not getattr(self, "_step_cache", None):
            _ = self.begin_step(phase_code, round_num, agents)
        self._step_cache["payload"] = None if payload_idx is None else int(payload_idx)
        self._step_cache["choice_type"] = choice_type
        return payload_idx, "TALK_INTENT" if choice_type == "TALK_INTENT" else choice_type

    @torch.no_grad()
    def finalize_step(self, agents: List["BaseAgent"], z_next: Optional[torch.Tensor] = None):
        if not getattr(self, "_step_cache", None):
            raise RuntimeError("finalize_step() called before begin_step()/decide().")

        z_t = self._step_cache.get("z_t")
        phase = self._step_cache.get("phase")
        payload = self._step_cache.get("payload")
        choice_type = self._step_cache.get("choice_type")
        aux = self._step_cache.get("aux")

        if z_next is None:
            round_num = int(self._step_cache.get("round", 0))
            z_next = self.encode_current_belief(round_num, agents)

        z_t_cpu = z_t.detach().cpu()
        z_next_cpu = z_next.detach().cpu()

        row = (
            z_t_cpu,
            int(phase),
            int(payload) if payload is not None else 0,
            z_next_cpu,
            (self.role or "Unknown"),
            choice_type,
            aux,
        )
        self.reset_step_cache()
        return row
