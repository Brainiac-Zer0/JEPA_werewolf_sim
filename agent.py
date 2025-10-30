import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from typing import List, Dict, Optional, Set
import yaml

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
    FactorizedPlanner,      # ✅ real multi-head wrapper from encoders
)

# NEW: unified mouthpiece (LLM router + bandit)
from speaker import SpeakerPolicy, DEFAULT_TEMPLATES

# NEW: social influence module (Stage A)
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

# Config values (env overrides take precedence; YAML is the fallback)
NUM_AGENTS         = _env_int("NUM_AGENTS", int(config.get("sim", {}).get("num_agents", config.get("NUM_AGENTS", 6))))
MAX_MEMORY         = _env_int("MAX_MEMORY", int(config.get("MAX_MEMORY", 20)))
USE_LANGUAGE       = _env_bool("USE_LANGUAGE", bool(config.get("USE_LANGUAGE", True)))
SPEAKER_ENABLED    = _env_bool("SPEAKER_ENABLED", bool(config.get("SPEAKER_ENABLED", False)))
SPEAKER_LR         = _env_float("SPEAKER_LR", float(config.get("SPEAKER_LR", 1e-3)))
SPEAKER_HIST_K     = _env_int("SPEAKER_HIST_K", int(config.get("SPEAKER_HIST_K", 3)))
NUM_TALK_CATS      = _env_int("NUM_TALK_CATS", int(config.get("NUM_TALK_CATS", 5)))
# Lightweight persona/style knobs (NEW)
SPEAKER_TEMP_SCALE = _env_float("SPEAKER_TEMP_SCALE", float(config.get("SPEAKER_TEMP_SCALE", 1.0)))
SPEAKER_BIAS_SCALE = _env_float("SPEAKER_BIAS_SCALE", float(config.get("SPEAKER_BIAS_SCALE", 1.0)))

# Phase-5 knobs
BIAS_LR            = _env_float("BIAS_LR", float(config.get("BIAS_LR", 1e-3)))
FUSION_ALPHA_DEF   = _env_float("TALK_FUSION_ALPHA", float(config.get("sim", {}).get("talk_fusion_alpha", 0.5)))

# --- Social influence knobs (Stage A) ---
# Base social toggle from root "social.enabled"
SOCIAL_ENABLED     = _env_bool("SOCIAL_ENABLED", bool(config.get("social", {}).get("enabled", False)))
SOCIAL_SCALE       = _env_float("SOCIAL_SCALE",  float(config.get("social", {}).get("scale", 0.05)))
SOCIAL_LAMBDA_REG  = _env_float("SOCIAL_LAMBDA_REG", float(config.get("social", {}).get("lambda_reg", 1e-3)))
SOCIAL_TRUST_MODE  = str(config.get("social", {}).get("trust", "none")).lower()
SOCIAL_TAU         = _env_float("SOCIAL_TAU", float(config.get("social", {}).get("tau", 0.5)))
SOCIAL_MAX_STEP    = _env_float("SOCIAL_MAX_STEP", float(config.get("social", {}).get("max_step", 0.25)))
# Optional external multiplier λ_social (applied on top of the module’s internal scale)
SOCIAL_LAMBDA_EXT  = _env_float("SOCIAL_LAMBDA_EXT", float(config.get("social", {}).get("lambda_ext", 1.0)))

# Prefer more-specific sim.social.enabled if present (env > YAML)
_SIM_SOCIAL_ENABLED_YAML = config.get("sim", {}).get("social", {})
if not isinstance(_SIM_SOCIAL_ENABLED_YAML, dict):
    _SIM_SOCIAL_ENABLED_YAML = {}
SIM_SOCIAL_ENABLED_CFG = _SIM_SOCIAL_ENABLED_YAML.get("enabled", None)
SIM_SOCIAL_ENABLED = _env_bool("SIM_SOCIAL_ENABLED", bool(SIM_SOCIAL_ENABLED_CFG)) if SIM_SOCIAL_ENABLED_CFG is not None else None

# Phases
PHASES = {"DISCUSS": 0, "VOTE": 1, "NIGHT": 2}

# Phase-1 logging toggle (read by sim.py)
TELEMETRY_ENABLED = _env_bool("TELEMETRY_ENABLED", bool(config.get("TELEMETRY_ENABLED", True)))

# ───────────────────────── TALK CATEGORY MAPPING (deterministic) ─────────────────────────
# Index convention for TALK_CATEGORIES: ["accuse", "defend", "hedge", "question", "vote"]
ACCUSE_CAT_ID    = 0
DEFEND_CAT_ID    = 1
HEDGE_CAT_ID     = 2
QUESTION_CAT_ID  = 3
VOTE_CAT_ID      = 4

# Map templates → talk category ids (truncate/clip to templates length)
TEMPLATE_TO_CAT_ID = [ACCUSE_CAT_ID, DEFEND_CAT_ID, HEDGE_CAT_ID, QUESTION_CAT_ID, VOTE_CAT_ID]
if len(TEMPLATE_TO_CAT_ID) < len(DEFAULT_TEMPLATES):
    TEMPLATE_TO_CAT_ID = TEMPLATE_TO_CAT_ID + [HEDGE_CAT_ID] * (len(DEFAULT_TEMPLATES) - len(TEMPLATE_TO_CAT_ID))
else:
    TEMPLATE_TO_CAT_ID = TEMPLATE_TO_CAT_ID[:len(DEFAULT_TEMPLATES)]


class BaseAgent:
    """Single Werewolf/Villager agent driven by JEPA components + a mouthpiece (LLM or trainable Speaker)."""

    def __init__(
        self,
        name: str,
        encoder: Optional[MLPBeliefEncoder] = None,
        world_model: Optional[WorldModelMLP] = None,
        action_encoder: Optional[ActionEncoder] = None,
        planner: Optional[PlannerHead] = None,                  # legacy single-head (vote only)
        planner_factorized: Optional[FactorizedPlanner] = None, # NEW: multi-head (talk/vote/kill)
        phase_action_encoder: Optional[PhaseActionEncoder] = None, # NEW: stored for training
    ) -> None:
        self.name = name
        self.role: Optional[str] = None
        self.alive: bool = True
        self.last_message: str = ""

        # Legacy LLM hooks (kept for back-compat; not used by SpeakerPolicy)
        self.llm_fn = None
        self.llm_tokenizer = None

        # Speech bookkeeping (read by sim.py)
        self.talk_category_last: int = -1  # ← maintained for legacy fill-ins
        self.speaker_mode: str = "none"   # {"bandit","llm","none"}
        self.persona_norm: float = 0.0
        self.persona_effects: Dict[str, float] = {}  # ← exposed for steering/training

        # Lightweight persona/style knobs (NEW)
        self.speaker_temp_scale: float = float(SPEAKER_TEMP_SCALE)
        self.bias_scale: float = float(SPEAKER_BIAS_SCALE)

        # JEPA sub-modules
        self.message_encoder = MessageEncoder()  # may be overwritten with shared instance by sim
        self.encoder        = encoder or MLPBeliefEncoder(input_dim=INPUT_DIM, latent_dim=LATENT_DIM)
        self.world_model    = world_model or WorldModelMLP(latent_dim=LATENT_DIM, action_dim=ACTION_DIM)
        self.action_encoder = action_encoder or ActionEncoder(num_actions=NUM_AGENTS, action_dim=ACTION_DIM)

        # Legacy single-head planner (kept for back-compat paths)
        self.planner        = planner or PlannerHead(latent_dim=LATENT_DIM, num_agents=NUM_AGENTS)
        # Preferred path: factorized heads
        self.planner_factorized = planner_factorized or FactorizedPlanner(
            latent_dim=LATENT_DIM, num_agents=NUM_AGENTS, num_talk_cats=NUM_TALK_CATS
        )
        # Stored for training (not consumed by agent logic)
        self.phase_action_encoder = phase_action_encoder

        # Memories
        self.vote_history: List[str] = []
        self.latent_history: List[torch.Tensor] = []
        self.heard_messages: Dict[str, str] = {}
        self.message_memory: deque[tuple[str, str]] = deque(maxlen=MAX_MEMORY)

        # Training buffer for message-level rewards (judge fills reward later)
        # {mode,z,role_bit,hist_feats,template_id,talk_intent,text,round,phase_code,reward}
        self.msg_buffer: List[dict] = []  # ← guaranteed to exist

        # Minimal per-step telemetry container (sim writes CSV from this)
        self.telemetry: Dict = {}

        # Pack awareness
        self.is_wolf: bool = False
        self.wolf_ids: Set[int] = set()

        # Per-step cache for Phase-4 rollout construction
        self._step_cache: Dict = {}

        # Unified mouthpiece (routes LLM ↔ bandit, applies hygiene, trainable)
        self.speaker = SpeakerPolicy(latent_dim=LATENT_DIM, templates=DEFAULT_TEMPLATES)
        self.speaker.attach_optimizers(bandit_lr=SPEAKER_LR, bias_lr=BIAS_LR)

        # Back-compat aliases used by external/fusion paths
        self.llm_bias_head = getattr(self.speaker.bias, "head", None)
        # NEW: expose bias_head (used by sim.py for fused intent)
        self.bias_head = getattr(self.speaker.bias, "head", None)

        # Optional: honor env toggle to enable/disable LLM route at init
        try:
            self.speaker.use_llm = bool(config.get("llm", {}).get("speaker_enabled", False))
        except Exception:
            pass

        # NEW: soft prefix / named-target hint (consumed once by speak())
        self.named_target_hint: Optional[str] = None

        # --- Social Influence (Stage A) ---
        # Prefer sim.social.enabled if explicitly set; else fall back to SOCIAL_ENABLED
        if SIM_SOCIAL_ENABLED is not None:
            self.social_enabled: bool = bool(SIM_SOCIAL_ENABLED)
        else:
            self.social_enabled: bool = bool(SOCIAL_ENABLED)

        self.social_lambda_ext: float = float(SOCIAL_LAMBDA_EXT)  # external λ_social multiplier
        self.last_delta_social_norm: float = 0.0  # telemetry
        self._last_social: Dict[str, object] = {  # for tests/invariants
            "delta": None,
            "info": {},
            "disabled": not self.social_enabled,
        }

        # Instantiate influence model; safe even if disabled
        self.social: Optional[SocialInfluence] = SocialInfluence(
            latent_dim=LATENT_DIM,
            scale=SOCIAL_SCALE,
            hidden=64,
            reg_lambda=SOCIAL_LAMBDA_REG,
            trust_mode=SOCIAL_TRUST_MODE,
            tau=SOCIAL_TAU,
            max_step=SOCIAL_MAX_STEP,
        )

    # ───────────────────────── pack/role helpers ─────────────────────────
    def set_role(self, role: str) -> None:
        self.role = role
        r = (role or "").lower()
        self.is_wolf = (r in ("werewolf", "wolf", "traitor"))

    @property
    def role_bit(self) -> float:
        """Simple scalar for mouthpiece inputs: wolf=1.0, villager=0.0."""
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
        """Setter so sims/tests can steer personality and temperature nudges."""
        self.persona_effects = effects or {}

    def set_style_knobs(self, *, speaker_temp_scale: Optional[float] = None, bias_scale: Optional[float] = None):
        """Lightweight per-agent style knobs (NEW)."""
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
        """
        Store a one-shot hint like 'Agent_3' that will lightly steer the next utterance.
        The hint is cleared immediately after speak() consumes it.
        """
        if isinstance(name, str) and name.startswith("Agent_"):
            self.named_target_hint = name
        else:
            self.named_target_hint = None

    def _prefix_from_hint(self) -> str:
        """
        Turn a stored target hint into a tiny, natural-language nudge.
        Mouthpiece hygiene keeps replies single-sentence and in-world.
        """
        named = getattr(self, "named_target_hint", None)
        if not named:
            return ""
        # Extremely light steer; no punctuation to avoid overconstraining the LLM.
        return f"I think {named} "

    def _latent_prompt_for_llm(self, z_t: torch.Tensor, agents: List["BaseAgent"], *, prefix: str = "") -> str:
        """
        Optional helper: reuse llm_script latent→prompt builder and add a prefix nudge.
        Not used by default (SpeakerPolicy builds prompts), but handy for tests.
        """
        try:
            from llm_script import _latent_prompt_from_agent
            tok = getattr(self, "llm_tokenizer", None)
            base = _latent_prompt_from_agent(tok if tok is not None else None, z_t, self)
            return (prefix + base) if prefix else base
        except Exception:
            # Fallback: minimal inline prompt if llm_script is unavailable
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
        """
        Back-compat shim:
          - toggles the LLM route inside SpeakerPolicy
          - preserves old signature (llm_fn, tokenizer) even though the policy
            manages its own backend internally
        """
        try:
            self.speaker.use_llm = bool(enabled)
        except Exception:
            pass
        # Keep these for external tools/tests that probe attributes
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

    # NEW: message buffer hook used by sim.py day discussion & night chat
    def buffer_message(
        self,
        speaker_name: str,
        text: str,
        *,
        private: bool = False,
        phase: str | None = None,
    ) -> None:
        """
        Store a heard message into this agent's rolling memory.

        - speaker_name: who said it (e.g., "Agent_3")
        - text: the utterance (already sanitized by caller)
        - private: True for werewolf night-chat messages
        - phase: optional phase tag ("DISCUSS", "NIGHT", etc.) for future use
        """
        t = (text or "").strip()
        if not t:
            return

        # Avoid immediate duplicates from the same speaker
        try:
            if self.message_memory and self.message_memory[-1][0] == speaker_name and self.message_memory[-1][1] == t:
                pass
            else:
                self.message_memory.append((speaker_name, t))
        except Exception:
            # Fallback append without dedupe
            try:
                self.message_memory.append((speaker_name, t))
            except Exception:
                pass

        # Optional "heard last" map
        try:
            self.heard_messages[speaker_name] = t
        except Exception:
            pass

        # Private night chat log (queried by smoke tests)
        if private:
            if not hasattr(self, "_night_log"):
                self._night_log = []
            self._night_log.append(t)

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
            logits = self.planner_factorized.talk(z)
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
        """Return intent id from TalkHead masked argmax, or None if unavailable."""
        try:
            logits = self.planner_factorized.talk(z)
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
        """
        Pull visible last_message snippets from others.
        Hygiene: only record non-empty messages and avoid immediate duplicates.
        """
        observed: List[tuple[str, str]] = []
        for a in agents:
            if not a.alive or a.name == self.name:
                continue
            msg = (a.last_message or "").strip()
            if not msg:
                continue
            observed.append((a.name, msg))
            self.heard_messages[a.name] = msg
            # Deduplicate consecutive repeats per speaker
            if self.message_memory and self.message_memory[-1][0] == a.name and self.message_memory[-1][1] == msg:
                continue
            self.message_memory.append((a.name, msg))
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
            # Legal if alive, not self, and not a wolf
            legal = a.alive and a.name != self.name and (idx not in self.wolf_ids)
            mask[idx] = bool(legal)
        return mask

    # ───────────────────────── Phase normalization ─────────────────────────
    def _normalize_phase_code(self, code) -> int:
        """
        Map incoming phase identifiers to our internal indices:
          0/DISCUSS*, 1/VOTE*, 2/NIGHT*
        Unrecognized → DISCUSS (0) so we still attempt to talk.
        """
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
        Returns (payload_idx, choice_type) where:
          - DISCUSS: payload_idx ∈ [0..NUM_TALK_CATS-1], choice_type="TALK_INTENT"
          - VOTE:    payload_idx ∈ [0..NUM_AGENTS-1], choice_type="VOTE_TARGET"
          - NIGHT:   payload_idx ∈ [0..NUM_AGENTS-1], choice_type="KILL_TARGET" (wolves only)
        """
        z = self.encode_current_belief(round_num, agents)
        phase = self._normalize_phase_code(phase_code)

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
            logits = self.planner_factorized.talk(z)
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
            logits = self.planner_factorized.vote(z)
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
            logits = self.planner_factorized.kill(z)
            if logits.dim() > 1:
                logits = logits.squeeze(0)
            mask = self.build_kill_mask(agents).to(logits.device)
            choice, legal_idx, probs = _masked_argmax(logits, mask, head="kill")
            if choice is None:
                return None, None
            return choice, "KILL_TARGET"

        # Fallback
        return None, None

    def choose_night_target(self, agents: List["BaseAgent"]):
        candidates = [a for a in agents if a.alive and a.name != self.name]
        return candidates[0].name if candidates else None

    # ───────────────────────── Latent belief encoding ─────────────────────────
    def encode_current_belief(self, round_num: int, agents: List["BaseAgent"]):
        """
        Public belief encoder used throughout the sim (guaranteed for fallbacks).
        """
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

    # ───────────────────────── Social influence update (Stage A) ─────────────────────────
    @torch.no_grad()
    def compute_social_update(
        self,
        z_self: torch.Tensor,
        neighbors: List["BaseAgent"],
    ):
        """
        Stage A:
          1) gather neighbors' most recent latents,
          2) δ_social = self.social(z_self, z_neighbors),
          3) z' = z_self + λ_social · δ_social,
          4) store self.last_delta_social_norm for telemetry.

        Returns (z_updated, info_dict). If disabled or no neighbors, returns (z_self, info_zero).
        Also records details for tests in self._last_social = {"delta","info","disabled"}.
        """
        device = z_self.device

        # If social is disabled or module missing, produce a zero-delta tuple and a clean flag.
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

        # Collect neighbor latents (latest), align to device
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

        # If no usable neighbors, behave like a clean zero-delta step (not disabled).
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

        # Influence step is explicitly inference-time: no grads
        delta, info = self.social(
            z_self=z_self.detach().to(device),
            z_neighbors=z_neighbors,
            self_role=self.role,
            neighbor_roles=neighbor_roles,
        )

        lam = float(self.social_lambda_ext)
        z_updated = z_self + lam * delta

        # Telemetry and info dict expected by sim logger
        dn = float(delta.norm().item()) * lam
        self.last_delta_social_norm = dn
        info_out = dict(info or {})
        info_out.setdefault("trust_mode", getattr(self.social, "trust_mode", "none"))
        info_out.setdefault("scale", float(getattr(self.social, "scale", 0.0)))
        info_out["n_neighbors"] = info_out.get("n_neighbors", len(z_neighbors))
        info_out["delta_norm"] = dn
        info_out["lambda_ext"] = lam

        # Persist for tests/telemetry
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

        # Back-compat: if a pipeline expects in-place update (self.z_t), honor it
        try:
            self.z_t = z_updated.detach()
        except Exception:
            pass

        return z_updated, info_out

    # ───────────────────────── Rollout aux snapshot ─────────────────────────
    def make_aux(self, agents: List["BaseAgent"]) -> Dict:
        alive = [bool(a.alive) for a in agents]
        wolves_attached = [getattr(a, "is_wolf", False) for a in agents]
        if any(wolves_attached):
            wolves = wolves_attached
        else:
            wset = self.wolf_ids
            wolves = [(int(a.name.split("_")[1]) in wset) for a in agents]
        return {
            "alive": alive,
            "self_idx": self._self_idx(),
            "wolves": wolves,
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
        Unified mouthpiece call:
          - natural, in-scenario dialogue via LLM route (with hygiene & bias)
          - stable fallback via Bandit templates
        Always logs a training record into msg_buffer for Judge/REINFORCE + bias-head.
        Adds: soft prefix steer from a one-shot named-target_hint, then clears it.
        """
        z = self.encode_current_belief(round_num, agents)
        self._update_persona_norm_if_present()

        candidate_targets = [a.name for a in agents if a.alive and a.name != self.name]

        # NEW: soft prefix from hint (propagated to LLM via persona_effects key `_prefix_hint`)
        prefix_hint = self._prefix_from_hint()
        pe = dict(self.persona_effects or {})
        if prefix_hint:
            pe["_prefix_hint"] = prefix_hint  # SpeakerPolicy/SpeakerLLM will prepend this if present

        # NEW: provide role_bit and lightweight style knobs to mouthpiece (non-breaking via persona_effects)
        pe["_role_bit"] = float(self.role_bit)
        pe["_temp_scale"] = float(self.speaker_temp_scale)
        pe["_bias_scale"] = float(self.bias_scale)

        text, meta = self.speaker.generate(
            z_t=z.detach(),
            role=self.role or "Unknown",
            recent_texts=self._recent_texts(),
            candidate_targets=candidate_targets,
            self_name=self.name,
            phase_code=phase_code,
            persona_effects=pe,
        )

        # One-shot: clear the hint so it doesn't leak into later phases
        self.named_target_hint = None

        self.last_message = text
        self.speaker_mode = meta.get("mode", "none")

        # --- NEW: choose a talk_intent label for bias-head supervision ---
        intent_id: Optional[int] = None

        # (a) If bandit/template, map template_id → category.
        template_id = meta.get("template_id", -1)
        if isinstance(template_id, int) and template_id >= 0:
            if template_id < len(TEMPLATE_TO_CAT_ID):
                intent_id = int(TEMPLATE_TO_CAT_ID[template_id])
            else:
                intent_id = HEDGE_CAT_ID  # safe default

        # (b) Otherwise (LLM/non-template), use TalkHead masked argmax on current z.
        if intent_id is None:
            z_for_head = meta.get("z", z.detach())
            intent_id = self._talk_intent_from_head(z_for_head)

        # (c) Lightweight text fallback if head was unavailable.
        if intent_id is None:
            intent_id = int(self._infer_talk_category(text))

        # Keep local state consistent for legacy paths
        self.talk_category_last = int(intent_id)

        # Default role_bit for logs if mouthpiece didn't include one
        default_role_bit_tensor = torch.tensor(self.role_bit, dtype=torch.float32)

        # Record into training buffer (now with talk_intent)
        self.msg_buffer.append({
            "mode": self.speaker_mode,
            "z": meta.get("z", z.detach().cpu()),
            "role_bit": meta.get("role_bit", default_role_bit_tensor),
            "hist_feats": meta.get("hist_feats", torch.tensor([0.0, 0.0])),
            "template_id": template_id,
            "talk_intent": int(intent_id),  # ← NEW: label for bias-head training
            "text": text,
            "round": round_num,
            "phase_code": phase_code if phase_code is not None else -1,
            "reward": None,
        })

        if TELEMETRY_ENABLED:
            self.telemetry.update({
                "speak_text_len": len(text or ""),
                "talk_category_last": int(self.talk_category_last),
                "talk_intent_id": int(intent_id),  # ← optional telemetry
                "speaker_mode": self.speaker_mode,
                "persona_norm": float(self.persona_norm),
                "role_bit": float(self.role_bit),
                "temp_scale": float(self.speaker_temp_scale),
                "bias_scale": float(self.bias_scale),
            })
        return text

    # ───────────────────────── Phase-4 rollout helpers ─────────────────────────
    def reset_for_new_game(self):
        self.alive = True
        self.last_message = ""
        self.talk_category_last = -1
        self.vote_history.clear()
        self.latent_history.clear()
        self.heard_messages.clear()
        self.message_memory.clear()
        self.telemetry.clear()
        self._step_cache = {}

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
        return payload_idx, choice_type

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
