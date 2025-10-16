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

# NEW: speaker import
from speaker import SpeakerBandit, DEFAULT_TEMPLATES  # make_hist_feats not needed here

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
NUM_AGENTS       = _env_int("NUM_AGENTS", int(config.get("NUM_AGENTS", 6)))
MAX_MEMORY       = _env_int("MAX_MEMORY", int(config.get("MAX_MEMORY", 20)))
USE_LANGUAGE     = _env_bool("USE_LANGUAGE", bool(config.get("USE_LANGUAGE", True)))
SPEAKER_ENABLED  = _env_bool("SPEAKER_ENABLED", bool(config.get("SPEAKER_ENABLED", False)))
SPEAKER_LR       = _env_float("SPEAKER_LR", float(config.get("SPEAKER_LR", 1e-3)))
SPEAKER_HIST_K   = _env_int("SPEAKER_HIST_K", int(config.get("SPEAKER_HIST_K", 3)))
NUM_TALK_CATS    = _env_int("NUM_TALK_CATS", int(config.get("NUM_TALK_CATS", 5)))

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

# Map SpeakerBandit template ids → talk category ids (truncate/clip to templates length)
TEMPLATE_TO_CAT_ID = [ACCUSE_CAT_ID, DEFEND_CAT_ID, HEDGE_CAT_ID, QUESTION_CAT_ID, VOTE_CAT_ID]
if len(TEMPLATE_TO_CAT_ID) < len(DEFAULT_TEMPLATES):
    # Extend with hedge for any extra templates to be safe
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
        self.llm_fn = None  # (z, self) -> str  • attached from llm_script

        # NEW: phase-aware speech bookkeeping (read by sim.py)
        self.talk_category_last: int = -1
        self.speaker_mode: str = "none"   # {"bandit","llm","none"}
        self.persona_norm: float = 0.0

        # JEPA sub-modules
        self.message_encoder = MessageEncoder()  # may be overwritten with shared instance by sim
        self.encoder        = encoder or MLPBeliefEncoder(input_dim=INPUT_DIM, latent_dim=LATENT_DIM)
        self.world_model    = world_model or WorldModelMLP(latent_dim=LATENT_DIM, action_dim=ACTION_DIM)
        self.action_encoder = action_encoder or ActionEncoder(num_actions=NUM_AGENTS, action_dim=ACTION_DIM)

        # Legacy single-head planner (kept for back-compat paths)
        self.planner        = planner or PlannerHead(latent_dim=LATENT_DIM, num_agents=NUM_AGENTS)
        # NEW: factorized heads (preferred path)
        self.planner_factorized = planner_factorized or FactorizedPlanner(
            latent_dim=LATENT_DIM, num_agents=NUM_AGENTS, num_talk_cats=NUM_TALK_CATS
        )
        # NEW: stored for future use in training (not consumed by agent logic)
        self.phase_action_encoder = phase_action_encoder

        # Memories
        self.vote_history: List[str] = []
        self.latent_history: List[torch.Tensor] = []
        self.heard_messages: Dict[str, str] = {}
        self.message_memory: deque[tuple[str, str]] = deque(maxlen=MAX_MEMORY)

        # Speaker (optional) + buffer for training
        self.speaker = None
        self.speaker_opt = None
        self.msg_buffer: List[dict] = []  # {z, role_bit, hist_feats, template_id, text, round, reward}

        # NEW: minimal per-step telemetry container (sim reads this to write CSV)
        self.telemetry: Dict = {}

        # NEW: pack awareness
        self.is_wolf: bool = False
        self.wolf_ids: Set[int] = set()   # indices of known packmates (incl. self if you want)

        # NEW: per-step cache for Phase-4 rollout construction
        self._step_cache: Dict = {}

        if SPEAKER_ENABLED:
            self._init_speaker()

    # ───────────────────────── pack/role helpers ─────────────────────────
    def set_role(self, role: str) -> None:
        self.role = role
        r = (role or "").lower()
        self.is_wolf = (r in ("werewolf", "wolf", "traitor"))

    def set_packmates(self, names: List[str]) -> None:
        ids: Set[int] = set()
        for n in names or []:
            try:
                ids.add(int(n.split("_")[1]))
            except Exception:
                continue
        self.wolf_ids = ids

    def _self_idx(self) -> int:
        try:
            return int(self.name.split("_")[1])
        except Exception:
            return 0

    # ───────────────────────── internal helpers ─────────────────────────
    def _init_speaker(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.speaker = SpeakerBandit(latent_dim=LATENT_DIM, num_templates=len(DEFAULT_TEMPLATES))
        self.speaker.to(device)
        self.speaker_opt = torch.optim.Adam(self.speaker.parameters(), lr=SPEAKER_LR)

    def _recent_texts(self) -> List[str]:
        """Last K observed lines for conditioning the speaker."""
        if not self.message_memory:
            return []
        return [m for (_, m) in list(self.message_memory)[-SPEAKER_HIST_K:] if m]

    def _alive_others(self, agents: List["BaseAgent"]) -> List["BaseAgent"]:
        return [a for a in agents if a.alive and a.name != self.name]

    def _alive_indices(self, agents: List["BaseAgent"]) -> List[int]:
        return [int(a.name.split("_")[1]) for a in self._alive_others(agents)]

    def _update_persona_norm_if_present(self):
        """Recompute persona vector norm if a persona has been attached."""
        if hasattr(self, "persona_vec"):
            try:
                pv = torch.as_tensor(self.persona_vec, dtype=torch.float32)
                self.persona_norm = float(pv.norm().item())
            except Exception:
                self.persona_norm = 0.0
        else:
            self.persona_norm = 0.0

    def _infer_talk_category(self, text: str) -> int:
        """
        Frozen, rule-based mapping from free text to category id.
        Keeps LLM path deterministic for clean ablations.
        """
        t = (text or "").lower().strip()

        # Strong vote cues
        if any(kw in t for kw in ["we should vote", "vote to", "vote out", "vote ", "eliminate ", "lynch "]):
            return VOTE_CAT_ID

        # Interrogatives / requests for justification
        if any(kw in t for kw in ["why", "how", "what about", "explain", "because?"]):
            return QUESTION_CAT_ID

        # Clear defense cues
        if any(kw in t for kw in ["i trust", "not a wolf", "innocent", "seems fine", "defend"]):
            return DEFEND_CAT_ID

        # Accusation / suspicion cues
        if any(kw in t for kw in ["is a wolf", "suspect", "guilty", "looks bad", "suspicious"]):
            return ACCUSE_CAT_ID

        # Default: hedge
        return HEDGE_CAT_ID

    # ───────────────────────── Perception ─────────────────────────
    def observe(self, agents: List["BaseAgent"]):
        """Pull others' latest messages into short-term memory."""
        observed: List[tuple[str, str]] = []
        for a in agents:
            if a.alive and a.name != self.name:
                observed.append((a.name, a.last_message))
                self.heard_messages[a.name] = a.last_message
                self.message_memory.append((a.name, a.last_message))
        return observed

    # ───────────────────────── Mask builders ─────────────────────────
    def build_talk_mask(self) -> torch.Tensor:
        """Allow all talk categories for now (override if you want to ban some)."""
        return torch.ones(NUM_TALK_CATS, dtype=torch.bool)

    def build_vote_mask(self, agents: List["BaseAgent"]) -> torch.Tensor:
        """Legal = alive & not self."""
        mask = torch.zeros(NUM_AGENTS, dtype=torch.bool)
        for a in agents:
            if a.alive and a.name != self.name:
                try:
                    mask[int(a.name.split("_")[1])] = True
                except Exception:
                    continue
        return mask

    def build_kill_mask(self, agents: List["BaseAgent"]) -> torch.Tensor:
        """
        Wolves can kill only alive non-wolves, non-self.
        Villagers produce an all-False mask (no-op at night).
        """
        mask = torch.zeros(NUM_AGENTS, dtype=torch.bool)
        if not self.is_wolf:
            return mask  # villagers don't issue kills

        for a in agents:
            try:
                idx = int(a.name.split("_")[1])
            except Exception:
                continue
            legal = a.alive and a.name != self.name and (idx not in self.wolf_ids)
            mask[idx] = bool(legal)
        return mask

    # ───────────────────────── Action selection (phase-aware) ─────────────────────────
    @torch.no_grad()
    def choose_action_by_phase(self, phase_code: int, round_num: int, agents: List["BaseAgent"]):
        """
        Returns (payload_idx, choice_type) where:
          - DISCUSS: payload_idx ∈ [0..NUM_TALK_CATS-1], choice_type="TALK_INTENT"
          - VOTE:    payload_idx ∈ [0..NUM_AGENTS-1], choice_type="VOTE_TARGET"
          - NIGHT:   payload_idx ∈ [0..NUM_AGENTS-1], choice_type="KILL_TARGET" (wolves only)
        When no legal action exists, returns (None, None).
        """
        z = self.encode_current_belief(round_num, agents)

        def _masked_argmax(logits_1d: torch.Tensor, legal_mask_1d: torch.Tensor, head: str):
            if legal_mask_1d is None:
                # If mask missing, treat as all-true with same shape
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

        # DISCUSS → TalkHead
        if phase_code == PHASES["DISCUSS"]:
            logits = self.planner_factorized.talk(z)           # [C] or [1,C]
            if logits.dim() > 1:
                logits = logits.squeeze(0)
            mask = self.build_talk_mask().to(logits.device)    # [C]
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

        # VOTE → VoteHead (masked to alive non-self)
        if phase_code == PHASES["VOTE"]:
            logits = self.planner_factorized.vote(z)           # [N]
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
            # compact history
            self.vote_history.append(f"Agent_{choice}")
            if len(self.vote_history) > MAX_MEMORY:
                self.vote_history.pop(0)
            return choice, "VOTE_TARGET"

        # NIGHT → KillHead (wolves only)
        if phase_code == PHASES["NIGHT"]:
            logits = self.planner_factorized.kill(z)           # [N]
            if logits.dim() > 1:
                logits = logits.squeeze(0)
            mask = self.build_kill_mask(agents).to(logits.device)
            choice, legal_idx, probs = _masked_argmax(logits, mask, head="kill")
            if choice is None:
                return None, None
            if TELEMETRY_ENABLED:
                self.telemetry.update({
                    "kill_mask_idx": legal_idx,
                    "kill_choice_idx": choice,
                    "kill_probs": probs,
                })
            return choice, "KILL_TARGET"

        # Unknown phase
        return None, None

    # Legacy vote helper (kept; internally routes through factorized head)
    @torch.no_grad()
    def plan_vote(self, z: torch.Tensor, agents: List["BaseAgent"]):
        """
        Back-compat wrapper. Uses factorized VoteHead with proper masks.
        """
        logits = self.planner_factorized.vote(z)               # [N]
        if logits.dim() > 1:
            logits = logits.squeeze(0)
        mask = self.build_vote_mask(agents).to(logits.device)
        masked = torch.full_like(logits, float("-inf"))
        masked[mask] = logits[mask]
        chosen_idx = int(torch.argmax(masked).item())
        alive = self._alive_others(agents)
        chosen = next((a for a in alive if a.name == f"Agent_{chosen_idx}"), alive[0] if alive else self)

        # Telemetry (softmax only over legal set)
        probs = torch.softmax(logits[mask], dim=-1).detach().cpu().tolist()
        if TELEMETRY_ENABLED:
            self.telemetry.update({
                "vote_alive_idx": torch.where(mask)[0].tolist(),
                "vote_probs": probs,
                "vote_choice_idx": chosen_idx,
            })

        # Compact history
        self.vote_history.append(chosen.name)
        if len(self.vote_history) > MAX_MEMORY:
            self.vote_history.pop(0)

        return chosen

    def choose_night_target(self, agents: List["BaseAgent"]):
        """
        Night kill policy (legacy fallback). Prefer factorized head via choose_action_by_phase.
        """
        candidates = [a for a in agents if a.alive and a.name != self.name]
        return candidates[0].name if candidates else None

    # ───────────────────────── Latent belief encoding ─────────────────────────
    def encode_current_belief(self, round_num: int, agents: List["BaseAgent"]):
        """
        Encode the current belief state z_t from internal + social features.
        Phase-1 fixes:
        - BUGFIX: pass the *local* self_msg_embed into package_features (was previously wrong)
        - Device hygiene: create tensors on z/encoder device
        - Telemetry: store z norms and message lengths for CSV
        """
        device = next(self.encoder.parameters()).device

        # Self message embedding
        self_msg_embed = self.message_encoder(self.last_message).squeeze().to(device)
        if torch.isnan(self_msg_embed).any():
            self_msg_embed = torch.zeros_like(self_msg_embed)

        # Neighbour messages embedding (ablated if USE_LANGUAGE==False)
        neighbour_msgs = [msg for _, msg in self.observe(agents) if msg] if USE_LANGUAGE else []
        if neighbour_msgs:
            neighbour_embed = self.message_encoder(neighbour_msgs).mean(dim=0).to(device)
        else:
            neighbour_embed = torch.zeros_like(self_msg_embed)

        # Vote history vector (normalized counts over agent indices)
        vote_vec = torch.zeros(NUM_AGENTS, device=device)
        for name in self.vote_history[-MAX_MEMORY:]:
            try:
                vote_vec[int(name.split("_")[1])] += 1.0
            except Exception:
                continue
        if float(vote_vec.sum().item()) > 0.0:
            vote_vec = vote_vec / vote_vec.sum()

        # Memory summary over past latents
        if self.latent_history:
            memory_summary = torch.stack([t.to(device) for t in self.latent_history]).mean(dim=0)
        else:
            memory_summary = torch.zeros(LATENT_DIM, device=device)

        # ✅ Correct packaging (single call, correct variable)
        x = package_features(
            agent_alive=self.alive,
            round_num=round_num,
            self_msg_embed=self_msg_embed,
            neighbor_msg_embed=neighbour_embed,
            vote_vector=vote_vec,
            memory_summary=memory_summary,
        )

        z = self.encoder(x)

        # NaN guard in z
        if torch.isnan(z).any() or torch.isinf(z).any():
            raise RuntimeError("NaN/Inf in z latent")

        # store for temporal context
        self.latent_history.append(z.detach())
        if len(self.latent_history) > MAX_MEMORY:
            self.latent_history.pop(0)

        # Telemetry
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

    # ───────────────────────── Rollout aux snapshot ─────────────────────────
    def make_aux(self, agents: List["BaseAgent"]) -> Dict:
        alive = [bool(a.alive) for a in agents]
        # Try to build wolves vector if roles are attached; else fall back to self.wolf_ids
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

    # ───────────────────────── Speak ─────────────────────────
    def speak(self, round_num: int, agents: List["BaseAgent"]):
        """
        If SPEAKER_ENABLED=1, use trainable Speaker (template bandit).
        Otherwise fall back to llm_script mouthpiece (frozen).
        """
        # Always encode z first (also updates message_memory via observe)
        z = self.encode_current_belief(round_num, agents)

        # Refresh persona norm if a persona vector has been attached
        self._update_persona_norm_if_present()

        # Reset talk category for this utterance
        self.talk_category_last = -1

        if self.speaker is not None:
            # Persona-driven exploration tweak (light)
            if hasattr(self, "persona_effects"):
                scale = float(self.persona_effects.get("speaker_temp_scale", 1.0))
                self.speaker.temperature = max(0.3, min(2.0, 1.0 * scale))

            # Build conditioning inputs
            recent = self._recent_texts()
            cand_targets = [a.name for a in agents if a.alive]  # include self; speaker avoids self as target
            text, meta = self.speaker.generate(
                z_t=z.detach(),
                role=self.role or "Unknown",
                recent_texts=recent,
                templates=DEFAULT_TEMPLATES,
                candidate_targets=cand_targets,
                self_name=self.name,
                persona_effects=getattr(self, "persona_effects", None),
            )
            # Buffer line for training
            self.msg_buffer.append({
                "z": z.detach().cpu(),
                "role_bit": meta.get("role_bit", 0),
                "hist_feats": meta.get("hist_feats"),
                "template_id": meta.get("template_id"),
                "text": text,
                "round": round_num,
                "reward": None,
            })
            self.last_message = text
            self.speaker_mode = "bandit"

            # Map to talk category id (prefer meta, else template mapping, else hedge)
            cat_from_meta = meta.get("category_id", None)
            if isinstance(cat_from_meta, int):
                self.talk_category_last = int(cat_from_meta)
            else:
                tid = int(meta.get("template_id", -1)) if meta.get("template_id", None) is not None else -1
                if 0 <= tid < len(TEMPLATE_TO_CAT_ID):
                    self.talk_category_last = int(TEMPLATE_TO_CAT_ID[tid])
                else:
                    self.talk_category_last = HEDGE_CAT_ID

            # Telemetry
            if TELEMETRY_ENABLED:
                self.telemetry.update({
                    "speak_template_id": int(meta.get("template_id", -1)),
                    "speak_text_len": len(text or ""),
                    "talk_category_last": int(self.talk_category_last),
                    "speaker_mode": self.speaker_mode,
                    "persona_norm": float(self.persona_norm),
                })
            return text

        # Fallback: frozen mouthpiece
        if not self.llm_fn:
            self.last_message = "…"
            self.speaker_mode = "none"
            if TELEMETRY_ENABLED:
                self.telemetry.update({
                    "speak_text_len": 1,
                    "talk_category_last": HEDGE_CAT_ID,
                    "speaker_mode": self.speaker_mode,
                    "persona_norm": float(self.persona_norm),
                })
            self.talk_category_last = HEDGE_CAT_ID
            return "…"

        response = self.llm_fn(z, self)
        self.last_message = response
        self.speaker_mode = "llm"
        self.talk_category_last = self._infer_talk_category(response)

        if TELEMETRY_ENABLED:
            self.telemetry.update({
                "speak_text_len": len(response or ""),
                "talk_category_last": int(self.talk_category_last),
                "speaker_mode": self.speaker_mode,
                "persona_norm": float(self.persona_norm),
            })
        return response

    # ───────────────────────── NEW: Phase-4 rollout helpers ─────────────────────────
    def reset_for_new_game(self):
        """Clear state between games."""
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
        """Clear per-step cached values (z_t, phase, payload, choice_type, aux, round)."""
        self._step_cache = {}

    @torch.no_grad()
    def begin_step(self, phase_code: int, round_num: int, agents: List["BaseAgent"]) -> torch.Tensor:
        """
        Capture z_t and aux at the start of a sim phase-step.
        Returns z_t for convenience (some sims like to log it).
        """
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
        """
        Thin wrapper to choose and cache the action for this phase.
        Returns (payload_idx, choice_type). If no legal action, (None, None).
        """
        payload_idx, choice_type = self.choose_action_by_phase(phase_code, round_num, agents)
        if not getattr(self, "_step_cache", None):
            _ = self.begin_step(phase_code, round_num, agents)
        self._step_cache["payload"] = None if payload_idx is None else int(payload_idx)
        self._step_cache["choice_type"] = choice_type
        return payload_idx, choice_type

    @torch.no_grad()
    def finalize_step(
        self,
        agents: List["BaseAgent"],
        z_next: Optional[torch.Tensor] = None,
    ):
        """
        Package a training row:
          (z_t, phase_code, action_payload, z_{t+1}, role, choice_type, aux)

        If z_next is None, we re-encode belief now (post-environment update).
        Detaches tensors to CPU for log-friendly storage.
        """
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
