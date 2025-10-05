import os
import torch
import torch.nn.functional as F
from collections import deque
import yaml

from encoders import (
    MessageEncoder,
    MLPBeliefEncoder,
    WorldModelMLP,
    ActionEncoder,
    package_features,
    INPUT_DIM,
    LATENT_DIM,
    PlannerHead,
)

# NEW: speaker import
from speaker import SpeakerBandit, DEFAULT_TEMPLATES  # make_hist_feats not needed here

# Load config once
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f) or {}

# Config values
NUM_AGENTS     = int(config.get("NUM_AGENTS", 6))
MAX_MEMORY     = int(config.get("MAX_MEMORY", 20))
USE_LANGUAGE   = bool(config.get("USE_LANGUAGE", True))
SPEAKER_ENABLED= bool(config.get("SPEAKER_ENABLED", False))
SPEAKER_LR     = float(config.get("SPEAKER_LR", 1e-3))
SPEAKER_HIST_K = int(config.get("SPEAKER_HIST_K", 3))

# Phase-1 logging toggle (read by sim.py)
TELEMETRY_ENABLED = bool(config.get("TELEMETRY_ENABLED", True))


class BaseAgent:
    """Single Werewolf/Villager agent driven by JEPA components + a mouthpiece (LLM or trainable Speaker)."""

    def __init__(
        self,
        name: str,
        encoder: MLPBeliefEncoder | None = None,
        world_model: WorldModelMLP | None = None,
        action_encoder: ActionEncoder | None = None,
        planner: PlannerHead | None = None,
    ) -> None:
        self.name = name
        self.role: str | None = None
        self.alive: bool = True
        self.last_message: str = ""
        self.llm_fn = None  # (z, self) -> str  • attached from llm_script

        # JEPA sub-modules
        self.message_encoder = MessageEncoder()  # may be overwritten with shared instance by sim
        self.encoder        = encoder or MLPBeliefEncoder(input_dim=INPUT_DIM, latent_dim=LATENT_DIM)
        self.world_model    = world_model or WorldModelMLP(latent_dim=LATENT_DIM, action_dim=8)
        self.action_encoder = action_encoder or ActionEncoder(num_actions=NUM_AGENTS, action_dim=8)
        self.planner        = planner or PlannerHead(latent_dim=LATENT_DIM, num_agents=NUM_AGENTS)

        # Memories
        self.vote_history: list[str] = []
        self.latent_history: list[torch.Tensor] = []
        self.heard_messages: dict[str, str] = {}
        self.message_memory: deque[tuple[str, str]] = deque(maxlen=MAX_MEMORY)

        # Speaker (optional) + buffer for training
        self.speaker = None
        self.speaker_opt = None
        self.msg_buffer: list[dict] = []  # {z, role_bit, hist_feats, template_id, text, round, reward}

        # NEW: minimal per-step telemetry container (sim reads this to write CSV)
        self.telemetry: dict = {}

        if SPEAKER_ENABLED:
            self._init_speaker()

    # ───────────────────────── internal helpers ─────────────────────────
    def _init_speaker(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.speaker = SpeakerBandit(latent_dim=LATENT_DIM, num_templates=len(DEFAULT_TEMPLATES))
        self.speaker.to(device)
        self.speaker_opt = torch.optim.Adam(self.speaker.parameters(), lr=SPEAKER_LR)

    def _recent_texts(self) -> list[str]:
        """Last K observed lines for conditioning the speaker."""
        if not self.message_memory:
            return []
        return [m for (_, m) in list(self.message_memory)[-SPEAKER_HIST_K:] if m]

    def _alive_others(self, agents: list["BaseAgent"]) -> list["BaseAgent"]:
        return [a for a in agents if a.alive and a.name != self.name]

    def _alive_indices(self, agents: list["BaseAgent"]) -> list[int]:
        return [int(a.name.split("_")[1]) for a in self._alive_others(agents)]

    # ───────────────────────── Perception ─────────────────────────
    def observe(self, agents: list["BaseAgent"]):
        """Pull others' latest messages into short-term memory."""
        observed: list[tuple[str, str]] = []
        for a in agents:
            if a.alive and a.name != self.name:
                observed.append((a.name, a.last_message))
                self.heard_messages[a.name] = a.last_message
                self.message_memory.append((a.name, a.last_message))
        return observed

    # ───────────────────────── Action selection ─────────────────────────
    @torch.no_grad()
    def plan_vote(self, z: torch.Tensor, agents: list["BaseAgent"]):
        """
        Mask-aware, deterministic vote selection:
        - compute logits over all NUM_AGENTS
        - set illegal targets (self/dead) to -inf
        - argmax among remaining (deterministic)
        - store telemetry (mask indices + probs over legal set)
        """
        alive = self._alive_others(agents)
        if not alive:
            return self  # fallback to self if alone

        logits = self.planner(z)                     # [NUM_AGENTS]
        device = logits.device
        alive_idx = self._alive_indices(agents)

        masked = torch.full_like(logits, float("-inf"))
        masked[alive_idx] = logits[alive_idx]
        # deterministic pick among legal targets
        chosen_idx = int(torch.argmax(masked).item())
        chosen = next(a for a in alive if a.name == f"Agent_{chosen_idx}")

        # Telemetry (softmax only over legal set)
        probs = torch.softmax(logits[alive_idx], dim=-1).detach().cpu().tolist()
        if TELEMETRY_ENABLED:
            self.telemetry.update({
                "vote_alive_idx": alive_idx,
                "vote_probs": probs,
                "vote_choice_idx": chosen_idx,
            })

        # Compact history
        self.vote_history.append(chosen.name)
        if len(self.vote_history) > MAX_MEMORY:
            self.vote_history.pop(0)

        return chosen

    def choose_night_target(self, agents: list["BaseAgent"]):
        """
        Night kill policy (kept simple/deterministic for Phase-1):
        pick the *lowest index* legal non-self, non-wolf target among living.
        """
        candidates = [a for a in agents if a.alive and a.name != self.name]
        return candidates[0].name if candidates else None

    # ───────────────────────── Latent belief encoding ─────────────────────────
    def encode_current_belief(self, round_num: int, agents: list["BaseAgent"]):
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

    # ───────────────────────── Human-readable decode (optional) ─────────────────────────
    def decode_z(self, z: torch.Tensor) -> str:
        mean, std = z.mean().item(), z.std().item()
        mood = ("bad feeling about someone." if mean > 0.2
                else "quiet… too quiet." if mean < -0.2
                else "uncertain.")
        confidence = "unsure who to trust." if std > 0.5 else "confident in my suspicions."
        return f"The group seems {mood} I am {confidence}"

    # ───────────────────────── Speak ─────────────────────────
    def speak(self, round_num: int, agents: list["BaseAgent"]):
        """
        If SPEAKER_ENABLED=1, use trainable Speaker (template bandit).
        Otherwise fall back to llm_script mouthpiece (frozen).
        """
        # Always encode z first (also updates message_memory via observe)
        z = self.encode_current_belief(round_num, agents)

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

            # Telemetry
            if TELEMETRY_ENABLED:
                self.telemetry.update({
                    "speak_template_id": int(meta.get("template_id", -1)),
                    "speak_text_len": len(text or ""),
                })
            return text

        # Fallback: frozen mouthpiece
        if not self.llm_fn:
            self.last_message = "…"
            if TELEMETRY_ENABLED:
                self.telemetry.update({"speak_text_len": 1})
            return "…"

        response = self.llm_fn(z, self)
        self.last_message = response
        if TELEMETRY_ENABLED:
            self.telemetry.update({"speak_text_len": len(response or "")})
        return response
