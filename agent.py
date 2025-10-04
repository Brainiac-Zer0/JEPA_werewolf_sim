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

# Load config file once at startup
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Access values
NUM_AGENTS = config.get("NUM_AGENTS")
MAX_MEMORY = config.get("MAX_MEMORY")
USE_LANGUAGE = bool(config.get("USE_LANGUAGE", True))
SPEAKER_ENABLED = bool(config.get("SPEAKER_ENABLED", False))
SPEAKER_LR = float(config.get("SPEAKER_LR", 1e-3))
SPEAKER_HIST_K = int(config.get("SPEAKER_HIST_K", 3))


class BaseAgent:
    """Single Werewolf / Villager agent driven by JEPA components + an LLM mouthpiece (or trainable Speaker)."""

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
        self.message_encoder = MessageEncoder()  # may be overwritten with shared instance
        self.encoder = encoder or MLPBeliefEncoder(input_dim=INPUT_DIM, latent_dim=LATENT_DIM)
        self.world_model = world_model or WorldModelMLP(latent_dim=LATENT_DIM, action_dim=8)
        self.action_encoder = action_encoder or ActionEncoder(num_actions=6, action_dim=8)
        self.planner = planner or PlannerHead(latent_dim=LATENT_DIM, num_agents=NUM_AGENTS)

        # Memories
        self.vote_history: list[str] = []
        self.latent_history: list[torch.Tensor] = []
        self.heard_messages: dict[str, str] = {}
        self.message_memory: deque[tuple[str, str]] = deque(maxlen=20)

        # NEW: Speaker (trainable mouthpiece) + buffer for training
        self.speaker = None
        self.speaker_opt = None
        self.msg_buffer: list[dict] = []  # each: {z, role_bit, hist_feats, template_id, text, round, reward}

        if SPEAKER_ENABLED:
            self._init_speaker()

    # ───────────────────────── internal helpers ─────────────────────────
    def _init_speaker(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.speaker = SpeakerBandit(latent_dim=LATENT_DIM, num_templates=len(DEFAULT_TEMPLATES))
        self.speaker.to(device)
        self.speaker_opt = torch.optim.Adam(self.speaker.parameters(), lr=SPEAKER_LR)

    def _recent_texts(self) -> list[str]:
        """Return last K observed lines (string only) for conditioning the speaker."""
        if not self.message_memory:
            return []
        return [m for (_, m) in list(self.message_memory)[-SPEAKER_HIST_K:] if m]

    # ───────────────────────── Perception ─────────────────────────
    def observe(self, agents: list["BaseAgent"]):
        observed: list[tuple[str, str]] = []
        for a in agents:
            if a.alive and a.name != self.name:
                observed.append((a.name, a.last_message))
                self.heard_messages[a.name] = a.last_message
                self.message_memory.append((a.name, a.last_message))
        return observed

    # ───────────────────────── Action selection ─────────────────────────
    def plan_vote(self, z: torch.Tensor, agents: list["BaseAgent"]):
        alive = [a for a in agents if a.alive and a.name != self.name]
        if not alive:
            return self  # fallback

        logits = self.planner(z)  # → R^{NUM_AGENTS}
        # keep on same device as logits to avoid device mismatch
        device = logits.device
        alive_idx = [int(a.name.split("_")[1]) for a in alive]
        filtered = torch.tensor([logits[i] for i in alive_idx], device=device)
        chosen = alive[torch.argmax(filtered).item()]

        self.vote_history.append(chosen.name)
        if len(self.vote_history) > MAX_MEMORY:
            self.vote_history.pop(0)
        return chosen

    def choose_night_target(self, agents):
        candidates = [a for a in agents if a.alive and a.name != self.name]
        return candidates[0].name if candidates else None

    # ───────────────────────── Latent belief encoding ─────────────────────────
    def encode_current_belief(self, round_num: int, agents: list["BaseAgent"]):
        """Encode the current belief state z_t from internal + social features."""
        self_msg_embed = self.message_encoder(self.last_message).squeeze()

        # neighbour messages embedding (ablated if USE_LANGUAGE==False)
        neighbour_msgs = [msg for _, msg in self.observe(agents) if msg] if USE_LANGUAGE else []
        if neighbour_msgs:
            neighbour_embed = self.message_encoder(neighbour_msgs).mean(dim=0)
        else:
            neighbour_embed = torch.zeros_like(self_msg_embed)

        # vote history vector
        vote_vec = torch.zeros(NUM_AGENTS)
        for name in self.vote_history[-MAX_MEMORY:]:
            try:
                vote_vec[int(name.split("_")[1])] += 1.0
            except Exception:
                continue
        if vote_vec.sum() > 0:
            vote_vec /= vote_vec.sum()

        # memory summary
        memory_summary = (
            torch.stack(self.latent_history).mean(dim=0) if self.latent_history else torch.zeros(32)
        )

        # package + encode  ⟵ FIX: pass the local `self_msg_embed`
        x = package_features(
            agent_alive=self.alive,
            round_num=round_num,
            self_msg_embed=self_msg_embed,       # <-- incorrect placeholder in previous version
            neighbor_msg_embed=neighbour_embed,  # ok to pass British var name into US arg
            vote_vector=vote_vec,
            memory_summary=memory_summary,
        )
        # Replace with the correct line:
        x = package_features(
            agent_alive=self.alive,
            round_num=round_num,
            self_msg_embed=self_msg_embed if False else self_msg_embed,  # clarity
            neighbor_msg_embed=neighbour_embed,
            vote_vector=vote_vec,
            memory_summary=memory_summary,
        )

        z = self.encoder(x)

        # NaN guard in z
        if torch.isnan(z).any():
            raise RuntimeError("NaN in z latent")

        # store for temporal context
        self.latent_history.append(z.detach())
        if len(self.latent_history) > MAX_MEMORY:
            self.latent_history.pop(0)
        return z

    # ───────────────────────── Human-readable decode (optional) ─────────────────────────
    def decode_z(self, z: torch.Tensor) -> str:
        mean, std = z.mean().item(), z.std().item()
        mood = (
            "bad feeling about someone." if mean > 0.2 else
            "quiet… too quiet." if mean < -0.2 else
            "uncertain."
        )
        confidence = "unsure who to trust." if std > 0.5 else "confident in my suspicions."
        return f"The group seems {mood} I am {confidence}"

    # ───────────────────────── Speak ─────────────────────────
    def speak(self, round_num: int, agents: list["BaseAgent"]):
        """
        If SPEAKER_ENABLED=1, use trainable Speaker (template bandit).
        Otherwise fall back to llm_script mouthpiece (frozen).
        """
        # Always encode z first (also updates message_memory via observe inside)
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
                persona_effects=getattr(self, "persona_effects", None),  # NEW (handled if speaker supports it)
            )
            # Log for training
            self.msg_buffer.append({
                "z": z.detach().cpu(),
                "role_bit": meta["role_bit"],
                "hist_feats": meta["hist_feats"],
                "template_id": meta["template_id"],
                "text": text,
                "round": round_num,
                "reward": None,  # to be filled by training after judge scoring
            })
            self.last_message = text
            return text

        # Fallback: frozen mouthpiece
        if not self.llm_fn:
            self.last_message = "…"
            return "…"
        response = self.llm_fn(z, self)
        self.last_message = response
        return response
