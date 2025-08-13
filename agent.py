import os
import torch
import torch.nn.functional as F
from collections import deque
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

NUM_AGENTS = 6
MAX_MEMORY = 5
USE_LANGUAGE = os.environ.get("USE_LANGUAGE", "1") != "0"  # ablation flag

class BaseAgent:
    """Single Werewolf / Villager agent driven by JEPA components + an LLM mouthpiece."""

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

        # JEPA sub‑modules
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

    # Perception
    def observe(self, agents: list["BaseAgent"]):
        observed: list[tuple[str, str]] = []
        for a in agents:
            if a.alive and a.name != self.name:
                observed.append((a.name, a.last_message))
                self.heard_messages[a.name] = a.last_message
                self.message_memory.append((a.name, a.last_message))
        return observed

    # Action selection
    def plan_vote(self, z: torch.Tensor, agents: list["BaseAgent"]):
        alive = [a for a in agents if a.alive and a.name != self.name]
        if not alive:
            return self  # fallback

        logits = self.planner(z)  # → R^{NUM_AGENTS}
        alive_idx = [int(a.name.split("_")[1]) for a in alive]
        filtered = torch.tensor([logits[i] for i in alive_idx])
        chosen = alive[torch.argmax(filtered).item()]

        self.vote_history.append(chosen.name)
        if len(self.vote_history) > MAX_MEMORY:
            self.vote_history.pop(0)
        return chosen

    def choose_night_target(self, agents):
        candidates = [a for a in agents if a.alive and a.name != self.name]
        return candidates[0].name if candidates else None

    # Latent belief encoding
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

        # package + encode
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
        if torch.isnan(z).any():
            raise RuntimeError("NaN in z latent")

        # store for temporal context
        self.latent_history.append(z.detach())
        if len(self.latent_history) > MAX_MEMORY:
            self.latent_history.pop(0)
        return z

    # Human‑readable decode (optional)
    def decode_z(self, z: torch.Tensor) -> str:
        mean, std = z.mean().item(), z.std().item()
        mood = (
            "bad feeling about someone." if mean > 0.2 else
            "quiet… too quiet." if mean < -0.2 else
            "uncertain."  )
        confidence = "unsure who to trust." if std > 0.5 else "confident in my suspicions."
        return f"The group seems {mood} I am {confidence}"

    # Speak
    def speak(self, round_num: int, agents: list["BaseAgent"]):
        if not self.llm_fn:
            return "…"
        z = self.encode_current_belief(round_num, agents)
        response = self.llm_fn(z, self)
        self.last_message = response
        return response
