import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from functools import lru_cache
import yaml

# ───────────────────────────────────────────────────────────────────────────────
# Config & constants
# ───────────────────────────────────────────────────────────────────────────────
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f) or {}

# Core dims
INPUT_DIM   = int(config.get("INPUT_DIM", 808))
LATENT_DIM  = int(config.get("LATENT_DIM", 32))
ACTION_DIM  = int(config.get("ACTION_DIM", 8))
NUM_ACTIONS = int(config.get("NUM_ACTIONS", 6))
NUM_AGENTS  = int(config.get("NUM_AGENTS", 6))

# Optional talk categories (for future phase-aware planner split)
TALK_CATEGORIES = config.get("TALK_CATEGORIES", ["accuse", "defend", "hedge", "question", "vote"])
NUM_TALK_CATS   = int(config.get("NUM_TALK_CATS", len(TALK_CATEGORIES)))

# Phase scaffolding (for Day: Discuss/Vote; Night: Kill)
PHASES = {"DISCUSS": 0, "VOTE": 1, "NIGHT": 2}
NUM_PHASES = 3

def phase_onehot(phase_code: int) -> torch.Tensor:
    v = torch.zeros(NUM_PHASES, dtype=torch.float32)
    if 0 <= phase_code < NUM_PHASES:
        v[phase_code] = 1.0
    return v

# ───────────────────────────────────────────────────────────────────────────────
# TEXT ENCODER MODULE  (shared; includes tiny LRU for single-string calls)
# ───────────────────────────────────────────────────────────────────────────────
class MessageEncoder(nn.Module):
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        self.output_dim = self.transformer.config.hidden_size
        self.transformer.eval()
        for p in self.transformer.parameters():
            p.requires_grad_(False)

    @lru_cache(maxsize=4096)
    def _encode_cached_one(self, text: str):
        # Returns a CPU tensor (D,) for determinism & low VRAM
        encoded_input = self.tokenizer([text], padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.transformer(**encoded_input)
        pooled = self.mean_pooling(model_output, encoded_input['attention_mask'])[0]  # (D,)
        return pooled.detach().cpu()

    def forward(self, texts):
        # Fast path for a single string (cacheable)
        if isinstance(texts, str):
            return self._encode_cached_one(texts).unsqueeze(0)  # (1,D)
        # Batch path (not cached)
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.transformer(**encoded_input)
        pooled = self.mean_pooling(model_output, encoded_input['attention_mask'])
        return pooled.detach()

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state  # (B,T,D)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())  # (B,T,1)->(B,T,D)
        denom = input_mask_expanded.sum(1).clamp_min(1.0)  # avoid div/0
        return (token_embeddings * input_mask_expanded).sum(1) / denom  # (B,D)

# ───────────────────────────────────────────────────────────────────────────────
# Latent/state encoders & world model (unchanged APIs for compatibility)
# ───────────────────────────────────────────────────────────────────────────────
class MLPBeliefEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )

    def forward(self, input_vector):
        return self.encoder(input_vector)

class ActionEncoder(nn.Module):
    """
    Backward-compatible action index embedder.
    (For phase-aware work, you can later swap to PhaseActionEncoder without
     changing training_utils if you map indices consistently.)
    """
    def __init__(self, num_actions, action_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_actions, action_dim)

    def forward(self, action_idx):
        return self.embedding(action_idx)

class WorldModelMLP(nn.Module):
    def __init__(self, latent_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )

    def forward(self, z_t, a_t_embed):
        combined = torch.cat([z_t, a_t_embed], dim=-1)
        return self.model(combined)

class PlannerHead(nn.Module):
    """
    Classic planner: logits over NUM_AGENTS (kept for compatibility).
    """
    def __init__(self, latent_dim, num_agents):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_agents)
        )

    def forward(self, z):
        return self.net(z)

# ───────────────────────────────────────────────────────────────────────────────
# Phase-aware optional heads (added, not used by old code until you switch)
# ───────────────────────────────────────────────────────────────────────────────
class TalkHead(nn.Module):
    """Logits over talk categories (NUM_TALK_CATS)."""
    def __init__(self, latent_dim: int, num_cats: int = NUM_TALK_CATS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.Tanh(),
            nn.Linear(64, num_cats)
        )
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

class VoteHead(nn.Module):
    """Logits over agent indices (mask self/dead outside before softmax)."""
    def __init__(self, latent_dim: int, num_agents: int = NUM_AGENTS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.Tanh(),
            nn.Linear(64, num_agents)
        )
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

class KillHead(VoteHead):
    """Identical shape to VoteHead; semantics differ (wolves only)."""
    pass

# (Optional) Phase-aware action embedding builder — not wired into old code yet.
class PhaseActionEncoder(nn.Module):
    """
    Map (phase_onehot, payload_onehot) → ACTION_DIM via small MLP.
    Keeps a fixed ACTION_DIM regardless of which subspace produced the payload.
    """
    def __init__(self, action_dim: int = ACTION_DIM, num_agents: int = NUM_AGENTS, num_talk: int = NUM_TALK_CATS):
        super().__init__()
        self.num_agents = num_agents
        self.num_talk   = num_talk
        in_dim = NUM_PHASES + max(num_agents, num_talk)
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, phase_code: torch.Tensor, payload_idx: torch.Tensor, *, is_talk: bool) -> torch.Tensor:
        """
        phase_code: (B,) int in [0..2]
        payload_idx: (B,) int  (either talk_cat_id or agent_id)
        is_talk: which space payload_idx refers to
        """
        B = phase_code.shape[0]
        ph = torch.zeros(B, NUM_PHASES, dtype=torch.float32, device=phase_code.device)
        ph[torch.arange(B, device=phase_code.device), phase_code.clamp(0, NUM_PHASES-1)] = 1.0

        space = self.num_talk if is_talk else self.num_agents
        oh = torch.zeros(B, space, dtype=torch.float32, device=payload_idx.device)
        pid = payload_idx.clamp(0, space-1)
        oh[torch.arange(B, device=payload_idx.device), pid] = 1.0

        # pad to max(num_agents, num_talk) so the input dim is constant
        if space < max(self.num_agents, self.num_talk):
            pad_cols = max(self.num_agents, self.num_talk) - space
            oh = torch.cat([oh, torch.zeros(B, pad_cols, dtype=oh.dtype, device=oh.device)], dim=1)

        x = torch.cat([ph, oh], dim=1)
        return self.net(x)  # (B, ACTION_DIM)

# ───────────────────────────────────────────────────────────────────────────────
# Feature packaging (add meta-friendly variant for logging)
# ───────────────────────────────────────────────────────────────────────────────
def package_features(agent_alive, round_num, self_msg_embed, neighbor_msg_embed, vote_vector, memory_summary):
    alive_flag = torch.tensor([1.0 if agent_alive else 0.0])
    round_norm = torch.tensor([round_num / 10.0])
    return torch.cat([
        alive_flag,
        round_norm,
        self_msg_embed,
        neighbor_msg_embed,
        vote_vector,
        memory_summary
    ], dim=0)

def package_features_with_meta(agent_alive, round_num, self_msg_embed, neighbor_msg_embed, vote_vector, memory_summary):
    """
    Same as package_features, but also returns a small meta dict that you can
    write to CSV/JSONL alongside the model’s decision row.
    """
    x = package_features(agent_alive, round_num, self_msg_embed, neighbor_msg_embed, vote_vector, memory_summary)
    meta = {
        "alive": bool(agent_alive),
        "round": int(round_num),
        "self_msg_norm": float(self_msg_embed.norm().item()) if torch.is_tensor(self_msg_embed) else 0.0,
        "neighbor_msg_norm": float(neighbor_msg_embed.norm().item()) if torch.is_tensor(neighbor_msg_embed) else 0.0,
        "vote_hist_mass": float(vote_vector.sum().item()) if torch.is_tensor(vote_vector) else 0.0,
        "memory_norm": float(memory_summary.norm().item()) if torch.is_tensor(memory_summary) else 0.0,
    }
    return x, meta
