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
# Social influence: text → bounded latent delta  (NEW)
# ───────────────────────────────────────────────────────────────────────────────
class SocialInfluence(nn.Module):
    """
    Projects a mean-pooled neighbor text embedding into a bounded latent delta δ_social.
    Accepts (D_text,) or (B, D_text) and returns (LATENT_DIM,) or (B, LATENT_DIM).
    Uses tanh(+scale) to keep coupling stable.
    """
    def __init__(self, text_dim: int, latent_dim: int = LATENT_DIM, hidden: int = 64, scale: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(text_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, latent_dim)
        )
        self.scale = float(scale)

    def forward(self, mean_text_embed: torch.Tensor) -> torch.Tensor:
        x = mean_text_embed
        if x.dim() == 1:
            x = x.unsqueeze(0)
        delta = self.net(x)
        delta = torch.tanh(delta) * self.scale
        return delta.squeeze(0) if delta.size(0) == 1 else delta

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
# Mask utilities (shared by all heads)
# ───────────────────────────────────────────────────────────────────────────────
def mask_logits(logits: torch.Tensor, legal_mask: torch.Tensor | None) -> torch.Tensor:
    """
    legal_mask: boolean tensor, same trailing shape as logits.
    Returns logits with illegal positions = -inf (broadcast-safe).
    """
    if legal_mask is None:
        return logits
    if legal_mask.dtype != torch.bool:
        legal_mask = legal_mask.bool()
    if logits.dim() == 2 and legal_mask.dim() == 1:
        legal_mask = legal_mask.unsqueeze(0).expand(logits.size(0), -1)
    return logits.masked_fill(~legal_mask, float("-inf"))

@torch.no_grad()
def illegal_softmax_mass(logits: torch.Tensor, legal_mask: torch.Tensor | None) -> float:
    """
    Compute softmax mass that would fall on illegal indices (diagnostic).
    If you correctly call mask_logits BEFORE softmax during inference, this
    should be near zero on masked logits.
    """
    if legal_mask is None:
        return 0.0
    probs = torch.softmax(logits, dim=-1)
    if probs.dim() == 2 and legal_mask.dim() == 1:
        legal_mask = legal_mask.unsqueeze(0).expand_as(probs)
    illegal = ~legal_mask.bool()
    return float(probs.masked_select(illegal).sum().item())

# ───────────────────────────────────────────────────────────────────────────────
# Phase-aware heads (factorized)
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
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.net(z)
        return mask_logits(logits, mask)

class VoteHead(nn.Module):
    """Logits over agent indices (mask self/dead outside before softmax)."""
    def __init__(self, latent_dim: int, num_agents: int = NUM_AGENTS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.Tanh(),
            nn.Linear(64, num_agents)
        )
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.net(z)
        return mask_logits(logits, mask)

class KillHead(VoteHead):
    """Identical shape to VoteHead; semantics differ (wolves only)."""
    pass

# ───────────────────────────────────────────────────────────────────────────────
# Multi-head containers
# ───────────────────────────────────────────────────────────────────────────────
class FactorizedPlanner(nn.Module):
    """
    Back-compat simple factorized planner: one Talk, one Vote, one Kill.
    Temperature scales all head logits uniformly.
    """
    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        num_agents: int = NUM_AGENTS,
        num_talk_cats: int = NUM_TALK_CATS,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.talk = TalkHead(latent_dim, num_cats=num_talk_cats)
        self.vote = VoteHead(latent_dim, num_agents=num_agents)
        self.kill = KillHead(latent_dim, num_agents=num_agents)
        self.temperature = float(temperature)

    def _apply_temp(self, logits: torch.Tensor) -> torch.Tensor:
        if self.temperature and self.temperature != 1.0:
            return logits * (1.0 / max(1e-6, self.temperature))
        return logits

    def forward(
        self,
        z: torch.Tensor,
        *,
        talk_mask: torch.Tensor | None = None,
        vote_mask: torch.Tensor | None = None,
        kill_mask: torch.Tensor | None = None,
    ) -> dict:
        t_logits = self._apply_temp(self.talk(z))
        v_logits = self._apply_temp(self.vote(z))
        k_logits = self._apply_temp(self.kill(z))

        t_logits = mask_logits(t_logits, talk_mask)
        v_logits = mask_logits(v_logits, vote_mask)
        k_logits = mask_logits(k_logits, kill_mask)

        return {"talk": t_logits, "vote": v_logits, "kill": k_logits}

class PlannerHeads(nn.Module):
    """
    Talk/Vote + Kill (shared or independent coalition).
    .forward(...) returns dict of masked logits: {'talk', 'vote', 'kill'}.
    """
    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        num_agents: int = NUM_AGENTS,
        num_talk_cats: int = NUM_TALK_CATS,
        coalition_mode: str = "shared",     # 'shared' | 'independent'
        temperature: float = 1.0,
    ):
        super().__init__()
        self.talk = TalkHead(latent_dim, num_cats=num_talk_cats)
        self.vote = VoteHead(latent_dim, num_agents=num_agents)
        self.temperature = float(temperature)
        coalition_mode = (coalition_mode or "shared").lower()
        assert coalition_mode in ("shared", "independent")
        self.coalition_mode = coalition_mode
        self.num_agents = int(num_agents)
        self._latent_dim = int(latent_dim)

        if coalition_mode == "shared":
            self.kill_shared = KillHead(latent_dim, num_agents=num_agents)
            self.kill_independent = None
        else:
            self.kill_shared = None
            self.kill_independent = nn.ModuleDict()  # keyed by wolf agent name/id

    def ensure_wolf_head(self, wolf_key: str):
        """For 'independent' coalition mode, make sure a per-wolf KillHead exists."""
        if self.coalition_mode != "independent":
            return
        if wolf_key not in self.kill_independent:
            self.kill_independent[wolf_key] = KillHead(self._latent_dim, num_agents=self.num_agents)

    def _apply_temp(self, logits: torch.Tensor) -> torch.Tensor:
        if self.temperature and self.temperature != 1.0:
            return logits * (1.0 / max(1e-6, self.temperature))
        return logits

    def forward(
        self,
        z: torch.Tensor,
        *,
        talk_mask: torch.Tensor | None = None,
        vote_mask: torch.Tensor | None = None,
        kill_mask: torch.Tensor | None = None,
        wolf_key: str | None = None,   # required if coalition_mode='independent'
    ) -> dict[str, torch.Tensor]:
        t = self._apply_temp(self.talk(z))
        v = self._apply_temp(self.vote(z))
        t = mask_logits(t, talk_mask)
        v = mask_logits(v, vote_mask)

        if self.coalition_mode == "shared":
            k_raw = self._apply_temp(self.kill_shared(z))
        else:
            if wolf_key is None:
                raise ValueError("wolf_key must be provided for coalition_mode='independent'")
            self.ensure_wolf_head(wolf_key)
            k_raw = self._apply_temp(self.kill_independent[wolf_key](z))

        k = mask_logits(k_raw, kill_mask)
        return {"talk": t, "vote": v, "kill": k}

    @classmethod
    def from_config(cls, cfg: dict) -> "PlannerHeads":
        """
        Build from unified config:
          model.latent_dim / .num_actions or .num_agents / .num_talk_cats
          planner.coalitions ('shared'|'independent')
          planner.temperature
        """
        mcfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
        pconf = cfg.get("planner", {}) if isinstance(cfg, dict) else {}
        latent = int(mcfg.get("latent_dim", LATENT_DIM))
        # prefer explicit num_agents; else fall back to num_actions; else global NUM_AGENTS
        num_agents = int(mcfg.get("num_agents", mcfg.get("num_actions", NUM_AGENTS)))
        num_talk  = int(mcfg.get("num_talk_cats", NUM_TALK_CATS))
        coalitions = (pconf.get("coalitions", "shared") or "shared").lower()
        temp = float(pconf.get("temperature", 1.0))
        return cls(
            latent_dim=latent,
            num_agents=num_agents,
            num_talk_cats=num_talk,
            coalition_mode=coalitions,
            temperature=temp,
        )

# ───────────────────────────────────────────────────────────────────────────────
# (Optional) Phase-aware action embedding builder — not wired into old code yet.
# ───────────────────────────────────────────────────────────────────────────────
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
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

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
