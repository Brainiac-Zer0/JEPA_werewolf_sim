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
ACTION_DIM  = int(config.get("ACTION_DIM", 8))  # legacy default
NUM_ACTIONS = int(config.get("NUM_ACTIONS", 6))
NUM_AGENTS  = int(config.get("NUM_AGENTS", 6))

# Optional talk categories (for phase-aware planner split)
TALK_CATEGORIES = config.get("TALK_CATEGORIES", ["accuse", "defend", "hedge", "question", "vote"])
NUM_TALK_CATS   = int(config.get("NUM_TALK_CATS", len(TALK_CATEGORIES)))

# Phase scaffolding (for Day: Discuss/Vote; Night: Kill)
PHASES = {"DISCUSS": 0, "VOTE": 1, "NIGHT": 2}
NUM_PHASES = 3

# Phase-4 knobs (with safe fallbacks)
WORLD_INPUT_MODE  = (config.get("WORLD_INPUT_MODE", "z_plus_action") or "z_plus_action").lower()  # 'z_plus_action'|'z_only'
ACTION_EMBED_KIND = (config.get("ACTION_EMBED_KIND", "onehot") or "onehot").lower()               # 'onehot'|'learned'
ACTION_EMBED_DIM  = int(config.get("ACTION_EMBED_DIM", ACTION_DIM))                               # final a_embed width

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
# Social influence: text → bounded latent delta
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
# Latent/state encoders & world model
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
    Backward-compatible action index embedder (legacy).
    For phase-aware JEPA, prefer PhaseAwareActionEmbedder below.
    """
    def __init__(self, num_actions, action_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_actions, action_dim)

    def forward(self, action_idx):
        return self.embedding(action_idx)

class WorldModelMLP(nn.Module):
    """
    World dynamics over latent state.
    mode='z_plus_action' → uses [z ; a_embed]
    mode='z_only'       → ignores a_embed (for ablations/back-compat)
    """
    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        *,
        mode: str = WORLD_INPUT_MODE,   # 'z_plus_action'|'z_only'
        hidden: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.mode = (mode or "z_plus_action").lower()
        assert self.mode in ("z_plus_action", "z_only")

        in_dim = latent_dim + (action_dim if self.mode == "z_plus_action" else 0)
        layers = [nn.Linear(in_dim, hidden), nn.GELU()]
        if dropout and dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden, latent_dim))
        self.model = nn.Sequential(*layers)

        # init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    @classmethod
    def from_config(cls, cfg: dict, *, latent_dim: int | None = None, action_dim: int | None = None):
        mcfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
        wcfg = cfg.get("world", {}) if isinstance(cfg, dict) else {}
        ld = int(latent_dim or mcfg.get("latent_dim", LATENT_DIM))
        ad = int(action_dim or cfg.get("ACTION_EMBED_DIM", ACTION_EMBED_DIM))
        mode = (wcfg.get("input_mode", cfg.get("WORLD_INPUT_MODE", WORLD_INPUT_MODE)) or "z_plus_action").lower()
        return cls(ld, ad, mode=mode)

    def forward(self, z_t: torch.Tensor, a_t_embed: torch.Tensor | None = None) -> torch.Tensor:
        if self.mode == "z_only":
            if z_t.dim() == 1:
                z_t = z_t.unsqueeze(0)
            return self.model(z_t)
        # z_plus_action
        if a_t_embed is None:
            raise ValueError("WorldModelMLP(mode='z_plus_action') requires a_t_embed, got None.")
        if z_t.dim() == 1:
            z_t = z_t.unsqueeze(0)
        if a_t_embed.dim() == 1:
            a_t_embed = a_t_embed.unsqueeze(0)
        if a_t_embed.size(0) != z_t.size(0):
            raise ValueError(f"Batch mismatch z:{z_t.size(0)} vs a:{a_t_embed.size(0)}")
        if a_t_embed.size(-1) != self.action_dim:
            raise ValueError(f"action_dim mismatch: expected {self.action_dim}, got {a_t_embed.size(-1)}")
        a_t_embed = a_t_embed.to(dtype=z_t.dtype, device=z_t.device)
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
# Phase-aware action embedding (Phase-4)
# ───────────────────────────────────────────────────────────────────────────────
class PhaseAwareActionEmbedder(nn.Module):
    """
    Build an action embedding that encodes both phase and payload (talk cat or agent id).
    Two strategies:
      - kind='onehot' : [onehot(phase); onehot(payload padded)] → MLP → a_embed
      - kind='learned': concat(Emb(phase), Emb_talk|Emb_agent) → proj → a_embed
    API:
      forward_b(phase: Long[B], payload: Long[B]) -> Float[B, ACTION_EMBED_DIM]
    """
    def __init__(
        self,
        *,
        kind: str = ACTION_EMBED_KIND,           # 'onehot' | 'learned'
        a_dim: int = ACTION_EMBED_DIM,
        num_agents: int = NUM_AGENTS,
        num_talk: int = NUM_TALK_CATS,
    ):
        super().__init__()
        self.kind = (kind or "onehot").lower()
        assert self.kind in ("onehot", "learned")
        self.a_dim = int(a_dim)
        self.num_agents = int(num_agents)
        self.num_talk = int(num_talk)
        self.max_payload = max(self.num_agents, self.num_talk)

        if self.kind == "onehot":
            in_dim = NUM_PHASES + self.max_payload
            self.net = nn.Sequential(
                nn.Linear(in_dim, 64),
                nn.ReLU(),
                nn.Linear(64, self.a_dim),
            )
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)
        else:
            # learned: small embeddings + projection
            d_phase = max(4, self.a_dim // 4)
            d_pay   = max(8, self.a_dim // 2)
            self.phase_emb = nn.Embedding(NUM_PHASES, d_phase)
            self.talk_emb  = nn.Embedding(self.num_talk, d_pay)
            self.agent_emb = nn.Embedding(self.num_agents, d_pay)
            self.proj = nn.Linear(d_phase + d_pay, self.a_dim)
            nn.init.xavier_uniform_(self.proj.weight); nn.init.zeros_(self.proj.bias)

    @staticmethod
    def _one_hot(indices: torch.Tensor, depth: int) -> torch.Tensor:
        B = indices.shape[0]
        out = torch.zeros(B, depth, dtype=torch.float32, device=indices.device)
        idx = indices.clamp_(0, depth - 1)
        out[torch.arange(B, device=indices.device), idx] = 1.0
        return out

    def forward_b(self, phase_codes: torch.Tensor, payload_idx: torch.Tensor) -> torch.Tensor:
        """
        phase_codes: (B,) Long in {0:DISCUSS, 1:VOTE, 2:NIGHT}
        payload_idx: (B,) Long, cat_id for DISCUSS; agent_id for VOTE or NIGHT
        """
        if phase_codes.dim() != 1 or payload_idx.dim() != 1:
            raise ValueError("phase_codes and payload_idx must be 1D tensors of shape (B,)")

        if self.kind == "onehot":
            ph = self._one_hot(phase_codes.long(), NUM_PHASES)  # (B,3)

            # Build payload one-hots per phase and place into a fixed-width block [max_payload]
            B = payload_idx.shape[0]
            oh = torch.zeros(B, self.max_payload, dtype=torch.float32, device=payload_idx.device)

            is_discuss = (phase_codes.long() == PHASES["DISCUSS"])
            is_other   = ~is_discuss

            # Discuss rows → talk space
            if is_discuss.any():
                p_talk = payload_idx[is_discuss].clamp(0, self.num_talk - 1)
                rows = torch.nonzero(is_discuss, as_tuple=False).squeeze(1)
                oh[rows, p_talk] = 1.0

            # Vote/Kill rows → agent space (also left-aligned)
            if is_other.any():
                p_agent = payload_idx[is_other].clamp(0, self.num_agents - 1)
                rows = torch.nonzero(is_other, as_tuple=False).squeeze(1)
                oh[rows, p_agent] = 1.0

            x = torch.cat([ph, oh], dim=1)  # (B, 3+max_payload)
            return self.net(x)

        # learned
        ph_e = self.phase_emb(phase_codes.long())  # (B, d_phase)
        is_discuss = (phase_codes.long() == PHASES["DISCUSS"])
        pay_discuss = payload_idx.clamp(0, self.num_talk - 1)
        pay_other   = payload_idx.clamp(0, self.num_agents - 1)

        talk_vec  = self.talk_emb(pay_discuss)   # (B, d_pay)
        agent_vec = self.agent_emb(pay_other)    # (B, d_pay)
        # Row-wise select: if discuss → talk_vec else agent_vec
        payload_e = torch.where(is_discuss.unsqueeze(1), talk_vec, agent_vec)
        a_raw = torch.cat([ph_e, payload_e], dim=-1)
        return self.proj(a_raw)                  # (B, a_dim)

    @classmethod
    def from_config(cls, cfg: dict) -> "PhaseAwareActionEmbedder":
        mcfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
        kind = (cfg.get("ACTION_EMBED_KIND", ACTION_EMBED_KIND) or "onehot").lower()
        a_dim = int(cfg.get("ACTION_EMBED_DIM", ACTION_EMBED_DIM))
        n_agents = int(mcfg.get("num_agents", NUM_AGENTS))
        n_talk   = int(mcfg.get("num_talk_cats", NUM_TALK_CATS))
        return cls(kind=kind, a_dim=a_dim, num_agents=n_agents, num_talk=n_talk)

# ───────────────────────────────────────────────────────────────────────────────
# Deprecation shim: PhaseActionEncoder → PhaseAwareActionEmbedder
# ───────────────────────────────────────────────────────────────────────────────
class PhaseActionEncoder(nn.Module):
    """
    [DEPRECATED] Kept for backward compatibility.
    Wraps PhaseAwareActionEmbedder. Old API:
      forward(phase_code, payload_idx, *, is_talk: bool) -> (B, ACTION_DIM)
    New usage should call PhaseAwareActionEmbedder.forward_b(phase, payload).
    """
    def __init__(self, action_dim: int = ACTION_EMBED_DIM, num_agents: int = NUM_AGENTS, num_talk: int = NUM_TALK_CATS):
        super().__init__()
        self.inner = PhaseAwareActionEmbedder(kind=ACTION_EMBED_KIND, a_dim=action_dim,
                                              num_agents=num_agents, num_talk=num_talk)

    def forward(self, phase_code: torch.Tensor, payload_idx: torch.Tensor, *, is_talk: bool | None = None) -> torch.Tensor:
        # Accept scalars or (B,)
        if not torch.is_tensor(phase_code):
            phase_code = torch.tensor([int(phase_code)], dtype=torch.long)
        if not torch.is_tensor(payload_idx):
            payload_idx = torch.tensor([int(payload_idx)], dtype=torch.long)
        if phase_code.dim() == 0:
            phase_code = phase_code.unsqueeze(0)
        if payload_idx.dim() == 0:
            payload_idx = payload_idx.unsqueeze(0)
        # Ignore is_talk (phase determines space); kept for signature compat.
        return self.inner.forward_b(phase_code.long(), payload_idx.long())

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
