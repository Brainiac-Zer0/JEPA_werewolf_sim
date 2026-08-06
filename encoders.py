# encoders.py — Phase-5 ready encoders & heads (situated dialog + stable APIs)
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from functools import lru_cache
from typing import List, Tuple, Optional, Dict
import yaml

__all__ = [
    # dims/constants/helpers
    "INPUT_DIM", "LATENT_DIM", "ACTION_DIM", "NUM_ACTIONS", "NUM_AGENTS",
    "TALK_CATEGORIES", "NUM_TALK_CATS", "PHASES", "NUM_PHASES",
    "phase_onehot",
    # encoders / models
    "MessageEncoder", "DialogContextEncoder", "SocialInfluence",
    "social_delta_from_dialog",
    "MLPBeliefEncoder", "ActionEncoder", "WorldModelMLP",
    # heads/containers
    "PlannerHead", "TalkHead", "VoteHead", "KillHead",
    "FactorizedPlanner", "PlannerHeads",
    # action embedding (and back-compat)
    "PhaseAwareActionEmbedder", "PhaseActionEncoder",
    # regularizers & feature packers
    "talk_entropy_loss", "talk_kl_to_uniform",
    "package_features", "package_features_with_meta",
]

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

# ── Encoders Phase-5 knobs (optional, safe defaults)
ENC_CFG = (config.get("encoders", {}) or {})
ENC_CTX = (ENC_CFG.get("context", {}) or {})
CTX_WINDOW      = int(ENC_CTX.get("window", 6))
CTX_DECAY       = float(ENC_CTX.get("decay", 0.85))
CTX_SELF_GATE   = float(ENC_CTX.get("self_gate", 1.15))
CTX_OTHER_GATE  = float(ENC_CTX.get("other_gate", 1.0))
CTX_ADD_ROLE    = bool(ENC_CTX.get("add_role_bit", True))
CTX_L2_NORM     = bool(ENC_CTX.get("l2_norm", True))

ENC_TEXT        = (ENC_CFG.get("text", {}) or {})
TEXT_L2_NORM    = bool(ENC_TEXT.get("l2_norm", True))
TEXT_CACHE_SIZE = int(ENC_TEXT.get("cache_size", 8192))

ENC_REG         = (ENC_CFG.get("regularization", {}) or {})
TALK_ENTROPY_W  = float(ENC_REG.get("talk_entropy_w", 0.0))
TALK_KL_UNIF_W  = float(ENC_REG.get("talk_kl_uniform_w", 0.0))

def phase_onehot(phase_code: int) -> torch.Tensor:
    v = torch.zeros(NUM_PHASES, dtype=torch.float32)
    if 0 <= phase_code < NUM_PHASES:
        v[phase_code] = 1.0
    return v

# ───────────────────────────────────────────────────────────────────────────────
# TEXT ENCODER MODULE  (shared; includes L2 + larger cache)
# ───────────────────────────────────────────────────────────────────────────────
class MessageEncoder(nn.Module):
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        self.output_dim = self.transformer.config.hidden_size
        self.transformer.eval()
        for p in self.transformer.parameters():
            p.requires_grad_(False)

    @lru_cache(maxsize=TEXT_CACHE_SIZE)
    def _encode_cached_one(self, text: str):
        # Returns a CPU tensor (D,) for determinism & low VRAM
        encoded_input = self.tokenizer([text], padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.transformer(**encoded_input)
        pooled = self.mean_pooling(model_output, encoded_input['attention_mask'])[0]  # (D,)
        vec = pooled.detach().cpu()
        if TEXT_L2_NORM:
            vec = F.normalize(vec, dim=0, eps=1e-8)
        return vec

    def forward(self, texts):
        # Fast path for a single string (cacheable)
        if isinstance(texts, str):
            return self._encode_cached_one(texts).unsqueeze(0)  # (1,D)
        # Batch path (not cached)
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.transformer(**encoded_input)
        pooled = self.mean_pooling(model_output, encoded_input['attention_mask'])
        out = pooled.detach()
        if TEXT_L2_NORM:
            out = F.normalize(out, dim=1, eps=1e-8)
        return out

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state  # (B,T,D)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())  # (B,T,1)->(B,T,D)
        denom = input_mask_expanded.sum(1).clamp_min(1.0)  # avoid div/0
        return (token_embeddings * input_mask_expanded).sum(1) / denom  # (B,D)

# ───────────────────────────────────────────────────────────────────────────────
# Dialog context encoder: (speaker,text) window → one context vector
# ───────────────────────────────────────────────────────────────────────────────
class DialogContextEncoder(nn.Module):
    """
    Turn a short window of recent dialog into one context vector.
    - Per-utterance embed via MessageEncoder (shared).
    - Exponential decay on recency.
    - Self/other gating.
    - Optional role bit concatenation (Werewolf/Worker ~ 1/0).
    Output: (D,) where D = msg_dim (+1 if add_role_bit).
    """
    def __init__(
        self,
        message_encoder: MessageEncoder,
        *,
        window: int = CTX_WINDOW,
        decay: float = CTX_DECAY,
        self_gate: float = CTX_SELF_GATE,
        other_gate: float = CTX_OTHER_GATE,
        add_role_bit: bool = CTX_ADD_ROLE,
        l2_norm: bool = CTX_L2_NORM,
    ):
        super().__init__()
        self.msg_enc = message_encoder
        self.window = int(window)
        self.decay = float(decay)
        self.self_gate = float(self_gate)
        self.other_gate = float(other_gate)
        self.add_role_bit = bool(add_role_bit)
        self.l2_norm = bool(l2_norm)
        self.output_dim = self.msg_enc.output_dim + (1 if self.add_role_bit else 0)

    @torch.no_grad()
    def forward_from_messages(
        self,
        messages: List[Tuple[str, str]],
        *,
        self_name: Optional[str] = None,
        role_bit: Optional[int] = None
    ) -> torch.Tensor:
        """
        messages: [(speaker_name, text), ...] most-recent LAST.
        self_name: current agent name (to gate self vs others).
        role_bit: optional {1 werewolf, 0 villager}; appended if add_role_bit=True.
        """
        if not messages:
            base = torch.zeros(self.msg_enc.output_dim)
            return self._maybe_cat_role(base, role_bit)

        msgs = messages[-self.window:]
        T = len(msgs)
        weights = torch.tensor([self.decay ** (T - 1 - i) for i in range(T)], dtype=torch.float32)
        weights = weights / weights.sum().clamp_min(1e-8)

        texts = [m[1] for m in msgs]
        embeds = self.msg_enc(texts)  # (T, D)

        if self_name is not None:
            gates = torch.tensor(
                [self.self_gate if (speaker == self_name) else self.other_gate for (speaker, _) in msgs],
                dtype=torch.float32
            ).unsqueeze(1)  # (T,1)
            embeds = embeds * gates

        ctx = (weights.unsqueeze(1) * embeds).sum(0)  # (D,)
        if self.l2_norm:
            ctx = F.normalize(ctx, dim=0, eps=1e-8)
        return self._maybe_cat_role(ctx, role_bit)

    def _maybe_cat_role(self, v: torch.Tensor, role_bit: Optional[int]) -> torch.Tensor:
        if self.add_role_bit:
            bit = torch.tensor([float(1.0 if role_bit else 0.0)], dtype=torch.float32)
            return torch.cat([v, bit], dim=0)
        return v

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

@torch.no_grad()
def social_delta_from_dialog(
    social_module: SocialInfluence,
    ctx_encoder: DialogContextEncoder,
    messages: List[Tuple[str, str]],
    *,
    self_name: Optional[str] = None,
    role_bit: Optional[int] = None,
) -> torch.Tensor:
    """
    Convenience: (speaker,text)[] → context vector → δ_social via SocialInfluence.
    If DialogContextEncoder added a role bit, it is stripped before SocialInfluence.
    """
    ctx_vec = ctx_encoder.forward_from_messages(messages, self_name=self_name, role_bit=role_bit)
    if ctx_encoder.add_role_bit:
        ctx_vec = ctx_vec[:-1]
    return social_module(ctx_vec)

# ───────────────────────────────────────────────────────────────────────────────
# Latent/state encoders & world model
# ───────────────────────────────────────────────────────────────────────────────
class MLPBeliefEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            # Output LayerNorm bounds the latent norm (~sqrt(latent_dim)). Without it
            # the encoder's output scale drifted wildly across runs (‖z‖ from ~5 to
            # ~1400) and could blow up, making Δz-MSE meaningless and destabilizing
            # training. It also removes the trivial large-constant collapse mode.
            nn.LayerNorm(latent_dim),
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
        if z.dim() == 1:
            z = z.unsqueeze(0)
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
    """Logits over talk categories (NUM_TALK_CATS). Always returns (B, C)."""
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
        if z.dim() == 1:
            z = z.unsqueeze(0)                 # ensure (B, D)
        logits = self.net(z)                    # (B, C)
        return mask_logits(logits, mask)        # (B, C) w/ mask broadcast

class VoteHead(nn.Module):
    """Logits over agent indices (mask self/dead outside before softmax). Always (B, A)."""
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
        if z.dim() == 1:
            z = z.unsqueeze(0)                 # ensure (B, D)
        logits = self.net(z)                    # (B, A)
        return mask_logits(logits, mask)        # mask compat with training_utils.py

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
        if z.dim() == 1:
            z = z.unsqueeze(0)  # (B,D) for consistent downstream shapes
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
    .forward(...) returns dict of masked logits: {'talk', 'vote', 'kill'} with (B,·) shapes.
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
        if z.dim() == 1:
            z = z.unsqueeze(0)
        t = self._apply_temp(self.talk(z))     # (B,C)
        v = self._apply_temp(self.vote(z))     # (B,A)
        t = mask_logits(t, talk_mask)
        v = mask_logits(v, vote_mask)

        if self.coalition_mode == "shared":
            k_raw = self._apply_temp(self.kill_shared(z))
        else:
            if wolf_key is None:
                raise ValueError("wolf_key must be provided for coalition_mode='independent'")
            self.ensure_wolf_head(wolf_key)
            k_raw = self._apply_temp(self.kill_independent[wolf_key](z))

        k = mask_logits(k_raw, kill_mask)      # (B,A)
        return {"talk": t, "vote": v, "kill": k}

    @classmethod
    def from_config(cls, cfg: dict) -> "PlannerHeads":
        """
        Build from unified config:
          model.latent_dim / .num_agents / .num_talk_cats
          planner.coalitions ('shared'|'independent')
          planner.temperature
        """
        mcfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
        pconf = cfg.get("planner", {}) if isinstance(cfg, dict) else {}
        latent = int(mcfg.get("latent_dim", LATENT_DIM))
        # IMPORTANT: do NOT fall back to model.num_actions here — that caused 6 vs 9 bugs.
        num_agents = int(mcfg.get("num_agents", config.get("NUM_AGENTS", NUM_AGENTS)))
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
    DISCUSS phase → payload is talk_category_id ∈ [0, NUM_TALK_CATS)
    VOTE/NIGHT   → payload is agent_id       ∈ [0, NUM_AGENTS)
    Strategies:
      - kind='onehot' : [onehot(phase); onehot(payload padded to max(num_agents,num_talk))] → MLP → a_embed
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
# Persona modulation helpers (opt-in; keep head APIs stable)
# ───────────────────────────────────────────────────────────────────────────────
class PersonaModulation:
    """
    Light-touch multipliers derived from agent.persona_effects.
    Apply outside heads (or as tiny logit nudges) to preserve interfaces.
    """
    def __init__(self, effects: Dict | None):
        eff = effects or {}
        self.talk_temperature_scale = float(eff.get("speaker_temp_scale", 1.0))
        self.coherence_weight_scale = float(eff.get("coherence_weight_scale", 1.0))
        self.accuse_prior_boost     = float(eff.get("accuse_bias_scale", 1.0)) - 1.0  # ~[-0.5,+0.5]→ small

    def apply_talk_prior(self, talk_logits: torch.Tensor, accuse_index: Optional[int]) -> torch.Tensor:
        """
        Optional: add a tiny bias to the 'accuse' logit.
        talk_logits: (B,C) or (C,)
        """
        if accuse_index is None or abs(self.accuse_prior_boost) < 1e-6:
            return talk_logits
        if talk_logits.dim() == 1:
            out = talk_logits.clone()
            out[accuse_index] = out[accuse_index] + self.accuse_prior_boost
            return out
        out = talk_logits.clone()
        out[:, accuse_index] = out[:, accuse_index] + self.accuse_prior_boost
        return out

# ───────────────────────────────────────────────────────────────────────────────
# Regularizers for Talk (opt-in; trainer composes with weights from config)
# ───────────────────────────────────────────────────────────────────────────────
def talk_entropy_loss(talk_logits: torch.Tensor) -> torch.Tensor:
    """
    Encourage mild exploration; returns positive scalar (mean over batch).
    Use as: loss += TALK_ENTROPY_W * talk_entropy_loss(t_logits)
    """
    p = torch.softmax(talk_logits, dim=-1)
    ent = -(p * (p.clamp_min(1e-12).log())).sum(dim=-1)
    return -ent.mean()  # negative entropy (so adding this increases entropy)

def talk_kl_to_uniform(talk_logits: torch.Tensor) -> torch.Tensor:
    """
    Tiny KL to uniform prior to avoid category collapse.
    """
    C = talk_logits.size(-1)
    p = torch.softmax(talk_logits, dim=-1)
    u = torch.full_like(p, 1.0 / C)
    kl = (p * (p.clamp_min(1e-12).log() - u.log())).sum(dim=-1)
    return kl.mean()

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
