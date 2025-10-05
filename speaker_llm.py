# speaker.py
# Deterministic, mask-aware template speaker with logging + REINFORCE.
from __future__ import annotations
import os, json, math, random, hashlib
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import torch, yaml
import torch.nn as nn
import torch.nn.functional as F

# ── Config
with open("config.yaml", "r") as f:
    CFG = yaml.safe_load(f) or {}

SPEAKER_SEED: Optional[int] = CFG.get("SPEAKER_SEED", None)
SPEAKER_DEBUG: bool = bool(CFG.get("SPEAKER_DEBUG", False))
SPEAKER_LOG_DIR: str = CFG.get("SPEAKER_LOG_DIR", "logs")
SPEAKER_TEMP: float = float(CFG.get("SPEAKER_TEMP", 1.0))
SPEAKER_EPS: float = float(CFG.get("SPEAKER_EPS", 0.05))       # ε-greedy on top of softmax
SPEAKER_HIST_K: int = int(CFG.get("SPEAKER_HIST_K", 3))        # recent lines to featurize
SPEAKER_ENTROPY_BONUS: float = float(CFG.get("SPEAKER_ENTROPY_BONUS", 0.01))
SPEAKER_LR: float = float(CFG.get("SPEAKER_LR", 1e-3))

# ── Stable, shared historical features (also imported by speaker_llm.py)
def make_hist_feats(recent_texts: List[str]) -> torch.Tensor:
    """
    Short, stable features from last K messages:
      [accuse_rate, mean_len_100]
    """
    if not recent_texts:
        return torch.tensor([0.0, 0.0], dtype=torch.float32)
    n = len(recent_texts)
    accuse_like = ("accuse", "suspect", "vote", "eliminate", "target", "lynch")
    acc = sum(any(w in t.lower() for w in accuse_like) for t in recent_texts) / max(1, n)
    mean_len = sum(len(t) for t in recent_texts) / max(1, n)
    return torch.tensor([acc, min(1.5, mean_len / 100.0)], dtype=torch.float32)

# ── Default templates (can be overridden in config)
DEFAULT_TEMPLATES: List[str] = CFG.get("DEFAULT_TEMPLATES", [
    "Accuse {target}",
    "Defend {ally}",
    "Ask {target} a question",
    "Express uncertainty",
    "Propose vote on {target}",
])

# Optional explicit mapping to coarse categories (helps logging/analytics)
TEMPLATE_CATS: List[str] = CFG.get("TEMPLATE_CATS", [
    "accuse", "defend", "question", "hedge", "vote"
])

def _role_to_bit(role: Optional[str]) -> float:
    r = (role or "").lower()
    return 1.0 if r.startswith("were") else 0.0  # 1 = werewolf, 0 = villager/worker

def _rng_for_agent(agent_name: str, fallback_seed: Optional[int]) -> random.Random:
    """
    Per-agent reproducible RNG: hash(agent_name) mixed with SPEAKER_SEED.
    Ensures: same config + same roster ⇒ same speaker choices.
    """
    base = 0 if fallback_seed is None else int(fallback_seed)
    h = int(hashlib.sha256(agent_name.encode("utf-8")).hexdigest(), 16) & 0xFFFFFFFF
    return random.Random((base ^ h) & 0xFFFFFFFF)

def _safe_choice(rng: random.Random, seq: List[str]) -> Optional[str]:
    if not seq:
        return None
    return seq[rng.randrange(len(seq))]

def _mask_targets(candidate_targets: List[str], self_name: str, alive_names: Optional[List[str]]) -> List[str]:
    alive = set(alive_names) if alive_names else set(candidate_targets)
    return [t for t in candidate_targets if t in alive and t != self_name]

def _softmax_sample(rng: random.Random, logits: torch.Tensor, eps: float = 0.0, temp: float = 1.0) -> int:
    if logits.numel() == 1:
        return 0
    x = logits / max(1e-6, float(temp))
    probs = torch.softmax(x, dim=-1)
    if rng.random() < eps:
        return rng.randrange(len(probs))
    # Gumbel trick with external RNG for determinism
    u = torch.tensor([max(1e-9, min(1.0-1e-9, rng.random())) for _ in range(len(probs))], dtype=probs.dtype)
    g = -torch.log(-torch.log(u))
    return int(torch.argmax(torch.log(probs + 1e-9) + g).item())

def _fill_template(
    tmpl: str,
    *,
    rng: random.Random,
    self_name: str,
    candidate_targets: List[str],
    alive_names: Optional[List[str]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Fill placeholders conservatively; never select self; prefer alive targets."""
    targets = _mask_targets(candidate_targets, self_name, alive_names)
    ally = _safe_choice(rng, targets) or _safe_choice(rng, candidate_targets) or self_name
    target = _safe_choice(rng, targets) or ally

    text = tmpl
    text = text.replace("{self}", self_name)
    text = text.replace("{target}", target if target else self_name)
    text = text.replace("{ally}", ally if ally else self_name)
    return text, {"target": target, "ally": ally}

# ───────────────────────── SpeakerBandit ─────────────────────────

class SpeakerBandit(nn.Module):
    """
    Tiny template selector conditioned on z, role_bit and short history features.
    - Deterministic per-agent RNG
    - Returns (text, meta) where meta holds tensors your train loop expects
    - learn_step: simple REINFORCE over template logits using judge rewards
    """
    def __init__(self, latent_dim: int, num_templates: int, hidden: int = 128):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.num_templates = int(num_templates)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1 + 2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, num_templates)
        )
        self.temperature: float = SPEAKER_TEMP

    def forward(self, z_t: torch.Tensor, role_bit: torch.Tensor, hist_feats: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_t, role_bit, hist_feats], dim=-1)
        return self.net(x)  # unnormalized logits over templates

    @torch.no_grad()
    def generate(
        self,
        *,
        z_t: torch.Tensor,
        role: Optional[str],
        recent_texts: List[str],
        templates: List[str],
        candidate_targets: List[str],
        self_name: str,
        persona_effects: Optional[Dict[str, float]] = None,
        alive_names: Optional[List[str]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Returns:
          text: filled utterance
          meta: {
             "template_id": int,
             "role_bit": Tensor[1],
             "hist_feats": Tensor[2],
             "choice_probs": Tensor[num_templates],
             "temperature": float,
             "eps": float,
             "persona_effects": dict,
             "cat": str,
          }
        """
        device = next(self.parameters()).device
        if z_t.dim() == 1:
            z_t = z_t.unsqueeze(0)
        z_t = z_t.to(device)

        rb = torch.tensor([[ _role_to_bit(role) ]], dtype=z_t.dtype, device=device)         # [1,1]
        h  = make_hist_feats(recent_texts).to(device).unsqueeze(0)                           # [1,2]

        logits = self.forward(z_t, rb, h).squeeze(0)                                         # [T]
        temp = max(0.2, float(self.temperature))
        eps  = max(0.0, min(0.5, SPEAKER_EPS))

        # Persona nudges (subtle; keeps behavior stable)
        bias = 0.0
        if persona_effects:
            bias = float(persona_effects.get("accuse_bias", 0.0))  # in [-0.2, +0.2]
        # If we know categories, add a tiny bias to the matching template ids
        if TEMPLATE_CATS and len(TEMPLATE_CATS) == len(templates):
            for i, cat in enumerate(TEMPLATE_CATS):
                if cat == "accuse":
                    logits[i] = logits[i] + bias

        rng = _rng_for_agent(self_name, SPEAKER_SEED)
        idx = _softmax_sample(rng, logits, eps=eps, temp=temp)

        # fill the chosen template
        tmpl = templates[idx] if 0 <= idx < len(templates) else templates[0]
        text, fill_meta = _fill_template(
            tmpl, rng=rng, self_name=self_name,
            candidate_targets=candidate_targets, alive_names=alive_names
        )

        probs = torch.softmax(logits / temp, dim=-1)

        cat = TEMPLATE_CATS[idx] if 0 <= idx < len(TEMPLATE_CATS) else "other"
        meta = {
            "template_id": int(idx),
            "role_bit": rb.squeeze(0).detach().cpu(),     # Tensor[1]
            "hist_feats": h.squeeze(0).detach().cpu(),    # Tensor[2]
            "choice_probs": probs.detach().cpu(),         # Tensor[T]
            "temperature": temp,
            "eps": eps,
            "persona_effects": dict(persona_effects or {}),
            "cat": cat,
            **fill_meta
        }

        if SPEAKER_DEBUG:
            try:
                os.makedirs(SPEAKER_LOG_DIR, exist_ok=True)
                with open(os.path.join(SPEAKER_LOG_DIR, "speaker_decisions.jsonl"), "a", encoding="utf-8") as f:
                    row = {
                        "agent": self_name,
                        "role": role,
                        "template": tmpl,
                        "filled": text,
                        "cat": cat,
                        "target": fill_meta.get("target"),
                        "ally": fill_meta.get("ally"),
                        "temp": temp,
                        "eps": eps,
                        "probs": [float(x) for x in probs.tolist()],
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            except Exception as e:
                print("[SPEAKER-DBG] log write failed:", e)

        return text, meta

    def learn_step(
        self,
        batch: List[Dict[str, Any]],
        opt: Optional[torch.optim.Optimizer] = None,
        *,
        entropy_bonus: float = SPEAKER_ENTROPY_BONUS,
        baseline: float = 0.0,
    ) -> Dict[str, float]:
        """
        batch[i]:
        {
          "z": Tensor[d],
          "role_bit": Tensor[1],
          "hist_feats": Tensor[2],
          "template_id": int,
          "reward": float,
        }
        """
        if not batch:
            return {"loss": 0.0, "entropy": 0.0, "R_mean": 0.0}

        device = next(self.parameters()).device
        zs   = torch.stack([b["z"] for b in batch]).to(device)                              # [B,d]
        rbit = torch.stack([b["role_bit"] for b in batch]).to(device)                       # [B,1]
        hfs  = torch.stack([b["hist_feats"] for b in batch]).to(device)                     # [B,2]
        acts = torch.tensor([int(b["template_id"]) for b in batch], device=device)          # [B]
        R    = torch.tensor([float(b["reward"]) for b in batch], device=device)             # [B]

        logits = self.forward(zs, rbit, hfs)                                                # [B,T]
        logp = F.log_softmax(logits, dim=-1).gather(1, acts.view(-1,1)).squeeze(1)          # [B]
        ent = -(F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()

        # REINFORCE objective
        loss = -( (R - baseline) * logp ).mean() - float(entropy_bonus) * ent

        if opt is None:
            opt = torch.optim.Adam(self.parameters(), lr=SPEAKER_LR)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        opt.step()

        return {"loss": float(loss.item()), "entropy": float(ent.item()), "R_mean": float(R.mean().item())}
