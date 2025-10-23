# roles.py
from __future__ import annotations
import random
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

import torch, yaml

# ── Load config
with open("config.yaml", "r") as f:
    CFG = yaml.safe_load(f) or {}

# ── Canonical role names (kept consistent with sim/agent/judge)
WEREWOLF: str = CFG.get("WEREWOLF", "Werewolf")
VILLAGER: str = CFG.get("VILLAGER", "Worker")

# Warn once if someone changes names elsewhere (helps keep judge prompts/logs aligned)
__warned_roles = False
def _validate_role_constants():
    global __warned_roles
    if __warned_roles:
        return
    if WEREWOLF != "Werewolf" or VILLAGER != "Worker":
        print(f"[roles.py] WARNING: Role labels deviated: WEREWOLF={WEREWOLF}, VILLAGER={VILLAGER}")
    __warned_roles = True

_validate_role_constants()

# ── Seed resolution (structured config first, legacy mirrors second)
DET_SEED     = CFG.get("determinism", {}).get("seed", None)
GLOBAL_SEED  = CFG.get("seeds", {}).get("global", None)
RUN_SEED_LEG = CFG.get("RUN_SEED", None)                 # legacy mirror

PERSONA_SEED_CFG = (
    CFG.get("persona", {}).get("seed", None)             # structured
    if CFG.get("persona", {}) is not None else None
)
PERSONA_SEED_LEG = CFG.get("PERSONA_SEED", None)         # legacy mirror

def _resolve_run_seed() -> Optional[int]:
    for s in (DET_SEED, GLOBAL_SEED, RUN_SEED_LEG):
        if s is not None: return int(s)
    return None

def _resolve_persona_seed() -> Optional[int]:
    for s in (PERSONA_SEED_CFG, PERSONA_SEED_LEG, DET_SEED, GLOBAL_SEED, RUN_SEED_LEG):
        if s is not None: return int(s)
    return None

def _rng(seed: Optional[int]) -> random.Random:
    return random.Random(int(seed)) if seed is not None else random.Random()

# ── Latent dimension aware priors
LATENT_DIM: int = int(CFG.get("LATENT_DIM", CFG.get("model", {}).get("latent_dim", 32)))
_ROLE_PRIORS = CFG.get("ROLE_PRIORS", {})

def _mk_prior(tag: str) -> torch.Tensor:
    mode = _ROLE_PRIORS.get(tag, "zeros")
    return torch.ones(LATENT_DIM) if mode == "ones" else torch.zeros(LATENT_DIM)

ROLE_PRIORS = {
    WEREWOLF: _mk_prior("WEREWOLF"),
    VILLAGER: _mk_prior("VILLAGER"),
}

# ── Personality config (structured first, keep legacy mirrors)
PERSONA_ENABLED: bool = bool(
    CFG.get("persona", {}).get("enabled", CFG.get("PERSONA_ENABLED", True))
)
PERSONA_SCALE: float = float(
    CFG.get("persona", {}).get("scale", CFG.get("PERSONA_SCALE", 0.2))
)

@dataclass
class Persona:
    extraversion: float
    agreeableness: float
    conscientiousness: float
    neuroticism: float
    openness: float
    def as_dict(self): return asdict(self)

# Small role biases (kept tiny so learned behavior dominates)
ROLE_PERSONA_BIASES = {
    WEREWOLF: {"agreeableness": -0.05, "conscientiousness": -0.05, "extraversion": +0.05},
    VILLAGER: {"agreeableness": +0.05, "conscientiousness": +0.05},
}

def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def _sample_persona(rng: random.Random) -> Persona:
    s = PERSONA_SCALE
    return Persona(
        extraversion      = rng.uniform(-s, s),
        agreeableness     = rng.uniform(-s, s),
        conscientiousness = rng.uniform(-s, s),
        neuroticism       = rng.uniform(-s, s),
        openness          = rng.uniform(-s, s),
    )

def _derive_effects(p: Dict[str, float]) -> Dict[str, float]:
    """
    Phase-5 compatible multiplicative scales:
      - speaker_temp_scale      in [0.7, 1.3]
      - accuse_bias_scale       in [0.5, 1.5]  (speaker_llm uses this)
      - coherence_weight_scale  in [0.8, 1.2]  (judge/trainer can read this)
    """
    # exploration: openness + extraversion
    temp = 1.0 + 0.5 * (p["openness"] + p["extraversion"])
    speaker_temp_scale = _clip(temp, 0.7, 1.3)

    # tendency to challenge others: lower agreeableness + some extraversion
    accuse = 1.0 + 0.8 * (-p["agreeableness"]) + 0.4 * p["extraversion"]
    accuse_bias_scale = _clip(accuse, 0.5, 1.5)

    # keep things tidy under stress: conscientiousness vs. neuroticism
    coh = 1.0 + 0.6 * p["conscientiousness"] - 0.4 * p["neuroticism"]
    coherence_weight_scale = _clip(coh, 0.8, 1.2)

    return {
        "speaker_temp_scale": float(speaker_temp_scale),
        "accuse_bias_scale": float(accuse_bias_scale),
        "coherence_weight_scale": float(coherence_weight_scale),
    }

# ─────────────────────────────────────────────────────────────────────────────

def assign_roles(agent_list: List[Any], num_werewolves: int) -> None:
    """
    Deterministic role assignment using config seeds (if provided).
    Mutates agents in-place; does NOT return.
    """
    n = len(agent_list)
    if num_werewolves < 0 or num_werewolves > n:
        raise ValueError(f"num_werewolves={num_werewolves} out of range for n_agents={n}")

    # Separate RNGs so shuffling and persona sampling are reproducible
    rng_roles   = _rng(_resolve_run_seed())
    rng_persona = _rng(_resolve_persona_seed())

    roles = [WEREWOLF] * num_werewolves + [VILLAGER] * (n - num_werewolves)
    rng_roles.shuffle(roles)

    for agent, role in zip(agent_list, roles):
        agent.role = role

        # Persona: either assign a role-biased random vector, or a neutral one
        if PERSONA_ENABLED:
            p = _sample_persona(rng_persona).as_dict()
            # tiny role bias
            if role == WEREWOLF:
                p["agreeableness"]      = _clip(p["agreeableness"] - 0.05, -1.0, 1.0)
                p["conscientiousness"]  = _clip(p["conscientiousness"] - 0.05, -1.0, 1.0)
                p["extraversion"]       = _clip(p["extraversion"] + 0.05, -1.0, 1.0)
            elif role == VILLAGER:
                p["agreeableness"]      = _clip(p["agreeableness"] + 0.05, -1.0, 1.0)
                p["conscientiousness"]  = _clip(p["conscientiousness"] + 0.05, -1.0, 1.0)
            agent.persona = p
            agent.persona_effects = _derive_effects(p)
        else:
            agent.persona = {
                "extraversion": 0.0, "agreeableness": 0.0, "conscientiousness": 0.0,
                "neuroticism": 0.0, "openness": 0.0
            }
            agent.persona_effects = {
                "speaker_temp_scale": 1.0,
                "accuse_bias_scale": 1.0,
                "coherence_weight_scale": 1.0,
            }

        # Stable id for logs
        agent.persona_id = f"P{rng_persona.randrange(1_000_000)}"

    # Optional: let werewolves know their teammates (helpful for downstream hints/logs)
    wolf_names = [a.name for a in agent_list if getattr(a, "role", None) == WEREWOLF]
    for a in agent_list:
        a.team_mates = [n for n in wolf_names if n != a.name] if a.role == WEREWOLF else []

def apply_personality(agent_list: List[Any]) -> None:
    """
    Hook kept for compatibility with existing calls in sim.py.
    Here it's a no-op because personas are already attached in assign_roles.
    """
    return

# ─────────────────────────── telemetry helpers ──────────────────────────────

def roles_meta(agent_list: List[Any]) -> Dict[str, Any]:
    """
    Compact, log-friendly summary of role/persona assignment.
    """
    counts = {WEREWOLF: 0, VILLAGER: 0}
    entries = []
    for a in agent_list:
        r = getattr(a, "role", "Unknown")
        counts[r] = counts.get(r, 0) + 1
        eff = getattr(a, "persona_effects", {}) or {}
        entries.append({
            "name": a.name,
            "role": r,
            "persona_id": getattr(a, "persona_id", None),
            "effects": {
                "speaker_temp_scale": eff.get("speaker_temp_scale", 1.0),
                "accuse_bias_scale": eff.get("accuse_bias_scale", 1.0),
                "coherence_weight_scale": eff.get("coherence_weight_scale", 1.0),
            },
        })
    return {"counts": counts, "assignments": entries}

def role_bit(agent: Any) -> int:
    """1 if werewolf, else 0 — handy for encoders and logs."""
    return 1 if getattr(agent, "role", None) == WEREWOLF else 0

def team_hint(agent: Any, all_agents: List[Any]) -> List[str]:
    """List of teammate names (wolves see wolves; villagers see nobody)."""
    if getattr(agent, "role", None) != WEREWOLF:
        return []
    return [a.name for a in all_agents if a is not agent and getattr(a, "role", None) == WEREWOLF]

__all__ = [
    "WEREWOLF", "VILLAGER",
    "ROLE_PRIORS",
    "assign_roles", "apply_personality",
    "roles_meta", "role_bit", "team_hint",
]
