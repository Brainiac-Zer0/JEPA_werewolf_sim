# roles.py
from __future__ import annotations
import random
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

import torch, yaml

# ── Load config
with open("config.yaml", "r") as f:
    CFG = yaml.safe_load(f) or {}

# ── Canonical role names
WEREWOLF: str = CFG.get("WEREWOLF", "Werewolf")
VILLAGER: str = CFG.get("VILLAGER", "Worker")

# Optional global seeds for reproducibility
RUN_SEED: Optional[int] = CFG.get("RUN_SEED", None)
PERSONA_SEED: Optional[int] = CFG.get("PERSONA_SEED", None)

# ── Role-conditioned latent priors (kept tiny; mostly placeholders)
_ROLE_PRIORS = CFG.get("ROLE_PRIORS", {})
ROLE_PRIORS = {
    WEREWOLF: torch.ones(32) if _ROLE_PRIORS.get("WEREWOLF", "ones") == "ones" else torch.zeros(32),
    VILLAGER: torch.ones(32) if _ROLE_PRIORS.get("VILLAGER", "zeros") == "ones" else torch.zeros(32),
}

# ── Personality config
PERSONA_ENABLED: bool = bool(CFG.get("PERSONA_ENABLED", True))
PERSONA_SCALE: float   = float(CFG.get("PERSONA_SCALE", 0.2))

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

def _clip(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))

def _rng(seed_pref: Optional[int], fallback: Optional[int]) -> random.Random:
    """Prefer run-level seed; otherwise persona seed; otherwise system RNG."""
    if seed_pref is not None:
        return random.Random(int(seed_pref))
    if fallback is not None:
        return random.Random(int(fallback))
    return random.Random()

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
    Small derived nudges other modules can use.
    - speaker_temp_scale: >1 => more exploratory speech, <1 => more conservative
    - accuse_bias: positive => more likely to accuse, negative => more conciliatory
    - coherence_weight_bonus: nudges toward coherent statements
    """
    speaker_temp_scale = _clip(1.0 + 0.5 * (p["openness"] + p["extraversion"]), 0.7, 1.3)
    accuse_bias = _clip(-p["agreeableness"] + 0.5 * p["extraversion"], -0.2, 0.2)
    coherence_weight_bonus = _clip(p["conscientiousness"] - 0.5 * p["neuroticism"], -0.2, 0.2)
    return {
        "speaker_temp_scale": speaker_temp_scale,
        "accuse_bias": accuse_bias,
        "coherence_weight_bonus": coherence_weight_bonus,
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
    rng_roles   = _rng(RUN_SEED, PERSONA_SEED)
    rng_persona = _rng(PERSONA_SEED, RUN_SEED)

    roles = [WEREWOLF] * num_werewolves + [VILLAGER] * (n - num_werewolves)
    rng_roles.shuffle(roles)

    for agent, role in zip(agent_list, roles):
        agent.role = role

        # Persona: either assign a role-biased random vector, or a neutral one
        if PERSONA_ENABLED:
            p = _sample_persona(rng_persona)
            # tiny role bias
            for k, delta in ROLE_PERSONA_BIASES.get(role, {}).items():
                setattr(p, k, _clip(getattr(p, k) + delta))
            agent.persona = p.as_dict()
            agent.persona_effects = _derive_effects(agent.persona)
        else:
            agent.persona = {
                "extraversion": 0.0, "agreeableness": 0.0, "conscientiousness": 0.0,
                "neuroticism": 0.0, "openness": 0.0
            }
            agent.persona_effects = {"speaker_temp_scale": 1.0, "accuse_bias": 0.0, "coherence_weight_bonus": 0.0}

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
        entries.append({
            "name": a.name,
            "role": r,
            "persona_id": getattr(a, "persona_id", None),
            "effects": getattr(a, "persona_effects", {}),
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
