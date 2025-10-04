# roles.py
import os
import torch
import random, yaml
from dataclasses import dataclass, asdict

# ── Load config
with open("config.yaml", "r") as f:
    CFG = yaml.safe_load(f)

# ── Roles
WEREWOLF = CFG.get("WEREWOLF", "Werewolf")
VILLAGER = CFG.get("VILLAGER", "Worker")

# ── Role-conditioned latent goal priors
_ROLE_PRIORS = CFG.get("ROLE_PRIORS", {})
ROLE_PRIORS = {
    WEREWOLF: torch.ones(32) if _ROLE_PRIORS.get("WEREWOLF", "ones") == "ones" else torch.zeros(32),
    VILLAGER: torch.ones(32) if _ROLE_PRIORS.get("VILLAGER", "zeros") == "ones" else torch.zeros(32),
}

# ── Personality config
PERSONA_SCALE = float(CFG.get("PERSONA_SCALE", 0.2))
PERSONA_SEED = CFG.get("PERSONA_SEED", None)
@dataclass
class Persona:
    extraversion: float
    agreeableness: float
    conscientiousness: float
    neuroticism: float
    openness: float

    def as_dict(self):
        return asdict(self)

# Small role biases (kept tiny so learned behavior dominates)
ROLE_PERSONA_BIASES = {
    WEREWOLF: {"agreeableness": -0.05, "conscientiousness": -0.05, "extraversion": +0.05},
    VILLAGER: {"agreeableness": +0.05, "conscientiousness": +0.05},
}

def _clip(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))

def _sample_persona(rng: random.Random) -> Persona:
    s = PERSONA_SCALE
    return Persona(
        extraversion     = rng.uniform(-s, s),
        agreeableness    = rng.uniform(-s, s),
        conscientiousness= rng.uniform(-s, s),
        neuroticism      = rng.uniform(-s, s),
        openness         = rng.uniform(-s, s),
    )

def _derive_effects(p: dict) -> dict:
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

def assign_roles(agent_list, num_werewolves):
    rng = random.Random(PERSONA_SEED)
    roles = [WEREWOLF] * num_werewolves + [VILLAGER] * (len(agent_list) - num_werewolves)
    rng.shuffle(roles)

    for agent, role in zip(agent_list, roles):
        agent.role = role

        # Sample and bias a small Big-Five-like persona
        p = _sample_persona(rng)
        for k, delta in ROLE_PERSONA_BIASES.get(role, {}).items():
            setattr(p, k, _clip(getattr(p, k) + delta))

        agent.persona = p.as_dict()            # raw traits in [-PERSONA_SCALE, +PERSONA_SCALE] (+ tiny role bias)
        agent.persona_effects = _derive_effects(agent.persona)  # small derived nudges for downstream use
        agent.persona_id = f"P{rng.randrange(1_000_000)}"       # handy for logs/ablation
