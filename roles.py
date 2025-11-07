# roles.py (Phase-7: persona → backend-aware variability)
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

# Optional: bit mapping for mouthpiece/encoders/logs (wolf=1, villager=0)
ROLE_TO_BIT: Dict[str, int] = {WEREWOLF: 1, VILLAGER: 0}
BIT_TO_ROLE: Dict[int, str] = {1: WEREWOLF, 0: VILLAGER}

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

# ── Backend provider (controls how "temperature" effects are routed)
LLM_PROVIDER: str = str(CFG.get("llm", {}).get("provider", "openai")).lower()

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

def _scale_to_bucket(scale: float) -> int:
    """
    Map a temperature-like scale (≈0.7..1.3) to a small discrete bucket {0,1,2}
    used for prompt-side entropy choices (templates/markers).
    """
    if scale < 0.9:  # cooler
        return 0
    if scale > 1.1:  # hotter
        return 2
    return 1

def _derive_effects(p: Dict[str, float]) -> Dict[str, float]:
    """
    Phase-5 compatible multiplicative scales + Phase-7 backend-aware routing.

    Always compute a 'raw' exploration factor from persona, but:
      - If LLM_PROVIDER == 'hf': forward it as speaker_temp_scale (HF decoding will use it)
      - If LLM_PROVIDER == 'openai': DO NOT rely on API temperature; convert to prompt-side
        variability knobs (template entropy, hedging, discourse variety).
    """
    # exploration: openness + extraversion
    temp_raw = 1.0 + 0.5 * (p["openness"] + p["extraversion"])  # ~[0.5, 1.5] before clip
    temp_raw = _clip(temp_raw, 0.7, 1.3)

    # tendency to challenge others: lower agreeableness + some extraversion
    accuse = 1.0 + 0.8 * (-p["agreeableness"]) + 0.4 * p["extraversion"]
    accuse_bias_scale = _clip(accuse, 0.5, 1.5)

    # keep things tidy under stress: conscientiousness vs. neuroticism
    coh = 1.0 + 0.6 * p["conscientiousness"] - 0.4 * p["neuroticism"]
    coherence_weight_scale = _clip(coh, 0.8, 1.2)

    effects: Dict[str, Any] = {
        "accuse_bias_scale": float(accuse_bias_scale),
        "coherence_weight_scale": float(coherence_weight_scale),
    }

    if LLM_PROVIDER == "hf":
        # HF path: allow decode temperature scaling directly
        effects["speaker_temp_scale"] = float(temp_raw)
        # legacy underscore alias for back-compat
        effects["_temp_scale"] = float(temp_raw)
        # Also expose a mild prompt-side nudge; HF callers may ignore these.
        effects["prompt_entropy_bucket"] = int(_scale_to_bucket(temp_raw))
        effects["hedge_prob_boost"] = float(_clip((temp_raw - 1.0) * 0.25, -0.08, 0.12))
        effects["discourse_variety_w"] = float(_clip((temp_raw - 1.0) * 1.2 + 1.0, 0.8, 1.3))
    else:
        # OpenAI path: DO NOT alter API temperature — convert to prompt-side variability only
        # Keep scale fields neutral so downstream sanitize never forwards them.
        effects["speaker_temp_scale"] = 1.0
        effects["_temp_scale"] = 1.0

        # Prompt-side variability knobs (used by speaker/sim planners):
        # - template selection entropy (0=conservative, 2=spicy)
        bucket = _scale_to_bucket(temp_raw)
        effects["prompt_entropy_bucket"] = int(bucket)

        # - hedging verbs probability boost (keeps OpenAI path within prompt-only changes)
        #   map 0.7..1.3 → approx -0.08..+0.12
        effects["hedge_prob_boost"] = float(_clip((temp_raw - 1.0) * 0.25, -0.08, 0.12))

        # - discourse marker variety weight (used to pick from a larger set of markers)
        effects["discourse_variety_w"] = float(_clip((temp_raw - 1.0) * 1.2 + 1.0, 0.8, 1.3))

        # - optional seed noise (downstream can sample a synonym / opener from this range)
        effects["seed_noise"] = float(_clip((temp_raw - 1.0) * 3.0, -0.9, 0.9))

    return effects

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
            # Neutral effects; temperature-related fields are safe defaults
            agent.persona_effects = {
                "speaker_temp_scale": 1.0,
                "_temp_scale": 1.0,
                "accuse_bias_scale": 1.0,
                "coherence_weight_scale": 1.0,
                "prompt_entropy_bucket": 1,
                "hedge_prob_boost": 0.0,
                "discourse_variety_w": 1.0,
                "seed_noise": 0.0,
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
                "prompt_entropy_bucket": eff.get("prompt_entropy_bucket", 1),
                "hedge_prob_boost": eff.get("hedge_prob_boost", 0.0),
                "discourse_variety_w": eff.get("discourse_variety_w", 1.0),
            },
        })
    return {"counts": counts, "assignments": entries}

def role_bit(agent: Any) -> int:
    """1 if werewolf, else 0 — handy for encoders and logs."""
    return ROLE_TO_BIT.get(getattr(agent, "role", None), 0)

def team_hint(agent: Any, all_agents: List[Any]) -> List[str]:
    """List of teammate names (wolves see wolves; villagers see nobody)."""
    if getattr(agent, "role", None) != WEREWOLF:
        return []
    return [a.name for a in all_agents if a is not agent and getattr(a, "role", None) == WEREWOLF]

__all__ = [
    "WEREWOLF", "VILLAGER",
    "ROLE_TO_BIT", "BIT_TO_ROLE",
    "ROLE_PRIORS",
    "assign_roles", "apply_personality",
    "roles_meta", "role_bit", "team_hint",
]
