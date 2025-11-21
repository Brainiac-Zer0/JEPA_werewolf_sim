from __future__ import annotations
from collections import Counter
from typing import Dict, List, Tuple, Optional, Iterable, Any
import math, random

from roles import WEREWOLF  # canonical role name from roles/config


# -------------------------- helpers: roster/legality --------------------------

def _alive_name_set(agents) -> set[str]:
    return {a.name for a in agents if getattr(a, "alive", False)}

def _living_agents(agents):
    return [a for a in agents if getattr(a, "alive", False)]

def _wolves_living(agents):
    return [a for a in _living_agents(agents) if getattr(a, "role", None) == WEREWOLF]

def _non_wolves_living(agents):
    return [a for a in _living_agents(agents) if getattr(a, "role", None) != WEREWOLF]


# -------------------------- night consensus helper ---------------------------

def _normalize_tally(
    tally: Dict[str, float] | Iterable[Tuple[str, float]] | Iterable[Any]
) -> List[Tuple[str, float]]:
    """
    Accepts:
      - dict  {name: count}
      - list  [(name, count), ...]
      - list  [name, name, ...]  (will count occurrences)
    Returns a list of (name, count) with counts >= 0, sorted by name ASC for stability.
    """
    if tally is None:
        return []
    if isinstance(tally, dict):
        items = [(str(k), float(v)) for k, v in tally.items()]
    else:
        # try sequence of pairs; if it's a flat list of names, count them
        try:
            first = next(iter(tally))  # type: ignore
        except StopIteration:
            return []
        except TypeError:
            # non-iterable
            return []

        # Rebuild iterator (since we consumed one element if it was iterable)
        if isinstance(tally, list) or isinstance(tally, tuple):
            seq = tally
        else:
            seq = list(tally)  # type: ignore

        if seq and isinstance(seq[0], (list, tuple)) and len(seq[0]) >= 2:
            items = [(str(a), float(b)) for (a, b, *_) in seq]  # tolerate extra columns
        else:
            c = Counter(str(x) for x in seq)
            items = [(k, float(v)) for k, v in c.items()]

    # sanitize & sort stable
    cleaned = [(name, max(0.0, float(cnt))) for name, cnt in items if isinstance(name, str)]
    cleaned.sort(key=lambda kv: kv[0])
    return cleaned

def consensus_target(
    tally: Dict[str, float] | Iterable[Tuple[str, float]] | Iterable[Any],
    *,
    temperature: float = 1.0,
    rng: Optional[random.Random] = None,
) -> Optional[str]:
    """
    Pick a consensus victim from wolf night-discussion tallies.

    Rule:
      1) If a strict majority exists (> 50% of total mass), return that target deterministically.
      2) Otherwise, draw from a softmax over counts (temperature-scaled):
            p_i ∝ exp((count_i / max_count) / temperature)
         If no RNG provided, uses a local deterministic RNG (seed=0).

    Args:
      tally: dict {name: count} OR list of (name, count) OR list of names (will be counted)
      temperature: >0 => smoother; →0 approaches argmax; <=0 treated as argmax
      rng: optional random.Random for reproducible sampling

    Returns:
      target name (str) or None if no candidates.
    """
    items = _normalize_tally(tally)
    if not items:
        return None

    total = sum(v for _, v in items)
    if total <= 0.0:
        # all zeros → fall back to alphabetical first for stability
        return items[0][0]

    # majority check
    # use deterministic order for tie-breaking (items sorted by name asc)
    items_by_count = sorted(items, key=lambda kv: (-kv[1], kv[0]))
    top_name, top_count = items_by_count[0]
    if top_count > (0.5 * total):
        return top_name

    # softmax fallback
    if temperature <= 0.0:
        # deterministic argmax with name-asc tie-break
        return top_name

    max_c = max(v for _, v in items)
    if max_c <= 0.0:
        return items[0][0]

    # compute probabilities in a numerically stable way
    scaled = []
    for _, v in items:
        # normalize counts by max to keep exponents tame
        scaled.append((v / max_c) / max(1e-6, temperature))

    m = max(scaled)
    exps = [math.exp(x - m) for x in scaled]
    Z = sum(exps)
    if Z <= 0.0 or not math.isfinite(Z):
        return top_name

    probs = [e / Z for e in exps]

    # deterministic by default unless rng is provided by caller
    r = rng if rng is not None else random.Random(0)
    u = r.random()
    acc = 0.0
    for (name, _), p in zip(items, probs):
        acc += p
        if u <= acc + 1e-12:
            return name
    return items[-1][0]  # numeric edge case


# -------------------------- voting (deterministic) ---------------------------

def tally_votes_deterministic(vote_dict: Dict[str, str]) -> List[Tuple[str, int]]:
    """
    Deterministic tally: return list of (target_name, count), sorted by
      1) count descending, then 2) target_name ascending (alpha).
    This avoids reliance on dict insertion order and Counter internals.
    """
    c = Counter(vote_dict.values())
    return sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))


def resolve_votes_detailed(
    vote_dict: Dict[str, str],
    alive_names: Optional[set[str]] = None,
) -> Dict[str, object]:
    """
    Resolve votes with legality reporting and deterministic ordering.

    Returns:
      {
        "winner": str | None,     # eliminated name or None on tie / no votes
        "tie": bool,
        "tally": List[(name, count)],  # deterministic order
        "illegal_votes": List[(voter, target)],  # non-alive or unknown targets
        "n_votes": int
      }
    """
    illegal: List[Tuple[str, str]] = []
    if alive_names is not None:
        # flag votes that target someone not alive/in-roster
        for voter, target in vote_dict.items():
            if target not in alive_names:
                illegal.append((voter, target))

    tally = tally_votes_deterministic(vote_dict) if vote_dict else []
    winner: Optional[str] = None
    tie = False

    if not tally:
        # no votes
        return {
            "winner": None,
            "tie": False,
            "tally": [],
            "illegal_votes": illegal,
            "n_votes": 0,
        }

    if len(tally) == 1:
        winner = tally[0][0]
    else:
        # top two by deterministic order (count desc, name asc)
        (name1, c1), (name2, c2) = tally[0], tally[1]
        if c1 > c2:
            winner = name1
        else:
            tie = True
            winner = None

    return {
        "winner": winner,
        "tie": tie,
        "tally": tally,
        "illegal_votes": illegal,
        "n_votes": sum(cnt for _, cnt in tally),
    }


# Backwards-compatible minimal resolver (kept for existing callers)
def resolve_votes(vote_dict):
    """
    Legacy API: return winner name or None on tie, or no votes.
    Uses the deterministic resolver above.
    """
    det = resolve_votes_detailed(vote_dict)
    return det["winner"]


# -------------------------- elimination side effect --------------------------

def eliminate_player(agent):
    agent.alive = False


# -------------------------- night phase (structured) -------------------------

def night_kill_detailed(agents) -> Dict[str, Optional[str]]:
    """
    Structured night kill:
      - chooses the first living werewolf as 'killer'
      - asks killer.choose_night_target(villagers) for target_name
      - eliminates if found & alive
    Returns:
      {
        "killer": str | None,
        "target": str | None,
        "performed": bool
      }
    """
    wolves = _wolves_living(agents)
    vill  = _non_wolves_living(agents)

    if not wolves or not vill:
        return {"killer": None, "target": None, "performed": False}

    killer = wolves[0]
    target_name = killer.choose_night_target(vill) if hasattr(killer, "choose_night_target") else None

    if target_name:
        for ag in agents:
            if ag.name == target_name and getattr(ag, "alive", False):
                eliminate_player(ag)
                return {"killer": killer.name, "target": target_name, "performed": True}

    return {"killer": killer.name, "target": None, "performed": False}


def night_kill(agents):
    """
    Legacy API: perform kill and return target name or None.
    Uses night_kill_detailed under the hood.
    """
    res = night_kill_detailed(agents)
    return res["target"]
