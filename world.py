from __future__ import annotations
from collections import Counter
from typing import Dict, List, Tuple, Optional

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


# -------------------------- voting (deterministic) ---------------------------

def tally_votes_deterministic(vote_dict: Dict[str, str]) -> List[Tuple[str, int]]:
    """
    Deterministic tally: return list of (target_name, count), sorted by
      1) count descending, then 2) target_name ascending (alpha).
    This avoids reliance on dict insertion order / Counter internals.
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
    Legacy API: return winner name or None on tie / no votes.
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
