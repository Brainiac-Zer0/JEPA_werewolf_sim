"""
Tests for Phase 3: genuine JEPA-only voting (world-model free-energy, no planner
head) and a language-independent heuristic (never falls back to the VoteHead).
"""
import types
import torch
import pytest

from encoders import (LATENT_DIM, ACTION_DIM, WorldModelMLP, PhaseActionEncoder,
                      FactorizedPlanner)
import sim


def _living(n=6, alive=None):
    alive = alive or [True] * n
    return [types.SimpleNamespace(name=f"Agent_{i}", alive=alive[i],
                                  vote_history=[], message_memory=[]) for i in range(n)]


def _jepa_agent():
    ag = types.SimpleNamespace(
        name="Agent_0", alive=True,
        world_model=WorldModelMLP(latent_dim=LATENT_DIM, action_dim=ACTION_DIM),
        phase_action_encoder=PhaseActionEncoder(action_dim=ACTION_DIM, num_agents=9, num_talk=5))
    return ag


# ============================ JEPA-only voting ============================= #
def test_jepa_only_returns_valid_alive_target():
    ag = _jepa_agent()
    living = _living(6)
    living[0] = ag  # ensure self present
    out = sim._jepa_only_vote(ag, torch.randn(LATENT_DIM), living)
    assert out is not None and out != ag.name
    assert out in {x.name for x in living if x.alive}


def test_jepa_only_needs_world_model():
    ag = types.SimpleNamespace(name="Agent_0", alive=True, world_model=None,
                               phase_action_encoder=None)
    assert sim._jepa_only_vote(ag, torch.randn(LATENT_DIM), _living(4)) is None


def test_jepa_only_is_argmin_free_energy():
    """The chosen target must minimize ||f(z, a_vote) - z|| among alive candidates."""
    ag = _jepa_agent()
    living = _living(6); living[0] = ag
    z = torch.randn(LATENT_DIM)
    out = sim._jepa_only_vote(ag, z, living)
    zc = z.unsqueeze(0)
    scores = {}
    with torch.no_grad():
        for x in living:
            if x is ag or not x.alive:
                continue
            idx = int(x.name.split("_")[1])
            a = ag.phase_action_encoder(torch.tensor([1]), torch.tensor([idx]), is_talk=False)
            scores[x.name] = float((ag.world_model(zc, a) - zc).norm())
    assert out == min(scores, key=scores.get)


def test_jepa_only_can_differ_from_planner():
    """JEPA-only (world model) should not be identical to the planner VoteHead
    top-1 across many latents — i.e. it is a genuinely different mechanism."""
    ag = _jepa_agent()
    fp = FactorizedPlanner(latent_dim=LATENT_DIM, num_agents=9, num_talk_cats=5)
    living = _living(9); living[0] = ag
    mask = torch.ones(1, 9, dtype=torch.bool); mask[0, 0] = False
    diffs = 0
    for _ in range(40):
        z = torch.randn(LATENT_DIM)
        j = sim._jepa_only_vote(ag, z, living)
        with torch.no_grad():
            v = fp.vote(z.unsqueeze(0), mask=mask).squeeze(0)
        masked = torch.full_like(v, float("-inf"))
        for x in living:
            if x is not ag and x.alive:
                masked[int(x.name.split("_")[1])] = v[int(x.name.split("_")[1])]
        p = f"Agent_{int(masked.argmax())}"
        diffs += (j != p)
    assert diffs > 0     # they disagree at least sometimes


# ============================ Heuristic (B5) =============================== #
def test_heuristic_bandwagon_targets_most_voted():
    living = _living(5)
    # everyone has voted for Agent_3 a lot
    for a in living:
        a.vote_history = ["Agent_3", "Agent_3", "Agent_1"]
    out = sim._heuristic_vote_choice(living[0], living)
    assert out == "Agent_3"


def test_heuristic_language_independent_never_none():
    """With no messages and no vote history, still returns a valid target via the
    deterministic fallback (not the VoteHead, not None)."""
    living = _living(5)
    out = sim._heuristic_vote_choice(living[0], living)
    assert out is not None and out != living[0].name
    assert out in {x.name for x in living if x.alive}


def test_heuristic_excludes_self_and_dead():
    living = _living(5, alive=[True, True, False, True, True])
    for a in living:
        a.vote_history = ["Agent_2", "Agent_0"]     # Agent_2 dead, Agent_0 is self
    out = sim._heuristic_vote_choice(living[0], living)
    assert out not in ("Agent_0", "Agent_2")


def test_heuristic_name_mention_suspicion():
    living = _living(5)
    living[0].message_memory = [("Agent_1", "I think Agent_4 is lying")]
    out = sim._heuristic_vote_choice(living[0], living)
    assert out == "Agent_4"
