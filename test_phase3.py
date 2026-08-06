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


# ==================== JEPA-only: edge cases + properties =================== #
def test_jepa_only_deterministic():
    ag = _jepa_agent(); living = _living(6); living[0] = ag
    z = torch.randn(LATENT_DIM)
    assert sim._jepa_only_vote(ag, z, living) == sim._jepa_only_vote(ag, z, living)


def test_jepa_only_accepts_2d_latent():
    ag = _jepa_agent(); living = _living(6); living[0] = ag
    z = torch.randn(LATENT_DIM)
    assert sim._jepa_only_vote(ag, z, living) == sim._jepa_only_vote(ag, z.unsqueeze(0), living)


def test_jepa_only_single_candidate():
    ag = _jepa_agent()
    living = _living(6, alive=[True, False, False, False, False, True]); living[0] = ag
    # only Agent_5 is an alive non-self candidate
    assert sim._jepa_only_vote(ag, torch.randn(LATENT_DIM), living) == "Agent_5"


def test_jepa_only_no_candidates_returns_none():
    ag = _jepa_agent()
    living = _living(6, alive=[True, False, False, False, False, False]); living[0] = ag
    assert sim._jepa_only_vote(ag, torch.randn(LATENT_DIM), living) is None


def test_jepa_only_never_targets_dead_or_self():
    ag = _jepa_agent()
    living = _living(7, alive=[True, True, False, True, False, True, True]); living[0] = ag
    dead = {"Agent_2", "Agent_4"}
    for _ in range(30):
        out = sim._jepa_only_vote(ag, torch.randn(LATENT_DIM), living)
        assert out not in dead and out != ag.name


def test_jepa_only_skips_malformed_names():
    ag = _jepa_agent()
    living = _living(4); living[0] = ag
    living.append(types.SimpleNamespace(name="Spectator", alive=True))   # no _index
    out = sim._jepa_only_vote(ag, torch.randn(LATENT_DIM), living)
    assert out is not None and out.startswith("Agent_")


def test_jepa_only_is_z_sensitive():
    """Over many latents the world-model vote spreads across candidates (uses z)."""
    ag = _jepa_agent(); living = _living(9); living[0] = ag
    picks = {sim._jepa_only_vote(ag, torch.randn(LATENT_DIM), living) for _ in range(80)}
    assert len(picks) >= 3


def test_jepa_only_works_without_planner_attr():
    """Mechanism independence: no planner_factorized needed."""
    ag = _jepa_agent()
    assert not hasattr(ag, "planner_factorized")
    living = _living(5); living[0] = ag
    assert sim._jepa_only_vote(ag, torch.randn(LATENT_DIM), living) is not None


# ==================== Heuristic: edge cases + properties ================== #
def test_heuristic_deterministic():
    living = _living(5)
    for a in living:
        a.vote_history = ["Agent_2", "Agent_3", "Agent_2"]
    assert sim._heuristic_vote_choice(living[0], living) == sim._heuristic_vote_choice(living[0], living)


def test_heuristic_name_mention_beats_bandwagon():
    living = _living(5)
    for a in living:
        a.vote_history = ["Agent_3"] * 5              # bandwagon → Agent_3
    living[0].message_memory = [("x", "Agent_1 is sus")]   # name mention → Agent_1 wins
    assert sim._heuristic_vote_choice(living[0], living) == "Agent_1"


def test_heuristic_fallback_is_fixed_rotation():
    living = _living(5)
    # Agent_0 → (0+1)%4 → sorted[Agent_1,Agent_2,Agent_3,Agent_4][1] → Agent_2
    assert sim._heuristic_vote_choice(living[0], living) == "Agent_2"
    # Agent_1 → (1+1)%4 → sorted[Agent_0,Agent_2,Agent_3,Agent_4][2] → Agent_3
    assert sim._heuristic_vote_choice(living[1], living) == "Agent_3"


def test_heuristic_single_other_returns_it():
    living = _living(5, alive=[True, False, True, False, False])
    assert sim._heuristic_vote_choice(living[0], living) == "Agent_2"


def test_heuristic_no_others_returns_none():
    living = _living(5, alive=[True, False, False, False, False])
    assert sim._heuristic_vote_choice(living[0], living) is None


def test_heuristic_vote_history_recency_window():
    """Only the last 9 votes count, so a swamped-but-stale target loses."""
    living = _living(5)
    for a in living:
        a.vote_history = ["Agent_1"] * 100 + ["Agent_3"] * 9   # last 9 = Agent_3
    assert sim._heuristic_vote_choice(living[0], living) == "Agent_3"


# ============================ In-sim integration =========================== #
def test_jepa_only_policy_game_runs(monkeypatch):
    """The jepa_only branch runs end-to-end and produces a valid outcome."""
    monkeypatch.setattr(sim, "_IS_JEPA_ONLY_POLICY", True)
    monkeypatch.setattr(sim, "_IS_RANDOM_POLICY", False)
    monkeypatch.setattr(sim, "_IS_HEURISTIC_POLICY", False)
    _, meta = sim.simulate_game(visual=False, seed=21)
    assert meta["outcome"]["winner"] in ("villagers", "werewolves")
