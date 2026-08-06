"""
Core mechanics + pipeline tests (game rules, night-kill, masks, metrics,
train->eval linkage, REINFORCE baseline, judge gate). Complements the
fix/phase regression tests to give broad coverage.
"""
import math
import os
import random
import types
import numpy as np
import torch
import pytest

from encoders import (LATENT_DIM, INPUT_DIM, ACTION_DIM, WorldModelMLP,
                      PhaseActionEncoder, FactorizedPlanner, MLPBeliefEncoder)
import training_utils as T


# ============================ Day vote / world ============================= #
def test_resolve_votes_strict_majority():
    from world import resolve_votes
    assert resolve_votes({"A": "B", "C": "B", "D": "B", "E": "C"}) == "B"


def test_resolve_votes_tie_no_elimination():
    from world import resolve_votes
    assert resolve_votes({"A": "B", "C": "C"}) is None


def test_resolve_votes_empty():
    from world import resolve_votes
    assert resolve_votes({}) is None


def test_eliminate_player_sets_dead():
    from world import eliminate_player
    ag = types.SimpleNamespace(name="Z", alive=True)
    eliminate_player(ag)
    assert ag.alive is False


# ============================ Night-kill softmax =========================== #
def test_consensus_softmax_prefers_higher_count():
    from world import consensus_target
    # p_i ∝ exp(c_i/max_c): A(3) should win far more than B(1).
    picks = [consensus_target({"Agent_1": 3, "Agent_2": 1}, temperature=1.0,
                              rng=random.Random(s)) for s in range(400)]
    a = picks.count("Agent_1")
    assert a > picks.count("Agent_2")
    assert 0.5 < a / 400 < 0.85           # ~0.66 expected, not deterministic


def test_consensus_no_majority_shortcut_by_default():
    from world import consensus_target
    # Even with A at 75% (>50%), default samples (does not deterministically pick A).
    picks = {consensus_target({"Agent_1": 3, "Agent_2": 1}, temperature=1.0,
                              rng=random.Random(s)) for s in range(200)}
    assert picks == {"Agent_1", "Agent_2"}


def test_consensus_majority_shortcut_when_enabled():
    from world import consensus_target
    out = {consensus_target({"Agent_1": 3, "Agent_2": 1}, temperature=1.0,
                            rng=random.Random(s), use_majority_shortcut=True) for s in range(50)}
    assert out == {"Agent_1"}


def test_consensus_zero_tally_safe():
    from world import consensus_target
    assert consensus_target({"Agent_1": 0, "Agent_2": 0}) is not None   # no crash


def test_consensus_rng_reproducible():
    from world import consensus_target
    t = {"Agent_1": 2, "Agent_2": 2, "Agent_3": 1}
    assert consensus_target(t, rng=random.Random(7)) == consensus_target(t, rng=random.Random(7))


# ============================ Legality masks =============================== #
def _rs(aux):
    return types.SimpleNamespace(aux=aux)


def test_vote_mask_excludes_self_and_dead():
    alive = [1] * 9; alive[3] = 0
    batch = [_rs({"alive": alive, "self_idx": 5})]
    m = T.build_vote_mask_from_aux(batch, 9)[0]
    assert not bool(m[5]) and not bool(m[3])      # self + dead masked
    assert bool(m[0]) and bool(m[8])              # others legal


def test_kill_mask_wolf_actor_targets_nonwolves():
    wolves = [0] * 9; wolves[7] = 1; wolves[8] = 1
    batch = [_rs({"alive": [1] * 9, "wolves": wolves, "self_idx": 7})]
    m = T.build_kill_mask_from_aux(batch, 9)[0]
    assert not bool(m[7]) and not bool(m[8])      # cannot kill wolves
    assert bool(m[0])                             # can kill a villager


def test_kill_mask_villager_actor_all_false():
    wolves = [0] * 9; wolves[7] = 1; wolves[8] = 1
    batch = [_rs({"alive": [1] * 9, "wolves": wolves, "self_idx": 0})]
    m = T.build_kill_mask_from_aux(batch, 9)[0]
    assert not m.any()


def test_votehead_masks_with_neg_inf():
    fp = FactorizedPlanner(latent_dim=LATENT_DIM, num_agents=9, num_talk_cats=5)
    mask = torch.ones(1, 9, dtype=torch.bool); mask[0, 2] = False; mask[0, 4] = False
    logits = fp.vote(torch.randn(1, LATENT_DIM), mask=mask).squeeze(0)
    assert torch.isinf(logits[2]) and logits[2] < 0     # -inf, not 0.0
    assert torch.isfinite(logits[0])


# ============================ Metrics ====================================== #
def test_talk_vote_alignment_real():
    import sim
    ag = types.SimpleNamespace(talk_category_last=0, talk_target_last_idx=5)   # accuse Agent_5
    assert sim._talk_vote_align_real(ag, 5) == 1.0                             # voted the accused
    assert sim._talk_vote_align_real(ag, 3) == 0.0                             # voted elsewhere
    ag.talk_category_last = 2                                                  # hedge (undirected)
    assert math.isnan(sim._talk_vote_align_real(ag, 5))                        # excluded


def test_simulate_game_outcome_dict():
    import sim
    _, meta = sim.simulate_game(visual=False, seed=11)
    oc = meta["outcome"]
    for k in ("winner", "villager_win", "vill_vote_accuracy", "judge_accept",
              "talk_vote_align", "rounds", "seed"):
        assert k in oc
    assert oc["winner"] in ("villagers", "werewolves")
    assert oc["villager_win"] == (oc["winner"] == "villagers")


# ============================ Roles / seeding ============================== #
def test_assign_roles_sets_is_wolf_and_counts():
    from roles import assign_roles, WEREWOLF
    agents = [types.SimpleNamespace(name=f"Agent_{i}", role=None) for i in range(9)]
    assign_roles(agents, 2, seed=1337)
    wolves = [a for a in agents if getattr(a, "is_wolf", False)]
    assert len(wolves) == 2 and all(a.role == WEREWOLF for a in wolves)


def test_per_game_seed_distinct_and_reproducible():
    from train import _game_seed_for
    s = [_game_seed_for(i) for i in range(6)]
    assert len(set(s)) == 6 and s == [_game_seed_for(i) for i in range(6)]


# ============================ Train -> eval linkage ======================== #
def test_belief_encoder_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "CHECKPOINT_DIR", str(tmp_path))
    enc = MLPBeliefEncoder(input_dim=INPUT_DIM, latent_dim=LATENT_DIM)
    with torch.no_grad():
        enc.encoder[0].weight.add_(1.0)                     # perturb so it's non-default
    torch.save({"belief_encoder": enc.state_dict(), "input_dim": INPUT_DIM,
                "latent_dim": LATENT_DIM}, os.path.join(str(tmp_path), "belief_encoder.pt"))
    enc2 = T.load_shared_belief_encoder()
    assert torch.allclose(enc.encoder[0].weight, enc2.encoder[0].weight)


def test_factorized_loader_returns_modules():
    wm, pae, fp = T.load_role_models_factorized("Worker")
    assert isinstance(wm, WorldModelMLP) and isinstance(fp, FactorizedPlanner)


def test_sim_shares_encoder_and_social_across_agents():
    import sim
    _, meta = sim.simulate_game(visual=False, seed=12)
    agents = meta["agents"]
    assert all(a.encoder is agents[0].encoder for a in agents)     # shared belief encoder
    socials = [getattr(a, "social", None) for a in agents]
    if socials[0] is not None:
        assert all(s is socials[0] for s in socials)               # shared social module


# ============================ REINFORCE baseline (M1) ====================== #
def test_speaker_bandit_ema_baseline_updates():
    from speaker import SpeakerBandit
    sb = SpeakerBandit(latent_dim=LATENT_DIM, num_templates=5)
    opt = torch.optim.SGD(sb.parameters(), lr=0.0) if list(sb.parameters()) else None
    batch = [{"z": torch.randn(LATENT_DIM), "role_bit": torch.zeros(1),
              "hist_feats": torch.zeros(3), "template_id": i % 5, "reward": float(i)}
             for i in range(8)]
    # first call builds the net; ensure an optimizer over real params
    sb.forward(torch.randn(1, LATENT_DIM), torch.zeros(1, 1), torch.zeros(1, 3))
    opt = torch.optim.SGD(sb.parameters(), lr=1e-3)
    assert sb._baseline_initialized is False
    sb.learn_step(batch, opt, baseline=None)
    assert sb._baseline_initialized is True
    b1 = sb.reward_baseline
    sb.learn_step(batch, opt, baseline=None)
    assert sb.reward_baseline != 0.0 and abs(sb.reward_baseline - b1) < abs(b1) + 10


# ============================ Offline judge gate =========================== #
def test_can_use_judge_provider_aware(monkeypatch):
    import sim
    monkeypatch.setattr(sim, "JUDGE_ENABLED", True)
    monkeypatch.setattr(sim, "JUDGE_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert sim._can_use_judge() is False                 # openai needs a key
    monkeypatch.setattr(sim, "JUDGE_PROVIDER", "hf")
    assert sim._can_use_judge() is True                  # local provider: no key needed
    monkeypatch.setattr(sim, "JUDGE_ENABLED", False)
    assert sim._can_use_judge() is False


# ============================ Ladder aggregation =========================== #
def test_bootstrap_ci_bounds_and_nan_filtering():
    from run_baseline_ladder import _bootstrap_ci
    m, lo, hi = _bootstrap_ci([0.0, 1.0, 1.0, 0.0, 1.0, float("nan")])
    assert lo <= m <= hi
    assert 0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0


def test_ladder_baseline_toggle_mapping():
    from run_baseline_ladder import BASELINES, LADDER_ORDER
    assert len(LADDER_ORDER) == 7
    assert BASELINES["B6_random"]["AGENTSIM_POLICY"] == "random_voting"
    assert BASELINES["B0_full"]["USE_LANGUAGE"] == "1"
    assert BASELINES["B2_jepa_planner"]["SOCIAL_ENABLED"] == "0"
    assert BASELINES["B1_planner_social"]["SOCIAL_ENABLED"] == "1"
