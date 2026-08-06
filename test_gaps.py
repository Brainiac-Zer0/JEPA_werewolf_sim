"""
Gap-closing tests identified during the suite review: win-condition boundaries,
the train->eval "trained weights are actually loaded" invariant (the original
headline bug), social non-inertness, rollout-aux plumbing, and an exact
night-kill softmax distribution check.
"""
import math
import random
import types
import torch
import pytest

import sim
import training_utils as T
from encoders import (LATENT_DIM, INPUT_DIM, ACTION_DIM, WorldModelMLP,
                      PhaseActionEncoder, FactorizedPlanner)

W = sim.WEREWOLF
V = "Worker"


def _mk(role, alive=True):
    return types.SimpleNamespace(role=role, alive=alive)


# ============================ Win conditions =============================== #
def test_game_over_villagers_win_when_no_wolves():
    ags = [_mk(W, False), _mk(W, False)] + [_mk(V) for _ in range(5)]
    assert sim._game_over(ags) == "villagers"


def test_game_over_wolves_win_at_parity():
    ags = [_mk(W), _mk(W), _mk(V), _mk(V)]           # 2 wolves, 2 villagers → parity
    assert sim._game_over(ags) == "werewolves"


def test_game_over_continues_when_wolves_outnumbered():
    ags = [_mk(W), _mk(W)] + [_mk(V) for _ in range(5)]   # 2 vs 5
    assert sim._game_over(ags) is None


def test_game_over_one_v_one_is_wolf_win():
    assert sim._game_over([_mk(W), _mk(V)]) == "werewolves"


def test_game_over_both_wolves_die_same_round():
    ags = [_mk(W, False), _mk(W, False), _mk(V), _mk(V), _mk(V)]
    assert sim._game_over(ags) == "villagers"


# =================== Train -> eval: trained weights loaded ================= #
def _worker_vote_rows(n=64):
    rows = []
    for i in range(n):
        aux = {"alive": [1] * 9, "wolves": [0] * 7 + [1, 1], "self_idx": i % 7,
               "x_t": torch.randn(INPUT_DIM), "x_next": torch.randn(INPUT_DIM)}
        rows.append((torch.randn(LATENT_DIM), torch.tensor(1), torch.tensor(i % 7),
                     torch.randn(LATENT_DIM), "Worker", "VOTE_TARGET", aux))
    return rows


def test_sim_loads_trained_planner_not_fresh(tmp_path, monkeypatch):
    """After training, a Worker agent in the sim must carry the TRAINED factorized
    planner weights (matching the checkpoint) — the original bug loaded nothing."""
    monkeypatch.setattr(T, "CHECKPOINT_DIR", str(tmp_path))
    fp = FactorizedPlanner(latent_dim=LATENT_DIM, num_agents=9, num_talk_cats=5)
    T.train_jepa_factorized(
        _worker_vote_rows(), WorldModelMLP(latent_dim=LATENT_DIM, action_dim=ACTION_DIM),
        PhaseActionEncoder(action_dim=ACTION_DIM, num_agents=9, num_talk=5),
        fp, role_name="Worker", epochs=2, batch_size=64, belief_encoder=None)
    ckpt = torch.load(str(tmp_path / "worker_jepa_factorized.pt"), map_location="cpu")
    key = next(k for k in ckpt["factorized_planner"] if "vote" in k and "weight" in k)
    trained_w = ckpt["factorized_planner"][key]

    _, meta = sim.simulate_game(visual=False, seed=3)
    worker = next(a for a in meta["agents"] if a.role != W)
    loaded_w = worker.planner_factorized.state_dict()[key]
    assert torch.allclose(loaded_w, trained_w), "sim did not load the trained planner"


# ============================ Social non-inertness ======================== #
def test_social_delta_can_flip_planner_vote():
    """A scaled social correction must be *capable* of changing the vote argmax
    (guards against the inert-δ regression)."""
    from social import SocialInfluence
    soc = SocialInfluence(latent_dim=LATENT_DIM); soc.scale = 0.3
    fp = FactorizedPlanner(latent_dim=LATENT_DIM, num_agents=9, num_talk_cats=5)
    torch.manual_seed(0)
    m = torch.ones(1, 9, dtype=torch.bool)
    flips = 0
    for _ in range(200):
        z = torch.randn(LATENT_DIM) * 6.0
        mu = torch.randn(LATENT_DIM)
        with torch.no_grad():
            d = soc.delta_from_inputs(z.unsqueeze(0), mu.unsqueeze(0), None).squeeze(0)
            a0 = int(fp.vote(z.unsqueeze(0), mask=m).argmax())
            a1 = int(fp.vote((z + d).unsqueeze(0), mask=m).argmax())
        flips += (a0 != a1)
    assert flips > 0


# ============================ Rollout aux plumbing ======================== #
def test_rollout_aux_carries_training_inputs():
    """Vote rollouts must carry the raw obs (x_t/x_next) and neighbor mean that
    feed encoder + social training."""
    rollout, _ = sim.simulate_game(visual=False, seed=8)
    vote_rows = [r for r in rollout if len(r) >= 7 and r[5] == "VOTE_TARGET"]
    assert vote_rows
    aux0 = vote_rows[0][6]
    assert isinstance(aux0.get("x_t"), torch.Tensor)
    assert isinstance(aux0.get("x_next"), torch.Tensor)
    assert any(isinstance((r[6] or {}).get("z_neigh_mean"), torch.Tensor) for r in vote_rows)


# ============================ Night-kill distribution ===================== #
def test_consensus_softmax_proportions_match_formula():
    """p_i ∝ exp(c_i/max_c): for A=2,B=1 the closed-form P(A)=e/(e+√e)≈0.622."""
    from world import consensus_target
    picks = [consensus_target({"Agent_1": 2, "Agent_2": 1}, temperature=1.0,
                              rng=random.Random(s)) for s in range(1000)]
    p_a = picks.count("Agent_1") / 1000.0
    expected = math.e / (math.e + math.sqrt(math.e))
    assert abs(p_a - expected) < 0.06, f"P(A)={p_a:.3f} expected≈{expected:.3f}"
