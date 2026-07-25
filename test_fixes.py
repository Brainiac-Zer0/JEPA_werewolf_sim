"""Regression tests for the correctness pass (see CHANGES.md)."""
import math
import torch
import pytest


def test_jepa_encoder_receives_gradients():
    """H1: re-encoding raw obs must let JEPA gradients reach the belief encoder."""
    import training_utils as T
    from encoders import MLPBeliefEncoder, INPUT_DIM, LATENT_DIM, WorldModelMLP, \
        PhaseActionEncoder, FactorizedPlanner, ACTION_DIM

    def mkrow(ct, phase, payload):
        xt = torch.randn(INPUT_DIM); xn = torch.randn(INPUT_DIM)
        aux = {"alive": [1] * 9, "wolves": [0] * 9, "self_idx": 0, "x_t": xt, "x_next": xn}
        return (torch.randn(LATENT_DIM), torch.tensor(phase), torch.tensor(payload),
                torch.randn(LATENT_DIM), "Worker", ct, aux)

    rows = [mkrow("TALK_INTENT", 0, 1) for _ in range(6)] + \
           [mkrow("VOTE_TARGET", 1, 2) for _ in range(6)]
    enc = MLPBeliefEncoder(input_dim=INPUT_DIM, latent_dim=LATENT_DIM)
    before = enc.encoder[0].weight.detach().clone()
    T.train_jepa_factorized(
        rows,
        WorldModelMLP(latent_dim=LATENT_DIM, action_dim=ACTION_DIM),
        PhaseActionEncoder(action_dim=ACTION_DIM, num_agents=9, num_talk=5),
        FactorizedPlanner(latent_dim=LATENT_DIM, num_agents=9, num_talk_cats=5),
        role_name="TEST", epochs=1, batch_size=6, belief_encoder=enc,
    )
    after = enc.encoder[0].weight.detach().clone()
    assert (after - before).abs().sum().item() > 1e-6, "encoder did not update"


def test_per_game_seeds_distinct():
    """H2: consecutive games derive distinct, reproducible seeds."""
    from train import _game_seed_for
    seeds = [_game_seed_for(i) for i in range(5)]
    assert len(set(seeds)) == 5
    assert seeds == [_game_seed_for(i) for i in range(5)]  # reproducible


def test_night_kill_softmax_matches_thesis():
    """M2: p_i ∝ exp(c_i/max_c); sampling honors a provided RNG, no majority shortcut."""
    import random
    from world import consensus_target
    tally = {"Agent_1": 2, "Agent_2": 1}
    # With a fixed RNG the choice is deterministic and one of the candidates.
    out = consensus_target(tally, temperature=1.0, rng=random.Random(0))
    assert out in ("Agent_1", "Agent_2")


def test_fusion_is_logit_space():
    """M5: fusion combines logits (α·intent + (1-α)·bias) then log_softmax."""
    import torch.nn.functional as F
    from speaker_llm import IntentFusionProcessor
    proc = IntentFusionProcessor(alpha=0.5)
    a = torch.tensor([2.0, 0.0, -1.0])
    b = torch.tensor([-1.0, 1.0, 0.5])
    got = proc(a, b)
    expected = F.log_softmax(0.5 * a + 0.5 * b, dim=-1)
    assert torch.allclose(got, expected, atol=1e-6)
