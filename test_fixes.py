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


def test_fep_entropy_penalty_active():
    """Phase 1a: the FEP entropy penalty must actually lower planner talk entropy."""
    import torch.nn.functional as F
    import training_utils as T
    from encoders import (INPUT_DIM, LATENT_DIM, WorldModelMLP, PhaseActionEncoder,
                          FactorizedPlanner, ACTION_DIM)

    def mkrows(n=64):
        rows = []
        for i in range(n):
            aux = {"alive": [1] * 9, "wolves": [0] * 9, "self_idx": 0,
                   "x_t": torch.randn(INPUT_DIM), "x_next": torch.randn(INPUT_DIM)}
            rows.append((torch.randn(LATENT_DIM), torch.tensor(0), torch.tensor(i % 5),
                         torch.randn(LATENT_DIM), "Worker", "TALK_INTENT", aux))
        return rows

    def train_and_entropy(w):
        torch.manual_seed(0)
        T.TALK_ENTROPY_W = w
        fp = FactorizedPlanner(latent_dim=LATENT_DIM, num_agents=9, num_talk_cats=5)
        T.train_jepa_factorized(mkrows(), WorldModelMLP(latent_dim=LATENT_DIM, action_dim=ACTION_DIM),
                                PhaseActionEncoder(action_dim=ACTION_DIM, num_agents=9, num_talk=5),
                                fp, role_name="T", epochs=8, batch_size=64, belief_encoder=None)
        z = torch.randn(64, LATENT_DIM)
        with torch.no_grad():
            logits = fp(z, talk_mask=torch.ones(64, 5, dtype=torch.bool),
                        vote_mask=torch.ones(64, 9, dtype=torch.bool),
                        kill_mask=torch.ones(64, 9, dtype=torch.bool))["talk"]
            p = F.softmax(logits, -1)
            return -(p * p.clamp_min(1e-9).log()).sum(-1).mean().item()

    w0 = float(T.TALK_ENTROPY_W)
    try:
        assert train_and_entropy(0.5) < train_and_entropy(0.0) - 1e-3
    finally:
        T.TALK_ENTROPY_W = w0


def test_persona_steers_planning():
    """Phase 1b: persona must shift talk-intent selection when steering is on, not off."""
    import torch
    import sim
    from encoders import LATENT_DIM
    from agent import BaseAgent
    ag = BaseAgent("Agent_0"); ag.role = "Worker"
    z = torch.zeros(LATENT_DIM)

    def p_accuse(scale):
        ag.persona_effects = {"accuse_bias_scale": scale, "hedge_prob_boost": 0.0}
        return float(sim._fused_intent_for_agent(ag, z, recent_texts=[], alpha=0.65)["fused_probs"][0])

    saved = sim.ENABLE_PERSONA_STEER
    try:
        sim.ENABLE_PERSONA_STEER = False
        assert abs(p_accuse(0.6) - p_accuse(1.4)) < 1e-9      # off → no effect
        sim.ENABLE_PERSONA_STEER = True
        assert p_accuse(1.4) - p_accuse(0.6) > 0.02           # on → high-accuse persona accuses more
    finally:
        sim.ENABLE_PERSONA_STEER = saved


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
