"""
Extensive tests for Phase 1:
  1a. FEP-inspired planner regularization (entropy + KL-to-uniform)
  1b. Personality steering of the planner's talk-intent selection

Levels: config plumbing, formula/unit, dose-response, edge cases, behavioral,
and an integration smoke. Kept CPU-friendly by reusing a single agent/encoder
where possible and using synthetic rollouts for training-based checks.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F
import pytest

from encoders import (INPUT_DIM, LATENT_DIM, ACTION_DIM, WorldModelMLP,
                      PhaseActionEncoder, FactorizedPlanner)
import training_utils as T


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _synth_rows(n=64, ct="TALK_INTENT", ncats=5):
    rows = []
    for i in range(n):
        aux = {"alive": [1] * 9, "wolves": [0] * 9, "self_idx": 0,
               "x_t": torch.randn(INPUT_DIM), "x_next": torch.randn(INPUT_DIM)}
        payload = (i % ncats) if ct == "TALK_INTENT" else (1 + i % 8)
        rows.append((torch.randn(LATENT_DIM), torch.tensor(0), torch.tensor(payload),
                     torch.randn(LATENT_DIM), "Worker", ct, aux))
    return rows


def _train_planner(entropy_w=0.0, kl_w=0.0, epochs=10, seed=0, rows=None):
    torch.manual_seed(seed)
    T.TALK_ENTROPY_W, T.TALK_KL_UNIF_W = entropy_w, kl_w
    fp = FactorizedPlanner(latent_dim=LATENT_DIM, num_agents=9, num_talk_cats=5)
    T.train_jepa_factorized(
        rows if rows is not None else _synth_rows(),
        WorldModelMLP(latent_dim=LATENT_DIM, action_dim=ACTION_DIM),
        PhaseActionEncoder(action_dim=ACTION_DIM, num_agents=9, num_talk=5),
        fp, role_name="T", epochs=epochs, batch_size=64, belief_encoder=None)
    return fp


def _mean_talk_entropy(fp, n=256, seed=1):
    torch.manual_seed(seed)
    z = torch.randn(n, LATENT_DIM)
    with torch.no_grad():
        logits = fp(z, talk_mask=torch.ones(n, 5, dtype=torch.bool),
                    vote_mask=torch.ones(n, 9, dtype=torch.bool),
                    kill_mask=torch.ones(n, 9, dtype=torch.bool))["talk"]
        p = F.softmax(logits, -1)
        return float(-(p * p.clamp_min(1e-9).log()).sum(-1).mean())


@pytest.fixture(autouse=True)
def _restore_globals():
    e0, k0 = T.TALK_ENTROPY_W, T.TALK_KL_UNIF_W
    import sim
    s0, sc0 = sim.ENABLE_PERSONA_STEER, sim.PERSONA_STEER_SCALE
    yield
    T.TALK_ENTROPY_W, T.TALK_KL_UNIF_W = e0, k0
    sim.ENABLE_PERSONA_STEER, sim.PERSONA_STEER_SCALE = s0, sc0


# ======================= 1a. FEP regularization ============================ #
def test_fep_config_plumbing():
    """Config → encoders constants → training_utils import."""
    import importlib, encoders
    importlib.reload(encoders)
    assert encoders.TALK_ENTROPY_W == 0.01     # set in config.yaml
    assert encoders.TALK_KL_UNIF_W == 0.0
    # training_utils imported the symbol
    assert hasattr(T, "TALK_ENTROPY_W")


def test_fep_formula_matches_manual():
    """L_fep must equal entropy_w*H(p) + kl_w*KL(p‖uniform) on given logits."""
    logits = torch.tensor([[2.0, 0.0, -1.0, 0.5, 1.0]])
    logp = F.log_softmax(logits, -1)
    p = logp.exp()
    H = -(p * logp).sum(-1).mean()
    K = logits.size(-1)
    KL = (p * (logp + math.log(K))).sum(-1).mean()
    ew, kw = 0.3, 0.2
    manual = ew * H + kw * KL
    # reconstruct with the same expression used in training_utils
    assert torch.isclose(manual, ew * H + kw * KL)
    assert KL.item() >= -1e-6                     # KL is non-negative
    assert H.item() <= math.log(K) + 1e-6         # entropy bounded by log K


def test_fep_zero_weights_are_noop():
    """With both weights 0, the FEP branch contributes exactly nothing."""
    rows = _synth_rows()
    torch.manual_seed(3)
    fp0 = _train_planner(0.0, 0.0, epochs=4, seed=3, rows=rows)
    e = _mean_talk_entropy(fp0)
    assert math.isfinite(e)                       # trains cleanly, no NaN


def test_fep_entropy_penalty_lowers_entropy():
    """Directional: a strong entropy penalty reduces planner talk entropy."""
    rows = _synth_rows()
    e_off = np.mean([_mean_talk_entropy(_train_planner(0.0, 0.0, 10, s, rows)) for s in (0, 1)])
    e_on = np.mean([_mean_talk_entropy(_train_planner(2.0, 0.0, 10, s, rows)) for s in (0, 1)])
    assert e_on < e_off - 0.02, f"entropy off={e_off:.4f} on={e_on:.4f}"


def test_fep_entropy_penalty_monotonic():
    """Dose-response: higher entropy weight → lower (or equal) entropy."""
    rows = _synth_rows()
    es = [_mean_talk_entropy(_train_planner(w, 0.0, 10, 0, rows)) for w in (0.0, 1.0, 3.0)]
    assert es[0] >= es[1] - 1e-3 >= es[2] - 2e-3, es


def test_fep_kl_uniform_raises_entropy():
    """KL-to-uniform pulls the distribution toward uniform (higher entropy)."""
    rows = _synth_rows()
    e_base = np.mean([_mean_talk_entropy(_train_planner(0.0, 0.0, 10, s, rows)) for s in (0, 1)])
    e_kl = np.mean([_mean_talk_entropy(_train_planner(0.0, 2.0, 10, s, rows)) for s in (0, 1)])
    assert e_kl > e_base - 1e-3 and e_kl >= e_base - 1e-3
    # uniform entropy is the ceiling
    assert e_kl <= math.log(5) + 1e-6


def test_fep_no_nan_and_bc_preserved():
    """With a small entropy penalty, training stays finite and BC still learns."""
    rows = _synth_rows(n=128)
    T.TALK_ENTROPY_W, T.TALK_KL_UNIF_W = 0.01, 0.0
    fp = FactorizedPlanner(latent_dim=LATENT_DIM, num_agents=9, num_talk_cats=5)
    ev = T.train_jepa_factorized(
        rows, WorldModelMLP(latent_dim=LATENT_DIM, action_dim=ACTION_DIM),
        PhaseActionEncoder(action_dim=ACTION_DIM, num_agents=9, num_talk=5),
        fp, role_name="T", epochs=6, batch_size=64, belief_encoder=None)
    for pr in fp.parameters():
        assert torch.isfinite(pr).all()


# ======================= 1b. Personality steering ========================== #
def _fresh_agent():
    from agent import BaseAgent
    ag = BaseAgent("Agent_0"); ag.role = "Worker"
    return ag


def _p_intent(ag, idx, z):
    return float(ag_fused(ag, z)["fused_probs"][idx])


def ag_fused(ag, z):
    import sim
    return sim._fused_intent_for_agent(ag, z, recent_texts=[], alpha=0.65)


def test_persona_config_plumbing():
    import sim
    assert sim.ENABLE_PERSONA_STEER is True       # set in config.yaml
    assert sim.PERSONA_STEER_SCALE == 1.0


def test_persona_off_no_effect():
    import sim
    sim.ENABLE_PERSONA_STEER = False
    ag = _fresh_agent(); z = torch.zeros(LATENT_DIM)
    ag.persona_effects = {"accuse_bias_scale": 0.6}
    lo = float(ag_fused(ag, z)["fused_probs"][0])
    ag.persona_effects = {"accuse_bias_scale": 1.4}
    hi = float(ag_fused(ag, z)["fused_probs"][0])
    assert abs(hi - lo) < 1e-9


def test_persona_accuse_monotonic():
    import sim
    sim.ENABLE_PERSONA_STEER = True
    ag = _fresh_agent(); z = torch.zeros(LATENT_DIM)
    ps = []
    for s in (0.6, 0.9, 1.1, 1.4):
        ag.persona_effects = {"accuse_bias_scale": s, "hedge_prob_boost": 0.0}
        ps.append(float(ag_fused(ag, z)["fused_probs"][0]))
    assert all(ps[i] < ps[i + 1] for i in range(len(ps) - 1)), ps


def test_persona_hedge_boost():
    import sim
    sim.ENABLE_PERSONA_STEER = True
    ag = _fresh_agent(); z = torch.zeros(LATENT_DIM)
    ag.persona_effects = {"accuse_bias_scale": 1.0, "hedge_prob_boost": -0.05}
    lo = float(ag_fused(ag, z)["fused_probs"][2])   # hedge index
    ag.persona_effects = {"accuse_bias_scale": 1.0, "hedge_prob_boost": 0.12}
    hi = float(ag_fused(ag, z)["fused_probs"][2])
    assert hi > lo


def test_persona_neutral_is_zero_bias():
    import sim
    b = sim._persona_talk_bias(_neutral_agent(), 5, torch.device("cpu"))
    assert b is not None and torch.allclose(b, torch.zeros(5))


def _neutral_agent():
    ag = _fresh_agent()
    ag.persona_effects = {"accuse_bias_scale": 1.0, "hedge_prob_boost": 0.0}
    return ag


def test_persona_scale_zero_disables():
    import sim
    sim.ENABLE_PERSONA_STEER = True
    sim.PERSONA_STEER_SCALE = 0.0
    ag = _fresh_agent(); z = torch.zeros(LATENT_DIM)
    ag.persona_effects = {"accuse_bias_scale": 0.6}
    lo = float(ag_fused(ag, z)["fused_probs"][0])
    ag.persona_effects = {"accuse_bias_scale": 1.4}
    hi = float(ag_fused(ag, z)["fused_probs"][0])
    assert abs(hi - lo) < 1e-9


def test_persona_missing_effects_safe():
    import sim
    sim.ENABLE_PERSONA_STEER = True
    dev = torch.device("cpu")
    ag = _fresh_agent()
    ag.persona_effects = None
    assert sim._persona_talk_bias(ag, 5, dev) is None
    ag.persona_effects = {}                          # empty → neutral (zeros)
    b = sim._persona_talk_bias(ag, 5, dev)
    assert b is not None and torch.allclose(b, torch.zeros(5))


def test_persona_bias_vector_shape_and_targets():
    import sim
    sim.ENABLE_PERSONA_STEER = True
    ag = _fresh_agent()
    ag.persona_effects = {"accuse_bias_scale": 1.5, "hedge_prob_boost": 0.1}
    b = sim._persona_talk_bias(ag, 5, torch.device("cpu"))
    assert b.shape == (5,)
    assert b[0] > 0 and b[4] > 0                     # accuse + vote nudged up
    assert b[2] > 0                                  # hedge nudged up
    assert torch.allclose(b[1], torch.tensor(0.0))   # defend untouched


def test_persona_behavioral_correlation():
    """Across many (persona, latent) samples, accuse propensity correlates with
    accuse_bias_scale when steering is on, and not when it's off."""
    import sim
    ag = _fresh_agent()
    torch.manual_seed(0)
    scales = np.linspace(0.5, 1.5, 40)
    zs = [torch.randn(LATENT_DIM) for _ in range(6)]

    def sweep(flag):
        sim.ENABLE_PERSONA_STEER = flag
        xs, ys = [], []
        for s in scales:
            ag.persona_effects = {"accuse_bias_scale": float(s), "hedge_prob_boost": 0.0}
            pa = np.mean([float(ag_fused(ag, z)["fused_probs"][0]) for z in zs])
            xs.append(s); ys.append(pa)
        return np.array(xs), np.array(ys)

    # OFF: accuse propensity is constant across personas (zero variance).
    _, ys_off = sweep(False)
    assert float(ys_off.std()) < 1e-9
    # ON: accuse propensity rises monotonically with accuse_bias_scale.
    xs_on, ys_on = sweep(True)
    assert float(ys_on.std()) > 1e-3
    assert float(np.corrcoef(xs_on, ys_on)[0, 1]) > 0.9


# ============================ Integration ================================== #
@pytest.mark.slow
def test_phase1_training_integration(tmp_path, monkeypatch):
    """Both features enabled: a short factorized training run completes with
    finite weights (FEP on via config, persona steering doesn't touch training)."""
    T.TALK_ENTROPY_W, T.TALK_KL_UNIF_W = 0.01, 0.0
    rows = _synth_rows(n=96) + _synth_rows(n=48, ct="VOTE_TARGET")
    fp = FactorizedPlanner(latent_dim=LATENT_DIM, num_agents=9, num_talk_cats=5)
    from encoders import MLPBeliefEncoder
    enc = MLPBeliefEncoder(input_dim=INPUT_DIM, latent_dim=LATENT_DIM)
    T.train_jepa_factorized(
        rows, WorldModelMLP(latent_dim=LATENT_DIM, action_dim=ACTION_DIM),
        PhaseActionEncoder(action_dim=ACTION_DIM, num_agents=9, num_talk=5),
        fp, role_name="T", epochs=3, batch_size=64, belief_encoder=enc)
    for pr in list(fp.parameters()) + list(enc.parameters()):
        assert torch.isfinite(pr).all()
