"""
Tests for Phase 2: social-influence module is trained (wolf-supervised),
message-aware, relatively scaled, checkpointed/loaded, and its correction both
changes and improves villager voting.
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
import pytest

from encoders import (INPUT_DIM, LATENT_DIM, ACTION_DIM, WorldModelMLP,
                      PhaseActionEncoder, FactorizedPlanner)
from social import SocialInfluence
import training_utils as T

WOLVES = [0] * 7 + [1, 1]        # agents 7,8 are wolves


def _fresh_social():
    """A fresh module so tests don't depend on ambient checkpoints/social.pt."""
    return SocialInfluence(latent_dim=LATENT_DIM)


def _villager_vote_rows(n=96):
    """Villager VOTE rows with wolf labels + a neighbor mean, for wolf-supervision."""
    rows = []
    for i in range(n):
        aux = {"alive": [1] * 9, "wolves": list(WOLVES), "self_idx": i % 7,  # self is a villager
               "x_t": torch.randn(INPUT_DIM), "x_next": torch.randn(INPUT_DIM),
               "z_neigh_mean": torch.randn(LATENT_DIM)}
        rows.append((torch.randn(LATENT_DIM) * 6.0, torch.tensor(1), torch.tensor(i % 7),
                     torch.randn(LATENT_DIM), "Worker", "VOTE_TARGET", aux))
    return rows


def test_delta_from_inputs_shapes_and_finite():
    soc = _fresh_social()
    z = torch.randn(4, LATENT_DIM); mu = torch.randn(4, LATENT_DIM)
    d = soc.delta_from_inputs(z, mu, None)
    assert d.shape == (4, LATENT_DIM) and torch.isfinite(d).all()


def test_delta_relative_to_z_norm():
    """||delta|| should be ~ scale * ||z_self|| (robust to encoder norm)."""
    soc = _fresh_social(); soc.scale = 0.15
    for zscale in (5.0, 30.0):
        z = torch.randn(1, LATENT_DIM); z = z / z.norm() * zscale
        mu = torch.randn(1, LATENT_DIM)
        d = soc.delta_from_inputs(z, mu, None)
        assert abs(float(d.norm()) - 0.15 * zscale) < 0.15 * zscale * 0.2  # within 20%


def test_message_pathway_changes_delta():
    """The message-content projection (thesis §3.9) affects δ."""
    soc = _fresh_social()
    assert soc.msg_proj is not None
    z = torch.randn(2, LATENT_DIM); mu = torch.randn(2, LATENT_DIM)
    msg = torch.randn(2, soc.msg_dim)
    d_no = soc.delta_from_inputs(z, mu, None)
    d_msg = soc.delta_from_inputs(z, mu, msg)
    assert not torch.allclose(d_no, d_msg, atol=1e-5)


def test_social_params_train():
    soc = _fresh_social()
    before = [p.detach().clone() for p in soc.parameters()]
    T.train_jepa_factorized(
        _villager_vote_rows(),
        WorldModelMLP(latent_dim=LATENT_DIM, action_dim=ACTION_DIM),
        PhaseActionEncoder(action_dim=ACTION_DIM, num_agents=9, num_talk=5),
        FactorizedPlanner(latent_dim=LATENT_DIM, num_agents=9, num_talk_cats=5),
        role_name="T", epochs=3, batch_size=64, belief_encoder=None, social_module=soc)
    after = [p.detach().clone() for p in soc.parameters()]
    assert sum((a - b).abs().sum().item() for a, b in zip(after, before)) > 1e-6


def test_social_checkpoint_written(tmp_path, monkeypatch):
    soc = _fresh_social()
    monkeypatch.setattr(T, "CHECKPOINT_DIR", str(tmp_path))
    T.train_jepa_factorized(
        _villager_vote_rows(32),
        WorldModelMLP(latent_dim=LATENT_DIM, action_dim=ACTION_DIM),
        PhaseActionEncoder(action_dim=ACTION_DIM, num_agents=9, num_talk=5),
        FactorizedPlanner(latent_dim=LATENT_DIM, num_agents=9, num_talk_cats=5),
        role_name="T", epochs=1, batch_size=32, belief_encoder=None, social_module=soc)
    assert os.path.exists(os.path.join(str(tmp_path), "social.pt"))


def test_wolf_supervision_shifts_votes_toward_wolves():
    """After wolf-supervised training, adding δ moves the villager vote toward a
    wolf more often than the base planner alone."""
    torch.manual_seed(0)
    soc = _fresh_social(); soc.scale = 0.2
    fp = FactorizedPlanner(latent_dim=LATENT_DIM, num_agents=9, num_talk_cats=5)
    rows = _villager_vote_rows(160)
    T.train_jepa_factorized(
        rows, WorldModelMLP(latent_dim=LATENT_DIM, action_dim=ACTION_DIM),
        PhaseActionEncoder(action_dim=ACTION_DIM, num_agents=9, num_talk=5),
        fp, role_name="T", epochs=6, batch_size=64, belief_encoder=None, social_module=soc)
    fp.eval()
    m = torch.ones(1, 9, dtype=torch.bool)
    base_wolf = soc_wolf = 0
    N = 300
    for _ in range(N):
        z = torch.randn(LATENT_DIM) * 6.0
        mu = torch.randn(LATENT_DIM)
        with torch.no_grad():
            d = soc.delta_from_inputs(z.unsqueeze(0), mu.unsqueeze(0), None).squeeze(0)
            a0 = int(fp.vote(z.unsqueeze(0), mask=m).argmax())
            a1 = int(fp.vote((z + d).unsqueeze(0), mask=m).argmax())
        base_wolf += WOLVES[a0]
        soc_wolf += WOLVES[a1]
    # Social correction should raise the wolf-vote rate.
    assert soc_wolf > base_wolf, f"base_wolf={base_wolf} soc_wolf={soc_wolf}"
