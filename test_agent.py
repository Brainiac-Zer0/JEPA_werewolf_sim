import torch
from agent import BaseAgent
from roles import assign_roles


def _make_agents(n=3):
    agents = [BaseAgent(f"Agent_{i}") for i in range(n)]
    assign_roles(agents, 1, seed=1337)
    return agents


def test_encode_belief_shape():
    agents = _make_agents(3)
    a = agents[0]
    z = a.encode_current_belief(round_num=1, agents=agents)
    assert torch.is_tensor(z)
    assert z.dim() == 1 and z.numel() > 0
    assert torch.isfinite(z).all()


def test_encode_stashes_raw_obs():
    # The trainer relies on _last_obs_x being captured for JEPA encoder training.
    agents = _make_agents(3)
    a = agents[0]
    a.encode_current_belief(round_num=1, agents=agents)
    assert hasattr(a, "_last_obs_x")
    assert torch.is_tensor(a._last_obs_x)


def test_observe_excludes_self_and_dead():
    agents = _make_agents(3)
    agents[1].alive = False
    observed = agents[0].observe(agents)
    names = {n for (n, _m) in observed}
    assert agents[0].name not in names
    assert agents[1].name not in names
