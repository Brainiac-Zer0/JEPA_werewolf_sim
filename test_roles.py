import pytest
from roles import assign_roles, WEREWOLF, VILLAGER

class Agent:
    _n = 0
    def __init__(self):
        self.role = None
        self.name = f"Agent_{Agent._n}"
        Agent._n += 1

def test_assign_roles_counts():
    agents = [Agent() for _ in range(10)]
    assign_roles(agents, 3, seed=1337)
    roles = [a.role for a in agents]
    assert roles.count(WEREWOLF) == 3, f"Expected 3 {WEREWOLF}, got {roles.count(WEREWOLF)}"
    assert roles.count(VILLAGER) == 7, f"Expected 7 {VILLAGER}, got {roles.count(VILLAGER)}"

def test_assign_roles_sets_is_wolf():
    agents = [Agent() for _ in range(9)]
    assign_roles(agents, 2, seed=42)
    wolves = [a for a in agents if getattr(a, "is_wolf", False)]
    assert len(wolves) == 2
    assert all(a.role == WEREWOLF for a in wolves)

def test_assign_roles_distinct_seeds_differ():
    a1 = [Agent() for _ in range(10)]
    a2 = [Agent() for _ in range(10)]
    assign_roles(a1, 4, seed=1)
    assign_roles(a2, 4, seed=2)
    # Distinct seeds should (almost surely) give a different arrangement.
    assert [a.role for a in a1] != [a.role for a in a2]

def test_assign_roles_reproducible():
    a1 = [Agent() for _ in range(10)]
    a2 = [Agent() for _ in range(10)]
    assign_roles(a1, 4, seed=7)
    assign_roles(a2, 4, seed=7)
    assert [a.role for a in a1] == [a.role for a in a2]

def test_assign_roles_zero_agents():
    agents = []
    assign_roles(agents, 0)
    assert agents == []

def test_assign_roles_zero_wolves():
    agents = [Agent() for _ in range(5)]
    assign_roles(agents, 0, seed=1)
    assert all(a.role == VILLAGER for a in agents)

def test_assign_roles_all_wolves():
    agents = [Agent() for _ in range(5)]
    assign_roles(agents, 5, seed=1)
    assert all(a.role == WEREWOLF for a in agents)
