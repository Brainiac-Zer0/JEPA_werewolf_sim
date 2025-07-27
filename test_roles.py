import pytest
from roles import assign_roles, DEFECTIVE, WORKER

class Agent:
    def __init__(self):
        self.role = None

def test_assign_roles_counts():
    print("Testing role counts...")
    agents = [Agent() for _ in range(10)]
    assign_roles(agents, 3)
    roles = [a.role for a in agents]
    print("Assigned roles:", roles)
    assert roles.count(DEFECTIVE) == 3, f"Expected 3 DEFECTIVE, got {roles.count(DEFECTIVE)}"
    assert roles.count(WORKER) == 7, f"Expected 7 WORKER, got {roles.count(WORKER)}"

def test_assign_roles_randomness():
    print("Testing randomness...")
    agents1 = [Agent() for _ in range(10)]
    agents2 = [Agent() for _ in range(10)]
    assign_roles(agents1, 4)
    assign_roles(agents2, 4)
    roles1 = [a.role for a in agents1]
    roles2 = [a.role for a in agents2]
    print("Roles1:", roles1)
    print("Roles2:", roles2)
    assert roles1 != roles2 or roles1.count(DEFECTIVE) == 4

def test_assign_roles_zero_agents():
    print("Testing zero-agent case...")
    agents = []
    assign_roles(agents, 0)
    print("Agents list remains empty:", agents)
    assert agents == []

def test_assign_roles_zero_defectives():
    print("Testing all WORKER case...")
    agents = [Agent() for _ in range(5)]
    assign_roles(agents, 0)
    roles = [a.role for a in agents]
    print("Assigned roles:", roles)
    assert all(a.role == WORKER for a in agents)

def test_assign_roles_all_defectives():
    print("Testing all DEFECTIVE case...")
    agents = [Agent() for _ in range(5)]
    assign_roles(agents, 5)
    roles = [a.role for a in agents]
    print("Assigned roles:", roles)
    assert all(a.role == DEFECTIVE for a in agents)

if __name__ == "__main__":
    test_assign_roles_counts()
    test_assign_roles_randomness()
    test_assign_roles_zero_agents()
    test_assign_roles_zero_defectives()
    test_assign_roles_all_defectives()
    print("✅ All tests passed.")
