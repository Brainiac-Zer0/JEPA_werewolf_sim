from world import resolve_votes, eliminate_player

class DummyAgent:
    def __init__(self, name):
        self.name = name
        self.alive = True

def test_resolve_votes_majority():
    print("Test: resolve_votes with clear majority")
    votes = {"A": "B", "C": "B", "D": "B", "E": "C"}
    result = resolve_votes(votes)
    print("Votes:", votes)
    print("Expected: B, Got:", result)
    assert result == "B"

def test_resolve_votes_tie():
    print("Test: resolve_votes with tie")
    votes = {"A": "B", "C": "C"}
    result = resolve_votes(votes)
    print("Votes:", votes)
    print("Expected: None, Got:", result)
    assert result is None

def test_resolve_votes_empty():
    print("Test: resolve_votes with empty vote dict")
    votes = {}
    result = resolve_votes(votes)
    print("Expected: None, Got:", result)
    assert result is None

def test_eliminate_player():
    print("Test: eliminate_player sets alive to False")
    agent = DummyAgent("Z")
    eliminate_player(agent)
    print(f"Agent {agent.name} alive: {agent.alive}")
    assert not agent.alive

if __name__ == "__main__":
    test_resolve_votes_majority()
    test_resolve_votes_tie()
    test_resolve_votes_empty()
    test_eliminate_player()
    print("✅ All world.py tests passed.")
