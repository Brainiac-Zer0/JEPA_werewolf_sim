from agent import BaseAgent

def test_agent_lifecycle():
    alice = BaseAgent("Alice")
    bob = BaseAgent("Bob")
    charlie = BaseAgent("Charlie")

    agents = [alice, bob, charlie]

    # Assign dummy roles for now
    alice.role = "Werewolf"
    bob.role = "Villager"
    charlie.role = "Villager"

    # Mark Bob as dead
    bob.alive = False

    # Assign dummy last messages
    alice.last_message = "I’m innocent."
    bob.last_message = "Trust me."
    charlie.last_message = "Lynch the werewolf."

    # Observe
    print("=== Alice's Observation ===")
    print(alice.observe(agents))  # Should not include Alice or dead Bob

    # Vote
    print("=== Charlie Votes ===")
    print("Charlie votes to eliminate:", charlie.vote(agents))  # Should not vote for self

    # Night target
    print("=== Alice (Werewolf) Chooses Night Target ===")
    print("Target:", alice.choose_night_target(agents))  # Should not pick self or Bob (dead)

    # Encode dummy beliefs
    print("=== Encoded belief vector ===")
    print(alice.encode_current_belief())  # Placeholder vector


if __name__ == "__main__":
    test_agent_lifecycle()
