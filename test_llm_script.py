# test_llm_script.py
from llm_script import chatgpt_llm_from_latent
import torch

class DummyAgent:
    def __init__(self, name, role):
        self.name = name
        self.role = role

def test_llm_responds():
    agent = DummyAgent("Alice", "Villager")
    z = torch.randn(32)

    message = chatgpt_llm_from_latent(z, agent)
    print(f"[LLM OUTPUT] {agent.name} says: {message}")

    assert isinstance(message, str)
    assert len(message) > 0
    assert "```" not in message and "{" not in message  # should not return code or JSON

def test_multiple_roles():
    roles = ["Villager", "Werewolf", "Seer"]
    for role in roles:
        agent = DummyAgent(name=role, role=role)
        z = torch.randn(32)
        msg = chatgpt_llm_from_latent(z, agent)
        print(f"[{role}] said: {msg}")

if __name__ == "__main__":
    print("[TEST] Running LLM messaging tests...")
    test_llm_responds()
    test_multiple_roles()
    print("[TEST] All tests completed.")
