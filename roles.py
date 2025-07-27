# roles.py
import torch

WEREWOLF = "Werewolf"
VILLAGER = "Worker"

# Role-conditioned latent goal priors (z_goal)
ROLE_PRIORS = {
    WEREWOLF: torch.ones(32),  # Disruptive, exploratory
    VILLAGER: torch.zeros(32),  # Stable, orderly
}

def assign_roles(agent_list, num_werewolves):
    import random
    roles = [WEREWOLF]*num_werewolves + [VILLAGER]*(len(agent_list) - num_werewolves)
    random.shuffle(roles)
    for agent, role in zip(agent_list, roles):
        agent.role = role