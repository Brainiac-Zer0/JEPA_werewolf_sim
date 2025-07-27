from collections import Counter

def resolve_votes(vote_dict):
    counts = Counter(vote_dict.values())
    if not counts:
        return None
    top = counts.most_common()
    if len(top) == 1 or top[0][1] > top[1][1]:
        return top[0][0]
    return None  # Tie

def eliminate_player(agent):
    agent.alive = False

def night_kill(agents):
    """Werewolf selects a target to eliminate during night phase."""
    werewolves = [a for a in agents if a.alive and getattr(a, 'role', None) == "Werewolf"]
    villagers = [a for a in agents if a.alive and getattr(a, 'role', None) != "Werewolf"]

    if not werewolves or not villagers:
        return None  # No kill possible

    # Pick first werewolf to choose (could expand to consensus voting later)
    killer = werewolves[0]
    target_name = killer.choose_night_target(villagers)
    if target_name:
        for agent in agents:
            if agent.name == target_name:
                eliminate_player(agent)
                return target_name
    return None
