import pygame
import sys
import random
import torch
from agent import BaseAgent
from roles import WEREWOLF, VILLAGER, assign_roles
from world import resolve_votes, eliminate_player
from llm_script import chatgpt_llm_from_latent
from train import train_jepa
from encoders import WorldModelMLP, ActionEncoder

# --- Simulation Parameters ---
NUM_AGENTS = 6
NUM_WEREWOLVES = 1
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 600
FPS = 1
TRAIN_INTERVAL = 3

# --- Pygame Setup ---
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("JEPA Werewolf Sim")
font = pygame.font.SysFont(None, 32)
font_small = pygame.font.SysFont(None, 24)
clock = pygame.time.Clock()

# --- Agent Visuals ---
AGENT_RADIUS = 30
MESSAGE_LOG_LIMIT = 12
MESSAGE_BOX_WIDTH = 360
MESSAGE_BOX_X = SCREEN_WIDTH - MESSAGE_BOX_WIDTH
MESSAGE_BOX_Y_BOTTOM = SCREEN_HEIGHT - 40

# Store messages globally
message_log = []

def draw_agents(agents):
    screen.fill((30, 30, 30))
    padding = 80
    spacing = (SCREEN_WIDTH - MESSAGE_BOX_WIDTH - 2 * padding) // max(1, (len(agents) - 1))
    y = SCREEN_HEIGHT // 3

    for i, agent in enumerate(agents):
        x = padding + i * spacing
        color = (200, 0, 0) if agent.alive else (80, 80, 80)
        pygame.draw.circle(screen, color, (x, y), AGENT_RADIUS)
        label = font.render(agent.name, True, (255, 255, 255))
        screen.blit(label, (x - AGENT_RADIUS, y + AGENT_RADIUS + 10))

    draw_message_log()
    pygame.display.flip()

def wrap_text(text, font, max_width):
    words = text.split(' ')
    lines = []
    current_line = ''
    for word in words:
        test_line = current_line + word + ' '
        if font.size(test_line)[0] <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word + ' '
    if current_line:
        lines.append(current_line)
    return lines

def draw_message_log():
    log_height = 30 * MESSAGE_LOG_LIMIT + 10
    pygame.draw.rect(screen, (20, 20, 20), (MESSAGE_BOX_X - 10, SCREEN_HEIGHT - log_height - 10, MESSAGE_BOX_WIDTH + 20, log_height))
    y = SCREEN_HEIGHT - 40
    for name, msg in reversed(message_log[-MESSAGE_LOG_LIMIT:]):
        wrapped_lines = wrap_text(f"{name}: {msg}", font_small, MESSAGE_BOX_WIDTH)
        for line in reversed(wrapped_lines):
            text_surface = font_small.render(line.strip(), True, (220, 220, 220))
            screen.blit(text_surface, (MESSAGE_BOX_X, y))
            y -= 20

def choose_night_target(werewolf, candidates):
    non_wolves = [a for a in candidates if a.alive and a.role != WEREWOLF]
    return random.choice(non_wolves) if non_wolves else None

def run_sim_and_collect_rollouts():
    global message_log
    shared_models = {
        WEREWOLF: {
            'world_model': WorldModelMLP(latent_dim=32, action_dim=8),
            'action_encoder': ActionEncoder(num_actions=6, action_dim=8)
        },
        VILLAGER: {
            'world_model': WorldModelMLP(latent_dim=32, action_dim=8),
            'action_encoder': ActionEncoder(num_actions=6, action_dim=8)
        }
    }

    agents = [BaseAgent(f"Agent_{i}") for i in range(NUM_AGENTS)]
    assign_roles(agents, NUM_WEREWOLVES)

    for agent in agents:
        role = agent.role
        agent.world_model = shared_models[role]['world_model']
        agent.action_encoder = shared_models[role]['action_encoder']
        agent.llm_fn = chatgpt_llm_from_latent

    print(f"[INFO] Assigned roles: {[agent.role for agent in agents]}")
    round_num = 1
    running = True

    rollout_by_role = {WEREWOLF: [], VILLAGER: []}
    prev_latents = {}

    while running and any(a.alive and a.role == WEREWOLF for a in agents) and any(a.alive and a.role != WEREWOLF for a in agents):
        print(f"\n[ROUND {round_num}] Discussion Phase")
        draw_agents(agents)
        pygame.time.delay(1500)

        for agent in agents:
            if agent.alive:
                z_t = agent.encode_current_belief(round_num, agents)
                prev_latents[agent.name] = z_t.detach()
                message = agent.speak()
                print(f"{agent.name}: {message}")
                message_log.append((agent.name, message))

        votes = {}
        actions = {}
        alive_agents = [a for a in agents if a.alive]
        for agent in alive_agents:
            target = agent.vote(alive_agents)
            if target:
                votes[agent.name] = target.name
                target_idx = int(target.name.split('_')[1])
                actions[agent.name] = torch.tensor([target_idx])

        eliminated_name = resolve_votes(votes)
        for agent in agents:
            if agent.name == eliminated_name:
                eliminate_player(agent)
                print(f"[VOTE] {agent.name} was eliminated.")
                break

        draw_agents(agents)
        pygame.time.delay(1500)

        for agent in agents:
            if agent.name in prev_latents and agent.alive:
                z_next = agent.encode_current_belief(round_num + 1, agents).detach()
                a_t = actions.get(agent.name, torch.zeros(1))
                rollout_by_role[agent.role].append((prev_latents[agent.name], a_t, z_next, agent.role))

        if round_num % TRAIN_INTERVAL == 0:
            for role in [WEREWOLF, VILLAGER]:
                data = rollout_by_role[role]
                if data:
                    print(f"[TRAINING] {role} model on {len(data)} samples")
                    train_jepa(
                        data,
                        shared_models[role]['world_model'],
                        shared_models[role]['action_encoder']
                    )
                    rollout_by_role[role] = []

        werewolves = [a for a in agents if a.alive and a.role == WEREWOLF]
        if werewolves:
            print(f"\n[NIGHT PHASE] Werewolves are selecting a target...")
            for ww in werewolves:
                target = choose_night_target(ww, agents)
                if target:
                    eliminate_player(target)
                    print(f"[NIGHT] {target.name} was killed in the night.")
                    break

        draw_agents(agents)
        pygame.time.delay(1500)

        round_num += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    print("\n[SIMULATION ENDED]")
    pygame.quit()
    return [], agents

if __name__ == "__main__":
    run_sim_and_collect_rollouts()