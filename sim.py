# sim.py  ── verbose, multithreaded, responsive Pygame
# -----------------------------------------------------------------------------
# This version fixes the rollout tuples to use the *true* post‑act z_{t+1},
# and prints an acceptance metric: mean ||Δz|| per day. It also shares the
# MessageEncoder across agents to save VRAM/CPU.
# -----------------------------------------------------------------------------

import sys
import os
import random
import concurrent.futures as cf
from collections import deque
from typing import Dict, Tuple

import pygame
import torch

from agent import BaseAgent
from roles import WEREWOLF, VILLAGER, assign_roles
from world import resolve_votes, eliminate_player
from training_utils import load_role_models
from encoders import MessageEncoder  # shared instance

# ───────────────────────── CONFIG
NUM_AGENTS, NUM_WEREWOLVES = 6, 1
SCREEN_W, SCREEN_H = 1200, 600
FPS, AGENT_R = 1, 30
MSG_LOG_LIMIT, MSG_BOX_W = 12, 360
MSG_BOX_X = SCREEN_W - MSG_BOX_W

# Language toggle (set with env var; PowerShell:  $env:USE_LANGUAGE="0")
USE_LANGUAGE = os.environ.get("USE_LANGUAGE", "1") != "0"

# ───────────────────────── runtime globals (populated iff visual=True)
screen = font = font_s = clock = None
msg_log: deque[tuple[str, str]] = deque(maxlen=200)

# ╭────────────────────────── UI HELPERS ───────────────────────────╮
def _wrap(text: str, fnt: pygame.font.Font, width: int) -> list[str]:
    out, cur = [], ""
    for word in text.split():
        test = f"{cur}{word} "
        if fnt.size(test)[0] <= width:
            cur = test
        else:
            out.append(cur)
            cur = f"{word} "
    if cur:
        out.append(cur)
    return out

def _draw_log() -> None:
    box_h = 30 * MSG_LOG_LIMIT + 10
    pygame.draw.rect(
        screen, (20, 20, 20),
        (MSG_BOX_X - 10, SCREEN_H - box_h - 10, MSG_BOX_W + 20, box_h),
    )
    y = SCREEN_H - 40
    for name, msg in list(msg_log)[-MSG_LOG_LIMIT:][::-1]:
        for line in _wrap(f"{name}: {msg}", font_s, MSG_BOX_W)[::-1]:
            screen.blit(font_s.render(line.strip(), True, (220, 220, 220)), (MSG_BOX_X, y))
            y -= 20

def draw_agents(agents: list[BaseAgent]) -> None:
    screen.fill((30, 30, 30))
    pad = 80
    spacing = (SCREEN_W - MSG_BOX_W - 2 * pad) // max(1, len(agents) - 1)
    y = SCREEN_H // 3
    for idx, ag in enumerate(agents):
        x = pad + idx * spacing
        colour = (200, 0, 0) if ag.alive else (80, 80, 80)
        pygame.draw.circle(screen, colour, (x, y), AGENT_R)
        screen.blit(font.render(ag.name, True, (255, 255, 255)), (x - AGENT_R, y + AGENT_R + 10))
    _draw_log()
    pygame.display.flip()

# ╭───────────────────────── GAME HELPERS ───────────────────────────╮
def choose_night_target(wolf: BaseAgent, agents: list[BaseAgent]):
    non_wolves = [a for a in agents if a.alive and a.role != WEREWOLF]
    return random.choice(non_wolves) if non_wolves else None

# ╭────────────────────────── MAIN LOOP ─────────────────────────────╮
def simulate_game(visual: bool = True):
    # ───── Pygame initialisation (only when visual) ─────
    if visual:
        pygame.init()
        global screen, font, font_s, clock
        screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("JEPA Werewolf Sim")
        font = pygame.font.SysFont(None, 32)
        font_s = pygame.font.SysFont(None, 24)
        clock = pygame.time.Clock()

    # ───── Agent creation + role assignment ─────
    agents = [BaseAgent(f"Agent_{i}") for i in range(NUM_AGENTS)]
    assign_roles(agents, NUM_WEREWOLVES)
    print("▶ Assigned roles:", ", ".join(f"{a.name}:{a.role}" for a in agents))

    # Share one MessageEncoder to reduce memory/latency
    shared_msg_encoder = MessageEncoder()
    for ag in agents:
        ag.message_encoder = shared_msg_encoder

    # Attach JEPA sub‑modules
    for ag in agents:
        wm, ae, planner = load_role_models(ag.role)
        ag.world_model, ag.action_encoder, ag.planner = wm, ae, planner

    # LLM hookup: if language is ablated, stub regardless of visual mode.
    if not USE_LANGUAGE:
        for ag in agents:
            ag.llm_fn = lambda z, self: "..."
    else:
        from llm_script import chatgpt_llm_from_latent
        for ag in agents:
            ag.llm_fn = chatgpt_llm_from_latent

    executor = cf.ThreadPoolExecutor(max_workers=NUM_AGENTS)
    rollout = []
    round_num = 0

    # ───── main day/night loop ─────
    while True:
        if visual: pygame.event.pump()
        round_num += 1
        living = [a for a in agents if a.alive]
        print(f"\n=== Day {round_num} ===")

        # ─── Asynchronous dialogue (one future per agent) ───
        futures = {executor.submit(a.speak, round_num, agents): a for a in living}
        while futures:
            done, _ = cf.wait(futures, timeout=0.05, return_when=cf.FIRST_COMPLETED)
            for fut in done:
                ag = futures.pop(fut)
                msg = fut.result() if fut.exception() is None else "..."
                print(f"{ag.name}: {msg}")
                if visual:
                    msg_log.append((ag.name, msg))
            if visual:
                draw_agents(agents)
                clock.tick(FPS)
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

        # ─── PRE‑ACT: collect (z_t, a_idx, role) and remember per agent ───
        pending: Dict[str, Tuple[torch.Tensor, torch.Tensor, str]] = {}
        vote_map = {}
        for ag in living:
            z_t = ag.encode_current_belief(round_num, agents)
            target = ag.plan_vote(z_t, living)
            vote_map[ag] = target
            a_idx = torch.tensor([int(target.name.split('_')[1])])
            pending[ag.name] = (z_t.detach(), a_idx, ag.role)
        print("✉ Votes:", ", ".join(f"{k.name}->{v.name}" for k, v in vote_map.items()))

        # resolve vote
        eliminated_name = resolve_votes({k.name: v.name for k, v in vote_map.items()})
        if eliminated_name:
            for ag in agents:
                if ag.name == eliminated_name:
                    eliminate_player(ag)
                    print(f"☠ {ag.name} was voted out.")
                    if visual:
                        msg_log.append(("System", f"{ag.name} eliminated."))
                    break

        # night kill
        wolves = [a for a in agents if a.alive and a.role == WEREWOLF]
        if wolves:
            victim = choose_night_target(wolves[0], agents)
            if victim:
                eliminate_player(victim)
                print(f"🌙 Night kill: {victim.name}")
                if visual:
                    msg_log.append(("Night", f"{victim.name} slain."))

        # ─── POST‑ACT: re‑encode to get z_{t+1} and append rollouts ───
        z_deltas = []
        for ag in agents:
            if ag.name in pending and ag.alive or ag.name in pending:
                z_next = ag.encode_current_belief(round_num + 1, agents).detach()
                z_t, a_idx, role = pending[ag.name]
                rollout.append((z_t, a_idx, z_next, role))
                z_deltas.append(torch.norm(z_next - z_t).item())
        if z_deltas:
            mean_delta = sum(z_deltas) / len(z_deltas)
            print(f"[Δz] mean ||z_{round_num+1}-z_{round_num}|| = {mean_delta:.4f}")

        # win check
        wolves_alive = [a for a in agents if a.alive and a.role == WEREWOLF]
        vill_alive = [a for a in agents if a.alive and a.role != WEREWOLF]
        if not wolves_alive or len(wolves_alive) >= len(vill_alive):
            break

    print("\n== Game over ==")
    executor.shutdown(wait=True)
    return rollout, {"rounds": round_num}


# ───────────────────────── CLI ─────────────────────────
if __name__ == "__main__":
    simulate_game(visual=True)
