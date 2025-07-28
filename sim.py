# sim.py  ── verbose, multithreaded, responsive Pygame
# -----------------------------------------------------------------------------
# This version calls the *round‑aware* BaseAgent.speak(round_num, agents) API, so
# every dialog line is generated from the correct latent belief state.  The
# code is heavily commented to clarify the multithreaded flow, the belief
# encoding pipeline, and the Pygame rendering loop.
# -----------------------------------------------------------------------------

import sys
import random
import concurrent.futures as cf
from collections import deque

import pygame
import torch

from agent import BaseAgent
from roles import WEREWOLF, VILLAGER, assign_roles
from world import resolve_votes, eliminate_player
from llm_script import chatgpt_llm_from_latent
from training_utils import load_role_models

# ───────────────────────── CONFIG
NUM_AGENTS, NUM_WEREWOLVES = 6, 1
SCREEN_W, SCREEN_H = 1200, 600
FPS, AGENT_R = 1, 30                     # FPS only matters in visual mode
MSG_LOG_LIMIT, MSG_BOX_W = 12, 360
MSG_BOX_X = SCREEN_W - MSG_BOX_W

# ───────────────────────── runtime globals (populated iff visual=True)
screen = font = font_s = clock = None    # pygame objects
msg_log: deque[tuple[str, str]] = deque(maxlen=200)  # text lines for side‑panel

# ╭────────────────────────── UI HELPERS ───────────────────────────╮
# | _wrap, _draw_log, draw_agents handle all Pygame visualisation. |
# ╰─────────────────────────────────────────────────────────────────╯

def _wrap(text: str, fnt: pygame.font.Font, width: int) -> list[str]:
    """Simple word‑wrap so long chat lines fit the message box."""
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
    """Render the scrolling chat log on the right side of the screen."""
    box_h = 30 * MSG_LOG_LIMIT + 10
    pygame.draw.rect(
        screen, (20, 20, 20),
        (MSG_BOX_X - 10, SCREEN_H - box_h - 10, MSG_BOX_W + 20, box_h),
    )
    y = SCREEN_H - 40
    for name, msg in list(msg_log)[-MSG_LOG_LIMIT:][::-1]:  # newest at bottom
        for line in _wrap(f"{name}: {msg}", font_s, MSG_BOX_W)[::-1]:
            screen.blit(font_s.render(line.strip(), True, (220, 220, 220)), (MSG_BOX_X, y))
            y -= 20


def draw_agents(agents: list[BaseAgent]) -> None:
    """Draw each agent as a circle + name."""
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
# | choose_night_target  – very dumb wolf kill policy                |
# ╰──────────────────────────────────────────────────────────────────╯

def choose_night_target(wolf: BaseAgent, agents: list[BaseAgent]):
    non_wolves = [a for a in agents if a.alive and a.role != WEREWOLF]
    return random.choice(non_wolves) if non_wolves else None

# ╭────────────────────────── MAIN LOOP ─────────────────────────────╮
# | simulate_game(visual=bool) runs a *single* game and returns the  |
# | JEPA rollout data plus simple meta‑stats.                        |
# ╰──────────────────────────────────────────────────────────────────╯

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

    # Attach JEPA sub‑modules & LLM
    for ag in agents:
        wm, ae, planner = load_role_models(ag.role)
        ag.world_model, ag.action_encoder, ag.planner = wm, ae, planner
        ag.llm_fn = chatgpt_llm_from_latent

    executor = cf.ThreadPoolExecutor(max_workers=NUM_AGENTS)
    rollout = []
    round_num = 0

    # ───── main day/night loop ─────
    while True:
        if visual: pygame.event.pump()  # keep window responsive
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

        # ─── Belief encode, vote, collect rollout ───
        vote_map = {}
        for ag in living:
            z_t = ag.encode_current_belief(round_num, agents)
            target = ag.plan_vote(z_t, living)
            vote_map[ag] = target
            rollout.append((z_t, torch.tensor([int(target.name.split('_')[1])]), z_t, ag.role))
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
