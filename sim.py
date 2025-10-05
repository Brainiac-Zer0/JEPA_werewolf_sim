# sim.py  ── verbose, multithreaded, responsive Pygame + Judge integration
# -----------------------------------------------------------------------------
# - Keeps the rollout tuples using the *true* post-act z_{t+1}
# - Prints acceptance metric: mean ||Δz|| per day
# - Shares a single MessageEncoder across agents to save VRAM/CPU
# - Calls LLM-as-Judge on planner top-k vote targets, picks final vote, logs subscores
# - Returns agents in meta so train.py can run speaker learning
# - Applies personality randomization per agent
# - NEW (Phase 1: Stabilization & Logging)
#   * Deterministic seeds + run metadata snapshot
#   * Phase-aware + mask-aware telemetry
#   * Per-decision CSV rows (TALK / VOTE / KILL) with Δz for votes
#   * Optional judge debug JSONL is handled in judge.py (not here)
# -----------------------------------------------------------------------------

import sys
import os
import random
import concurrent.futures as cf
from collections import deque
from typing import Dict, Tuple, List

import pygame
import torch, yaml
import torch.nn.functional as F

# NEW: logging & determinism helpers
import uuid, time, json, csv, pathlib
import numpy as np

from agent import BaseAgent
from roles import WEREWOLF, VILLAGER, assign_roles
# NEW: persona randomization hook
try:
    from roles import apply_personality
except Exception:
    apply_personality = None

from world import resolve_votes, eliminate_player
from training_utils import load_role_models
from encoders import MessageEncoder  # shared instance

# Judge imports
from judge import JudgeRubric, score_batch

# ── Load config
with open("config.yaml", "r") as f:
    CFG = yaml.safe_load(f)

# Core sim settings
NUM_AGENTS = int(CFG.get("NUM_AGENTS", 6))
NUM_WEREWOLVES = int(CFG.get("NUM_WEREWOLVES", 1))

# Screen/display
SCREEN_W = int(CFG.get("SCREEN_W", 1200))
SCREEN_H = int(CFG.get("SCREEN_H", 600))
FPS = int(CFG.get("FPS", 1))
AGENT_R = int(CFG.get("AGENT_R", 30))

# Message log
MSG_LOG_LIMIT = int(CFG.get("MSG_LOG_LIMIT", 12))
MSG_BOX_W = int(CFG.get("MSG_BOX_W", 360))
MSG_BOX_X = int(CFG.get("MSG_BOX_X", SCREEN_W - MSG_BOX_W))

# Language toggle
USE_LANGUAGE = bool(CFG.get("USE_LANGUAGE", True))

# Judge settings
RUBRIC_PATH = CFG.get("RUBRIC_PATH", "judge_rubric.yaml")
PLANNER_TOPK = int(CFG.get("PLANNER_TOPK", 3))

# NEW: logging & seeds
LOG_CFG = CFG.get("logging", {}) if isinstance(CFG.get("logging", {}), dict) else {}
LOG_DIR       = LOG_CFG.get("dir", "logs")
METRICS_CSV   = LOG_CFG.get("metrics_csv", f"{LOG_DIR}/metrics.csv")
RUN_CFG_PATH  = LOG_CFG.get("run_config", f"{LOG_DIR}/run_config.yaml")
RUN_META_PATH = LOG_CFG.get("run_meta",   f"{LOG_DIR}/run_meta.json")
SAVE_CFG_SNAPSHOT = bool(CFG.get("runtime", {}).get("save_config_snapshot", True))
SEEDS = CFG.get("seeds", {}) if isinstance(CFG.get("seeds", {}), dict) else {}
SEED_GLOBAL = int(SEEDS.get("global", 123))

# ── runtime globals (populated iff visual=True)
screen = font = font_s = clock = None
msg_log: deque[tuple[str, str]] = deque(maxlen=200)

# ╭────────────────────────── NEW: UTILITIES ───────────────────────────╮
def set_seed(seed: int):
    """Deterministic-ish seeding across python/numpy/torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)

def write_config_snapshot(cfg: dict, path: str):
    ensure_dir(path)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)

def append_csv_rows(path: str, rows: list[dict]):
    if not rows:
        return
    ensure_dir(path)
    header = list(rows[0].keys())
    new_file = not pathlib.Path(path).exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if new_file:
            w.writeheader()
        for r in rows:
            w.writerow(r)

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

def _planner_topk_for_agent(ag: BaseAgent, z_t: torch.Tensor, agents: list[BaseAgent], k: int) -> List[tuple[str, float]]:
    """Returns top-k (target_name, prob) among currently alive opponents for agent ag."""
    with torch.no_grad():
        logits = ag.planner(z_t.unsqueeze(0)).squeeze(0)  # [num_agents]
        probs  = torch.softmax(logits, dim=-1)

    alive = [x for x in agents if x.alive and x.name != ag.name]
    if not alive:
        return []
    alive_idx = [int(x.name.split("_")[1]) for x in alive]

    # mask dead/self in the logits, then top-k
    masked = torch.full_like(logits, float("-inf"))
    masked[alive_idx] = logits[alive_idx]
    k_eff = min(k, len(alive_idx))
    _, topi = torch.topk(masked, k=k_eff)

    out: List[tuple[str, float]] = []
    for idx in topi.tolist():
        name = f"Agent_{idx}"
        out.append((name, float(probs[idx].item())))
    return out

def _agent_context_block(ag: BaseAgent, max_lines: int = 6) -> str:
    """Build a compact context string from the agent's recent heard messages."""
    lines = []
    for n, m in list(ag.message_memory)[-max_lines:]:
        if not m.strip():
            continue
        lines.append(f"- {n}: {m.strip()}")
    return "\n".join(lines) if lines else "- (no recent messages heard)"

def _candidate_text(target_name: str) -> str:
    """Turn an action (vote target) into a short candidate utterance for the judge."""
    return f"We should vote to eliminate {target_name}."

# ╭────────────────────────── MAIN LOOP ─────────────────────────────╮
def simulate_game(visual: bool = True):
    # NEW: run id + determinism + log dirs + config snapshot
    run_id = f"run_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    set_seed(SEED_GLOBAL)
    pathlib.Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    if SAVE_CFG_SNAPSHOT:
        write_config_snapshot(CFG, RUN_CFG_PATH)
    with open(RUN_META_PATH, "w") as f:
        json.dump({
            "run_id": run_id,
            "seed": SEED_GLOBAL,
            "timestamp": int(time.time()),
            "device": str(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")),
        }, f)

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
    # NEW: apply personality randomization (if available)
    if apply_personality is not None:
        apply_personality(agents)

    print("▶ Assigned roles:", ", ".join(f"{a.name}:{a.role}" for a in agents))

    # Share one MessageEncoder to reduce memory/latency
    shared_msg_encoder = MessageEncoder()
    for ag in agents:
        ag.message_encoder = shared_msg_encoder

    # Attach JEPA sub-modules
    for ag in agents:
        wm, ae, planner = load_role_models(ag.role)
        ag.world_model, ag.action_encoder, ag.planner = wm, ae, planner

    # LLM hookup: choose mouthpiece by env (baseline or bias version)
    if not USE_LANGUAGE:
        for ag in agents:
            ag.llm_fn = lambda z, self: "..."
    else:
        from llm_script import llm_fn_from_env
        mouth_fn = llm_fn_from_env()
        for ag in agents:
            ag.llm_fn = mouth_fn

    # Load judge rubric once
    try:
        judge_rubric = JudgeRubric.load(RUBRIC_PATH)
    except Exception as e:
        raise RuntimeError(f"Failed to load judge rubric at {RUBRIC_PATH}: {e}")

    executor = cf.ThreadPoolExecutor(max_workers=NUM_AGENTS)
    rollout = []
    round_num = 0

    # NEW: collectors for Phase-1 logging
    metrics_rows: list[dict] = []   # one row per talk/vote/kill
    phase_log: list[dict] = []      # [{"round":i,"phase":"..."}]
    mask_logs: list[dict] = []      # [{"round":i,"phase":"...","actor":"Agent_k","mask":[...]}]

    # ───── main day/night loop ─────
    while True:
        if visual:
            pygame.event.pump()
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

                # NEW: log TALK row (phase-aware)
                phase_log.append({"round": round_num, "phase": "DAY_DISCUSS"})
                metrics_rows.append({
                    "run_id": run_id,
                    "round": round_num,
                    "phase": "DAY_DISCUSS",
                    "agent": ag.name,
                    "role": ag.role,
                    "choice_type": "TALK_INTENT",
                    "choice_payload": getattr(ag, "talk_category_last", "") or "",
                    "mask_size": "",
                    "judge_score": "", "coh": "", "truth": "", "role_score": "", "safety": "",
                    "dz_l2": "", "dz_1mcos": "",
                    "speaker_mode": getattr(ag, "speaker_mode", "") or "",
                    "persona_norm": getattr(ag, "persona_norm", 0.0),
                })

            if visual:
                draw_agents(agents)
                clock.tick(FPS)
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

        # ─── PRE-ACT: (z_t cache), LLM Judge over planner top-k, final vote choice ───
        pending: Dict[str, Tuple[torch.Tensor, torch.Tensor, str]] = {}
        vote_map: Dict[BaseAgent, BaseAgent] = {}

        # First compute z_t for all living
        z_map: Dict[BaseAgent, torch.Tensor] = {ag: ag.encode_current_belief(round_num, agents) for ag in living}

        for ag in living:
            z_t = z_map[ag]

            # planner top-k candidates (names + probs)
            topk = _planner_topk_for_agent(ag, z_t, living, PLANNER_TOPK)
            if not topk:
                continue

            # NEW: record legal vote mask (names) + phase tick
            alive_names = [x.name for x in living if x.name != ag.name]
            mask_logs.append({"round": round_num, "phase": "DAY_VOTE", "actor": ag.name, "mask": alive_names})
            phase_log.append({"round": round_num, "phase": "DAY_VOTE"})

            # Build judge items for this agent (same context, different candidate strings)
            context_block = _agent_context_block(ag, max_lines=3)
            judge_items = [{
                "context": context_block,
                "role": ag.role or "Unknown",
                "candidate": _candidate_text(name),
            } for (name, _p) in topk]

            # Score with judge (batched per agent)
            judged = score_batch(judge_items, judge_rubric)

            # Pick best by judge score
            best_idx = max(range(len(judged)), key=lambda i: judged[i].get("score", 0.0))
            best_name = topk[best_idx][0]
            # map to actual agent object
            target = next((x for x in living if x.name == best_name), None)
            if target is None:
                # fallback: original planner argmax among alive
                target = next((x for x in living if x.name != ag.name), living[0])

            vote_map[ag] = target
            a_idx = torch.tensor([int(target.name.split('_')[1])])
            pending[ag.name] = (z_t.detach(), a_idx, ag.role)

            # Log judge decision + subscores
            subs = judged[best_idx].get("subscores", {})
            s = judged[best_idx].get("score", 0.0)
            log_line = (f"Judge→ {ag.name} votes {target.name} "
                        f"[score={s:.2f} | coh={subs.get('coherence',0.0):.2f} "
                        f"truth={subs.get('truthfulness',0.0):.2f} "
                        f"role={subs.get('role_alignment',0.0):.2f} "
                        f"safety={subs.get('social_safety',0.0):.2f}]")
            print(log_line)
            if visual:
                msg_log.append(("Judge", log_line))

            # NEW: append VOTE row (Δz filled post-act)
            metrics_rows.append({
                "run_id": run_id,
                "round": round_num,
                "phase": "DAY_VOTE",
                "agent": ag.name,
                "role": ag.role,
                "choice_type": "VOTE_TARGET",
                "choice_payload": target.name if target else "",
                "mask_size": len(alive_names),
                "judge_score": f"{s:.4f}",
                "coh": f"{subs.get('coherence',0.0):.4f}",
                "truth": f"{subs.get('truthfulness',0.0):.4f}",
                "role_score": f"{subs.get('role_alignment',0.0):.4f}",
                "safety": f"{subs.get('social_safety',0.0):.4f}",
                "dz_l2": "", "dz_1mcos": "",
                "speaker_mode": getattr(ag, "speaker_mode", "") or "",
                "persona_norm": getattr(ag, "persona_norm", 0.0),
            })

        if vote_map:
            print("✉ Votes (judge-selected):", ", ".join(f"{k.name}->{v.name}" for k, v in vote_map.items()))
        else:
            print("✉ Votes: (none)")

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
            # NEW: record night kill legal mask + phase tick
            phase_log.append({"round": round_num, "phase": "NIGHT_KILL"})
            wolf = wolves[0]
            legal_targets = [a.name for a in agents if a.alive and a.name != wolf.name and a.role != WEREWOLF]
            mask_logs.append({"round": round_num, "phase": "NIGHT_KILL", "actor": wolf.name, "mask": legal_targets})

            victim = choose_night_target(wolf, agents)
            if victim:
                eliminate_player(victim)
                print(f"🌙 Night kill: {victim.name}")
                if visual:
                    msg_log.append(("Night", f"{victim.name} slain."))
                # NEW: append KILL row
                metrics_rows.append({
                    "run_id": run_id,
                    "round": round_num,
                    "phase": "NIGHT_KILL",
                    "agent": wolf.name,
                    "role": wolf.role,
                    "choice_type": "KILL_TARGET",
                    "choice_payload": victim.name,
                    "mask_size": len(legal_targets),
                    "judge_score": "", "coh": "", "truth": "", "role_score": "", "safety": "",
                    "dz_l2": "", "dz_1mcos": "",
                    "speaker_mode": getattr(wolf, "speaker_mode", "") or "",
                    "persona_norm": getattr(wolf, "persona_norm", 0.0),
                })

        # ─── POST-ACT: re-encode to get z_{t+1} and append rollouts ───
        z_deltas = []
        cos_deltas = []
        # NEW: per-agent Δz to fill back into their most recent vote row
        dz_by_agent: Dict[str, float] = {}
        cos_by_agent: Dict[str, float] = {}

        for ag in agents:
            if ag.name in pending:
                z_next = ag.encode_current_belief(round_num + 1, agents).detach()
                z_t, a_idx, role = pending[ag.name]
                rollout.append((z_t, a_idx, z_next, role))
                l2 = torch.norm(z_next - z_t).item()
                z_deltas.append(l2)
                cos_val = F.cosine_similarity(z_next.unsqueeze(0), z_t.unsqueeze(0)).item()
                cos_deltas.append(1.0 - cos_val)
                dz_by_agent[ag.name] = l2
                cos_by_agent[ag.name] = 1.0 - cos_val

        if z_deltas:
            mean_l2 = sum(z_deltas) / len(z_deltas)
            mean_1mcos = (sum(cos_deltas) / len(cos_deltas)) if cos_deltas else 0.0
            print(f"[Δz] L2={mean_l2:.4f}  (1-cos)={mean_1mcos:.4f}")

            # NEW: fill Δz columns for the vote rows of this round
            for row in reversed(metrics_rows):
                if row["round"] != round_num:
                    break  # earlier rounds
                if row["phase"] == "DAY_VOTE":
                    ag_name = row["agent"]
                    if ag_name in dz_by_agent:
                        row["dz_l2"] = f"{dz_by_agent[ag_name]:.6f}"
                        row["dz_1mcos"] = f"{cos_by_agent[ag_name]:.6f}"

        # win check
        wolves_alive = [a for a in agents if a.alive and a.role == WEREWOLF]
        vill_alive = [a for a in agents if a.alive and a.role != WEREWOLF]
        if not wolves_alive or len(wolves_alive) >= len(vill_alive):
            break

    print("\n== Game over ==")

    # NEW: write metrics and print tiny summary
    append_csv_rows(METRICS_CSV, metrics_rows)
    by_phase: Dict[str, int] = {}
    for r in metrics_rows:
        by_phase[r["phase"]] = by_phase.get(r["phase"], 0) + 1
    print("[SUMMARY] rows:", len(metrics_rows), "by_phase:", by_phase)

    executor.shutdown(wait=True)
    # include agents + phase/mask logs in meta for verifiers / downstream
    meta_out = {
        "rounds": round_num,
        "agents": agents,
        "run_id": run_id,
        "phases": phase_log,
        "mask_logs": mask_logs,
    }
    return rollout, meta_out


# ───────────────────────── CLI ─────────────────────────
if __name__ == "__main__":
    simulate_game(visual=True)
