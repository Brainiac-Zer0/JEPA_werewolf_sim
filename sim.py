# sim.py  ── verbose, multithreaded, responsive Pygame + Judge integration
# -----------------------------------------------------------------------------
# - Keeps the rollout tuples using the *true* post-act z_{t+1}
# - Prints acceptance metric: mean ||Δz|| per day
# - Shares a single MessageEncoder across agents to save VRAM/CPU
# - Calls LLL-as-Judge on planner top-k vote targets, picks final vote, logs subscores
# - Returns agents in meta so train.py can run speaker learning
# - Applies personality randomization per agent
# - NEW (Phase 1: Stabilization & Logging)
#   * Deterministic seeds + run metadata snapshot
#   * Phase-aware + mask-aware telemetry
#   * Per-decision CSV rows (TALK / VOTE / KILL) with Δz for votes
#   * Optional judge debug JSONL is handled in judge.py (not here)
# - NEW (Phase 3: Multi-head)
#   * Use TalkHead / VoteHead / KillHead when available
#   * Proper boolean masks (self/dead/wolves)
#   * Judge re-ranking over VoteHead top-k
#   * Phase-aware rollout tuples: (z_t, phase_code, payload_idx, z_{t+1}, role)
#   * (This build) Planner×Judge mixing, social influence knobs, talk mask logging,
#                  optional Δz for TALK and KILL
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
# NEW: bring in SocialInfluence (falls back cleanly if not present yet)
try:
    from encoders import MessageEncoder, SocialInfluence  # shared instance + social coupling
except Exception:
    from encoders import MessageEncoder
    SocialInfluence = None  # will skip δ_social if encoders.py not patched yet

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

# NEW: sim-level knobs (planner×judge mixing, Δz logs)
SIM_CFG = CFG.get("sim", {}) if isinstance(CFG.get("sim"), dict) else {}
VOTE_MIX_ALPHA: float = float(SIM_CFG.get("vote_mix_alpha", 0.0))  # 0.0 judge-only among top-k; 1.0 planner-only
LOG_DZ_TALK: bool = bool(SIM_CFG.get("log_dz_talk", False))
LOG_DZ_KILL: bool = bool(SIM_CFG.get("log_dz_kill", False))

# NEW: social influence knobs
# Prefer modern `sim.social`, fallback to legacy `social_influence` if present.
_SOC_SECTION = SIM_CFG.get("social", None)
if not isinstance(_SOC_SECTION, dict):
    _SOC_SECTION = CFG.get("social_influence", {}) if isinstance(CFG.get("social_influence"), dict) else {}
SOC_CFG = _SOC_SECTION
SOC_ENABLED: bool = bool(SOC_CFG.get("enabled", True))
SOC_K: int = int(SOC_CFG.get("K", 6))                 # number of recent utterances
SOC_SCALE: float = float(SOC_CFG.get("scale", 1.0))   # scaling on δ_social

# ── ENV OVERRIDES (job script shims)
def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name, None)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")

def _env_int(name: str, default: int) -> int:
    v = os.getenv(name, None)
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        return default

def _env_float(name: str, default: float) -> float:
    v = os.getenv(name, None)
    if v is None:
        return default
    try:
        return float(v)
    except Exception:
        return default

USE_LANGUAGE   = _env_bool("USE_LANGUAGE", USE_LANGUAGE)
PLANNER_TOPK   = _env_int("PLANNER_TOPK", PLANNER_TOPK)
VOTE_MIX_ALPHA = _env_float("VOTE_MIX_ALPHA", VOTE_MIX_ALPHA)
LOG_DZ_TALK    = _env_bool("LOG_DZ_TALK", LOG_DZ_TALK)
LOG_DZ_KILL    = _env_bool("LOG_DZ_KILL", LOG_DZ_KILL)

# Social overrides
SOC_ENABLED = _env_bool("SOC_ENABLED", SOC_ENABLED)
SOC_K       = _env_int("SOC_K", SOC_K)
SOC_SCALE   = _env_float("SOC_SCALE", SOC_SCALE)

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

# NEW: uniform event logger (adds phase_code + numeric payloads)
def emit_event(rows, *, run_id, round_num, phase_code, phase_str, agent, role,
               choice_type, payload_idx, mask_names=None, judge=None,
               dz=None, speaker_mode="", persona_norm=0.0):
    rows.append({
        "run_id": run_id,
        "round": round_num,
        "phase": phase_str,
        "phase_code": phase_code,                         # NEW
        "agent": agent,
        "role": role,
        "choice_type": choice_type,
        "choice_payload": payload_idx,                    # numeric id (cat_id or agent_idx)
        "mask_size": (len(mask_names) if mask_names is not None else ""),
        "judge_score": "" if not judge else f"{judge.get('score', 0):.4f}",
        "coh": "" if not judge else f"{judge.get('coherence', 0):.4f}",
        "truth": "" if not judge else f"{judge.get('truthfulness', 0):.4f}",
        "role_score": "" if not judge else f"{judge.get('role_alignment', 0):.4f}",
        "safety": "" if not judge else f"{judge.get('social_safety', 0):.4f}",
        "dz_l2": "" if not dz else f"{dz.get('l2',0):.6f}",
        "dz_1mcos": "" if not dz else f"{dz.get('1mcos',0):.6f}",
        "speaker_mode": speaker_mode,
        "persona_norm": persona_norm,
    })

# NEW: mix helpers (planner × judge)
def _safe_norm_probs(x: torch.Tensor) -> torch.Tensor:
    s = float(x.sum().item())
    if s <= 0.0 or not torch.isfinite(x).all():
        return torch.full_like(x, 1.0 / max(1, x.numel()))
    return x / s

def _mix_topk_scores(topk_probs_planner: List[float], judged: List[dict], alpha: float) -> int:
    """
    Return index into top-k list using convex mix:
      alpha * planner_probs + (1-alpha) * normalized_judge_scores
    """
    if alpha <= 0.0:
        return max(range(len(judged)), key=lambda i: float(judged[i].get("score", 0.0)))
    if alpha >= 1.0:
        return int(torch.tensor(topk_probs_planner).argmax().item())

    p_pl = torch.tensor([max(0.0, float(p)) for p in topk_probs_planner], dtype=torch.float32)
    p_pl = _safe_norm_probs(p_pl)

    p_j = torch.tensor([max(0.0, float(x.get("score", 0.0))) for x in judged], dtype=torch.float32)
    p_j = _safe_norm_probs(p_j)

    mix = alpha * p_pl + (1.0 - alpha) * p_j
    return int(torch.argmax(mix).item())

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

# ╭───────────────────────── MULTI-HEAD HELPERS ─────────────────────────╮
def _get_heads(ag: BaseAgent):
    """
    Prefer the agent's factorized planner if present; otherwise return the legacy planner.
    """
    if hasattr(ag, "planner_factorized") and ag.planner_factorized is not None:
        return ag.planner_factorized
    return ag.planner

def _vote_mask_for(ag: BaseAgent, agents: list[BaseAgent], num_agents: int) -> torch.Tensor:
    """True where legal to vote: alive and not self."""
    mask = torch.zeros(num_agents, dtype=torch.bool)
    for x in agents:
        if x.alive and x.name != ag.name:
            idx = int(x.name.split("_")[1])
            if 0 <= idx < num_agents:
                mask[idx] = True
    return mask

def _kill_mask_for(wolf: BaseAgent, agents: list[BaseAgent], num_agents: int) -> torch.Tensor:
    """True where legal to kill: alive, not self, not a werewolf."""
    mask = torch.zeros(num_agents, dtype=torch.bool)
    for x in agents:
        if x.alive and x.name != wolf.name and x.role != WEREWOLF:
            idx = int(x.name.split("_")[1])
            if 0 <= idx < num_agents:
                mask[idx] = True
    return mask

def _talk_mask(num_cats: int) -> torch.Tensor:
    """All talk categories are currently legal."""
    return torch.ones(num_cats, dtype=torch.bool)

def _vote_topk_for_agent(ag: BaseAgent, z_t: torch.Tensor, agents: list[BaseAgent], k: int) -> List[tuple[str, float]]:
    """Top-k (target_name, prob) using VoteHead when available, else legacy planner."""
    alive = [x for x in agents if x.alive and x.name != ag.name]
    if not alive:
        return []
    alive_idx = [int(x.name.split("_")[1]) for x in alive]

    with torch.no_grad():
        fp = _get_heads(ag)
        if hasattr(fp, "vote"):
            # Try to infer num_agents from the head; else fall back to config
            try:
                num_agents = int(fp.vote.net[-1].out_features)  # type: ignore[attr-defined]
            except Exception:
                num_agents = NUM_AGENTS
            vmask = _vote_mask_for(ag, agents, num_agents).unsqueeze(0)  # (1, N)
            logits = fp.vote(z_t.unsqueeze(0), mask=vmask).squeeze(0)    # (N,)
        else:
            logits = ag.planner(z_t.unsqueeze(0)).squeeze(0)             # legacy over N

        probs  = torch.softmax(logits, dim=-1)

        # top-k over masked alive indices
        masked = torch.full_like(logits, float("-inf"))
        for idx in alive_idx:
            if 0 <= idx < logits.numel():
                masked[idx] = logits[idx]

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

    # NEW: instantiate social influence (if module available)
    social = None
    if SocialInfluence is not None and hasattr(shared_msg_encoder, "output_dim"):
        try:
            social = SocialInfluence(text_dim=shared_msg_encoder.output_dim)
        except Exception:
            social = None  # fail-safe: proceed without δ_social

    # Attach JEPA sub-modules (legacy loader; agents also carry factorized in agent.py init)
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

        # ─── optional: cache z pre-talk for Δz(TALK)
        z_pre_talk: Dict[str, torch.Tensor] = {}
        if LOG_DZ_TALK:
            for ag in living:
                z_pre_talk[ag.name] = ag.encode_current_belief(round_num, agents).detach()

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

                # If TalkHead is available, infer a talk category for logging/telemetry
                num_cats = 5  # fallback
                try:
                    fp = _get_heads(ag)
                    if hasattr(fp, "talk"):
                        with torch.no_grad():
                            z_t_talk = ag.encode_current_belief(round_num, agents)
                            try:
                                num_cats = int(fp.talk.net[-1].out_features)  # type: ignore[attr-defined]
                            except Exception:
                                num_cats = 5
                            tmask = _talk_mask(num_cats).unsqueeze(0)  # (1, C)
                            t_logits = fp.talk(z_t_talk.unsqueeze(0), mask=tmask).squeeze(0)
                            ag.talk_category_last = int(torch.argmax(t_logits).item())
                    else:
                        ag.talk_category_last = int(getattr(ag, "talk_category_last", -1))
                except Exception:
                    ag.talk_category_last = int(getattr(ag, "talk_category_last", -1))

                # NEW: log TALK mask (category IDs) for this speaker
                try:
                    mask_logs.append({
                        "round": round_num,
                        "phase": "DAY_DISCUSS",
                        "actor": ag.name,
                        "mask": [f"cat_{i}" for i in range(int(num_cats))]
                    })
                except Exception:
                    pass

                # NEW: log TALK row (phase-aware + numeric payload)
                phase_log.append({"round": round_num, "phase": "DAY_DISCUSS"})
                emit_event(
                    metrics_rows,
                    run_id=run_id, round_num=round_num,
                    phase_code=0, phase_str="DAY_DISCUSS",
                    agent=ag.name, role=ag.role,
                    choice_type="TALK_INTENT",
                    payload_idx=int(getattr(ag, "talk_category_last", -1)),
                    speaker_mode=getattr(ag, "speaker_mode", "") or "",
                    persona_norm=getattr(ag, "persona_norm", 0.0),
                )

                # optional Δz for TALK: measure after the agent has spoken
                if LOG_DZ_TALK and ag.name in z_pre_talk:
                    try:
                        z_post = ag.encode_current_belief(round_num, agents).detach()
                        dz_l2 = float(torch.norm(z_post - z_pre_talk[ag.name]).item())
                        dz_1mcos = float(1.0 - F.cosine_similarity(
                            z_post.unsqueeze(0), z_pre_talk[ag.name].unsqueeze(0)
                        ).item())
                        # backfill into the latest DAY_DISCUSS row for this agent
                        for row in reversed(metrics_rows):
                            if row["round"] != round_num:
                                break
                            if row["phase"] == "DAY_DISCUSS" and row["agent"] == ag.name and not row.get("dz_l2"):
                                row["dz_l2"] = f"{dz_l2:.6f}"
                                row["dz_1mcos"] = f"{dz_1mcos:.6f}"
                                break
                    except Exception:
                        pass

            if visual:
                draw_agents(agents)
                clock.tick(FPS)
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

        # ─── PRE-ACT: (z_t cache), social coupling, LLM Judge over planner top-k, final vote choice ───
        # Always use phase-aware pending: name -> (z_t, phase_code, payload_idx, role)
        pending: Dict[str, Tuple[torch.Tensor, int, torch.Tensor, str]] = {}
        vote_map: Dict[BaseAgent, BaseAgent] = {}

        # First compute z_t for all living
        z_map: Dict[BaseAgent, torch.Tensor] = {ag: ag.encode_current_belief(round_num, agents) for ag in living}

        # NEW: Apply language→state coupling once per agent (between Discuss and Vote)
        if social is not None and SOC_ENABLED:
            for ag in living:
                neighbors = [(n, m) for (n, m) in list(ag.message_memory)[-SOC_K:]
                             if n != ag.name and m and m.strip()]
                if not neighbors:
                    continue
                texts = [m for (_n, m) in neighbors]
                with torch.no_grad():
                    t_embed = shared_msg_encoder(texts).mean(dim=0)   # (D_text,)
                    delta = social(t_embed) * SOC_SCALE               # (LATENT_DIM,)
                    z_map[ag] = (z_map[ag] + delta).detach()

        for ag in living:
            z_t = z_map[ag]

            # planner top-k candidates (names + probs) — use VoteHead when available
            topk = _vote_topk_for_agent(ag, z_t, living, PLANNER_TOPK)
            if not topk:
                continue

            # NEW: record legal vote mask (names) + phase tick
            alive_names = [x.name for x in living if x.name != ag.name]
            mask_logs.append({"round": round_num, "phase": "DAY_VOTE", "actor": ag.name, "mask": alive_names})
            phase_log.append({"round": round_num, "phase": "DAY_VOTE"})

            # No-legal guard (shouldn't happen, but safe)
            if not alive_names:
                emit_event(
                    metrics_rows, run_id=run_id, round_num=round_num,
                    phase_code=1, phase_str="DAY_VOTE",
                    agent=ag.name, role=ag.role, choice_type="VOTE_TARGET",
                    payload_idx=-1, mask_names=[],
                    speaker_mode=getattr(ag, "speaker_mode", "") or "",
                    persona_norm=getattr(ag, "persona_norm", 0.0),
                )
                continue

            # Build judge items for this agent (same context, different candidate strings)
            context_block = _agent_context_block(ag, max_lines=3)
            judge_items = [{
                "context": context_block,
                "role": ag.role or "Unknown",
                "candidate": _candidate_text(name),
            } for (name, _p) in topk]

            # Score with judge (batched per agent)
            judged = score_batch(judge_items, judge_rubric)

            # Mixed selection among top-k (planner × judge)
            mix_idx = _mix_topk_scores([p for (_n, p) in topk], judged, VOTE_MIX_ALPHA)
            best_name = topk[mix_idx][0]
            best_j = judged[mix_idx]

            # map to actual agent object
            target = next((x for x in living if x.name == best_name), None)
            if target is None:
                target = next((x for x in living if x.name != ag.name), living[0])

            vote_map[ag] = target
            tgt_idx = int(target.name.split('_')[1])
            a_idx = torch.tensor(int(tgt_idx))  # scalar tensor
            # PHASE-AWARE: code 1 for DAY_VOTE
            pending[ag.name] = (z_t.detach(), 1, a_idx, ag.role)

            # Log judge decision + subscores
            subs = best_j.get("subscores", {})
            s = best_j.get("score", 0.0)
            log_line = (f"Judge→ {ag.name} votes {target.name} "
                        f"[score={s:.2f} | coh={subs.get('coherence',0.0):.2f} "
                        f"truth={subs.get('truthfulness',0.0):.2f} "
                        f"role={subs.get('role_alignment',0.0):.2f} "
                        f"safety={subs.get('social_safety',0.0):.2f}]")
            print(log_line)
            if visual:
                msg_log.append(("Judge", log_line))

            # NEW: append VOTE row (Δz filled post-act) with numeric payload + phase_code
            emit_event(
                metrics_rows,
                run_id=run_id, round_num=round_num,
                phase_code=1, phase_str="DAY_VOTE",
                agent=ag.name, role=ag.role,
                choice_type="VOTE_TARGET",
                payload_idx=int(target.name.split("_")[1]) if target else -1,
                mask_names=alive_names,
                judge={
                    "score": s,
                    "coherence": subs.get("coherence", 0.0),
                    "truthfulness": subs.get("truthfulness", 0.0),
                    "role_alignment": subs.get("role_alignment", 0.0),
                    "social_safety": subs.get("social_safety", 0.0),
                },
                speaker_mode=getattr(ag, "speaker_mode", "") or "",
                persona_norm=getattr(ag, "persona_norm", 0.0),
            )

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
            # simple coalition: pick first wolf to act (extend here for sequential/coalitions)
            wolf = wolves[0]
            legal_targets = [a.name for a in agents if a.alive and a.name != wolf.name and a.role != WEREWOLF]
            # keep mask logging (useful even if no kill occurs later)
            mask_logs.append({"round": round_num, "phase": "NIGHT_KILL", "actor": wolf.name, "mask": legal_targets})

            # Prefer KillHead when available; fallback to random target
            victim = None
            fp_w = _get_heads(wolf)
            if hasattr(fp_w, "kill"):
                try:
                    with torch.no_grad():
                        z_t_w = z_map.get(wolf, wolf.encode_current_belief(round_num, agents))
                        try:
                            nA = int(fp_w.kill.net[-1].out_features)  # type: ignore[attr-defined]
                        except Exception:
                            nA = NUM_AGENTS
                        kmask = _kill_mask_for(wolf, agents, nA).unsqueeze(0)  # (1,N)
                        k_logits = fp_w.kill(z_t_w.unsqueeze(0), mask=kmask).squeeze(0)  # (N,)
                        tgt_idx = int(torch.argmax(k_logits).item())
                        victim = next((a for a in agents if a.name == f"Agent_{tgt_idx}" and a.alive), None)
                        if victim is not None:
                            # phase-aware pending for NIGHT_KILL (2)
                            pending[wolf.name] = (z_t_w.detach(), 2, torch.tensor(int(tgt_idx)), wolf.role)
                except Exception:
                    victim = None

            if victim is None:
                # Fallback random non-wolf
                non_wolves = [a for a in agents if a.alive and a.role != WEREWOLF]
                victim = random.choice(non_wolves) if non_wolves else None
                if victim is not None:
                    # still record a phase-aware pending tuple (simulate a chosen idx)
                    z_t_w = z_map.get(wolf, wolf.encode_current_belief(round_num, agents)).detach()
                    pending[wolf.name] = (z_t_w, 2, torch.tensor(int(victim.name.split('_')[1])), wolf.role)

            if victim:
                eliminate_player(victim)
                print(f"🌙 Night kill: {victim.name}")
                if visual:
                    msg_log.append(("Night", f"{victim.name} slain."))
                # NEW: append KILL row with numeric payload + phase_code
                emit_event(
                    metrics_rows,
                    run_id=run_id, round_num=round_num,
                    phase_code=2, phase_str="NIGHT_KILL",
                    agent=wolf.name, role=wolf.role,
                    choice_type="KILL_TARGET",
                    payload_idx=int(victim.name.split("_")[1]),
                    mask_names=legal_targets,
                    speaker_mode=getattr(wolf, "speaker_mode", "") or "",
                    persona_norm=getattr(wolf, "persona_norm", 0.0),
                )
                # NEW: explicit actor/target trace for NIGHT_KILL (single authoritative entry)
                phase_log.append({
                    "round": round_num,
                    "phase": "NIGHT_KILL",
                    "actor": wolf.name,
                    "target": victim.name
                })

        # ─── POST-ACT: re-encode to get z_{t+1} and append rollouts ───
        z_deltas = []
        cos_deltas = []
        # NEW: per-agent Δz to fill back into their most recent rows
        dz_by_agent: Dict[str, float] = {}
        cos_by_agent: Dict[str, float] = {}

        for ag in agents:
            if ag.name in pending:
                z_next = ag.encode_current_belief(round_num + 1, agents).detach()
                z_t, ph_code, payload_idx, role = pending[ag.name]
                # Phase-aware rollout tuple (ensure scalars for phase & payload)
                rollout.append((z_t, torch.tensor(int(ph_code)), torch.tensor(int(payload_idx)), z_next, role))
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

            # NEW: optionally backfill Δz into kill rows (actor's change)
            if LOG_DZ_KILL:
                for row in reversed(metrics_rows):
                    if row["round"] != round_num:
                        break
                    if row["phase"] == "NIGHT_KILL":
                        ag_name = row["agent"]  # the wolf who acted
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
