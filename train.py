# train.py ── offline JEPA + optional speaker learning (Phase-1 stabilization/logging)
# -----------------------------------------------------------------------------
# What this adds (no behavior change):
#   • Determinism: set_global_determinism(seed) at start
#   • Run ID + config snapshot: logs/<RUN_ID>/config.snapshot.yaml
#   • Per-epoch CSV logging: logs/metrics_train.csv (MSE/BC/|grad|/lr/role/epoch)
#   • Integrity summary JSON: logs/<RUN_ID>/run_summary.json
#   • Console integrity print: rollout counts + Δz stats per role
# -----------------------------------------------------------------------------

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Any

import torch, yaml

# Make project root importable when train.py is launched directly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from roles import WEREWOLF, VILLAGER  # noqa: E402
from training_utils import (         # noqa: E402
    load_role_models,
    run_sim_and_collect_rollouts,
    train_jepa,
    TrainingEpochLogger,
    set_global_determinism,
    save_run_config,
)
from judge import score_batch, JudgeRubric  # noqa: E402

# ── Load config
with open("config.yaml", "r") as f:
    CFG = yaml.safe_load(f) or {}

# ── Hyper-parameters
N_GAMES: int = int(CFG.get("N_GAMES", 50))  # per-role

# ── Paths
CHECKPOINT_DIR = str(CFG.get("CHECKPOINT_DIR", "checkpoints"))
Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
LOGS_DIR = "logs"
Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)

# ── Toggles and judge config
SPEAKER_ENABLED: bool = bool(CFG.get("SPEAKER_ENABLED", False))
JUDGE_RUBRIC_PATH: str = str(CFG.get("JUDGE_RUBRIC_PATH", "judge_rubric.yaml"))

# ── Seed / determinism
RUN_SEED: int = int(CFG.get("RUN_SEED", 1337))

# ─────────────────────────────── helpers: speaker reward / training
def _weights_from_cfg(section: str, default: dict) -> dict:
    w = CFG.get(section, default)
    return {k: float(v) for k, v in w.items()}

VILLAGER_W = _weights_from_cfg("VILLAGER_W", {"truthfulness": 0.5, "coherence": 0.3, "social_safety": 0.2})
WEREWOLF_W = _weights_from_cfg("WEREWOLF_W", {"truthfulness": -0.3, "coherence": 0.5, "social_safety": 0.2})

def _role_reward(subs: dict, role: str, persona_effects: dict | None = None) -> float:
    """Compute a simple role-conditioned reward from judge subscores + tiny persona nudge."""
    role_l = (role or "").lower()
    is_vill = role_l.startswith("vill") or role_l.startswith("work")
    W = VILLAGER_W if is_vill else WEREWOLF_W

    r = 0.0
    for k, w in W.items():
        r += float(w) * float(subs.get(k, 0.0))

    if persona_effects:
        r += 0.1 * float(persona_effects.get("coherence_weight_bonus", 0.0)) * float(subs.get("coherence", 0.0))

    return max(-1.0, min(1.0, r))

def _train_speakers_from_agents(agents: List[Any], rubric: JudgeRubric) -> None:
    """Score pending messages with Judge, assign rewards, run a REINFORCE step per agent speaker."""
    if not SPEAKER_ENABLED or not agents:
        return

    items, ptrs = [], []  # ptrs keep (agent, idx_in_buffer)
    for ag in agents:
        if not hasattr(ag, "msg_buffer") or not ag.msg_buffer:
            continue
        for i, m in enumerate(ag.msg_buffer):
            if m.get("reward", None) is None:
                items.append({"context": "", "role": ag.role or "Unknown", "candidate": m.get("text", "")})
                ptrs.append((ag, i))

    if not items:
        return

    results = score_batch(items, rubric)

    for (ag, i), res in zip(ptrs, results):
        subs = res.get("subscores", {}) if isinstance(res, dict) else {}
        persona_effects = getattr(ag, "persona_effects", None)
        R = _role_reward(subs, ag.role or "Unknown", persona_effects)
        ag.msg_buffer[i]["reward"] = float(R)

    for ag in agents:
        if not getattr(ag, "speaker", None) or not getattr(ag, "speaker_opt", None):
            continue
        batch = [m for m in ag.msg_buffer if m.get("reward") is not None]
        if not batch:
            continue
        stats = ag.speaker.learn_step(batch, ag.speaker_opt, entropy_bonus=0.01, baseline=0.0)
        ag.msg_buffer.clear()
        print(f"[SPEAKER] {ag.name} loss={stats['loss']:.4f} ent={stats['entropy']:.3f} R={stats['R_mean']:.3f}")

# ─────────────────────────────── rollout collection (unchanged API)
def collect_rollouts_for_role(
    role: str,
    n_games: int,
    rubric: JudgeRubric | None = None,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]]:
    """
    Run `n_games` simulations and grab only the rollout tuples whose *actor*
    has `role == role`. If the simulator returns agents in meta, also trains speakers.
    """
    all_rollouts: list = []

    for _ in range(n_games):
        sim_ret = run_sim_and_collect_rollouts(visual=False)

        if isinstance(sim_ret, tuple):
            rollouts, meta = sim_ret
        else:
            rollouts, meta = sim_ret, {}

        if SPEAKER_ENABLED and rubric is not None:
            agents = meta.get("agents") if isinstance(meta, dict) else None
            if agents:
                _train_speakers_from_agents(agents, rubric)

        all_rollouts.extend(r for r in rollouts if r[3] == role)

    return all_rollouts

# ─────────────────────────────── integrity helpers
def _delta_stats(rollouts: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]]) -> dict:
    if not rollouts:
        return {"count": 0, "mean_L2": 0.0, "mean_1mcos": 0.0}
    import torch.nn.functional as F
    l2s, one_minus_cos = [], []
    for z_t, _a, z_next, _role in rollouts:
        d = (z_next - z_t).norm().item()
        l2s.append(d)
        c = float(1.0 - F.cosine_similarity(z_next.unsqueeze(0), z_t.unsqueeze(0)).item())
        one_minus_cos.append(c)
    return {
        "count": len(rollouts),
        "mean_L2": float(sum(l2s) / max(1, len(l2s))),
        "mean_1mcos": float(sum(one_minus_cos) / max(1, len(one_minus_cos))),
    }

# ─────────────────────────────── main training routine
def main() -> None:
    # 0) Determinism & run id
    set_global_determinism(RUN_SEED)
    run_id = f"train_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_seed{RUN_SEED}"
    save_run_config(run_id, CFG)
    epoch_logger = TrainingEpochLogger()

    # 1) Judge rubric (for optional speaker learning)
    rubric = None
    if SPEAKER_ENABLED:
        try:
            rubric = JudgeRubric.load(JUDGE_RUBRIC_PATH)
            print(f"[SPEAKER] Loaded judge rubric: {JUDGE_RUBRIC_PATH}")
        except Exception as e:
            print(f"[SPEAKER] WARNING: failed to load rubric ({e}); speaker learning will be skipped.")
            rubric = None

    # 2) Train per role with integrity prints and epoch CSV logging
    run_summary = {"run_id": run_id, "seed": RUN_SEED, "roles": {}}
    for role_name in (WEREWOLF, VILLAGER):
        world_model, action_encoder, planner = load_role_models(role_name)

        print(f"[JEPA] Simulating {N_GAMES} games for role: {role_name}")
        role_rollouts = collect_rollouts_for_role(role_name, N_GAMES, rubric=rubric)
        stats = _delta_stats(role_rollouts)
        print(f"[JEPA] Collected {stats['count']} roll-outs for {role_name} | "
              f"Δz L2={stats['mean_L2']:.4f}  (1-cos)={stats['mean_1mcos']:.4f}")

        print(f"[JEPA] Training JEPA modules for role: {role_name}")
        train_jepa(
            rollout_data=role_rollouts,
            world_model=world_model,
            action_encoder=action_encoder,
            planner=planner,
            role_name=role_name,
            run_id=run_id,
            epoch_logger=epoch_logger,
        )

        run_summary["roles"][role_name] = stats

    # 3) Persist integrity summary
    run_dir = os.path.join(LOGS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)
    print("\n=== Integrity Summary ===")
    print(json.dumps(run_summary, indent=2))
    print("\n[JEPA] All roles trained and checkpoints updated.")

# ─────────────────────────────── CLI entry
if __name__ == "__main__":
    main()
