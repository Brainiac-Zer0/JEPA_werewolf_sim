# train.py ── offline JEPA + optional speaker learning (Phase-1→4: stabilization → multi-head)
# -----------------------------------------------------------------------------
# Adds/keeps:
#   • Determinism: set_global_determinism(seed) at start
#   • Run ID + config snapshot: logs/<RUN_ID>/config.snapshot.yaml
#   • Per-epoch CSV logging: logs/metrics_train.csv (MSE/BC/|grad|/lr/role/epoch)
#   • Integrity summary JSON: logs/<RUN_ID>/run_summary.json
#   • Accept both rollout schemas (legacy & phase-aware)
#   • Phase-4: routeable training modes → legacy | phase | factorized | auto
#   • New: CLI overrides; post-train evaluation with per-head accuracy & illegal mass
# -----------------------------------------------------------------------------

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Any, Dict

import torch, yaml

# Make project root importable when train.py is launched directly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from roles import WEREWOLF, VILLAGER  # noqa: E402
from training_utils import (         # noqa: E402
    load_role_models,
    load_role_models_phase,
    load_role_models_factorized,
    run_sim_and_collect_rollouts,
    train_jepa,
    train_jepa_phaseaware,
    train_jepa_factorized,
    evaluate_jepa,
    evaluate_jepa_phase,
    evaluate_jepa_factorized,
    TrainingEpochLogger,
    set_global_determinism,
    save_run_config,
)
from judge import score_batch, JudgeRubric  # noqa: E402

# ── Load config
with open("config.yaml", "r") as f:
    CFG = yaml.safe_load(f) or {}

# --------- OS ENV SHIM HELPERS (env overrides YAML; safe parsing) ----------
def _env_bool(key: str, default: bool) -> bool:
    v = os.getenv(key)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")

def _env_int(key: str, default: int) -> int:
    v = os.getenv(key)
    try:
        return int(v) if v is not None else default
    except Exception:
        return default

def _env_float(key: str, default: float) -> float:
    v = os.getenv(key)
    try:
        return float(v) if v is not None else default
    except Exception:
        return default

def _env_str(key: str, default: str) -> str:
    v = os.getenv(key)
    return v if v is not None else default
# --------------------------------------------------------------------------

# ── Hyper-parameters (config defaults; CLI can override; ENV can override both)
N_GAMES: int = _env_int("N_GAMES", int(CFG.get("N_GAMES", 50)))  # per-role

# Training mode & knobs (Phase-4)
TR_CFG = CFG.get("training", {}) if isinstance(CFG.get("training"), dict) else {}
# default to "phase" if PHASE_AWARE_JEPA legacy knob is on; else "legacy"
_mode_default = (TR_CFG.get("mode") or ("phase" if CFG.get("PHASE_AWARE_JEPA", False) else "legacy")).lower()
# Allow TRAIN_MODE or MODE env to override
MODE = _env_str("TRAIN_MODE", _env_str("MODE", _mode_default)).lower()
EPOCHS = _env_int("EPOCHS", int(TR_CFG.get("epochs", 5)))
BATCH_SIZE = _env_int("BATCH_SIZE", int(TR_CFG.get("batch_size", 64)))
LR = _env_float("LR", float(TR_CFG.get("lr", 1.0e-3)))

# Optional coalitions knobs (for shared vs independent kill comparisons)
COAL = CFG.get("coalitions", {}) or {}
COAL_COMPARE = _env_bool("COAL_COMPARE", bool(COAL.get("compare", False)))
COAL_SHARED_KILL = _env_bool("COAL_SHARED_KILL", bool(COAL.get("shared_kill", True)))  # baseline = shared

# ── Paths
CHECKPOINT_DIR = _env_str("CHECKPOINT_DIR", str(CFG.get("CHECKPOINT_DIR", "checkpoints")))
Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
LOGS_DIR = _env_str("LOGS_DIR", "logs")
Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)

# ── Toggles and judge config
SPEAKER_ENABLED: bool = _env_bool("SPEAKER_ENABLED", bool(CFG.get("SPEAKER_ENABLED", False)))
JUDGE_RUBRIC_PATH: str = _env_str("JUDGE_RUBRIC_PATH", str(CFG.get("JUDGE_RUBRIC_PATH", "judge_rubric.yaml")))

# (legacy flags kept for BC; MODE overrides routing)
PHASE_AWARE_JEPA: bool = _env_bool("PHASE_AWARE_JEPA", bool(CFG.get("PHASE_AWARE_JEPA", False)))
TRAIN_PHASE_HEADS: bool = _env_bool("TRAIN_PHASE_HEADS", bool(CFG.get("TRAIN_PHASE_HEADS", False)))  # placeholder

# ── Seed / determinism
RUN_SEED: int = _env_int("RUN_SEED", _env_int("SEED", int(CFG.get("RUN_SEED", 1337))))


# ============================== Speaker helpers ===============================

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


# ============================ Rollout collection ==============================

def collect_rollouts_for_role(
    role: str,
    n_games: int,
    rubric: JudgeRubric | None = None,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]]:
    """
    Run `n_games` simulations and grab only the rollout tuples whose *actor*
    has `role == role`. If the simulator returns agents in meta, also trains speakers.

    Supports both schemas:
      • legacy: (z_t, a_idx, z_next, role)
      • phase:  (z_t, phase_code, payload_idx, z_next, role[, choice_type[, aux]])
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

        for r in rollouts:
            try:
                if len(r) == 4 and r[3] == role:
                    all_rollouts.append(r)
                elif len(r) >= 5 and r[4] == role:
                    all_rollouts.append(r)
            except Exception:
                continue
    return all_rollouts


# ============================== Integrity helpers =============================

def _delta_stats(rollouts: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]]) -> dict:
    if not rollouts:
        return {"count": 0, "mean_L2": 0.0, "mean_1mcos": 0.0}
    import torch.nn.functional as F
    l2s, one_minus_cos = [], []
    for r in rollouts:
        if len(r) == 4:
            z_t, _a, z_next, _role = r
        elif len(r) >= 5:
            z_t, _ph, _pay, z_next, _role = r[:5]
        else:
            continue
        d = (z_next - z_t).norm().item()
        l2s.append(d)
        c = float(1.0 - F.cosine_similarity(z_next.unsqueeze(0), z_t.unsqueeze(0)).item())
        one_minus_cos.append(c)
    return {
        "count": len(l2s),
        "mean_L2": float(sum(l2s) / max(1, len(l2s))),
        "mean_1mcos": float(sum(one_minus_cos) / max(1, len(one_minus_cos))),
    }

def _delta_stats_by_phase(rollouts) -> dict:
    import torch.nn.functional as F
    buckets: dict[int, list] = {}
    for r in rollouts:
        if len(r) >= 5:
            z_t, ph, _pay, z_next, _role = r[:5]
            try:
                key = int(ph)
            except Exception:
                continue
            buckets.setdefault(key, []).append((z_t, z_next))
    out = {}
    for k, pairs in buckets.items():
        if not pairs:
            continue
        l2s, one_minus_cos = [], []
        for z_t, z_next in pairs:
            l2s.append((z_next - z_t).norm().item())
            one_minus_cos.append(float(1.0 - F.cosine_similarity(z_next.unsqueeze(0), z_t.unsqueeze(0)).item()))
        out[k] = {
            "count": len(pairs),
            "mean_L2": float(sum(l2s) / max(1, len(l2s))),
            "mean_1mcos": float(sum(one_minus_cos) / max(1, len(one_minus_cos))),
        }
    return out


# =================================== Main =====================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train JEPA modules (legacy/phase/factorized).")
    p.add_argument("--mode", type=str, default=MODE, choices=["legacy", "phase", "factorized", "auto"],
                   help=f"Training mode (default from config/env: {MODE})")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--n_games", type=int, default=N_GAMES)
    p.add_argument("--seed", type=int, default=RUN_SEED)
    return p.parse_args()

def main() -> None:
    # 0) CLI overrides + determinism & run id
    args = parse_args()
    effective_mode_cfg = (args.mode or MODE).lower()
    n_games = int(args.n_games)
    epochs = int(args.epochs)
    batch_size = int(args.batch_size)
    lr = float(args.lr)
    seed = int(args.seed)

    set_global_determinism(seed)
    run_id = f"train_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_seed{seed}"
    save_run_config(run_id, CFG)
    epoch_logger = TrainingEpochLogger()

    # 1) Judge rubric (for optional speaker learning)
    rubric = None
    if SPEAKER_ENABLED:
        try:
            rubric = JudgeRubric.load(JUDGE_RUBRIC_PATH)
            print(f"[SPEAKER] Loaded judge rubric: {JUDGE_RUBRIC_PATH}")
        except Exception as e:
            print(f"[SPEAKER] WARNING: failed to load rubric ({e}); speaker learning will be skipped).")
            rubric = None

    # 2) Train per role with integrity prints and epoch CSV logging
    run_summary: Dict[str, Any] = {"run_id": run_id, "seed": seed, "roles": {}, "config": {
        "mode": effective_mode_cfg, "epochs": epochs, "batch_size": batch_size, "lr": lr, "n_games": n_games
    }}

    for role_name in (WEREWOLF, VILLAGER):
        print(f"[JEPA] Simulating {n_games} games for role: {role_name}")
        role_rollouts = collect_rollouts_for_role(role_name, n_games, rubric=rubric)

        if not role_rollouts:
            print(f"[WARN] No rollouts for {role_name}. Skipping training for this role.")
            run_summary["roles"][role_name] = {"overall": {"count": 0}}
            continue

        stats = _delta_stats(role_rollouts)
        print(f"[JEPA] Collected {stats['count']} roll-outs for {role_name} | "
              f"Δz L2={stats['mean_L2']:.4f}  (1-cos)={stats['mean_1mcos']:.4f}")

        ph_stats = _delta_stats_by_phase(role_rollouts)
        if ph_stats:
            try:
                pretty = ", ".join(
                    f"phase={k}: n={v['count']} L2={v['mean_L2']:.4f} (1-cos)={v['mean_1mcos']:.4f}"
                    for k, v in sorted(ph_stats.items())
                )
                print(f"[JEPA] Per-phase Δz stats for {role_name} → {pretty}")
            except Exception:
                pass

        # 3) Choose training path (auto-upgrade to 'phase' on richer data)
        has_phase_rows = any(len(r) >= 5 for r in role_rollouts)
        effective_mode = effective_mode_cfg
        if effective_mode_cfg == "auto":
            effective_mode = "phase" if has_phase_rows else "legacy"

        print(f"[JEPA] Training JEPA modules for role: {role_name} (mode={effective_mode})")

        eval_metrics: Dict[str, float] = {}
        if effective_mode == "legacy":
            world_model, action_encoder, planner = load_role_models(role_name)
            train_jepa(
                rollout_data=role_rollouts,
                world_model=world_model,
                action_encoder=action_encoder,
                planner=planner,
                role_name=role_name,
                run_id=run_id,
                epoch_logger=epoch_logger,
                epochs=epochs, batch_size=batch_size, learning_rate=lr,
            )
            eval_metrics = evaluate_jepa(role_rollouts, world_model, action_encoder, planner)

        elif effective_mode == "phase":
            world_model, phase_action_encoder, planner = load_role_models_phase(role_name)
            train_jepa_phaseaware(
                rollout_data_phaseaware=role_rollouts,
                world_model=world_model,
                planner=planner,
                role_name=role_name,
                run_id=run_id,
                epoch_logger=epoch_logger,
                phase_action_encoder=phase_action_encoder,
                epochs=epochs, batch_size=batch_size, learning_rate=lr,
            )
            eval_metrics = evaluate_jepa_phase(role_rollouts, world_model, phase_action_encoder, planner)

        elif effective_mode == "factorized":
            world_model, phase_action_encoder, fplanner = load_role_models_factorized(role_name)
            train_jepa_factorized(
                rollout_data_phaseaware=role_rollouts,
                world_model=world_model,
                phase_action_encoder=phase_action_encoder,
                planner_factorized=fplanner,
                role_name=role_name,
                run_id=run_id,
                epoch_logger=epoch_logger,
                epochs=epochs, batch_size=batch_size, learning_rate=lr,
            )
            eval_metrics = evaluate_jepa_factorized(role_rollouts, world_model, phase_action_encoder, fplanner)

            # (Optional) Coalition probe — independent specialists vs shared, Werewolf only.
            if role_name == WEREWOLF and COAL_COMPARE:
                print("[COAL] Probe: IndependentKillHeads vs SharedKillHead (see console/CSV for loss trends)")
                from collections import defaultdict
                groups = defaultdict(list)
                for r in role_rollouts:
                    if len(r) >= 7 and isinstance(r[6], dict) and "self_idx" in r[6]:
                        groups[int(r[6]["self_idx"])].append(r)
                if not groups:
                    print("[COAL] No self_idx in aux; skipping independent probe.")
                else:
                    for wolf_id, rows in groups.items():
                        wm_i, pae_i, fplanner_i = load_role_models_factorized(role_name)  # fresh init
                        train_jepa_factorized(
                            rollout_data_phaseaware=rows,
                            world_model=wm_i,
                            phase_action_encoder=pae_i,
                            planner_factorized=fplanner_i,
                            role_name=f"{role_name}-wolf{wolf_id}",
                            run_id=run_id,
                            epoch_logger=None,
                            epochs=max(1, epochs // 5),
                            batch_size=min(16, batch_size),
                            learning_rate=lr,
                        )
                # Shared already trained above. Use logs to compare.

        else:
            raise ValueError(f"Unknown training mode: {effective_mode_cfg}")

        # Log evaluation metrics
        if eval_metrics:
            print(f"[EVAL] {role_name} ({effective_mode}) → {json.dumps(eval_metrics, indent=2)}")

        # record summary
        role_entry = {"overall": stats}
        if ph_stats:
            role_entry["per_phase"] = ph_stats
        if eval_metrics:
            role_entry["eval"] = eval_metrics
        run_summary["roles"][role_name] = role_entry

    # 4) Persist integrity summary
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
