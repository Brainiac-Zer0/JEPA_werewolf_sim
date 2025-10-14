# train.py ── offline JEPA + optional speaker learning (Phase-1→3: stabilization → multi-head)
# -----------------------------------------------------------------------------
# Adds/keeps:
#   • Determinism: set_global_determinism(seed) at start
#   • Run ID + config snapshot: logs/<RUN_ID>/config.snapshot.yaml
#   • Per-epoch CSV logging: logs/metrics_train.csv (MSE/BC/|grad|/lr/role/epoch)
#   • Integrity summary JSON: logs/<RUN_ID>/run_summary.json
#   • Accept both rollout schemas (legacy & phase-aware)
#   • Phase-3: routeable training modes → legacy | phase | factorized (multi-head)
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
    load_role_models_phase,
    load_role_models_factorized,   # NEW
    run_sim_and_collect_rollouts,
    train_jepa,
    train_jepa_phaseaware,
    train_jepa_factorized,         # NEW
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

# Training mode & knobs (Phase-3)
TR_CFG = CFG.get("training", {}) if isinstance(CFG.get("training"), dict) else {}
# default to "phase" if PHASE_AWARE_JEPA legacy knob is on; else "legacy"
MODE = (TR_CFG.get("mode") or ("phase" if CFG.get("PHASE_AWARE_JEPA", False) else "legacy")).lower()
EPOCHS = int(TR_CFG.get("epochs", 5))
BATCH_SIZE = int(TR_CFG.get("batch_size", 64))
LR = float(TR_CFG.get("lr", 1.0e-3))

# Optional coalitions knobs (for future shared vs independent kill comparisons)
COAL = CFG.get("coalitions", {}) or {}
COAL_COMPARE = bool(COAL.get("compare", False))
COAL_SHARED_KILL = bool(COAL.get("shared_kill", True))  # baseline = shared

# ── Paths
CHECKPOINT_DIR = str(CFG.get("CHECKPOINT_DIR", "checkpoints"))
Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
LOGS_DIR = "logs"
Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)

# ── Toggles and judge config
SPEAKER_ENABLED: bool = bool(CFG.get("SPEAKER_ENABLED", False))
JUDGE_RUBRIC_PATH: str = str(CFG.get("JUDGE_RUBRIC_PATH", "judge_rubric.yaml"))

# (legacy flags kept for BC; MODE overrides routing)
PHASE_AWARE_JEPA: bool = bool(CFG.get("PHASE_AWARE_JEPA", False))
TRAIN_PHASE_HEADS: bool = bool(CFG.get("TRAIN_PHASE_HEADS", False))  # placeholder (no-op here)

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

# ─────────────────────────────── rollout collection (schema-aware, minimal change)
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

        # Keep both formats, filter by role robustly
        for r in rollouts:
            try:
                if len(r) == 4:
                    if r[3] == role:
                        all_rollouts.append(r)
                elif len(r) >= 5:
                    if r[4] == role:
                        all_rollouts.append(r)
                # else ignore malformed rows silently
            except Exception:
                continue

    return all_rollouts

# ─────────────────────────────── integrity helpers (schema-aware)
def _delta_stats(rollouts: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]]) -> dict:
    if not rollouts:
        return {"count": 0, "mean_L2": 0.0, "mean_1mcos": 0.0}
    import torch.nn.functional as F
    l2s, one_minus_cos = [], []
    for r in rollouts:
        # legacy: (z_t, a_idx, z_next, role) ; phase: (z_t, phase, payload, z_next, role[, ...])
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
    """
    Optional integrity: when phase-format is present, report per-phase Δz.
    Returns {phase_code_int: {'count':..., 'mean_L2':..., 'mean_1mcos':...}, ...}
    """
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
            print(f"[SPEAKER] WARNING: failed to load rubric ({e}); speaker learning will be skipped).")
            rubric = None

    # 2) Train per role with integrity prints and epoch CSV logging
    run_summary = {"run_id": run_id, "seed": RUN_SEED, "roles": {}}
    for role_name in (WEREWOLF, VILLAGER):
        print(f"[JEPA] Simulating {N_GAMES} games for role: {role_name}")
        role_rollouts = collect_rollouts_for_role(role_name, N_GAMES, rubric=rubric)

        stats = _delta_stats(role_rollouts)
        print(f"[JEPA] Collected {stats['count']} roll-outs for {role_name} | "
              f"Δz L2={stats['mean_L2']:.4f}  (1-cos)={stats['mean_1mcos']:.4f}")

        # Optional integrity: per-phase stats if available
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

        # 3) Choose training path based on config & observed schema
        has_phase_rows = any(len(r) >= 5 for r in role_rollouts)
        effective_mode = MODE
        if MODE == "legacy" and has_phase_rows:
            # Auto-upgrade to phase if richer data exists
            effective_mode = "phase"

        print(f"[JEPA] Training JEPA modules for role: {role_name} (mode={effective_mode})")

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
                epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LR,
            )

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
                epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LR,
            )

        elif effective_mode == "factorized":
            world_model, phase_action_encoder, fplanner = load_role_models_factorized(role_name)
            # Baseline: shared heads (FactorizedPlanner). Training includes TALK/VOTE/KILL CE with masks.
            train_jepa_factorized(
                rollout_data_phaseaware=role_rollouts,
                world_model=world_model,
                phase_action_encoder=phase_action_encoder,
                planner_factorized=fplanner,
                role_name=role_name,
                run_id=run_id,
                epoch_logger=epoch_logger,
                epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LR,
            )

            # (Optional) Coalition probe — independent specialists vs shared, Werewolf only.
            if role_name == WEREWOLF and COAL_COMPARE:
                print("[COAL] Probe: IndependentKillHeads vs SharedKillHead (see console/CSV for loss trends)")
                # Minimal probe: split rows per actor if aux.self_idx present and train tiny specialists.
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
                            epochs=max(1, EPOCHS // 5),
                            batch_size=min(16, BATCH_SIZE),
                            learning_rate=LR,
                        )
                # Shared already trained above. Use logs to compare.

        else:
            raise ValueError(f"Unknown training mode: {MODE}")

        # record summary
        role_entry = {"overall": stats}
        if ph_stats:
            role_entry["per_phase"] = ph_stats
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
