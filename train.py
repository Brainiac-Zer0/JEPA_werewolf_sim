# train.py ── offline JEPA + optional speaker learning (Phase-1→4: stabilization → multi-head)
# -----------------------------------------------------------------------------
# Adds/keeps:
#   • Determinism: set_global_determinism(seed) at start
#   • Run ID + config snapshot: logs/<RUN_ID>/config.snapshot.yaml
#   • Per-epoch CSV logging: logs/metrics_train.csv (MSE/BC/|grad|/lr/role/epoch)
#   • Integrity summary JSON: logs/<RUN_ID>/run_summary.json
#   • Accept both rollout schemas (legacy & phase-aware)
#   • Phase-4: routeable training modes → legacy | phase | factorized | auto
#   • NEW: Outer simulate→train cycles, optional mouthpiece (speaker & bias-head) training
#   • NEW (Phase-5): λ-weighted rewards + repetition penalty, post-cycle speaker/bias updates,
#                    mouthpiece save/load.
#   • PATCHES:
#       - Robust unpacking of simulator return (supports 2-tuple, 3-tuple, or dict)
#       - Drop datetime.utcnow() deprecation: use timezone-aware UTC timestamp
#       - Robust unpacking of mouthpiece load (supports 2- or 3-item returns)
#       - Pass through mouthpiece meta to save() when supported
#       - Phase-6: social telemetry print and jsonl logging when meta["social_stats"] is present
# -----------------------------------------------------------------------------

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone  # PATCH: timezone-aware UTC
from typing import List, Tuple, Any, Dict, Optional

import torch, yaml

# Make project root importable when train.py is launched directly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from roles import WEREWOLF, VILLAGER  # noqa: E402
from training_utils import (         # noqa: E402
    load_role_models,
    load_role_models_phase,
    load_role_models_factorized,
    load_shared_belief_encoder,
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
    # Mouthpiece persistence (canonical)
    save_mouthpiece as mp_save,
    load_mouthpiece as mp_load,
)
from judge import score_batch, JudgeRubric  # noqa: E402

# Optional: bandit trainer entrypoint (if provided by your repo)
try:
    from training_utils import train_speaker_bandit as _train_speaker_bandit_api  # type: ignore
except Exception:
    _train_speaker_bandit_api = None

# Speaker mouthpiece classes are optional (only for construction if no checkpoint found).
try:
    from speaker_llm import LogitBiasHead, SpeakerBandit  # noqa: E402
except Exception:
    LogitBiasHead = None
    SpeakerBandit = None

# ── Load config
with open("config.yaml", "r") as f:
    CFG = yaml.safe_load(f) or {}

# Global run-id for auxiliary logging from helpers
RUN_ID: Optional[str] = None

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
N_GAMES: int = _env_int("N_GAMES", int(CFG.get("N_GAMES", 5)))  # per-role

# Training mode & knobs (Phase-4)
TR_CFG = CFG.get("training", {}) if isinstance(CFG.get("training"), dict) else {}
# default to "phase" if PHASE_AWARE_JEPA legacy knob is on; else "legacy"
_mode_default = (TR_CFG.get("mode") or ("phase" if CFG.get("PHASE_AWARE_JEPA", False) else "legacy")).lower()
# Allow TRAIN_MODE or MODE env to override
MODE = _env_str("TRAIN_MODE", _env_str("MODE", _mode_default)).lower()
EPOCHS = _env_int("EPOCHS", int(TR_CFG.get("epochs", 2)))
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

# Reward mixing weights for Phase-5 speaker learning (env overrides YAML)
LAMBDA_J = _env_float("LAMBDA_J", float(CFG.get("speaker", {}).get("lambda_j", 1.0)))
LAMBDA_C = _env_float("LAMBDA_C", float(CFG.get("speaker", {}).get("lambda_c", 0.25)))
LAMBDA_O = _env_float("LAMBDA_O", float(CFG.get("speaker", {}).get("lambda_o", 0.2)))

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

    # ----- Phase-5 reward mix: judge + consistency + outcome + tiny repetition penalty -----
    for (ag, i), res in zip(ptrs, results):
        subs = res.get("subscores", {}) if isinstance(res, dict) else {}
        persona_effects = getattr(ag, "persona_effects", None)

        # 1) Role-conditioned judge reward
        R_judge = _role_reward(subs, ag.role or "Unknown", persona_effects)

        # 2) Consistency (talk→vote alignment) and Outcome (did named target get eliminated?)
        m = ag.msg_buffer[i] if (hasattr(ag, "msg_buffer") and i < len(ag.msg_buffer)) else {}
        align_tv = float(m.get("alignment_vote", 0.0)) if isinstance(m, dict) else 0.0
        elim = 1.0 if (isinstance(m, dict) and m.get("elim")) else 0.0

        # 3) Base reward mix
        R_total = (LAMBDA_J * R_judge) + (LAMBDA_C * align_tv) + (LAMBDA_O * elim)

        # 4) Optional repetition penalty from LLM meta (discourage dull outputs)
        #    Support either direct field or nested under m["meta"]
        rp = 0.0
        try:
            if isinstance(m, dict):
                rp = float(m.get("repetition_penalty", 0.0) or m.get("meta", {}).get("repetition_penalty", 0.0))
        except Exception:
            rp = 0.0
        R_total = R_total - 0.05 * rp  # small deduction

        # Persist reward (with component breakdown)
        ag.msg_buffer[i]["reward"] = float(R_total)
        ag.msg_buffer[i]["reward_components"] = {
            "R_judge": float(R_judge),
            "align_tv": float(align_tv),
            "elim": float(elim),
            "lambda_j": float(LAMBDA_J),
            "lambda_c": float(LAMBDA_C),
            "lambda_o": float(LAMBDA_O),
            "repetition_penalty": float(rp),
            "rp_deduction": float(-0.05 * rp),
        }

    for ag in agents:
        if not getattr(ag, "speaker", None) or not getattr(ag, "speaker_opt", None):
            continue
        batch = [m for m in ag.msg_buffer if m.get("reward") is not None]
        if not batch:
            continue
        # Use the bandit's EMA reward baseline (baseline=None activates it) for
        # REINFORCE variance reduction, per the thesis, instead of a fixed 0.0.
        stats = ag.speaker.learn_step(batch, ag.speaker_opt, entropy_bonus=0.01, baseline=None)

        # Optional: compact component stats to sanity-check variation
        try:
            comps = [m.get("reward_components", {}) for m in batch if isinstance(m, dict)]
            if comps:
                mean_Rj = sum(c.get("R_judge", 0.0) for c in comps) / max(1, len(comps))
                mean_Al = sum(c.get("align_tv", 0.0) for c in comps) / max(1, len(comps))
                mean_Oc = sum(c.get("elim", 0.0) for c in comps) / max(1, len(comps))
                print(f"[SPEAKER] comps: R_j={mean_Rj:.3f} align={mean_Al:.3f} out={mean_Oc:.3f}")
        except Exception:
            pass

        ag.msg_buffer.clear()
        print(f"[SPEAKER] {ag.name} loss={stats['loss']:.4f} ent={stats['entropy']:.3f} R={stats['R_mean']:.3f}")

def _collect_utterance_dataset_from_agents(agents: List[Any]) -> List[Dict[str, Any]]:
    """
    Build a per-utterance dataset from agents' msg_buffer for optional bias-head supervision.
    Each row (dict) may include:
      text, role, talk_intent (optional int), template_id (optional), reward (if already scored)
    """
    ds: List[Dict[str, Any]] = []
    for ag in agents or []:
        buf = getattr(ag, "msg_buffer", None)
        if not buf:
            continue
        for m in buf:
            ds.append({
                "text": m.get("text", ""),
                "role": ag.role or "Unknown",
                "talk_intent": m.get("talk_intent", None),
                "template_id": m.get("template_id", None),
                "reward": m.get("reward", None),
            })
    return ds

# Public alias matching the spec text
def collect_utterance_dataset(agents: List[Any]) -> List[Dict[str, Any]]:
    return _collect_utterance_dataset_from_agents(agents)

def _train_bias_head_on_intents(
    role_name: str,
    agents: List[Any],
    bias_head_for_role: Any | None,
    *,
    lr: float = 1.0e-4,
    epochs: int = 1,
) -> Dict[str, float]:
    """
    Minimal supervised step for a logit-bias head if present.
    Expects bias_head to expose a simple 'train_step' method; otherwise no-ops.
    Returns small stats dict for logging.
    """
    if bias_head_for_role is None or not hasattr(bias_head_for_role, "train_step"):
        return {"count": 0}

    ds = _collect_utterance_dataset_from_agents(agents)
    labeled = [r for r in ds if isinstance(r.get("talk_intent", None), int)]
    if not labeled:
        return {"count": 0}

    texts = [r["text"] for r in labeled]
    intents = [int(r["talk_intent"]) for r in labeled]
    try:
        stats = bias_head_for_role.train_step(texts, intents, lr=lr, epochs=epochs)
    except Exception as e:
        print(f"[SPEAKER/BIAS] {role_name}: bias-head train_step failed: {e}")
        return {"count": 0}

    out = {"count": float(len(labeled))}
    if isinstance(stats, dict):
        out.update({k: float(v) for k, v in stats.items()})
    return out

# Mouthpiece registries (per role), populated lazily
SPEAKER_BY_ROLE: Dict[str, Any] = {}
BIAS_BY_ROLE: Dict[str, Any] = {}
MOUTHPIECE_META_BY_ROLE: Dict[str, Any] = {}  # NEW: keep optional meta payload

def _ensure_mouthpiece_for_role(role_name: str):
    """Create or load optional mouthpiece modules for a role (loads if resuming)."""
    if role_name in SPEAKER_BY_ROLE and role_name in BIAS_BY_ROLE:
        return
    speaker, bias, meta = None, None, None
    # Load checkpoint if present (robust to 2- or 3-item returns)
    try:
        ret = mp_load(role_name, speaker=None, bias_head=None)
        if isinstance(ret, tuple):
            if len(ret) >= 3:
                speaker, bias, meta = ret[:3]
            elif len(ret) == 2:
                speaker, bias = ret
        if meta is not None:
            print(f"[SPEAKER] Loaded mouthpiece for role={role_name} (with meta)")
        elif speaker is not None or bias is not None:
            print(f"[SPEAKER] Loaded mouthpiece for role={role_name}")
    except Exception as e:
        print(f"[SPEAKER] Load mouthpiece failed for role={role_name}: {e}")
        speaker, bias, meta = None, None, None
    # Create missing pieces if classes are available
    if speaker is None and SpeakerBandit is not None:
        try:
            speaker = SpeakerBandit()
            print(f"[SPEAKER] Created new SpeakerBandit for role={role_name}")
        except Exception:
            speaker = None
    if bias is None and LogitBiasHead is not None:
        try:
            bias = LogitBiasHead(latent_dim=int(CFG.get("LATENT_DIM", 32)))
            print(f"[SPEAKER] Created new LogitBiasHead for role={role_name}")
        except Exception:
            bias = None
    SPEAKER_BY_ROLE[role_name] = speaker
    BIAS_BY_ROLE[role_name] = bias
    MOUTHPIECE_META_BY_ROLE[role_name] = meta  # NEW

def _save_mouthpiece_for_role(role_name: str):
    """Persist mouthpiece modules if available; pass through meta when supported."""
    try:
        from inspect import signature
        sig = signature(mp_save)
        kwargs = dict(
            speaker=SPEAKER_BY_ROLE.get(role_name),
            bias_head=BIAS_BY_ROLE.get(role_name),
        )
        if "meta" in sig.parameters:
            kwargs["meta"] = MOUTHPIECE_META_BY_ROLE.get(role_name)
        mp_save(role_name, **kwargs)
    except Exception as e:
        print(f"[SPEAKER] WARNING: failed to save mouthpiece for {role_name}: {e}")

# ============================ Simulator unpacking =============================

def _split_rollouts_meta(sim_ret: Any) -> Tuple[Any, Dict[str, Any]]:
    """
    Robustly extract (rollouts, meta_dict) from simulator return.
    Accepts:
      - (rollouts, meta)
      - (rollouts, meta, extra...)
      - just rollouts
      - dict with keys {"rollouts": ..., ...}
    """
    rollouts, meta = [], {}
    try:
        if isinstance(sim_ret, tuple):
            # 2-tuple or 3+-tuple, first is rollouts, second may be meta
            rollouts = sim_ret[0]
            meta_candidate = sim_ret[1] if len(sim_ret) >= 2 else {}
            meta = meta_candidate if isinstance(meta_candidate, dict) else {}
        elif isinstance(sim_ret, dict):
            # Some simulators may package everything in a dict
            rollouts = sim_ret.get("rollouts", sim_ret.get("data", []))
            meta = sim_ret
        else:
            # Legacy: function returns rollouts directly
            rollouts = sim_ret
    except Exception:
        rollouts, meta = [], {}
    return rollouts, (meta or {})

# ============================ Social stats logging ============================

def _log_social_stats_if_any(meta: Dict[str, Any]) -> None:
    """Best-effort print, and jsonl write for meta['social_stats'].""" 
    try:
        if not isinstance(meta, dict):
            return
        ss = meta.get("social_stats")
        if not ss:
            return
        mean = ss.get("delta_norm_mean", ss.get("mean_norm", 0.0))
        var  = ss.get("delta_norm_var", ss.get("var_norm", 0.0))
        n    = ss.get("n", ss.get("count", 0))
        print(f"[SOCIAL] ||δ_social|| mean={float(mean):.4f}, var={float(var):.4f}, n={int(n)}")
        # Write to logs/<RUN_ID>/social_stats.jsonl (or logs/social_stats.jsonl if run id missing)
        out_dir = os.path.join("logs", RUN_ID) if RUN_ID else "logs"
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "social_stats.jsonl"), "a", encoding="utf-8") as f:
            json.dump(ss, f)
            f.write("\n")
    except Exception:
        pass

# ============================ Progress helpers ================================

_SIM_EXPECTED: int = 0
_SIM_DONE: int = 0
# Base seed for the run; each collected game derives a distinct seed from this so
# that repeated games are independent draws (fixes the identical-games bug where
# every game reset to the same constant seed).
RUN_BASE_SEED: int = 1337

def _game_seed_for(index: int) -> int:
    """Deterministic-but-distinct per-game seed derived from the run base seed."""
    return (int(RUN_BASE_SEED) * 1_000_003 + int(index) * 9973 + 1) % (2**31 - 1)

def _compute_expected_sims(outer_cycles: int, games_per_cycle: int, speaker_enabled: bool) -> int:
    """
    Per cycle accounting:
      core collection = 2 * games_per_cycle  (two roles)
      speaker extras  = 3 per cycle          (per role bias fetch = 2, post-cycle = 1)
    """
    per_cycle = (2 * games_per_cycle) + (3 if speaker_enabled else 0)
    return outer_cycles * per_cycle

def _run_sim_and_count(label: str = "core"):
    """Wrapper that runs one simulation, increments the global counter, and prints progress."""
    global _SIM_DONE, _SIM_EXPECTED
    game_seed = _game_seed_for(_SIM_DONE)
    sim_ret = run_sim_and_collect_rollouts(visual=False, seed=game_seed)
    _SIM_DONE += 1
    try:
        print(f"[PROGRESS] game {_SIM_DONE}/{_SIM_EXPECTED} ({label})")
    except Exception:
        pass
    return sim_ret

# ============================ Rollout collection ==============================

def collect_rollouts_for_role(
    role: str,
    n_games: int,
    rubric: JudgeRubric | None = None,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]]:
    """
    Run `n_games` simulations and grab only the rollout tuples whose actor
    has `role == role`. If the simulator returns agents in meta, also trains speakers.

    Supports both schemas:
      • legacy: (z_t, a_idx, z_next, role)
      • phase:  (z_t, phase_code, payload_idx, z_next, role[, choice_type[, aux]])
    """
    all_rollouts: list = []
    for _ in range(n_games):
        sim_ret = _run_sim_and_count(label=f"core:{role}")
        rollouts, meta = _split_rollouts_meta(sim_ret)
        # Optional social telemetry print, and write
        _log_social_stats_if_any(meta)

        if SPEAKER_ENABLED and rubric is not None:
            agents = meta.get("agents") if isinstance(meta, dict) else None
            if agents:
                _train_speakers_from_agents(agents, rubric)

        for r in rollouts or []:
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

    # NEW: outer cycles + mouthpiece toggles
    p.add_argument("--outer_cycles", type=int,
                   default=_env_int("OUTER_CYCLES", int(TR_CFG.get("outer_cycles", 1))),
                   help="How many simulate→train cycles to run (default 1).")
    p.add_argument("--games_per_cycle", type=int,
                   default=_env_int("GAMES_PER_CYCLE", N_GAMES),
                   help="Number of games per cycle per role (default: config N_GAMES).")
    p.add_argument("--speaker", type=int, choices=[0, 1],
                   default=1 if SPEAKER_ENABLED else 0,
                   help="Enable(1)/disable(0) mouthpiece training for this run.")
    p.add_argument("--speaker_only", action="store_true",
                   help="Only train mouthpiece; skip JEPA modules.")
    return p.parse_args()

def main() -> None:
    # 0) CLI overrides, determinism, and run id
    args = parse_args()
    effective_mode_cfg = (args.mode or MODE).lower()
    n_games = int(args.n_games)
    epochs = int(args.epochs)
    batch_size = int(args.batch_size)
    lr = float(args.lr)
    seed = int(args.seed)

    set_global_determinism(seed)
    global RUN_BASE_SEED
    RUN_BASE_SEED = int(seed)
    # PATCH: timezone-aware UTC
    run_id = f"train_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_seed{seed}"
    save_run_config(run_id, CFG)
    # Expose run id to helpers for social telemetry logging
    global RUN_ID
    RUN_ID = run_id
    epoch_logger = TrainingEpochLogger()

    # 1) Judge rubric, for optional speaker learning
    rubric = None
    if bool(args.speaker):
        try:
            rubric = JudgeRubric.load(JUDGE_RUBRIC_PATH)
            print(f"[SPEAKER] Loaded judge rubric: {JUDGE_RUBRIC_PATH}")
        except Exception as e:
            print(f"[SPEAKER] WARNING: failed to load rubric ({e}); speaker learning will be skipped).")
            rubric = None

    # 2) Orchestrate simulate→train cycles
    run_summary: Dict[str, Any] = {"run_id": run_id, "seed": seed, "roles": {}, "config": {
        "mode": effective_mode_cfg, "epochs": epochs, "batch_size": batch_size, "lr": lr,
        "outer_cycles": int(getattr(args, "outer_cycles", 1)),
        "games_per_cycle": int(getattr(args, "games_per_cycle", n_games)),
        "speaker": int(getattr(args, "speaker", 1 if SPEAKER_ENABLED else 0)),
        "speaker_only": bool(getattr(args, "speaker_only", False)),
        "lambda_j": float(LAMBDA_J),
        "lambda_c": float(LAMBDA_C),
        "lambda_o": float(LAMBDA_O),
    }}

    outer_cycles = int(getattr(args, "outer_cycles", 1))
    games_per_cycle = int(getattr(args, "games_per_cycle", n_games))
    speaker_enabled = bool(getattr(args, "speaker", 1 if SPEAKER_ENABLED else 0))
    speaker_only = bool(getattr(args, "speaker_only", False))

    # Progress announcement
    global _SIM_EXPECTED, _SIM_DONE
    _SIM_DONE = 0
    _SIM_EXPECTED = _compute_expected_sims(outer_cycles, games_per_cycle, speaker_enabled)
    print(f"[PLAN] expected total games = {_SIM_EXPECTED} "
          f"(outer_cycles={outer_cycles}, games_per_cycle={games_per_cycle}, speaker_enabled={int(speaker_enabled)})")

    # Pre-create or load mouthpieces, if available
    for role_name in (WEREWOLF, VILLAGER):
        _ensure_mouthpiece_for_role(role_name)

    # Shared JEPA belief encoder (single representational space across roles/agents).
    # Trained in-place across both roles each cycle so gradients reach the encoder,
    # then persisted to checkpoints/belief_encoder.pt for the simulator to load.
    shared_belief_encoder = load_shared_belief_encoder()

    # For JEPA eval aggregation across cycles
    eval_cache: Dict[str, Dict[str, float]] = {}

    for cyc in range(1, outer_cycles + 1):
        print(f"\n===== CYCLE {cyc}/{outer_cycles} =====")

        for role_name in (WEREWOLF, VILLAGER):
            print(f"[JEPA] Simulating {games_per_cycle} games for role: {role_name}")
            # Speaker REINFORCE happens inside collect if rubric is provided
            effective_rubric = rubric if speaker_enabled else None

            role_rollouts = collect_rollouts_for_role(role_name, games_per_cycle, rubric=effective_rubric)

            if not role_rollouts:
                print(f"[WARN] No rollouts for {role_name} in cycle {cyc}.")
                run_summary["roles"][role_name] = {"overall": {"count": 0}}
                continue

            # Integrity prints
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

            # Choose training path per role
            has_phase_rows = any(len(r) >= 5 for r in role_rollouts)
            effective_mode = effective_mode_cfg
            if effective_mode_cfg == "auto":
                effective_mode = "phase" if has_phase_rows else "legacy"

            # Train JEPA unless speaker-only
            eval_metrics: Dict[str, float] = {}
            if not speaker_only:
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
                        belief_encoder=shared_belief_encoder,
                    )
                    eval_metrics = evaluate_jepa_factorized(role_rollouts, world_model, phase_action_encoder, fplanner, belief_encoder=shared_belief_encoder)

                    # Optional coalition probe, Werewolf only
                    if role_name == WEREWOLF and COAL_COMPARE:
                        print("[COAL] Probe: IndependentKillHeads vs SharedKillHead (see console and CSV for loss trends)")
                        from collections import defaultdict
                        groups = defaultdict(list)
                        for r in role_rollouts:
                            if len(r) >= 7 and isinstance(r[6], dict) and "self_idx" in r[6]:
                                groups[int(r[6]["self_idx"])].append(r)
                        if not groups:
                            print("[COAL] No self_idx in aux; skipping independent probe.")
                        else:
                            for wolf_id, rows in groups.items():
                                wm_i, pae_i, fplanner_i = load_role_models_factorized(role_name)
                                train_jepa_factorized(
                                    rollout_data_phaseaware=rows,
                                    world_model=wm_i,
                                    phase_action_encoder=pae_i,
                                    planner_factorized=fplanner_i,
                                    role_name=f"{role_name}-wolf{wolf_id}",
                                    run_id=run_id,
                                    epoch_logger=None,
                                    epochs=max(1, epochs // 2),
                                    batch_size=min(16, batch_size),
                                    learning_rate=lr,
                                )
                        # Shared already trained above. Use logs to compare.

                else:
                    raise ValueError(f"Unknown training mode: {effective_mode_cfg}")

                if eval_metrics:
                    eval_cache.setdefault(role_name, {})
                    eval_cache[role_name] = eval_metrics
                    print(f"[EVAL] {role_name} ({effective_mode}) → {json.dumps(eval_metrics, indent=2)}")

            # ===== Mouthpiece (speaker + bias-head) =====
            if speaker_enabled:
                # 1) REINFORCE already triggered during rollout via _train_speakers_from_agents
                # 2) Optional supervised bias-head step if present and we have labeled intents
                try:
                    agents_for_bias = None
                    try:
                        # Re-run a tiny sim to get meta-agents for labeled intents, if available
                        sim_ret_extra = _run_sim_and_count(label=f"bias:{role_name}")
                        _rolls_extra, meta_extra = _split_rollouts_meta(sim_ret_extra)
                        _log_social_stats_if_any(meta_extra)
                        agents_for_bias = meta_extra.get("agents", None) if isinstance(meta_extra, dict) else None
                    except Exception:
                        agents_for_bias = None

                    if agents_for_bias:
                        bias_stats = _train_bias_head_on_intents(
                            role_name=role_name,
                            agents=agents_for_bias,
                            bias_head_for_role=BIAS_BY_ROLE.get(role_name),
                            lr=float(CFG.get("BIAS_LR", 1e-4)),
                            epochs=int(CFG.get("BIAS_EPOCHS", 1)),
                        )
                        if bias_stats.get("count", 0) > 0:
                            print(f"[SPEAKER/BIAS] {role_name}: {bias_stats}")
                except Exception as e:
                    print(f"[SPEAKER] Mouthpiece step skipped due to error: {e}")

                # Save mouthpiece pieces if any updated
                _save_mouthpiece_for_role(role_name)

            # record or update summary entry
            role_entry = {"overall": stats}
            if ph_stats:
                role_entry["per_phase"] = ph_stats
            if eval_cache.get(role_name):
                role_entry["eval"] = eval_cache[role_name]
            run_summary["roles"][role_name] = role_entry

        # === Post-cycle speaker dataset and trainers (spec requirement) ===========
        if speaker_enabled:
            try:
                sim_ret = _run_sim_and_count(label="postcycle")
                _rolls_cycle, meta_cycle = _split_rollouts_meta(sim_ret)
                _log_social_stats_if_any(meta_cycle)
                agents_cycle = meta_cycle.get("agents", []) if isinstance(meta_cycle, dict) else []
                if agents_cycle:
                    ds = collect_utterance_dataset(agents_cycle)
                    if _train_speaker_bandit_api is not None and ds:
                        # Signature may vary repo-to-repo, call defensively
                        try:
                            _train_speaker_bandit_api(ds)
                            print(f"[SPEAKER/BANDIT] post-cycle updated on {len(ds)} samples.")
                        except TypeError:
                            _train_speaker_bandit_api(dataset=ds)
                            print(f"[SPEAKER/BANDIT] post-cycle updated on {len(ds)} samples.")
                    # Also refresh bias-head one more time using all agents of the cycle
                    for role_name in (WEREWOLF, VILLAGER):
                        _ = _train_bias_head_on_intents(
                            role_name=role_name,
                            agents=agents_cycle,
                            bias_head_for_role=BIAS_BY_ROLE.get(role_name),
                            lr=float(CFG.get("BIAS_LR", 1e-4)),
                            epochs=int(CFG.get("BIAS_EPOCHS", 1)),
                        )
                    # Persist mouthpieces after the post-cycle updates
                    for role_name in (WEREWOLF, VILLAGER):
                        _save_mouthpiece_for_role(role_name)
            except Exception as e:
                print(f"[SPEAKER] Post-cycle speaker step skipped: {e}")

    # 3) Persist integrity summary at end
    #    Belt and suspenders: save mouthpieces once more after all cycles
    for role_name in (WEREWOLF, VILLAGER):
        try:
            _save_mouthpiece_for_role(role_name)
        except Exception:
            pass

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
