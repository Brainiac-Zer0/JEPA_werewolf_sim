#!/usr/bin/env python
"""
Baseline-ladder runner (Table 1).

Reproduces the thesis ablation ladder B0..B6 by evaluating the (trained)
simulator under different component toggles and aggregating per-game metrics
with 95% bootstrap confidence intervals.

Because sim.py reads its configuration from environment variables at import
time, each baseline is evaluated in its own subprocess (the orchestrator sets
the env, the worker imports sim fresh and runs the games).

Usage
-----
  # Evaluate the whole ladder (train checkpoints first with train.py):
  python run_baseline_ladder.py --games 450 --seeds 1337,2718,3141

  # Quick smoke:
  python run_baseline_ladder.py --games 6 --seeds 1337 --baselines B6_random,B0_full

Outputs (under logs/baselines/):
  <Bx>/per_game.csv, <Bx>/winners.jsonl, <Bx>/summary.json
  baseline_summary.csv          # Table 1
"""
from __future__ import annotations
import argparse, csv, json, os, subprocess, sys
from pathlib import Path

# ----------------------------------------------------------------------------
# Baseline -> component toggle mapping (thesis ladder).
#   AGENTSIM_POLICY: "" = full planner (factorized heads); "jepa_only" = JEPA
#   value head w/o planner; "random_voting"/"heuristic" = non-learned voting.
#   USE_LANGUAGE / JUDGE_ENABLED / SOCIAL_ENABLED gate the language, LLM-judge
#   and social-influence pathways respectively.
# ----------------------------------------------------------------------------
# Each baseline carries (a) its EVAL toggles and (b) `ckpt`: the checkpoint namespace
# holding a model TRAINED with exactly this baseline's components (retrain-per-condition,
# so no baseline is evaluated on a model that never saw its components — the fair design).
BASELINES: dict[str, dict[str, str]] = {
    "B6_random":        {"AGENTSIM_POLICY": "random_voting", "USE_LANGUAGE": "0", "JUDGE_ENABLED": "0", "SOCIAL_ENABLED": "0", "ckpt": "ck_base"},
    "B5_heuristic":     {"AGENTSIM_POLICY": "heuristic",     "USE_LANGUAGE": "0", "JUDGE_ENABLED": "0", "SOCIAL_ENABLED": "0", "ckpt": "ck_base"},
    "B4_llm_only":      {"AGENTSIM_POLICY": "random_voting", "USE_LANGUAGE": "1", "JUDGE_ENABLED": "1", "SOCIAL_ENABLED": "0", "ckpt": "ck_llm"},
    "B3_jepa_only":     {"AGENTSIM_POLICY": "jepa_only",     "USE_LANGUAGE": "0", "JUDGE_ENABLED": "0", "SOCIAL_ENABLED": "0", "ckpt": "ck_base"},
    "B2_jepa_planner":  {"AGENTSIM_POLICY": "",              "USE_LANGUAGE": "0", "JUDGE_ENABLED": "0", "SOCIAL_ENABLED": "0", "ckpt": "ck_base"},
    "B1_planner_social":{"AGENTSIM_POLICY": "",              "USE_LANGUAGE": "0", "JUDGE_ENABLED": "0", "SOCIAL_ENABLED": "1", "ckpt": "ck_social"},
    "B0_full":          {"AGENTSIM_POLICY": "",              "USE_LANGUAGE": "1", "JUDGE_ENABLED": "1", "SOCIAL_ENABLED": "1", "ckpt": "ck_full"},
}
LADDER_ORDER = ["B6_random", "B5_heuristic", "B4_llm_only", "B3_jepa_only",
                "B2_jepa_planner", "B1_planner_social", "B0_full"]

# TRAINING config per checkpoint variant (deduped — 4 models cover the 7 rungs).
# ck_llm / ck_full train with language on (speaker+judge) → need OPENAI_API_KEY.
TRAIN_VARIANTS: dict[str, dict[str, str]] = {
    "ck_base":   {"SOCIAL_ENABLED": "0", "SOCIAL_INTEGRATE": "0", "USE_LANGUAGE": "0", "JUDGE_ENABLED": "0", "speaker": "0"},
    "ck_social": {"SOCIAL_ENABLED": "1", "SOCIAL_INTEGRATE": "1", "USE_LANGUAGE": "0", "JUDGE_ENABLED": "0", "speaker": "0"},
    "ck_llm":    {"SOCIAL_ENABLED": "0", "SOCIAL_INTEGRATE": "0", "USE_LANGUAGE": "1", "JUDGE_ENABLED": "1", "speaker": "1"},
    "ck_full":   {"SOCIAL_ENABLED": "1", "SOCIAL_INTEGRATE": "1", "USE_LANGUAGE": "1", "JUDGE_ENABLED": "1", "speaker": "1"},
}

OUT_ROOT = Path("logs") / "baselines"
CKPT_ROOT = Path("checkpoints_ablation")


def _bootstrap_ci(values, n_boot: int = 2000, seed: int = 12345):
    """Mean and 95% percentile bootstrap CI over finite values."""
    import numpy as np
    v = np.array([x for x in values if x == x], dtype=float)  # drop NaN
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(v.mean())
    if v.size == 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    boots = rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1)
    lo = float(np.percentile(boots, 2.5))
    hi = float(np.percentile(boots, 97.5))
    return mean, lo, hi


# ============================ WORKER ========================================
def run_worker(baseline: str, games: int, seeds: list[int]) -> dict:
    """Runs `games` games split across `seeds`; writes per-baseline artifacts."""
    # Import sim only now: env toggles are already applied by the orchestrator.
    import torch
    import sim
    from training_utils import (
        load_role_models_factorized, load_shared_belief_encoder,
        evaluate_jepa_factorized,
    )
    from roles import VILLAGER

    enc = load_shared_belief_encoder()
    # A world model + planner to score one-step JEPA prediction error (Δz MSE).
    wm, pae, fplanner = load_role_models_factorized(VILLAGER)

    out_dir = OUT_ROOT / baseline
    out_dir.mkdir(parents=True, exist_ok=True)

    per_seed = max(1, games // max(1, len(seeds)))
    rows: list[dict] = []
    winners: list[dict] = []
    by_seed: dict[str, int] = {}

    game_counter = 0
    for s in seeds:
        for g in range(per_seed):
            game_seed = int(s) * 1_000_003 + g * 9973 + 1
            rollouts, meta = sim.simulate_game(visual=False, seed=game_seed)
            oc = meta.get("outcome", {}) if isinstance(meta, dict) else {}
            # One-step JEPA prediction error on this game's rollouts.
            try:
                dz = float(evaluate_jepa_factorized(
                    rollouts, wm, pae, fplanner, belief_encoder=enc).get("mse", float("nan")))
            except Exception:
                dz = float("nan")
            run_id = meta.get("run_id", f"{baseline}_s{s}_g{g}")
            rows.append({
                "run_id": run_id,
                "seed": s,
                "villager_win": int(bool(oc.get("villager_win", False))),
                "vill_vote_accuracy": oc.get("vill_vote_accuracy", float("nan")),
                "judge_accept": oc.get("judge_accept", float("nan")),
                "talk_vote_align": oc.get("talk_vote_align", float("nan")),
                "dz_mse": dz,
            })
            winners.append({"run_id": run_id, "winner": oc.get("winner", "unknown")})
            by_seed[str(s)] = by_seed.get(str(s), 0) + 1
            game_counter += 1
            if game_counter % 25 == 0:
                print(f"[{baseline}] {game_counter} games done", flush=True)

    # Write per-game CSV + winners
    with open(out_dir / "per_game.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(out_dir / "winners.jsonl", "w") as f:
        for wj in winners:
            f.write(json.dumps(wj) + "\n")

    # Aggregate with bootstrap CIs
    def col(name):
        return [r[name] for r in rows]
    win_m, win_lo, win_hi = _bootstrap_ci([float(r["villager_win"]) for r in rows])
    va_m, va_lo, va_hi = _bootstrap_ci(col("vill_vote_accuracy"))
    ja_m, ja_lo, ja_hi = _bootstrap_ci(col("judge_accept"))
    tv_m, tv_lo, tv_hi = _bootstrap_ci(col("talk_vote_align"))
    dz_m, dz_lo, dz_hi = _bootstrap_ci(col("dz_mse"))

    summary = {
        "baseline": baseline,
        "games_counted": len(rows),
        "by_seed": by_seed,
        "villager_win_rate_mean": win_m, "villager_win_rate_lo": win_lo, "villager_win_rate_hi": win_hi,
        "vote_accuracy_mean": va_m, "vote_accuracy_lo": va_lo, "vote_accuracy_hi": va_hi,
        "judge_accept_mean": ja_m, "judge_accept_lo": ja_lo, "judge_accept_hi": ja_hi,
        "talk_vote_align_mean": tv_m, "talk_vote_align_lo": tv_lo, "talk_vote_align_hi": tv_hi,
        "dz_mse_mean": dz_m, "dz_mse_lo": dz_lo, "dz_mse_hi": dz_hi,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[{baseline}] DONE  win={win_m:.3f} vote={va_m:.3f} judge={ja_m} dz={dz_m:.5f}", flush=True)
    return summary


# ============================ TRAINING (retrain-per-condition) ==============
def _eval_env_for(b: str) -> dict:
    """Eval toggles for a baseline (BASELINES minus the non-env `ckpt` key)."""
    return {k: v for k, v in BASELINES[b].items() if k != "ckpt"}


def train_variants(baselines: list[str], train_games: int, train_cycles: int,
                   train_epochs: int, seed: int, force: bool):
    """Train the distinct checkpoint variants needed by `baselines`, each with the
    exact components it will be evaluated with (retrain-per-condition). Deduped: the
    4 variants in TRAIN_VARIANTS cover all 7 rungs, so shared bases train once."""
    CKPT_ROOT.mkdir(parents=True, exist_ok=True)
    needed = []
    for ck in dict.fromkeys(BASELINES[b]["ckpt"] for b in baselines):  # ordered-unique
        if ck not in needed:
            needed.append(ck)
    for ck in needed:
        ck_dir = CKPT_ROOT / ck
        tvar = TRAIN_VARIANTS[ck]
        done_marker = ck_dir / "worker_jepa_factorized.pt"
        if done_marker.exists() and not force:
            print(f"\n===== TRAIN[{ck}] SKIP (exists; use --retrain to force) =====", flush=True)
            continue
        env = dict(os.environ)
        env.update({k: str(v) for k, v in tvar.items() if k != "speaker"})
        env["CHECKPOINT_DIR"] = str(ck_dir)
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("PYTHONUTF8", "1")
        lang = tvar.get("USE_LANGUAGE", "0") == "1"
        print(f"\n===== TRAIN[{ck}] -> {ck_dir}  cfg={tvar}  (language={'ON' if lang else 'off'}) =====", flush=True)
        cmd = [sys.executable, os.path.abspath(os.path.join(os.path.dirname(__file__), "train.py")),
               "--mode", "factorized",
               "--games_per_cycle", str(train_games),
               "--outer_cycles", str(train_cycles),
               "--epochs", str(train_epochs),
               "--speaker", str(tvar.get("speaker", "0")),
               "--seed", str(seed)]
        subprocess.run(cmd, env=env, check=True)


# ============================ ORCHESTRATOR ==================================
def run_orchestrator(games: int, seeds: list[int], baselines: list[str], extra_env: dict,
                     retrain: bool = False, train_games: int = 40, train_cycles: int = 2,
                     train_epochs: int = 4, train_seed: int = 1337, force_train: bool = False):
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if retrain:
        train_variants(baselines, train_games, train_cycles, train_epochs, train_seed, force_train)
    summaries = []
    for b in baselines:
        env = dict(os.environ)
        env.update({k: str(v) for k, v in extra_env.items()})
        env.update(_eval_env_for(b))
        if retrain:
            # Evaluate this rung on the model trained with its own components.
            env["CHECKPOINT_DIR"] = str(CKPT_ROOT / BASELINES[b]["ckpt"])
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("PYTHONUTF8", "1")
        ck_note = f"  ckpt={env.get('CHECKPOINT_DIR', 'checkpoints')}"
        print(f"\n===== {b}  toggles={_eval_env_for(b)}{ck_note} =====", flush=True)
        cmd = [sys.executable, os.path.abspath(__file__),
               "--run-one", b, "--games", str(games), "--seeds", ",".join(str(s) for s in seeds)]
        subprocess.run(cmd, env=env, check=True)
        sp = OUT_ROOT / b / "summary.json"
        if sp.exists():
            summaries.append(json.loads(sp.read_text()))

    # Table 1
    if summaries:
        keys = ["baseline", "games_counted",
                "villager_win_rate_mean", "villager_win_rate_lo", "villager_win_rate_hi",
                "vote_accuracy_mean", "vote_accuracy_lo", "vote_accuracy_hi",
                "judge_accept_mean", "judge_accept_lo", "judge_accept_hi",
                "dz_mse_mean", "dz_mse_lo", "dz_mse_hi"]
        with open(OUT_ROOT / "baseline_summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for s in summaries:
                w.writerow({k: s.get(k) for k in keys})
        print(f"\n[LADDER] wrote {OUT_ROOT/'baseline_summary.csv'} ({len(summaries)} baselines)")


def main():
    ap = argparse.ArgumentParser(description="Baseline ladder (Table 1) runner")
    ap.add_argument("--games", type=int, default=450, help="total games per baseline")
    ap.add_argument("--seeds", type=str, default="1337,2718,3141", help="comma-separated seeds")
    ap.add_argument("--baselines", type=str, default=",".join(LADDER_ORDER),
                    help="comma-separated baseline names (default: full ladder)")
    ap.add_argument("--run-one", type=str, default=None, help="(internal) run a single baseline worker")
    ap.add_argument("--retrain", action="store_true",
                    help="retrain-per-condition: train each baseline's own model into a "
                         "namespaced checkpoint dir, then evaluate it against that model")
    ap.add_argument("--force-train", action="store_true", help="retrain even if the checkpoint exists")
    ap.add_argument("--train-games", type=int, default=40, help="games per training cycle")
    ap.add_argument("--train-cycles", type=int, default=2, help="outer training cycles")
    ap.add_argument("--train-epochs", type=int, default=4, help="epochs per cycle")
    ap.add_argument("--train-seed", type=int, default=1337, help="training seed")
    args = ap.parse_args()

    seeds = [int(x) for x in str(args.seeds).split(",") if str(x).strip()]
    if args.run_one:
        run_worker(args.run_one, args.games, seeds)
    else:
        bl = [b.strip() for b in str(args.baselines).split(",") if b.strip() in BASELINES]
        run_orchestrator(args.games, seeds, bl, extra_env={},
                         retrain=args.retrain, train_games=args.train_games,
                         train_cycles=args.train_cycles, train_epochs=args.train_epochs,
                         train_seed=args.train_seed, force_train=args.force_train)


if __name__ == "__main__":
    main()
