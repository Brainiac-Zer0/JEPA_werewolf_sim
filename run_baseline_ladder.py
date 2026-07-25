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
BASELINES: dict[str, dict[str, str]] = {
    "B6_random":        {"AGENTSIM_POLICY": "random_voting", "USE_LANGUAGE": "0", "JUDGE_ENABLED": "0", "SOCIAL_ENABLED": "0"},
    "B5_heuristic":     {"AGENTSIM_POLICY": "heuristic",     "USE_LANGUAGE": "0", "JUDGE_ENABLED": "0", "SOCIAL_ENABLED": "0"},
    "B4_llm_only":      {"AGENTSIM_POLICY": "random_voting", "USE_LANGUAGE": "1", "JUDGE_ENABLED": "0", "SOCIAL_ENABLED": "0"},
    "B3_jepa_only":     {"AGENTSIM_POLICY": "jepa_only",     "USE_LANGUAGE": "0", "JUDGE_ENABLED": "0", "SOCIAL_ENABLED": "0"},
    "B2_jepa_planner":  {"AGENTSIM_POLICY": "",              "USE_LANGUAGE": "0", "JUDGE_ENABLED": "0", "SOCIAL_ENABLED": "0"},
    "B1_planner_social":{"AGENTSIM_POLICY": "",              "USE_LANGUAGE": "0", "JUDGE_ENABLED": "0", "SOCIAL_ENABLED": "1"},
    "B0_full":          {"AGENTSIM_POLICY": "",              "USE_LANGUAGE": "1", "JUDGE_ENABLED": "1", "SOCIAL_ENABLED": "1"},
}
LADDER_ORDER = ["B6_random", "B5_heuristic", "B4_llm_only", "B3_jepa_only",
                "B2_jepa_planner", "B1_planner_social", "B0_full"]

OUT_ROOT = Path("logs") / "baselines"


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


# ============================ ORCHESTRATOR ==================================
def run_orchestrator(games: int, seeds: list[int], baselines: list[str], extra_env: dict):
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summaries = []
    for b in baselines:
        env = dict(os.environ)
        env.update({k: str(v) for k, v in extra_env.items()})
        env.update(BASELINES[b])
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("PYTHONUTF8", "1")
        print(f"\n===== {b}  toggles={BASELINES[b]} =====", flush=True)
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
    args = ap.parse_args()

    seeds = [int(x) for x in str(args.seeds).split(",") if str(x).strip()]
    if args.run_one:
        run_worker(args.run_one, args.games, seeds)
    else:
        bl = [b.strip() for b in str(args.baselines).split(",") if b.strip() in BASELINES]
        run_orchestrator(args.games, seeds, bl, extra_env={})


if __name__ == "__main__":
    main()
