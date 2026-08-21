#!/usr/bin/env python
"""
Sweep runner (Table 2) — reproduces the S1 (social regularization) and
S2 (intent–bias fusion weight) robustness sweeps.

The original driver that produced logs/sweeps/** was never committed; this
recreates it with the same output schema:
  logs/sweeps/<family>/<condition>/per_game.csv
  logs/sweeps/<family>/<condition>/winners.jsonl
  logs/sweeps/<family>/<condition>/summary.json
  logs/sweeps/<family>/<condition>/status.json
  logs/sweeps/sweep_summary.csv

Each condition sets environment variables that sim.py reads at import time, so
conditions run in isolated subprocesses.

  S1 (lambda_reg):        varies the social-correction magnitude (SOCIAL_SCALE,
                          read at eval by load_shared_social); the thesis's
                          lambda_reg is a training-time regularizer with no eval
                          effect, so eval-time robustness is probed via the scale.
  S2 (alpha_fusion):      varies ALPHA_INTENT_BIAS (planner-intent vs bias-head blend).
  S3 (persona_diversity): varies PERSONA_SCALE, the spread of Big-Five personas
                          across agents (0 = homogeneous population). Backs the
                          "personality diversity" claim in the title.

Usage:
  python run_sweeps.py --games 300 --seeds 1337,2718
  python run_sweeps.py --games 6 --seeds 1337 --only alpha_fusion   # smoke
"""
from __future__ import annotations
import argparse, csv, json, os, subprocess, sys
from pathlib import Path

OUT_ROOT = Path("logs") / "sweeps"

# family -> list of (condition_name, env_overrides)
SWEEPS: dict[str, list[tuple[str, dict]]] = {
    "lambda_reg": [
        ("S1_scale0.00", {"SOCIAL_ENABLED": "1", "SOCIAL_SCALE": "0.00", "USE_LANGUAGE": "0", "JUDGE_ENABLED": "0"}),
        ("S1_scale0.15", {"SOCIAL_ENABLED": "1", "SOCIAL_SCALE": "0.15", "USE_LANGUAGE": "0", "JUDGE_ENABLED": "0"}),
        ("S1_scale0.30", {"SOCIAL_ENABLED": "1", "SOCIAL_SCALE": "0.30", "USE_LANGUAGE": "0", "JUDGE_ENABLED": "0"}),
    ],
    "alpha_fusion": [
        ("S2_alpha0.3", {"ALPHA_INTENT_BIAS": "0.3", "USE_LANGUAGE": "1"}),
        ("S2_alpha0.7", {"ALPHA_INTENT_BIAS": "0.7", "USE_LANGUAGE": "1"}),
    ],
    # S3 (personality diversity): PERSONA_SCALE governs how far each agent's
    # Big-Five persona is drawn from neutral (roles.py:_sample_persona samples
    # each trait ~ U[-scale, scale]). scale=0 -> homogeneous agents; larger ->
    # more diverse. Eval-time sensitivity of the full (language-on) system to the
    # spread of personalities in the population. This is the experiment behind the
    # "personality diversity" claim in the title.
    "persona_diversity": [
        ("S3_homogeneous", {"PERSONA_SCALE": "0.0", "USE_LANGUAGE": "1", "JUDGE_ENABLED": "1"}),
        ("S3_moderate",    {"PERSONA_SCALE": "0.2", "USE_LANGUAGE": "1", "JUDGE_ENABLED": "1"}),
        ("S3_diverse",     {"PERSONA_SCALE": "0.4", "USE_LANGUAGE": "1", "JUDGE_ENABLED": "1"}),
    ],
}


def _bootstrap_ci(values, n_boot: int = 2000, seed: int = 12345):
    import numpy as np
    v = np.array([x for x in values if x == x], dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(v.mean())
    if v.size == 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    boots = rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1)
    return mean, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def run_worker(family: str, condition: str, games: int, seeds: list[int]):
    import sim
    from training_utils import (load_role_models_factorized, load_shared_belief_encoder,
                                 evaluate_jepa_factorized)
    from roles import VILLAGER
    enc = load_shared_belief_encoder()
    wm, pae, fplanner = load_role_models_factorized(VILLAGER)

    out_dir = OUT_ROOT / family / condition
    out_dir.mkdir(parents=True, exist_ok=True)
    per_seed = max(1, games // max(1, len(seeds)))
    rows, winners, by_seed = [], [], {}
    for s in seeds:
        for g in range(per_seed):
            game_seed = int(s) * 1_000_003 + g * 9973 + 1
            rollouts, meta = sim.simulate_game(visual=False, seed=game_seed)
            oc = meta.get("outcome", {}) if isinstance(meta, dict) else {}
            try:
                dz = float(evaluate_jepa_factorized(rollouts, wm, pae, fplanner, belief_encoder=enc).get("mse", float("nan")))
            except Exception:
                dz = float("nan")
            rid = meta.get("run_id", f"{condition}_s{s}_g{g}")
            rows.append({"dz_mse": dz, "judge_accept": oc.get("judge_accept", float("nan")),
                         "run_id": rid, "talk_vote_align": oc.get("talk_vote_align", float("nan")),
                         "vill_vote_accuracy": oc.get("vill_vote_accuracy", float("nan")),
                         "villager_win": int(bool(oc.get("villager_win", False)))})
            winners.append({"run_id": rid, "winner": oc.get("winner", "unknown")})
            by_seed[str(s)] = by_seed.get(str(s), 0) + 1

    with open(out_dir / "per_game.csv", "w", newline="") as f:
        cols = ["dz_mse", "judge_accept", "run_id", "talk_vote_align", "vill_vote_accuracy"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    with open(out_dir / "winners.jsonl", "w") as f:
        for wj in winners:
            f.write(json.dumps(wj) + "\n")
    with open(out_dir / "status.json", "w") as f:
        json.dump({"completed_games": len(rows), "by_seed": by_seed,
                   "target_per_seed": per_seed}, f, indent=2)

    def col(n): return [r[n] for r in rows]
    win_m, win_lo, win_hi = _bootstrap_ci([float(r["villager_win"]) for r in rows])
    va_m, va_lo, va_hi = _bootstrap_ci(col("vill_vote_accuracy"))
    ja_m, ja_lo, ja_hi = _bootstrap_ci(col("judge_accept"))
    tv_m, tv_lo, tv_hi = _bootstrap_ci(col("talk_vote_align"))
    dz_m, dz_lo, dz_hi = _bootstrap_ci(col("dz_mse"))
    summary = {"condition": condition, "family": family, "games_counted": len(rows),
               "wins_counted": len(rows), "seeds": ",".join(str(s) for s in seeds),
               "target_per_seed": per_seed,
               "villager_win_rate_mean": win_m, "villager_win_rate_lo": win_lo, "villager_win_rate_hi": win_hi,
               "vote_accuracy_mean": va_m, "vote_accuracy_lo": va_lo, "vote_accuracy_hi": va_hi,
               "judge_accept_mean": ja_m, "judge_accept_lo": ja_lo, "judge_accept_hi": ja_hi,
               "talk_vote_align_mean": tv_m, "talk_vote_align_lo": tv_lo, "talk_vote_align_hi": tv_hi,
               "dz_mse_mean": dz_m, "dz_mse_lo": dz_lo, "dz_mse_hi": dz_hi}
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[{condition}] DONE win={win_m:.3f} vote={va_m:.3f} tv={tv_m:.3f} dz={dz_m:.5f}", flush=True)


def run_orchestrator(games: int, seeds: list[int], families: list[str]):
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summaries = []
    for fam in families:
        for cond, overrides in SWEEPS[fam]:
            env = dict(os.environ)
            env.update({k: str(v) for k, v in overrides.items()})
            env.setdefault("PYTHONIOENCODING", "utf-8"); env.setdefault("PYTHONUTF8", "1")
            print(f"\n===== {fam}/{cond}  {overrides} =====", flush=True)
            subprocess.run([sys.executable, os.path.abspath(__file__),
                            "--run-one", f"{fam}:{cond}", "--games", str(games),
                            "--seeds", ",".join(str(s) for s in seeds)], env=env, check=True)
            sp = OUT_ROOT / fam / cond / "summary.json"
            if sp.exists():
                summaries.append(json.loads(sp.read_text()))
    if summaries:
        keys = ["condition", "family", "games_counted",
                "villager_win_rate_mean", "villager_win_rate_lo", "villager_win_rate_hi",
                "vote_accuracy_mean", "vote_accuracy_lo", "vote_accuracy_hi",
                "talk_vote_align_mean", "talk_vote_align_lo", "talk_vote_align_hi",
                "dz_mse_mean", "dz_mse_lo", "dz_mse_hi"]
        with open(OUT_ROOT / "sweep_summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for s in summaries:
                w.writerow({k: s.get(k) for k in keys})
        print(f"\n[SWEEP] wrote {OUT_ROOT/'sweep_summary.csv'} ({len(summaries)} conditions)")


def main():
    ap = argparse.ArgumentParser(description="Sweep runner (Table 2)")
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--seeds", type=str, default="1337,2718")
    ap.add_argument("--only", type=str, default=None,
                    help="single family: lambda_reg|alpha_fusion|persona_diversity")
    ap.add_argument("--run-one", type=str, default=None, help="(internal) family:condition")
    args = ap.parse_args()
    seeds = [int(x) for x in str(args.seeds).split(",") if str(x).strip()]
    if args.run_one:
        fam, cond = args.run_one.split(":", 1)
        run_worker(fam, cond, args.games, seeds)
    else:
        fams = [args.only] if args.only else list(SWEEPS.keys())
        run_orchestrator(args.games, seeds, fams)


if __name__ == "__main__":
    main()
