#!/usr/bin/env python3
"""Assess the retrain-per-condition ablation ladder.

Reads logs/baselines/<baseline>/per_game.csv and prints Table 1 plus the three
contrasts the ladder is designed to resolve, each with a bootstrap CI on the
paired difference and a significance verdict:

  * JEPA+planner value : B2_jepa_planner  vs  B6_random
  * Social value       : B1_planner_social vs B2_jepa_planner
  * Language+judge value: B0_full          vs B1_planner_social

Every rung is evaluated on a model trained with exactly its own components
(retrain-per-condition), so these contrasts are apples-to-apples.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT_ROOT = Path("logs") / "baselines"
ORDER = ["B0_full", "B1_planner_social", "B2_jepa_planner", "B3_jepa_only",
         "B4_llm_only", "B5_heuristic", "B6_random"]


def _ci(x, n_boot=5000, seed=12345):
    x = np.asarray([v for v in x if v == v], dtype=float)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    m = float(x.mean())
    if x.size == 1:
        return m, m, m
    rng = np.random.default_rng(seed)
    b = rng.choice(x, size=(n_boot, x.size), replace=True).mean(axis=1)
    return m, float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def _diff_ci(a, b, n_boot=5000, seed=999):
    """Unpaired bootstrap CI on mean(a) - mean(b) (games differ across rungs)."""
    a = np.asarray([v for v in a if v == v], dtype=float)
    b = np.asarray([v for v in b if v == v], dtype=float)
    if a.size == 0 or b.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    da = rng.choice(a, size=(n_boot, a.size), replace=True).mean(axis=1)
    db = rng.choice(b, size=(n_boot, b.size), replace=True).mean(axis=1)
    d = da - db
    return float(a.mean() - b.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float((d > 0).mean())


def _load(b):
    p = OUT_ROOT / b / "per_game.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def main():
    frames = {b: _load(b) for b in ORDER}
    frames = {b: d for b, d in frames.items() if d is not None and len(d)}
    if not frames:
        print(f"No per_game.csv under {OUT_ROOT}. Run the ladder first.", file=sys.stderr)
        sys.exit(1)

    print("\n================= TABLE 1  (retrain-per-condition ablation) =================")
    print(f"{'baseline':20s} {'n':>4s}  {'win_rate [95% CI]':>26s}  {'vote_acc':>9s}  {'dz_mse':>8s}")
    for b in ORDER:
        d = frames.get(b)
        if d is None:
            continue
        wm, wlo, whi = _ci(d['villager_win'].astype(float))
        vm, _, _ = _ci(d['vill_vote_accuracy'].astype(float))
        dm, _, _ = _ci(d['dz_mse'].astype(float)) if 'dz_mse' in d else (float('nan'),)*3
        print(f"{b:20s} {len(d):>4d}  {wm:>6.3f} [{wlo:.3f}, {whi:.3f}]     {vm:>7.4f}  {dm:>8.5f}")

    print("\n================= CONTRASTS (Δ villager win-rate) ==========================")
    contrasts = [
        ("JEPA+planner value", "B2_jepa_planner", "B6_random"),
        ("Social value",       "B1_planner_social", "B2_jepa_planner"),
        ("Language+judge value","B0_full",          "B1_planner_social"),
    ]
    for label, hi, lo in contrasts:
        if hi not in frames or lo not in frames:
            print(f"  {label:22s}: (missing {hi if hi not in frames else lo})")
            continue
        dm, dlo, dhi, pgt = _diff_ci(frames[hi]['villager_win'].astype(float),
                                     frames[lo]['villager_win'].astype(float))
        sig = "SIGNIFICANT" if (dlo > 0 or dhi < 0) else "n.s."
        sign = "helps" if dm > 0 else ("hurts" if dm < 0 else "neutral")
        print(f"  {label:22s}: {hi} - {lo} = {dm:+.3f}  [{dlo:+.3f}, {dhi:+.3f}]  "
              f"P(Δ>0)={pgt:.2f}  -> {sign} ({sig})")
    print("============================================================================\n")


if __name__ == "__main__":
    main()
