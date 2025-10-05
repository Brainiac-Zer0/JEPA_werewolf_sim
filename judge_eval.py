# judge_eval.py
# Bias & stability harness for LLM-as-Judge (Phase 1)
# - Loads a calibration set of items: JSONL with {"context","role","candidate", ["gold":0..1]}
# - Applies perturbations (order shuffle, truncation, padding) to each base item
# - Scores all with judge.score_batch(...) and computes:
#     * consistency@eps (fraction of perturbed scores within eps of base)
#     * per-item variance and global variance
#     * length-bias index (Pearson r between token length and score) per item and overall
#     * optional agreement with gold labels (MSE / Spearman, if "gold" present)
# - Saves results to logs/judge_eval_report.json and logs/judge_eval_records.csv
from __future__ import annotations

import os
import json
import csv
import math
import argparse
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

from judge import JudgeRubric, score_batch

# ------------------------------- config-aware logs dir ---------------------

try:
    import yaml
    with open("config.yaml", "r") as _f:
        _CFG = yaml.safe_load(_f)
except Exception:
    _CFG = {}

def _logs_dir_from_cfg(default: str = "logs") -> str:
    lg = _CFG.get("logging", {}) if isinstance(_CFG.get("logging", {}), dict) else {}
    return lg.get("dir", default)

# ------------------------------- I/O utils --------------------------------

def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    return items

def _ensure_logs_dir() -> str:
    out_dir = _logs_dir_from_cfg("logs")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

# ---------------------------- Perturbation suite --------------------------

@dataclass
class EvalItem:
    context: str
    role: str
    candidate: str
    gold: Optional[float] = None

def _split_context_lines(ctx: str) -> List[str]:
    # Split by newline, keep non-empty
    return [ln.strip() for ln in ctx.splitlines() if ln.strip()]

def _shuffle_order(ctx: str, seed: int) -> str:
    rng = random.Random(seed)
    lines = _split_context_lines(ctx)
    rng.shuffle(lines)
    return "\n".join(lines) if lines else ctx

def _truncate_context(ctx: str, frac: float) -> str:
    lines = _split_context_lines(ctx)
    if not lines:
        return ctx
    k = max(1, int(len(lines) * frac))
    return "\n".join(lines[:k])

def _pad_context(ctx: str, pad_reps: int = 1) -> str:
    # Add neutral filler lines that should not change the content
    pad = ["- (idle)"] * max(0, pad_reps)
    lines = _split_context_lines(ctx) + pad
    return "\n".join(lines)

def _pad_candidate(cand: str, pad_reps: int = 1) -> str:
    if not pad_reps:
        return cand
    return cand + " " + ("." * pad_reps)

@dataclass
class Perturbed:
    kind: str
    seed: int
    context: str
    role: str
    candidate: str

def make_perturbations(base: EvalItem, seeds: List[int]) -> List[Perturbed]:
    outs: List[Perturbed] = []
    for s in seeds:
        # Order shuffle
        outs.append(Perturbed("order", s, _shuffle_order(base.context, s), base.role, base.candidate))
        # Truncations
        for frac in (0.75, 0.5):
            outs.append(Perturbed(f"truncate_{int(frac*100)}", s, _truncate_context(base.context, frac), base.role, base.candidate))
        # Padding (context)
        outs.append(Perturbed("pad_ctx", s, _pad_context(base.context, pad_reps=2), base.role, base.candidate))
        # Padding (candidate length)
        outs.append(Perturbed("pad_cand", s, base.context, base.role, _pad_candidate(base.candidate, pad_reps=3)))
    return outs

# ------------------------------- Metrics ----------------------------------

def _pearsonr(x: List[float], y: List[float]) -> float:
    if len(x) < 2 or len(y) < 2:
        return 0.0
    xv, yv = np.array(x, dtype=float), np.array(y, dtype=float)
    if np.std(xv) == 0 or np.std(yv) == 0:
        return 0.0
    return float(np.corrcoef(xv, yv)[0, 1])

def _spearmanr(x: List[float], y: List[float]) -> float:
    if len(x) < 2 or len(y) < 2:
        return 0.0
    xr = np.argsort(np.argsort(x))
    yr = np.argsort(np.argsort(y))
    return _pearsonr(list(xr), list(yr))

def _mse(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    diff = np.array(a) - np.array(b)
    return float(np.mean(diff * diff))

# --------------------------- Scoring & Evaluation -------------------------

def _len_tokens(text: str) -> int:
    # simple whitespace token length proxy
    return len(text.split())

def evaluate_set(
    rubric: JudgeRubric,
    items: List[EvalItem],
    *,
    seeds_per_item: int = 3,
    eps_consistency: float = 0.05,
) -> Dict[str, Any]:
    """
    For each base item:
      - score base
      - generate perturbations (N seeds)
      - score perturbed
      - compute consistency@eps, variance, length correlation
    """
    rng = random.Random(1337)
    all_records: List[Dict[str, Any]] = []

    # Build full scoring batch for efficiency
    batch_inputs: List[Dict[str, str]] = []
    meta_index: List[Tuple[int, Optional[str]]] = []  # (global_idx, tag) tag=None for base

    expanded: List[Tuple[EvalItem, List[Perturbed]]] = []
    for it in items:
        seeds = [rng.randint(0, 1_000_000) for _ in range(seeds_per_item)]
        pert = make_perturbations(it, seeds)
        expanded.append((it, pert))

    # Prepare batch: base first, then all perts
    for bi, (base, perts) in enumerate(expanded):
        batch_inputs.append({"context": base.context, "role": base.role, "candidate": base.candidate})
        meta_index.append((bi, None))
        for p in perts:
            batch_inputs.append({"context": p.context, "role": p.role, "candidate": p.candidate})
            meta_index.append((bi, p.kind))

    # Score
    scores = score_batch(batch_inputs, rubric)

    # Aggregate per base
    per_item_stats: List[Dict[str, Any]] = []
    cursor = 0
    # build map base/pert indices
    per_base_scores: Dict[int, Dict[str, List[float]]] = {}

    for (bi, tag), sc in zip(meta_index, scores):
        score = float(sc.get("score", 0.0))
        subs = sc.get("subscores", {})
        rec = {
            "item_idx": bi,
            "tag": tag or "base",
            "score": score,
            "coherence": float(subs.get("coherence", 0.0)),
            "truthfulness": float(subs.get("truthfulness", 0.0)),
            "role_alignment": float(subs.get("role_alignment", 0.0)),
            "social_safety": float(subs.get("social_safety", 0.0)),
        }
        all_records.append(rec)

        if bi not in per_base_scores:
            per_base_scores[bi] = {"base": [], "pert": []}
        if tag is None:
            per_base_scores[bi]["base"].append(score)
        else:
            per_base_scores[bi]["pert"].append(score)

    # Compute metrics
    consist_list = []
    var_list = []
    len_corr_list = []
    gold_mse_list = []
    gold_spr_list = []

    # We also keep per-item CSV-friendly rows
    csv_rows: List[Dict[str, Any]] = []

    # length bias: record len tokens for context+candidate
    lengths = []
    scores_for_lengths = []

    idx_offset = 0
    for bi, (base, perts) in enumerate(expanded):
        base_scores = per_base_scores[bi]["base"]
        pert_scores = per_base_scores[bi]["pert"]

        base_score = base_scores[0] if base_scores else 0.0

        # Consistency@eps: fraction of pert scores within eps of base
        cons = 0.0
        if pert_scores:
            close = [abs(ps - base_score) <= eps_consistency for ps in pert_scores]
            cons = float(sum(close)) / len(pert_scores)
        consist_list.append(cons)

        # Variance across perts
        var = float(np.var(np.array(pert_scores))) if pert_scores else 0.0
        var_list.append(var)

        # Length correlations
        lens = [_len_tokens(base.context + " " + base.candidate)]
        scrs = [base_score]
        for p in perts:
            if p.kind == "pad_ctx" or p.kind == "pad_cand" or p.kind.startswith("truncate"):
                lens.append(_len_tokens(p.context + " " + p.candidate))
                # we'll rebuild score alignment below

        # Rebuild aligned lens/scrs arrays
        item_records = [r for r in all_records if r["item_idx"] == bi]
        # base first
        item_base_score = [r["score"] for r in item_records if r["tag"] == "base"][0]
        # map tag to list of scores (preserves multiple seeds)
        tag_to_scores: Dict[str, List[float]] = {}
        for r in item_records:
            if r["tag"] == "base":
                continue
            tag_to_scores.setdefault(r["tag"], []).append(r["score"])

        lens = [_len_tokens(base.context + " " + base.candidate)]
        scrs = [item_base_score]
        for kind in ("order", "truncate_75", "truncate_50", "pad_ctx", "pad_cand"):
            arr = tag_to_scores.get(kind, [])
            for sc in arr:
                # approximate length for this kind using a representative transform
                if kind == "order":
                    L = _len_tokens(base.context + " " + base.candidate)
                elif kind == "truncate_75":
                    L = _len_tokens(_truncate_context(base.context, 0.75) + " " + base.candidate)
                elif kind == "truncate_50":
                    L = _len_tokens(_truncate_context(base.context, 0.5) + " " + base.candidate)
                elif kind == "pad_ctx":
                    L = _len_tokens(_pad_context(base.context, 2) + " " + base.candidate)
                else:  # pad_cand
                    L = _len_tokens(base.context + " " + _pad_candidate(base.candidate, 3))
                lens.append(L)
                scrs.append(sc)

        r_len = _pearsonr(lens, scrs)
        len_corr_list.append(r_len)

        lengths.extend(lens)
        scores_for_lengths.extend(scrs)

        # Gold agreement if present
        if base.gold is not None:
            golds = [float(base.gold)] * len(scrs)
            gold_mse_list.append(_mse(scrs, golds))
            gold_spr_list.append(_spearmanr(scrs, golds))

        # CSV row for this item summary
        csv_rows.append({
            "item_idx": bi,
            "base_score": round(base_score, 4),
            "consistency_at_eps": round(cons, 4),
            "variance": round(var, 6),
            "length_corr": round(r_len, 4),
            "has_gold": int(base.gold is not None),
        })

    overall_len_corr = _pearsonr(lengths, scores_for_lengths) if lengths else 0.0
    mean_consistency = float(np.mean(consist_list)) if consist_list else 0.0
    mean_variance = float(np.mean(var_list)) if var_list else 0.0
    mean_len_corr = float(np.mean(len_corr_list)) if len_corr_list else 0.0
    mean_gold_mse = float(np.mean(gold_mse_list)) if gold_mse_list else None
    mean_gold_spr = float(np.mean(gold_spr_list)) if gold_spr_list else None

    return {
        "per_item_rows": csv_rows,
        "all_records": all_records,  # detailed rows
        "summary": {
            "n_items": len(items),
            "seeds_per_item": seeds_per_item,
            "eps_consistency": eps_consistency,
            "mean_consistency_at_eps": round(mean_consistency, 4),
            "mean_variance": round(mean_variance, 6),
            "mean_length_corr": round(mean_len_corr, 4),
            "overall_length_corr": round(overall_len_corr, 4),
            "mean_gold_mse": None if mean_gold_mse is None else round(mean_gold_mse, 4),
            "mean_gold_spearman": None if mean_gold_spr is None else round(mean_gold_spr, 4),
        },
    }

# ------------------------------- CLI entry --------------------------------

def main():
    ap = argparse.ArgumentParser(description="Judge bias/stability evaluation")
    ap.add_argument("--rubric", default="rubrics/judge_rubric.yaml", help="Rubric YAML/JSON path")
    ap.add_argument(
        "--examples",
        required=False,
        help="Calibration set (JSONL). Lines with: {context, role, candidate, [gold]}"
    )
    ap.add_argument("--seeds", type=int, default=3, help="Seeds per item for perturbations")
    ap.add_argument("--eps", type=float, default=0.05, help="Consistency epsilon")
    ap.add_argument("--out_prefix", default="judge_eval", help="Output file prefix in logs/")
    ap.add_argument("--audit_jsonl", action="store_true",
                    help="If set, write per-call judge audit JSONL alongside report.")
    args = ap.parse_args()

    rubric = JudgeRubric.load(args.rubric)

    # Load examples or use a tiny default set
    if args.examples and os.path.exists(args.examples):
        raw = _read_jsonl(args.examples)
        items = [EvalItem(
            context=r.get("context",""),
            role=r.get("role","Worker"),
            candidate=r.get("candidate",""),
            gold=r.get("gold", None),
        ) for r in raw]
    else:
        # Minimal fallback examples
        fallback = [
            {
                "context": "- Agent_2: I saw Agent_4 near the scene.\n- Agent_4: I was with Agent_1 the whole time.\n- Agent_1: That’s not true.",
                "role": "Worker",
                "candidate": "I think Agent_4 is lying about being with Agent_1.",
                "gold": 0.8,
            },
            {
                "context": "- Agent_3: We should vote quickly.\n- Agent_5: Rushing helps the wolves.\n- Agent_0: Let’s gather more info.",
                "role": "Werewolf",
                "candidate": "We should vote now before they change their minds.",
                "gold": 0.7,
            },
        ]
        items = [EvalItem(**x) for x in fallback]

    # Build and score
    report = evaluate_set(
        rubric=rubric,
        items=items,
        seeds_per_item=args.seeds,
        eps_consistency=args.eps,
    )

    logs_dir = _ensure_logs_dir()
    # JSON report
    json_path = os.path.join(logs_dir, f"{args.out_prefix}_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    # CSV detailed records
    csv_path = os.path.join(logs_dir, f"{args.out_prefix}_records.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["item_idx","tag","score","coherence","truthfulness","role_alignment","social_safety"])
        w.writeheader()
        for r in report["all_records"]:
            w.writerow(r)

    # NEW: optional audit jsonl for calibration runs (reuse judge.audit_judge_calls)
    if args.audit_jsonl:
        from datetime import datetime
        run_id = f"judge_eval_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"
        audit_path = os.path.join(logs_dir, "judge_eval_calls.jsonl")
        try:
            # Re-score in chunks to capture inputs+outputs with shared helper
            from judge import audit_judge_calls
            # Recreate the flat batch used inside evaluate_set
            flat_inputs: List[Dict[str, str]] = []
            rng = random.Random(1337)
            for it in items:
                seeds = [rng.randint(0, 1_000_000) for _ in range(args.seeds)]
                perts = make_perturbations(it, seeds)
                flat_inputs.append({"context": it.context, "role": it.role, "candidate": it.candidate})
                for p in perts:
                    flat_inputs.append({"context": p.context, "role": p.role, "candidate": p.candidate})

            # Score again (deterministic judge) to align lengths safely
            scores = score_batch(flat_inputs, rubric)
            CHUNK = 256
            for i in range(0, len(flat_inputs), CHUNK):
                audit_judge_calls(
                    run_id=run_id,
                    round_num=-1,       # not a sim round; mark as -1
                    phase="EVAL",
                    agent="__eval__",
                    items=flat_inputs[i:i+CHUNK],
                    results=scores[i:i+CHUNK],
                    jsonl_path=audit_path,
                )
        except Exception as e:
            print("[judge_eval] audit_jsonl failed:", e)

    # Console summary
    s = report["summary"]
    print("\n=== Judge Eval Summary ===")
    for k, v in s.items():
        print(f"{k}: {v}")
    print(f"\nSaved JSON → {json_path}")
    print(f"Saved CSV  → {csv_path}")

if __name__ == "__main__":
    main()
