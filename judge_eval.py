# judge_eval.py
# Bias & stability harness for LLM-as-Judge (Phase 7 metrics, strict-vote optional)
# - Loads a calibration set of items: JSONL with {"context","role","candidate", ["gold":0..1], ["vote_candidates":[...]]}
# - Two modes:
#     (A) Rubric mode (default): uses judge.score_batch(...) on candidate text
#     (B) Strict-vote mode (--strict or presence of vote_candidates): uses judge.strict_vote_decision(...)
#         • exact JSON keys: vote_target, confidence, rationale
#         • one retry with terse reminder, then fail-fast
#         • per-call records include strict_ok and retry_count
# - Applies perturbations (order shuffle, truncation, padding) to each base item
#   (strict-vote mode perturbs ONLY context; candidate list stays verbatim)
# - Computes, then writes JSON and CSV:
#     * consistency@eps (rubric: score deltas; strict: confidence deltas)
#     * per-item variance, and global variance
#     * length-bias index (Pearson r between token length and score/confidence)
#     * optional agreement with gold labels (MSE, and Spearman, if "gold" present; rubric only)
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

from judge import JudgeRubric, score_batch, strict_vote_decision

# ------------------------------- config-aware logs dir ---------------------

try:
    import yaml
    with open("config.yaml", "r") as _f:
        _CFG = yaml.safe_load(_f) or {}
except Exception:
    _CFG = {}

def _logs_dir_from_cfg(default: str = "logs") -> str:
    lg = _CFG.get("logging", {}) if isinstance(_CFG.get("logging", {}), dict) else {}
    return lg.get("dir", default) or default

# Pick rubric default from config if available, else fall back to repo root file.
RUBRIC_DEFAULT = (
    (_CFG.get("JUDGE_RUBRIC_PATH") if isinstance(_CFG, dict) else None)
    or (_CFG.get("RUBRIC_PATH") if isinstance(_CFG, dict) else None)
    or "judge_rubric.yaml"
)

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
    vote_candidates: Optional[List[str]] = None  # when provided, switches to strict-vote flow

def _split_context_lines(ctx: str) -> List[str]:
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

def make_perturbations(base: EvalItem, seeds: List[int], *, strict_vote: bool) -> List[Perturbed]:
    outs: List[Perturbed] = []
    for s in seeds:
        outs.append(Perturbed("order", s, _shuffle_order(base.context, s), base.role, base.candidate))
        for frac in (0.75, 0.5):
            outs.append(Perturbed(f"truncate_{int(frac*100)}", s, _truncate_context(base.context, frac), base.role, base.candidate))
        outs.append(Perturbed("pad_ctx", s, _pad_context(base.context, pad_reps=2), base.role, base.candidate))
        if not strict_vote:
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

def _len_tokens(text: str) -> int:
    return len((text or "").split())

_SUB_KEYS = ("coherence", "truthfulness", "role_alignment", "social_safety")

def _safe_subscores(d: Dict[str, Any]) -> Dict[str, float]:
    out = {}
    src = d if isinstance(d, dict) else {}
    for k in _SUB_KEYS:
        try:
            v = float(src.get(k, 0.0))
        except Exception:
            v = 0.0
        out[k] = max(0.0, min(1.0, v))
    return out

def _is_strict_mode(items: List[EvalItem], cli_force: bool) -> bool:
    if cli_force:
        return True
    return any(it.vote_candidates for it in items)

def _variance(xs: List[float]) -> float:
    if not xs:
        return 0.0
    arr = np.array(xs, dtype=float)
    return float(np.var(arr))

def _consistency_at_eps(vals: List[float], eps: float) -> float:
    """
    Fraction of pairwise deltas within eps.
    """
    if len(vals) < 2:
        return 1.0
    n = 0
    ok = 0
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            n += 1
            if abs(vals[i] - vals[j]) <= eps:
                ok += 1
    return float(ok / n) if n else 1.0

# --------------------------- Evaluation Core ------------------------------

def evaluate_set(
    rubric: JudgeRubric,
    items: List[EvalItem],
    *,
    seeds_per_item: int = 3,
    eps_consistency: float = 0.05,
    strict_vote: bool = False,
) -> Dict[str, Any]:
    rng = random.Random(1337)
    all_records: List[Dict[str, Any]] = []
    batch_inputs: List[Dict[str, str]] = []
    meta_index: List[Tuple[int, Optional[str], Dict[str, Any]]] = []

    expanded: List[Tuple[EvalItem, List[Perturbed]]] = []
    for it in items:
        seeds = [rng.randint(0, 1_000_000) for _ in range(seeds_per_item)]
        pert = make_perturbations(it, seeds, strict_vote=strict_vote)
        expanded.append((it, pert))

    # --- Strict-vote path ---
    if strict_vote:
        # Per-item aggregates
        per_item_conf_vals: List[List[float]] = []
        per_item_ctx_lens: List[List[int]] = []

        for bi, (base, perts) in enumerate(expanded):
            # base
            idx, base_rec = strict_vote_decision(
                context=base.context,
                role=base.role,
                candidates=base.vote_candidates or [],
                run_id="eval",
                round_num=-1,
                phase="EVAL",
                agent="__eval__",
            )
            base_rec = {
                "item_idx": bi,
                "tag": "base",
                "context": base.context,
                "role": base.role,
                "candidate": base.candidate,
                **base_rec,
            }
            all_records.append(base_rec)
            confs = [float(base_rec.get("confidence"))] if base_rec.get("confidence") is not None else []
            ctxlens = [_len_tokens(base.context)]

            # perts
            for p in perts:
                _idx, rec = strict_vote_decision(
                    context=p.context,
                    role=p.role,
                    candidates=base.vote_candidates or [],
                    run_id="eval",
                    round_num=-1,
                    phase="EVAL",
                    agent="__eval__",
                )
                rec = {
                    "item_idx": bi,
                    "tag": p.kind,
                    "context": p.context,
                    "role": p.role,
                    "candidate": base.candidate,
                    **rec,
                }
                all_records.append(rec)
                if rec.get("confidence") is not None:
                    confs.append(float(rec["confidence"]))
                ctxlens.append(_len_tokens(p.context))

            per_item_conf_vals.append(confs)
            per_item_ctx_lens.append(ctxlens)

        # Metrics
        # Pair context length and confidence only where confidence exists.
        pairs = [
            (_len_tokens(r.get("context", "")), float(r["confidence"]))
            for r in all_records
            if r.get("confidence") is not None
        ]
        if pairs:
            flat_ctx_len = [p[0] for p in pairs]
            flat_conf = [p[1] for p in pairs]
            conf_len_r = _pearsonr(flat_ctx_len, flat_conf)
        else:
            flat_ctx_len = []
            flat_conf = []
            conf_len_r = 0.0

        item_variances = [_variance(vs) for vs in per_item_conf_vals if vs]
        item_consistency = [_consistency_at_eps(vs, eps_consistency) for vs in per_item_conf_vals if vs]

        redo_counts = [r.get("retry_count", 0) for r in all_records if isinstance(r.get("retry_count"), (int, float))]
        strict_oks = [r.get("strict_ok", 0) for r in all_records]
        talk_vote_matches = [r.get("talk_vote_match", 0) for r in all_records]
        redo_p95 = float(np.percentile(redo_counts, 95)) if redo_counts else 0.0
        strict_ok_rate = float(np.mean(strict_oks)) if strict_oks else 0.0
        vote_align_rate = float(np.mean(talk_vote_matches)) if talk_vote_matches else 0.0

        return {
            "mode": "strict_vote",
            "summary": {
                "n_items": len(items),
                "redo_p95": round(redo_p95, 3),
                "strict_ok_rate": round(strict_ok_rate, 3),
                "vote_alignment_rate": round(vote_align_rate, 3),
                "consistency_at_eps": round(float(np.mean(item_consistency)) if item_consistency else 0.0, 3),
                "per_item_var_mean": round(float(np.mean(item_variances)) if item_variances else 0.0, 3),
                "global_var": round(_variance(flat_conf), 3) if flat_conf else 0.0,
                "length_bias_r": round(conf_len_r, 3),
                "ask_rate": None,
                "name_mention_rate": None,
            },
            "all_records": all_records,
        }

    # --- Rubric (language) path ---
    # Build batches, and keep metadata so we can tie outputs back to inputs
    for bi, (base, perts) in enumerate(expanded):
        batch_inputs.append({"context": base.context, "role": base.role, "candidate": base.candidate})
        meta_index.append((bi, "base", {"context": base.context, "role": base.role, "candidate": base.candidate}))
        for p in perts:
            batch_inputs.append({"context": p.context, "role": p.role, "candidate": p.candidate})
            meta_index.append((bi, p.kind, {"context": p.context, "role": p.role, "candidate": p.candidate}))

    scores = score_batch(batch_inputs, rubric)

    # Collect records with metadata
    for (bi, tag, meta), sc in zip(meta_index, scores):
        rec = {
            "item_idx": bi,
            "tag": tag or "base",
            **meta,
            **sc,
        }
        all_records.append(rec)

    # --- Metrics for rubric mode ---
    # Per-item score collections
    per_item_scores: Dict[int, List[float]] = {}
    per_item_cand_lens: Dict[int, List[int]] = {}
    gold_scores: List[float] = []
    pred_scores_for_gold: List[float] = []

    for r in all_records:
        bi = int(r["item_idx"])
        sc = r.get("score")
        if sc is not None:
            per_item_scores.setdefault(bi, []).append(float(sc))
            cand_len = _len_tokens(r.get("candidate", ""))
            per_item_cand_lens.setdefault(bi, []).append(cand_len)

    # Flatten for global metrics
    flat_scores = [s for arr in per_item_scores.values() for s in arr]
    flat_cand_len = [l for arr in per_item_cand_lens.values() for l in arr]

    # Consistency, and variance
    item_consistency = [_consistency_at_eps(vs, eps_consistency) for vs in per_item_scores.values() if vs]
    item_variances = [_variance(vs) for vs in per_item_scores.values() if vs]
    global_var = _variance(flat_scores)

    # Length-bias
    length_bias_r = _pearsonr(flat_cand_len, flat_scores) if flat_scores and flat_cand_len else 0.0

    # Agreement with gold, if provided
    # We use the base item score when available
    for i, (it, _perts) in enumerate(expanded):
        if it.gold is None:
            continue
        # pick base record
        base_recs = [r for r in all_records if r["item_idx"] == i and r["tag"] == "base" and r.get("score") is not None]
        if not base_recs:
            continue
        pred_scores_for_gold.append(float(base_recs[0]["score"]))
        try:
            gold_scores.append(float(it.gold))
        except Exception:
            continue

    gold_mse = _mse(pred_scores_for_gold, gold_scores) if gold_scores and pred_scores_for_gold else 0.0
    gold_spearman = _spearmanr(gold_scores, pred_scores_for_gold) if gold_scores and pred_scores_for_gold else 0.0

    # Language-pattern summaries
    texts = [r.get("raw_text", "") or r.get("candidate", "") for r in all_records]
    ask_rate = float(np.mean([("?" in t) for t in texts])) if texts else 0.0
    name_mentions = [r.get("name_mentioned", 0) for r in all_records if "name_mentioned" in r]
    name_mention_rate = float(np.mean(name_mentions)) if name_mentions else 0.0
    vote_matches = [r.get("talk_vote_match", 0) for r in all_records if "talk_vote_match" in r]
    vote_alignment_rate = float(np.mean(vote_matches)) if vote_matches else 0.0
    redo_counts = [r.get("retry_count", 0) for r in all_records if isinstance(r.get("retry_count"), (int, float))]
    redo_p95 = float(np.percentile(redo_counts, 95)) if redo_counts else 0.0
    strict_oks = [r.get("strict_ok", 0) for r in all_records]
    strict_ok_rate = float(np.mean(strict_oks)) if strict_oks else 0.0

    return {
        "mode": "rubric",
        "summary": {
            "n_items": len(items),
            "ask_rate": round(ask_rate, 3),
            "name_mention_rate": round(name_mention_rate, 3),
            "vote_alignment_rate": round(vote_alignment_rate, 3),
            "redo_p95": round(redo_p95, 3),
            "strict_ok_rate": round(strict_ok_rate, 3),
            "consistency_at_eps": round(float(np.mean(item_consistency)) if item_consistency else 0.0, 3),
            "per_item_var_mean": round(float(np.mean(item_variances)) if item_variances else 0.0, 3),
            "global_var": round(global_var, 3),
            "length_bias_r": round(length_bias_r, 3),
            "gold_mse": round(gold_mse, 4),
            "gold_spearman": round(gold_spearman, 3),
        },
        "all_records": all_records,
    }

# ------------------------------- CSV writer --------------------------------

def _write_csv(records: List[Dict[str, Any]], path: str) -> None:
    if not records:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return
    # Collect stable field order
    core = [
        "item_idx", "tag", "role", "vote_target", "confidence", "score",
        "rationale", "strict_ok", "retry_count", "redo_count",
        "name_mentioned", "talk_vote_match", "specificity", "repetition_penalty",
    ]
    aux = [
        "context", "candidate", "raw_text", "json", "error",
    ]
    # Include subscores if present
    sub_keys = set()
    for r in records:
        if isinstance(r.get("subscores"), dict):
            sub_keys.update(r["subscores"].keys())
    sub_fields = [f"sub_{k}" for k in sorted(sub_keys)]

    # Build rows
    rows = []
    for r in records:
        row = {k: r.get(k) for k in core}
        row.update({k: r.get(k) for k in aux})
        # flatten subscores
        if isinstance(r.get("subscores"), dict):
            for k in sub_keys:
                row[f"sub_{k}"] = r["subscores"].get(k)
        rows.append(row)

    # Final header
    header = core + aux + sub_fields

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

# ------------------------------- CLI entry --------------------------------

def main():
    ap = argparse.ArgumentParser(description="Judge bias, and stability evaluation")
    ap.add_argument("--rubric", default=RUBRIC_DEFAULT)
    ap.add_argument("--examples", required=False)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    if args.examples and os.path.exists(args.examples):
        raw = _read_jsonl(args.examples)
        items = [EvalItem(
            context=r.get("context",""),
            role=r.get("role","Worker"),
            candidate=r.get("candidate",""),
            gold=r.get("gold", None),
            vote_candidates=r.get("vote_candidates", None),
        ) for r in raw]
    else:
        items = [EvalItem(
            context="- Agent_1: Hello\n- Agent_2: I suspect Agent_3",
            role="Worker",
            candidate="Agent_3 seems off to me?",
        )]

    strict_mode = _is_strict_mode(items, args.strict)
    rubric = JudgeRubric.load(args.rubric) if not strict_mode else JudgeRubric(criteria={"coherence":{"w":1.0,"def":""}})
    report = evaluate_set(rubric, items, seeds_per_item=args.seeds, eps_consistency=args.eps, strict_vote=strict_mode)

    out_dir = _ensure_logs_dir()
    out_json = os.path.join(out_dir, "judge_eval_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    out_csv = os.path.join(out_dir, "judge_eval_records.csv")
    _write_csv(report.get("all_records", []), out_csv)

    print("\n=== Judge Eval Summary ===")
    for k, v in report["summary"].items():
        print(f"{k}: {v}")
    print(f"\nSaved JSON -> {out_json}")
    print(f"Saved CSV -> {out_csv}")

if __name__ == "__main__":
    main()
