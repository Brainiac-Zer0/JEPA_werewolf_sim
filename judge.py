# judge.py
from __future__ import annotations
import os, json, math, re
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import torch, yaml
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ────────────── Env ──────────────
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

JUDGE_MODEL_ID = config.get("JUDGE_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")
JUDGE_MAX_NEW = int(config.get("JUDGE_MAX_NEW", 128))
INCLUDE_RATIONALE = bool(config.get("INCLUDE_RATIONALE", True))
ENABLE_PERSONA_STEER = bool(config.get("ENABLE_PERSONA_STEER", False))
PERSONA_CONFIG_PATH = config.get("PERSONA_CONFIG_PATH", "configs/persona_vectors.yaml")
JUDGE_DEVICE = config.get("JUDGE_DEVICE", "").lower()
JUDGE_BATCH = max(1, int(config.get("JUDGE_BATCH", 3)))
JUDGE_DEBUG = bool(config.get("JUDGE_DEBUG", False))
JUDGE_DEBUG_DIR = config.get("JUDGE_DEBUG_DIR", "logs")

# ────────────── NEW: audit logging routed by config.logging (add-only) ──────────────
_LOGGING = config.get("logging", {}) if isinstance(config.get("logging", {}), dict) else {}
JUDGE_AUDIT_JSONL = _LOGGING.get("judge_jsonl", os.path.join(JUDGE_DEBUG_DIR, "judge_calls.jsonl"))
JUDGE_AUDIT_DEBUG = bool(_LOGGING.get("debug", JUDGE_DEBUG))  # prefer config.logging.debug, fallback to JUDGE_DEBUG

def judge_logging_enabled() -> bool:
    """
    Unified check to decide whether to write rich JSONL audit traces.
    Honors config.logging.debug (preferred) and falls back to JUDGE_DEBUG.
    """
    return bool(JUDGE_AUDIT_DEBUG or JUDGE_DEBUG)

def _ensure_parent_dir(path: str):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except Exception:
        pass

def audit_judge_calls(
    *,
    run_id: str,
    round_num: int,
    phase: str,
    agent: str,
    items: List[Dict[str, str]],
    results: List[Dict[str, Any]],
    jsonl_path: Optional[str] = None,
) -> None:
    """
    ADD-ONLY helper: write one JSON line per (item, result) pair.
    Call this AFTER score_batch(...) returns; no changes inside score_batch.

    Each line includes keys that line up with the sim CSV:
      run_id, round, phase, agent, context, role, candidate, subscores, score.
    """
    if not judge_logging_enabled():
        return
    if not items or not results:
        return
    path = jsonl_path or JUDGE_AUDIT_JSONL
    _ensure_parent_dir(path)

    n = min(len(items), len(results))
    try:
        with open(path, "a", encoding="utf-8") as f:
            for i in range(n):
                it  = items[i] or {}
                out = results[i] or {}
                rec = {
                    "run_id": run_id,
                    "round": int(round_num),
                    "phase": str(phase),
                    "agent": str(agent),
                    "context": it.get("context", ""),
                    "role": it.get("role", ""),
                    "candidate": it.get("candidate", ""),
                    "subscores": out.get("subscores", {}),
                    "score": float(out.get("score", 0.0)),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        # Never crash the caller on audit failures
        print("[JUDGE-DBG] audit_judge_calls write failed:", e)

# ────────────── Rubric ──────────────
@dataclass
class JudgeRubric:
    criteria: Dict[str, Dict[str, Any]]
    @classmethod
    def load(cls, path: str) -> "JudgeRubric":
        if not os.path.exists(path):
            raise FileNotFoundError(f"Rubric not found: {path}")
        if path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            try:
                import yaml  # type: ignore
            except Exception as e:
                raise RuntimeError("Install pyyaml or provide a .json rubric") from e
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        if "criteria" not in data or not isinstance(data["criteria"], dict):
            raise ValueError("Rubric must contain a 'criteria' dict.")
        total_w = 0.0
        for k, v in data["criteria"].items():
            if "w" not in v: raise ValueError(f"Rubric criteria '{k}' missing weight 'w'.")
            v["w"] = float(v["w"]); total_w += v["w"]
        if total_w <= 0: raise ValueError("Rubric weights must sum to > 0.")
        for v in data["criteria"].values(): v["w"] = v["w"] / total_w
        return cls(criteria=data["criteria"])

# ────────────── DEBUG ──────────────
def _dbg_print(*args):
    if JUDGE_DEBUG:
        print("[JUDGE-DBG]", *args)

def _dbg_write(record: Dict[str, Any]):
    if not JUDGE_DEBUG: return
    try:
        os.makedirs(JUDGE_DEBUG_DIR, exist_ok=True)
        with open(os.path.join(JUDGE_DEBUG_DIR, "judge_debug.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print("[JUDGE-DBG] write failed:", e)

# ────────────── LLM loader ──────────────
_tok = None
_model = None
_pipe = None
def _lazy_pipe():
    global _tok, _model, _pipe
    if _pipe is None:
        use_gpu = (JUDGE_DEVICE != "cpu") and torch.cuda.is_available()
        dev  = 0 if use_gpu else -1
        dtype = torch.float16 if use_gpu else torch.float32

        _tok   = AutoTokenizer.from_pretrained(JUDGE_MODEL_ID, use_fast=True)
        # ensure left padding for decoder-only models
        try:
            _tok.padding_side = "left"
        except Exception:
            pass

        _model = AutoModelForCausalLM.from_pretrained(JUDGE_MODEL_ID, torch_dtype=dtype)

        # ensure pad token for batching
        if _tok.pad_token_id is None:
            if _tok.eos_token is not None:
                _tok.pad_token = _tok.eos_token
            else:
                _tok.add_special_tokens({"pad_token": "<|pad|>"})
                try: _model.resize_token_embeddings(len(_tok))
                except Exception: pass
        if getattr(_model.config, "pad_token_id", None) is None:
            _model.config.pad_token_id = _tok.pad_token_id

        _pipe = pipeline("text-generation", model=_model, tokenizer=_tok, device=dev)
        # ⛔️ DO NOT set torch.set_grad_enabled(False) here; it breaks training elsewhere.
        _dbg_print(f"Loaded judge model={JUDGE_MODEL_ID} device={'cuda' if dev==0 else 'cpu'} "
                   f"pad_id={_tok.pad_token_id} eos_id={_tok.eos_token_id}")
    return _pipe, _tok


# ────────────── Prompt builder ──────────────
_SYSTEM_HDR = (
    "You are an impartial metrics judge for a hidden-role social deduction game.\n"
    "Evaluate ONLY the CANDIDATE utterance per the rubric. Output STRICT JSON with keys:\n"
    "  subscores (object of floats in [0,1] for each criterion),\n"
    "  score (float in [0,1], weighted by rubric weights),\n"
    "  rationale (short string).\n"
    "Your reply must start with '{' and end with '}'. No extra text."
)

def _rubric_lines(rubric: JudgeRubric) -> str:
    return "\n".join(
        f"- {name}: {str(cfg.get('def','')).strip()} (w={float(cfg.get('w',0.0)):.2f})"
        for name, cfg in rubric.criteria.items()
    )

def _make_prompt(context: str, role: str, candidate: str, rubric: JudgeRubric, tok) -> str:
    msgs = [
        {"role": "system", "content": _SYSTEM_HDR + "\n\nRubric:\n" + _rubric_lines(rubric)},
        {"role": "user",
         "content": f"Role: {role}\nContext:\n{context.strip()}\n\nCANDIDATE:\n{candidate.strip()}\n\n"
                    f"Begin JSON object immediately; do not add prose.\nJSON:\n{{"}  # pre-seed opening brace
    ]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        # fallback: raw concat
        return (
            f"<system>\n{_SYSTEM_HDR}\nRubric:\n{_rubric_lines(rubric)}\n</system>\n"
            f"Role: {role}\nContext:\n{context}\n\nCANDIDATE:\n{candidate}\n\nJSON:\n{{"
        )

# ────────────── JSON helpers ──────────────
_CODEFENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

def _extract_from_fence(text: str) -> Optional[str]:
    m = _CODEFENCE_RE.search(text)
    return m.group(1) if m else None

def _extract_last_json(text: str) -> Optional[str]:
    if not text: return None
    depth = 0; in_str = False; esc = False; start = None; last = None
    for i, ch in enumerate(text):
        if in_str:
            if esc: esc = False
            elif ch == "\\": esc = True
            elif ch == '"': in_str = False
        else:
            if ch == '"': in_str = True
            elif ch == "{":
                if depth == 0: start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    last = text[start:i+1]; start = None
    return last

def _autoclose_braces(s: str) -> str:
    depth = 0; in_str = False; esc = False
    for ch in s:
        if in_str:
            if esc: esc=False
            elif ch == "\\": esc=True
            elif ch == '"': in_str=False
        else:
            if ch == '"': in_str=True
            elif ch == "{": depth += 1
            elif ch == "}": depth = max(0, depth-1)
    if depth > 0:
        s = s + ("}" * depth)
    return s

def _repair_json_mild(s: str) -> str:
    s = s.strip().strip("`")
    s = re.sub(r'^\s*json\s*', '', s, flags=re.IGNORECASE)
    s = s.replace(": True", ": true").replace(": False", ": false").replace(": None", ": null")
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s

def _safe_parse_json(text: Optional[str]) -> Optional[dict]:
    if not text: return None
    for candidate in (text, _extract_from_fence(text)):
        if not candidate: continue
        for variant in (candidate, _repair_json_mild(candidate), _repair_json_mild(_autoclose_braces(candidate))):
            try:
                return json.loads(variant)
            except Exception:
                continue
    return None

_HEUR_SUBS_RE = re.compile(r'"subscores"\s*:\s*\{(.*?)\}', re.DOTALL)
def _heuristic_extract_scores(text: str, rubric: JudgeRubric) -> Optional[Tuple[Dict[str,float], Optional[float]]]:
    subs: Dict[str, float] = {}
    m = _HEUR_SUBS_RE.search(text)
    blob = m.group(1) if m else text
    ok_any = False
    for k in rubric.criteria.keys():
        rx = re.compile(rf'"{re.escape(k)}"\s*:\s*([0-9]*\.?[0-9]+)')
        mm = rx.search(blob)
        if mm:
            try:
                v = float(mm.group(1)); v = max(0.0, min(1.0, v))
                subs[k] = v
                ok_any = True
            except Exception:
                subs[k] = 0.0
        else:
            subs[k] = 0.0
    score = None
    ms = re.search(r'"score"\s*:\s*([0-9]*\.?[0-9]+)', text)
    if ms:
        try:
            score = max(0.0, min(1.0, float(ms.group(1))))
        except Exception:
            score = None
    return (subs, score) if ok_any else None

def _bounded_float(x: Any) -> float:
    try: v = float(x)
    except Exception: v = 0.0
    if math.isnan(v) or math.isinf(v): return 0.0
    return max(0.0, min(1.0, v))

# ────────────── Public API ──────────────
def score_batch(
    items: List[Dict[str, str]],
    rubric: JudgeRubric,
    *,
    persona_hook: Optional["PersonaHook"] = None,
) -> List[Dict[str, Any]]:
    if not items: return []
    pipe, tok = _lazy_pipe()

    prompts = [_make_prompt(i["context"], i["role"], i["candidate"], rubric, tok) for i in items]

    eos_id = tok.eos_token_id if tok.eos_token_id is not None else tok.pad_token_id
    gen_kwargs = dict(
        max_new_tokens=JUDGE_MAX_NEW,
        do_sample=False,                # deterministic judge
        pad_token_id=tok.pad_token_id,
        eos_token_id=eos_id,
        return_full_text=False,         # continuation only
        num_return_sequences=1,
        truncation=True,
    )

    _dbg_print(f"score_batch: n_items={len(items)} batch={min(len(prompts), JUDGE_BATCH)} max_new={JUDGE_MAX_NEW}")

    if persona_hook is not None and ENABLE_PERSONA_STEER:
        persona_hook.pre_infer(pipe)

    with torch.inference_mode():
        raw = pipe(prompts, batch_size=min(len(prompts), JUDGE_BATCH), **gen_kwargs)

    results: List[Dict[str, Any]] = []
    for idx, out_full in enumerate(raw):
        out  = out_full[0] if isinstance(out_full, list) else out_full
        cont = out.get("generated_text", "") or out.get("text", "")
        text = "{" + cont  # re-attach the pre-seeded opening brace

        # 1) Try strict parse (fenced or balanced, with mild repair/auto-close)
        json_candidate = _extract_from_fence(text) or _extract_last_json(text) or _autoclose_braces(text)
        parsed = _safe_parse_json(json_candidate)

        subs: Dict[str, float] = {}
        rationale = ""
        parsed_ok = isinstance(parsed, dict)

        if parsed_ok:
            raw_subs = parsed.get("subscores", {})
            if isinstance(raw_subs, dict):
                for k in rubric.criteria.keys():
                    subs[k] = _bounded_float(raw_subs.get(k, 0.0))
            rationale = str(parsed.get("rationale", "") or "")
        else:
            # 2) Heuristic salvage (regex) for truncated JSON
            salvage = _heuristic_extract_scores(text, rubric)
            if salvage:
                subs, score_hint = salvage
                rationale = ""
            else:
                for k in rubric.criteria.keys(): subs[k] = 0.0
                rationale = "Parse failure: judge did not return valid JSON."
                _dbg_print(f"parse_failed item#{idx} → head: {(cont[:160].replace(chr(10),' '))!r}")

        score = sum(rubric.criteria[k]["w"] * subs.get(k, 0.0) for k in rubric.criteria)
        if not INCLUDE_RATIONALE: rationale = ""

        # DEBUG record
        prompt_tail = prompts[idx][-400:]
        gen_snip = cont if len(cont) <= 600 else (cont[:250] + " ... " + cont[-250:])
        dbg = {
            "idx": idx,
            "role": items[idx].get("role"),
            "candidate": items[idx].get("candidate"),
            "prompt_tail": prompt_tail,
            "generated_snippet": gen_snip,
            "json_extracted": (json_candidate[:500] + "…") if isinstance(json_candidate, str) and len(json_candidate) > 500 else json_candidate,
            "parsed_ok": parsed_ok,
            "subscores": subs,
            "score": score,
        }
        _dbg_write(dbg)
        _dbg_print(f"item#{idx} parsed_ok={parsed_ok} score={score:.2f}")

        results.append({"subscores": subs, "score": score, "rationale": rationale})

    return results

def choose_best(
    contexts_roles_candidates: List[Tuple[str, str, str]],
    rubric: JudgeRubric,
    *,
    persona_hook: Optional["PersonaHook"] = None,
) -> Tuple[int, Dict[str, Any]]:
    items = [{"context": c, "role": r, "candidate": a} for (c, r, a) in contexts_roles_candidates]
    scored = score_batch(items, rubric, persona_hook=persona_hook)
    if not scored: return -1, {}
    def tie_key(d: Dict[str, Any]):
        subs = d.get("subscores", {})
        return (d.get("score", 0.0), subs.get("truthfulness", 0.0), subs.get("coherence", 0.0))
    best_idx = max(range(len(scored)), key=lambda i: tie_key(scored[i]))
    return best_idx, scored[best_idx]

# ────────────── Persona stub ──────────────
class PersonaHook:
    def __init__(self, config_path: str = PERSONA_CONFIG_PATH):
        self.config_path = config_path
        self._hooks: List[Any] = []
    def pre_infer(self, _pipe): return
    def close(self):
        for h in self._hooks:
            try: h.remove()
            except Exception: pass
        self._hooks.clear()

# ────────────── CLI ──────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Quick judge test")
    ap.add_argument("--rubric", required=True, help="Path to rubric YAML/JSON")
    ap.add_argument("--role", default="Worker", help="Role string")
    ap.add_argument("--context", required=True, help="Context text")
    ap.add_argument("--candidate", required=True, help="Candidate utterance")
    args = ap.parse_args()
    rubric = JudgeRubric.load(args.rubric)
    res = score_batch([{"context": args.context, "role": args.role, "candidate": args.candidate}], rubric)
    print(json.dumps(res[0], indent=2))
