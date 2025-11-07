# judge.py
from __future__ import annotations
import os, sys, json, math, re
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import yaml

# ────────────── Config + Env helpers ──────────────
def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None or (isinstance(v, str) and v.strip() == ""):
        return default
    try:
        return str(v).strip().lower() in ("1","true","yes","y","on")
    except Exception:
        return default

def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or (isinstance(v, str) and v.strip() == ""):
        return default
    try:
        return int(v)
    except Exception:
        return default

def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or (isinstance(v, str) and v.strip() == ""):
        return default
    try:
        return float(v)
    except Exception:
        return default

def _env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return default if v is None else str(v)

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f) or {}

# ────────────── Provider detect (mirrors llm_script) ──────────────
LLM_PROVIDER = _env_str("LLM_PROVIDER", str(config.get("LLM_PROVIDER", "hf"))).strip().lower()
LLM_MODEL_ID = _env_str("LLM_MODEL_ID", str(config.get("LLM_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2"))).strip()

def _using_openai_provider() -> bool:
    if LLM_PROVIDER in ("openai", "oai"):
        return True
    mid = LLM_MODEL_ID.lower()
    return mid.startswith(("gpt-", "o", "chatgpt"))

# ────────────── Base config values (from file) ──────────────
JUDGE_MODEL_ID       = config.get("JUDGE_MODEL_ID", "microsoft/Phi-3-mini-4k-instruct")
JUDGE_MAX_NEW        = int(config.get("JUDGE_MAX_NEW", 128))
INCLUDE_RATIONALE    = bool(config.get("INCLUDE_RATIONALE", True))
ENABLE_PERSONA_STEER = bool(config.get("ENABLE_PERSONA_STEER", False))
PERSONA_CONFIG_PATH  = config.get("PERSONA_CONFIG_PATH", "configs/persona_vectors.yaml")
JUDGE_BATCH          = max(1, int(config.get("JUDGE_BATCH", 3)))
JUDGE_DEBUG          = bool(config.get("JUDGE_DEBUG", False))
JUDGE_DEBUG_DIR      = config.get("JUDGE_DEBUG_DIR", "logs")

# ────────────── Phase-5 structured judge toggles ──────────────
_judge_cfg = config.get("judge", {}) if isinstance(config.get("judge", {}), dict) else {}
RERANK_TOPK         = bool(_judge_cfg.get("rerank_topk", True))
STORE_SUBSCORES     = bool(_judge_cfg.get("store_subscores", True))
TALK_VOTE_ALIGN_ON  = bool(_judge_cfg.get("talk_vote_alignment", True))

# NEW: repetition penalty config (audit, and optional scoring attenuation)
JUDGE_RP_WEIGHT     = float(_judge_cfg.get("rp_weight", 0.0))
JUDGE_RP_N          = int(_judge_cfg.get("rp_n", 2))

# ────────────── Env overrides ──────────────
JUDGE_MODEL_ID       = _env_str ("JUDGE_MODEL_ID",       JUDGE_MODEL_ID)
JUDGE_MAX_NEW        = _env_int ("JUDGE_MAX_NEW",        JUDGE_MAX_NEW)
INCLUDE_RATIONALE    = _env_bool("INCLUDE_RATIONALE",    INCLUDE_RATIONALE)
ENABLE_PERSONA_STEER = _env_bool("ENABLE_PERSONA_STEER", ENABLE_PERSONA_STEER)
PERSONA_CONFIG_PATH  = _env_str ("PERSONA_CONFIG_PATH",  PERSONA_CONFIG_PATH)
JUDGE_DEBUG          = _env_bool("JUDGE_DEBUG",          JUDGE_DEBUG)
JUDGE_DEBUG_DIR      = _env_str ("JUDGE_DEBUG_DIR",      JUDGE_DEBUG_DIR)
JUDGE_RP_WEIGHT      = _env_float("JUDGE_RP_WEIGHT",     JUDGE_RP_WEIGHT)
JUDGE_RP_N           = _env_int  ("JUDGE_RP_N",          JUDGE_RP_N)

# NEW: strict vote path toggle (optional runtime control for choose_best)
JUDGE_VOTE_STRICT     = _env_bool("JUDGE_VOTE_STRICT", False)
JUDGE_WARN_PARSE_ONCE = _env_bool("JUDGE_WARN_PARSE_ONCE", True)
_PARSE_WARNED_ONCE    = False

# ────────────── Logging paths ──────────────
_LOGGING = config.get("logging", {}) if isinstance(config.get("logging", {}), dict) else {}
def _join_default(path_fallback: str) -> str:
    return _LOGGING.get("judge_jsonl", os.path.join(JUDGE_DEBUG_DIR, path_fallback))
JUDGE_AUDIT_JSONL = _env_str("JUDGE_AUDIT_JSONL", _join_default("judge_calls.jsonl"))
JUDGE_AUDIT_DEBUG = _env_bool("JUDGE_AUDIT_DEBUG", bool(_LOGGING.get("debug", JUDGE_DEBUG)))

def judge_logging_enabled() -> bool:
    return bool(JUDGE_AUDIT_DEBUG or JUDGE_DEBUG)

def _ensure_parent_dir(path: str):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except Exception:
        pass

# ────────────── Name mention helper ──────────────
_AGENT_NAME_RE = re.compile(r"\bAgent_[0-9]+\b")
def _mentions_name(s: Optional[str]) -> bool:
    try:
        return bool(_AGENT_NAME_RE.search(s or ""))
    except Exception:
        return False

# ────────────── Responses kwargs filter (for safety) ──────────────
_ALLOWED_RESPONSES_KW = {"temperature", "top_p", "max_output_tokens", "stop"}
def _filter_responses_kwargs(kwargs: dict) -> dict:
    if not isinstance(kwargs, dict):
        return {}
    return {k: v for k, v in kwargs.items() if k in _ALLOWED_RESPONSES_KW}

# ────────────── Repetition penalty (lexical diversity) ──────────────
def repetition_penalty(text: str, n: int = 2) -> float:
    toks = [t for t in (text or "").strip().split() if t]
    if len(toks) < n + 1:
        return 0.0
    grams = [" ".join(toks[i:i+n]) for i in range(len(toks) - n + 1)]
    total = len(grams)
    uniq = len(set(grams))
    rep_frac = 1.0 - (uniq / max(1, total))
    return float(min(1.0, max(0.0, rep_frac)))

# ────────────── Specificity score ──────────────
def _specificity_score(text: str) -> float:
    t = (text or "").lower()
    toks = re.findall(r"[a-z']+", t)
    if not toks:
        return 0.0
    stop = set("the a an and or but if so to of in on for with that this it is are was were be been being do does did not no nor".split())
    content = [w for w in toks if w not in stop]
    if not content:
        return 0.0
    return min(1.0, max(0.0, len(set(content)) / len(content)))

# ────────────── Audit writer ──────────────
def audit_judge_calls(*, run_id: str, round_num: int, phase: str, agent: str,
                      items: List[Dict[str, str]], results: List[Dict[str, Any]],
                      jsonl_path: Optional[str] = None) -> None:
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
                rec: Dict[str, Any] = {
                    "run_id": run_id, "round": int(round_num), "phase": str(phase), "agent": str(agent),
                    "context": it.get("context", ""), "role": it.get("role", ""), "candidate": it.get("candidate", "")
                }
                for k in ("subscores","score","raw_text","json","align_tv","rp_applied","strict_ok","retry_count","redo_count","vote_target","rationale","error"):
                    if k in out and out[k] is not None:
                        rec[k] = out[k]
                if out.get("repetition_penalty") is not None:
                    rec["repetition_penalty"] = float(out["repetition_penalty"])
                if out.get("confidence") is not None:
                    rec["confidence"] = float(out["confidence"])
                if out.get("specificity") is not None:
                    rec["specificity"] = float(out["specificity"])
                if "name_mentioned" in out:
                    rec["name_mentioned"] = int(bool(out["name_mentioned"]))
                if "talk_vote_match" in out:
                    rec["talk_vote_match"] = int(bool(out["talk_vote_match"]))
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        if JUDGE_DEBUG:
            print("[JUDGE-DBG] audit_judge_calls write failed:", e, file=sys.stderr)

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
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        if "criteria" not in data or not isinstance(data["criteria"], dict):
            raise ValueError("Rubric must contain a 'criteria' dict.")
        total_w = 0.0
        for k, v in data["criteria"].items():
            if "w" not in v:
                raise ValueError(f"Rubric criteria '{k}' missing weight 'w'.")
            v["w"] = float(v["w"])
            total_w += v["w"]
        if total_w <= 0:
            raise ValueError("Rubric weights must sum to > 0.")
        for v in data["criteria"].values():
            v["w"] = v["w"] / total_w
        return cls(criteria=data["criteria"])

# ────────────── DEBUG ──────────────
def _dbg_print(*args):
    if JUDGE_DEBUG:
        print("[JUDGE-DBG]", *args, file=sys.stderr)

def _dbg_write(record: Dict[str, Any]):
    if not JUDGE_DEBUG:
        return
    try:
        os.makedirs(JUDGE_DEBUG_DIR, exist_ok=True)
        with open(os.path.join(JUDGE_DEBUG_DIR, "judge_debug.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print("[JUDGE-DBG] write failed:", e, file=sys.stderr)

# ────────────── LLM wrapper (shared mouth from llm_script) ──────────────
_mouth = None
def _get_mouth():
    global _mouth
    if _mouth is not None:
        return _mouth
    try:
        from llm_script import llm_fn_from_env
        _mouth = llm_fn_from_env()
        return _mouth
    except Exception as e:
        _mouth = None
        raise RuntimeError(f"judge mouth init failed: {e}")

# ────────────── Prompt builders ──────────────
def _rubric_lines(rubric: JudgeRubric) -> str:
    return "\n".join(
        f"- {name}: {str(cfg.get('def','')).strip()} (w={float(cfg.get('w',0.0)):.2f})"
        for name, cfg in rubric.criteria.items()
    )

def _make_prompt(context: str, role: str, candidate: str, rubric: JudgeRubric) -> str:
    example = (
        '{\n'
        '  "subscores": {"coherence": 0.82, "truthfulness": 0.73, "role_alignment": 0.61, "social_safety": 0.95},\n'
        '  "score": 0.76,\n'
        '  "rationale": "Short reason."\n'
        '}'
    )
    sys_hdr = (
        "You are an impartial metrics judge for a hidden-role social deduction game.\n"
        "Evaluate ONLY the CANDIDATE utterance per the rubric and return STRICT JSON.\n"
        "Output MUST be exactly ONE JSON object with keys:\n"
        "  - subscores: object of floats in [0,1] for each rubric criterion name\n"
        "  - score: float in [0,1] (weighted by rubric weights)\n"
        "  - rationale: short string\n"
        "Your reply MUST start with '{' and end with '}'.\n"
        "No prose, no explanations, no code fences, no comments, no extra keys.\n"
        "Example format (values illustrative only):\n"
        f"{example}\n"
        "Rubric:\n" + _rubric_lines(rubric) + "\n"
    )
    user = (
        f"Role: {role}\n"
        f"Context:\n{context.strip()}\n\n"
        f"CANDIDATE:\n{candidate.strip()}\n\n"
        "Begin JSON now. Output only one JSON object with keys subscores, score, rationale.\n"
        "JSON:\n"
    )
    return sys_hdr + "\n" + user

def _make_strict_vote_prompt(context: str, role: str, candidates: List[str]) -> str:
    opts = "\n".join([f"- {c}" for c in candidates])
    return (
        "You are the impartial vote arbiter for a hidden-role social deduction game.\n"
        "Given CONTEXT and ALLOWED CANDIDATES, choose ONE name to eliminate.\n"
        "Return STRICT JSON with EXACTLY these keys:\n"
        '  {"vote_target": <string, one of the allowed candidate names exactly>,\n'
        '   "confidence": <float in [0,1]>,\n'
        '   "rationale": <short string> }\n'
        "NO extra keys, NO extra text, NOT in code fences. Reply must start with '{' and end with '}'.\n\n"
        f"Role: {role}\n"
        f"Context:\n{context.strip()}\n\n"
        "ALLOWED CANDIDATES (choose exactly one, string must match one of these):\n"
        f"{opts}\n\n"
        "Begin JSON now:\n"
    )

# ────────────── JSON helpers ──────────────
_CODEFENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
def _extract_from_fence(text: str) -> Optional[str]:
    m = _CODEFENCE_RE.search(text)
    return m.group(1) if m else None

def _extract_last_json(text: str) -> Optional[str]:
    if not text:
        return None
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
        s += "}" * depth
    return s

def _repair_json_mild(s: str) -> str:
    s = s.strip().strip("`")
    s = re.sub(r'^\s*json\s*', '', s, flags=re.IGNORECASE)
    s = s.replace(": True", ": true").replace(": False", ": false").replace(": None", ": null")
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s

def _safe_parse_json(text: Optional[str]) -> Optional[dict]:
    if not text:
        return None
    candidate = _extract_from_fence(text) or _extract_last_json(text) or text
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
                subs[k] = v; ok_any = True
            except Exception:
                pass
    score = None
    ms = re.search(r'"score"\s*:\s*([0-9]*\.?[0-9]+)', text)
    if ms:
        try:
            score = max(0.0, min(1.0, float(ms.group(1))))
        except Exception:
            score = None
    return (subs, score) if ok_any else None

def _bounded_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        v = 0.0
    if math.isnan(v) or math.isinf(v):
        return 0.0
    return max(0.0, min(1.0, v))

def _parse_strict_vote_json(text: str, allowed: List[str]) -> Optional[Dict[str, Any]]:
    blob = _extract_last_json(text) or _extract_from_fence(text) or text
    try:
        obj = json.loads(blob)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    keys = set(obj.keys())
    if keys != {"vote_target", "confidence", "rationale"}:
        return None
    vt = obj.get("vote_target"); cf = obj.get("confidence"); rn = obj.get("rationale")
    if not isinstance(vt, str) or vt not in allowed:
        return None
    try:
        cf = float(cf)
    except Exception:
        return None
    if not (0.0 <= cf <= 1.0):
        return None
    if not isinstance(rn, str):
        return None
    return {"vote_target": vt, "confidence": cf, "rationale": rn}

# ────────────── Public API (rubric scorer) ──────────────
def _judge_call_kwargs(max_tokens: int) -> Dict[str, Any]:
    """
    Build safe kwargs for mouth(), conditional on provider.
    - For OpenAI: only pass max_output_tokens (no temperature or top_p).
    - For HF: pass deterministic knobs but keep it minimal.
    """
    if _using_openai_provider():
        return {"max_output_tokens": int(max_tokens)}
    return {
        "max_new_tokens": int(max_tokens),
        "temperature": 0.0,
        "top_p": 1.0,
    }

def _call_mouth_with_arg_gating(mouth_fn, prompt: str, gen_kwargs: Dict[str, Any]) -> str:
    """
    Call mouth(), then on OpenAI-style errors about unsupported temperature,
    drop 'temperature' and retry once.
    """
    try:
        return mouth_fn(prompt, generate_kwargs=gen_kwargs)
    except Exception as e:
        msg = getattr(e, "message", str(e))
        if ("Unsupported parameter" in msg or "invalid_request_error" in msg) and "temperature" in msg:
            g2 = dict(gen_kwargs)
            g2.pop("temperature", None)
            return mouth_fn(prompt, generate_kwargs=g2)
        raise

def score_batch(
    items: List[Dict[str, str]],
    rubric: JudgeRubric,
    *,
    persona_hook: Optional["PersonaHook"] = None,  # kept for API compatibility
    run_id: str = "",
    round_num: int = -1,
    phase: str = "",
    agent: str = "",
    alignment_values: Optional[List[Optional[float]]] = None,
) -> List[Dict[str, Any]]:

    if not items:
        return []
    mouth = _get_mouth()

    prompts = [_make_prompt(i["context"], i["role"], i["candidate"], rubric) for i in items]
    _dbg_print(f"score_batch: n_items={len(items)} provider={'openai' if _using_openai_provider() else 'hf'} max_new={JUDGE_MAX_NEW}")

    results: List[Dict[str, Any]] = []

    for idx, p in enumerate(prompts):
        gen_kwargs = _judge_call_kwargs(JUDGE_MAX_NEW)
        if _using_openai_provider():
            gen_kwargs = _filter_responses_kwargs(gen_kwargs)

        # Prompt sanity preview for debugging
        if JUDGE_DEBUG:
            preview = p[:600].replace("\n", "\\n")
            _dbg_print(f"[PROMPT-{idx}] {preview}")

        # Hard check for empty prompt
        if not p or len(p.strip()) < 10:
            rec = {
                "error": "prompt_empty",
                "rationale": "",
                "specificity": 0.0,
                "repetition_penalty": None,
                "rp_applied": False,
            }
            if judge_logging_enabled():
                rec["raw_text"] = ""
                rec["json"] = ""
            results.append(rec)
            continue

        error_tag: Optional[str] = None
        raw = ""
        parsed = None
        subs: Optional[Dict[str, float]] = None
        rationale = ""
        score: Optional[float] = None

        def _one_call(prompt_text: str) -> str:
            return _call_mouth_with_arg_gating(mouth, prompt_text, gen_kwargs)

        # First attempt
        try:
            raw = _one_call(p)
        except Exception as e:
            msg = f"{type(e).__name__}"
            if "presence_penalty" in str(e).lower():
                msg = f"{msg}:presence_penalty"
            error_tag = f"openai:{msg}"

        if not error_tag:
            parsed = _safe_parse_json(raw)

        # Heuristic salvage, or retry with stricter instruction if needed
        if not error_tag and not isinstance(parsed, dict):
            salvage = _heuristic_extract_scores(raw, rubric)
            if salvage is None:
                try:
                    raw2 = _one_call(p + "\nIMPORTANT: Respond strictly in JSON matching the schema, no extra text.")
                    raw = raw + "\n" + raw2
                    parsed = _safe_parse_json(raw2)
                    if not isinstance(parsed, dict):
                        salvage = _heuristic_extract_scores(raw2, rubric)
                except Exception as e:
                    if not error_tag:
                        error_tag = f"openai:{type(e).__name__}"
            if salvage and not isinstance(parsed, dict):
                ssubs, shint = salvage
                subs = ssubs if ssubs else None
                score = shint if shint is not None else None
                rationale = ""

        if not error_tag and isinstance(parsed, dict):
            raw_subs = parsed.get("subscores", {})
            if isinstance(raw_subs, dict):
                subs = {k: _bounded_float(raw_subs.get(k, None))
                        for k in rubric.criteria.keys()
                        if raw_subs.get(k, None) is not None}
                if subs and len(subs) == 0:
                    subs = None
            if "score" in parsed and parsed["score"] is not None:
                try:
                    score = _bounded_float(parsed["score"])
                except Exception:
                    score = None
            rationale = str(parsed.get("rationale", "") or "")

        cand_text = items[idx].get("candidate", "") or ""

        # Specificity bonus, applied to base score
        spec = _specificity_score(cand_text)
        if score is not None:
            score = max(0.0, min(1.0, float(score) + 0.05 * float(spec)))

        # Repetition attenuation
        rp_val = repetition_penalty(cand_text, n=max(1, int(JUDGE_RP_N)))
        rp_applied = False
        if (score is not None) and JUDGE_RP_WEIGHT > 0.0 and rp_val > 0.0:
            score = max(0.0, min(1.0, float(score) - float(JUDGE_RP_WEIGHT) * float(rp_val)))
            rp_applied = True

        rec: Dict[str, Any] = {
            "rationale": (rationale if INCLUDE_RATIONALE else ""),
            "repetition_penalty": rp_val if (score is not None) else None,
            "rp_applied": rp_applied if (score is not None) else False,
            "specificity": spec,
            "error": (error_tag or ""),
        }
        if subs is not None:
            rec["subscores"] = subs
        if score is not None:
            rec["score"] = score

        if TALK_VOTE_ALIGN_ON and alignment_values is not None and idx < len(alignment_values):
            rec["align_tv"] = alignment_values[idx] if alignment_values[idx] is not None else None
        rec["name_mentioned"] = int(_mentions_name(cand_text))
        tv_val = None
        if alignment_values is not None and idx < len(alignment_values) and alignment_values[idx] is not None:
            try:
                tv_val = float(alignment_values[idx])
            except Exception:
                tv_val = None
        rec["talk_vote_match"] = int((tv_val is not None and tv_val >= 0.5) or ("vote" in cand_text.lower() and rec["name_mentioned"]))

        if judge_logging_enabled():
            rec["raw_text"] = raw
            rec["json"] = _extract_from_fence(raw) or _extract_last_json(raw) or ""

        _dbg_write({
            "idx": idx,
            "parsed_ok": (isinstance(parsed, dict) and not error_tag),
            "has_score": (score is not None),
            "has_subscores": (subs is not None),
            "error": error_tag or "",
        })

        results.append(rec)

    if STORE_SUBSCORES:
        try:
            audit_judge_calls(
                run_id=run_id or "",
                round_num=int(round_num) if round_num is not None else -1,
                phase=str(phase or ""),
                agent=str(agent or ""),
                items=items,
                results=results,
                jsonl_path=None,
            )
        except Exception as e:
            _dbg_print("audit_judge_calls failed:", e)

    return results

# NEW: Strict vote decision API
def strict_vote_decision(
    *,
    context: str,
    role: str,
    candidates: List[str],
    run_id: str = "",
    round_num: int = -1,
    phase: str = "DAY_VOTE",
    agent: str = "",
    max_new: Optional[int] = None,
) -> Tuple[int, Dict[str, Any]]:
    if not candidates:
        return -1, {"strict_ok": 0, "retry_count": 0, "rationale": "No candidates provided."}

    mouth = _get_mouth()
    prompt = _make_strict_vote_prompt(context, role, candidates)

    # Prompt sanity preview for debugging
    if JUDGE_DEBUG:
        preview = prompt[:600].replace("\n", "\\n")
        _dbg_print(f"[STRICT-PROMPT] {preview}")

    attempts = 0
    raw_texts: List[str] = []
    parsed: Optional[Dict[str, Any]] = None
    error_tag: Optional[str] = None

    def _call(prompt_str: str) -> str:
        base = _judge_call_kwargs(max_new if isinstance(max_new, int) else min(128, JUDGE_MAX_NEW))
        if _using_openai_provider():
            base = _filter_responses_kwargs(base)
        return _call_mouth_with_arg_gating(mouth, prompt_str, base)

    for _try in range(2):
        attempts += 1
        try:
            cont = _call(prompt if _try == 0 else (prompt + "\nIMPORTANT: Respond strictly in JSON matching the schema, no extra text."))
            raw_texts.append(cont)
            parsed = _parse_strict_vote_json(cont, candidates)
            if parsed is not None:
                break
        except Exception as e:
            msg = f"{type(e).__name__}"
            if "presence_penalty" in str(e).lower():
                msg = f"{msg}:presence_penalty"
            error_tag = f"openai:{msg}"
            parsed = None
            break

    if parsed is None:
        idx = 0
        rec = {
            "vote_target": candidates[idx],
            "confidence": None,
            "rationale": "Strict parse failed; defaulted to planner top-1.",
            "strict_ok": 0,
            "retry_count": max(0, attempts - 1),
            "redo_count": max(0, attempts - 1),
            "name_mentioned": 1,
            "talk_vote_match": 0,
            "raw_text": raw_texts[-1] if raw_texts else "",
            "json": "",
            "subscores": None,
            "score": None,
            "error": (error_tag or ""),
        }
        if STORE_SUBSCORES:
            try:
                audit_judge_calls(
                    run_id=run_id or "",
                    round_num=int(round_num) if round_num is not None else -1,
                    phase=str(phase or ""),
                    agent=str(agent or ""),
                    items=[{"context": context, "role": role, "candidate": f"(strict vote) {candidates}"}],
                    results=[rec],
                )
            except Exception as e:
                _dbg_print("audit_judge_calls (strict) failed:", e)
        return idx, rec

    vt = parsed["vote_target"]
    idx = candidates.index(vt)
    rec = {
        "vote_target": vt,
        "confidence": float(parsed["confidence"]),
        "rationale": parsed["rationale"] if INCLUDE_RATIONALE else "",
        "strict_ok": 1,
        "retry_count": max(0, attempts - 1),
        "redo_count": max(0, attempts - 1),
        "name_mentioned": 1,
        "talk_vote_match": 1,
        "raw_text": raw_texts[-1] if raw_texts else "",
        "json": json.dumps(parsed, ensure_ascii=False),
        "score": float(parsed["confidence"]),
        "subscores": {},
        "error": "",
    }
    if STORE_SUBSCORES:
        try:
            audit_judge_calls(
                run_id=run_id or "",
                round_num=int(round_num) if round_num is not None else -1,
                phase=str(phase or ""),
                agent=str(agent or ""),
                items=[{"context": context, "role": role, "candidate": f"(strict vote) {candidates}"}],
                results=[rec],
            )
        except Exception as e:
            _dbg_print("audit_judge_calls (strict) failed:", e)
    return idx, rec

def choose_best(
    contexts_roles_candidates: List[Tuple[str, str, str]],
    rubric: JudgeRubric,
    *,
    persona_hook: Optional["PersonaHook"] = None,
    run_id: str = "",
    round_num: int = -1,
    phase: str = "",
    agent: str = "",
) -> Tuple[int, Dict[str, Any]]:
    if JUDGE_VOTE_STRICT:
        if not contexts_roles_candidates:
            return -1, {}
        ctx0, role0, _ = contexts_roles_candidates[0]
        candidates = [cand for (_c, _r, cand) in contexts_roles_candidates]
        idx, rec = strict_vote_decision(
            context=ctx0, role=role0, candidates=candidates,
            run_id=run_id, round_num=round_num, phase=phase or "DAY_VOTE", agent=agent,
        )
        return idx, rec

    items = [{"context": c, "role": r, "candidate": a} for (c, r, a) in contexts_roles_candidates]
    scored = score_batch(items, rubric, persona_hook=persona_hook, run_id=run_id, round_num=round_num, phase=phase, agent=agent)
    if not scored:
        return -1, {}
    if not RERANK_TOPK:
        return 0, scored[0]

    all_na = all(s.get("score") is None for s in scored)
    if all_na:
        neutral = dict(scored[0]);  neutral.setdefault("neutral_decision", 1)
        return 0, neutral

    def tie_key(d: Dict[str, Any]):
        subs = (d.get("subscores") or {}) if isinstance(d.get("subscores"), dict) else {}
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
            try:
                h.remove()
            except Exception:
                pass
        self._hooks.clear()

# ────────────── CLI ──────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Quick judge test (wrapper-backed)")
    ap.add_argument("--rubric", required=False, help="Path to rubric YAML/JSON (rubric path not needed for strict vote)")
    ap.add_argument("--role", default="Worker", help="Role string")
    ap.add_argument("--context", required=True, help="Context text")
    ap.add_argument("--candidate", help="Candidate utterance (rubric mode)")
    ap.add_argument("--strict", action="store_true", help="Use strict vote mode")
    ap.add_argument("--opts", nargs="*", help="Allowed candidate names for strict vote mode")
    args = ap.parse_args()

    if args.strict:
        opts = args.opts or []
        if not opts:
            raise SystemExit("Provide --opts for strict vote mode.")
        idx, rec = strict_vote_decision(context=args.context, role=args.role, candidates=opts)
        print(json.dumps({"index": idx, **rec}, ensure_ascii=False))
    else:
        if not args.rubric or not args.candidate:
            raise SystemExit("--rubric and --candidate are required for rubric mode.")
        rubric = JudgeRubric.load(args.rubric)
        res = score_batch([{"context": args.context, "role": args.role, "candidate": args.candidate}], rubric)
        print(json.dumps(res[0], ensure_ascii=False))
