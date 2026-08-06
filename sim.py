# sim.py  - verbose, multithreaded, responsive Pygame + Judge integration
# -----------------------------------------------------------------------------
# - Keeps the rollout tuples using the true post-act z_{t+1}
# - Prints acceptance metric: mean ||Δz|| per day
# - Shares a single MessageEncoder across agents to save VRAM/CPU
# - Calls LLL-as-Judge on planner top-k vote targets, picks final vote, logs subscores
# - Returns agents in meta so train.py can run speaker learning
# - Applies personality randomization per agent
# - NEW (Phase 1: Stabilization & Logging)
#   * Deterministic seeds + run metadata snapshot
#   * Phase-aware + mask-aware telemetry
#   * Per-decision CSV rows (TALK / VOTE / KILL) with Δz for votes
#   * Optional judge debug JSONL is handled in judge.py
# - NEW (Phase 3: Multi-head)
#   * Use TalkHead / VoteHead / KillHead when available
#   * Proper boolean masks (self/dead/wolves)
#   * Judge re-ranking over VoteHead top-k
#   * Phase-aware rollout tuples:
#       (z_t, phase_code, action_payload, z_{t+1}, role, choice_type, aux)
#   * Planner x Judge mixing, social influence knobs, talk mask logging,
#     optional Δz for TALK and KILL
# - NEW (Phase 5: Communication Alignment)
#   * Judge real utterances during DISCUSS and attach per-utterance subscores
#   * Log these judge fields in TALK rows
#   * Compute talk→vote alignment and log/backfill on VOTE rows
#   * Attach tokenizer when hooking up the LLM so talk x bias fusion engages
# - NEW (Phase 5c: Intent fusion + wolf night chat consensus)
#   * During DISCUSS, compute TalkHead logits + BiasHead category logits, fuse to p_intent
#   * Sample template_id, optional arg_id, and call LLM to get (text, meta)
#   * Push row to each agent.msg_buffer with p_intent and repetition_penalty
#   * Private NIGHT_DISCUSS loop among wolves; compute night_consensus and record per message
#   * Use consensus target as fallback for night_kill if needed
#   * Ensure aux bundles recent_texts and neighbor_texts for training masks and coupling
# - NEW (Stage A social): After DAY_DISCUSS, apply per-agent latent social coupling.
# - NEW (Phase 7: Lang CSV Emitter)
#   * Use training_utils.LangMetricsWriter for buffered lang_metrics.csv emission.
#   * Flush at episode end; rows include run_id and seed.
# - NEW (Phase 7b: Bias-first mouthpiece routing)
#   * Primary path uses bias fusion mouthpiece, fallback builds prompt via speaker_llm helpers,
#     final fallback uses SAFE_FALLBACK. Collapse is penalized via repetition metrics and judge.
# - NEW (Phase 7c hotfix): Agent-name hygiene for targetless speech and malformed mentions.
# - NEW (2.1/2.2/2.3): Allowed-name gating for LLM; deterministic day kill; wolf consensus night kill.
# - NEW (6: Logging print-site and CSV 'error' field).
# - NEW (Baseline toggle): Random voting policy via config or env to produce uniform targets.
# - NEW (Fixes): Judge can be disabled or unavailable; Heuristic policy uses local signals without judge;
#   vote rows backfill target and target_is_wolf so downstream summaries never NaN.
# - NEW (JEPA-only + social env override):
#   * POLICY in {"jepa_only", "jepa_random"} uses a JEPA-only vote branch without judge mixing.
#   * SOCIAL_ENABLED and SIM_SOCIAL_ENABLED env vars override config for Stage A social coupling.
# -----------------------------------------------------------------------------

import sys
import os
import random
import concurrent.futures as cf
from collections import deque, Counter
from typing import Dict, Tuple, List, Optional, Deque
import re

import pygame
import torch, yaml
import torch.nn.functional as F

import uuid, time, json, csv, pathlib
import numpy as np
from types import SimpleNamespace

from agent import BaseAgent
from roles import WEREWOLF, VILLAGER, assign_roles
try:
    from roles import apply_personality
except Exception:
    apply_personality = None

try:
    from world import resolve_votes_detailed, eliminate_player, _alive_name_set, consensus_target
except Exception:
    from world import resolve_votes as _resolve_votes_fallback, eliminate_player  # type: ignore
    def resolve_votes_detailed(votes: Dict[str, str], alive_names: Optional[List[str]] = None):
        try:
            winner = _resolve_votes_fallback(votes)  # type: ignore
        except Exception:
            winner = None
        return {"winner": winner, "tally": votes, "tie": winner is None}
    def _alive_name_set(agents):
        return [a.name for a in agents if getattr(a, "alive", False)]
    def consensus_target(tally_list: List[str], temperature: float = 0.7):
        if not tally_list:
            return None
        c = Counter(tally_list)
        return c.most_common(1)[0][0]

from training_utils import load_role_models, load_role_models_factorized, load_shared_belief_encoder, load_shared_social
try:
    from encoders import MessageEncoder
except Exception:
    raise

from judge import JudgeRubric, score_batch

SAFE_FALLBACK = "I need a moment to think."
try:
    from llm_script import SAFE_FALLBACK as _SAFE_FALLBACK_STR
    if isinstance(_SAFE_FALLBACK_STR, str) and _SAFE_FALLBACK_STR.strip():
        SAFE_FALLBACK = _SAFE_FALLBACK_STR.strip()
except Exception as _e_sf:
    print(f"[IMPORT] llm_script.SAFE_FALLBACK unavailable: {type(_e_sf).__name__}: {_e_sf}")

try:
    from speaker_llm import (
        LogitBiasHead,
        FUSION_ALPHA,
        CAT_ORDER,
        guard_and_shape,
        build_role_phase_prompt,
        build_prompt_and_controls,
        maybe_second_pass_kwargs,
        with_fused_bias_generate_kwargs,
        repetition_penalty,
        normalize_utterance,
        _set_allowed_names,
    )
    print("[IMPORT] speaker_llm import OK")
except Exception as e:
    print(f"[IMPORT] speaker_llm import FAILED: {type(e).__name__}: {e}")
    LogitBiasHead = None
    FUSION_ALPHA = 0.5
    CAT_ORDER = ["accuse","defend","hedge","question","vote"]

    def guard_and_shape(text_raw, plan, role, phase, cfg):
        text = text_raw if (text_raw and text_raw.strip()) else SAFE_FALLBACK
        return text, {"violated": False, "redo": False, "repetition_penalty": None}

    def build_role_phase_prompt(**_kwargs) -> str:
        return "[YOU]\n" + SAFE_FALLBACK

    def build_prompt_and_controls(**_kwargs):
        return "[YOU]\n" + SAFE_FALLBACK, {}

    def maybe_second_pass_kwargs(**_kwargs):
        return _kwargs.get("first_kwargs", {})

    def with_fused_bias_generate_kwargs(**_kwargs):
        return {"logits_processor": []}

    def repetition_penalty(text: str, n: int = 2) -> float:
        toks = [t for t in (text or "").split() if t]
        return 1.0 if len(toks) <= n else 0.0

    def normalize_utterance(s: str) -> str:
        return (s or "").strip()

    def _set_allowed_names(_names: List[str]):
        return None

try:
    from speaker import make_hist_feats as _mk_hist_feats  # type: ignore
    from speaker import build_plan_tuple  # type: ignore
except Exception as e:
    print(f"[IMPORT] speaker helper import partial FAILED: {type(e).__name__}: {e}")

    def _mk_hist_feats(_texts: List[str]) -> torch.Tensor:
        return torch.tensor([0.0, 0.0], dtype=torch.float32)

    def build_plan_tuple(**kwargs):
        return {
            "intent": kwargs.get("intent", "hedge"),
            "target": kwargs.get("target"),
            "shape": kwargs.get("shape", ""),
        }

try:
    from speaker import postprocess_text  # type: ignore
except Exception as e:
    print(f"[IMPORT] speaker.postprocess_text FAILED: {type(e).__name__}: {e}")

    def postprocess_text(text, role, cfg):
        return text if (text and text.strip()) else SAFE_FALLBACK

try:
    from llm_script import chatgpt_llm_with_bias, tok as _LLM_TOK
    print("[IMPORT] llm_script import OK")
except Exception as e:
    print(f"[IMPORT] llm_script import FAILED: {type(e).__name__}: {e}")

    def chatgpt_llm_with_bias(_z, _agent, *, named_target: Optional[str] = None, **_kwargs):
        return SAFE_FALLBACK, {"repetition_penalty": None}

    _LLM_TOK = None

try:
    from training_utils import LangMetricsWriter  # type: ignore
except Exception:
    LangMetricsWriter = None

with open("config.yaml", "r") as f:
    CFG = yaml.safe_load(f)

DEBUG_MODE = bool(CFG.get("logging", {}).get("debug", False) or CFG.get("DEBUG", False))

NUM_AGENTS = int(CFG.get("NUM_AGENTS", 6))
NUM_WEREWOLVES = int(CFG.get("NUM_WEREWOLVES", 1))

SCREEN_W = int(CFG.get("SCREEN_W", 1200))
SCREEN_H = int(CFG.get("SCREEN_H", 600))
FPS = int(CFG.get("FPS", 1))
AGENT_R = int(CFG.get("AGENT_R", 30))

MSG_LOG_LIMIT = int(CFG.get("MSG_LOG_LIMIT", 12))
MSG_BOX_W = int(CFG.get("MSG_BOX_W", 360))
MSG_BOX_X = int(CFG.get("MSG_BOX_X", SCREEN_W - MSG_BOX_W))

USE_LANGUAGE = bool(CFG.get("USE_LANGUAGE", True))

RUBRIC_PATH = CFG.get("RUBRIC_PATH", "judge_rubric.yaml")
PLANNER_TOPK = int(CFG.get("PLANNER_TOPK", 3))

LOG_CFG = CFG.get("logging", {}) if isinstance(CFG.get("logging", {}), dict) else {}
LOG_DIR       = LOG_CFG.get("dir", "logs")
METRICS_CSV   = LOG_CFG.get("metrics_csv", f"{LOG_DIR}/metrics.csv")
LANG_METRICS_CSV = LOG_CFG.get("lang_metrics_csv", f"{LOG_DIR}/lang_metrics.csv")
RUN_CFG_PATH  = LOG_CFG.get("run_config", f"{LOG_DIR}/run_config.yaml")
RUN_META_PATH = LOG_CFG.get("run_meta",   f"{LOG_DIR}/run_meta.json")
SAVE_CFG_SNAPSHOT = bool(CFG.get("runtime", {}).get("save_config_snapshot", True))
SEEDS = CFG.get("seeds", {}) if isinstance(CFG.get("seeds", {}), dict) else {}
SEED_GLOBAL = int(SEEDS.get("global", 123))

SIM_CFG = CFG.get("sim", {}) if isinstance(CFG.get("sim"), dict) else {}
VOTE_MIX_ALPHA: float = float(SIM_CFG.get("vote_mix_alpha", 0.0))
LOG_DZ_TALK: bool = bool(SIM_CFG.get("log_dz_talk", False))
LOG_DZ_KILL: bool = bool(SIM_CFG.get("log_dz_kill", False))
POLICY: str = str(SIM_CFG.get("policy", "") or "").lower()
# Number of discussion turns per day (each alive player speaks once per turn).
# Previously this config value was ignored and everyone spoke exactly once.
DISCUSS_TURNS: int = int(SIM_CFG.get("discuss_turns", CFG.get("TURNS_DAY_DISCUSS", 1)) or 1)

def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name, None)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")

def _env_int(name: str, default: int) -> int:
    v = os.getenv(name, None)
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        return default

def _env_float(name: str, default: float) -> float:
    v = os.getenv(name, None)
    if v is None:
        return default
    try:
        return float(v)
    except Exception:
        return default

def _env_str(name: str, default: str) -> str:
    v = os.getenv(name, None)
    return v if v is not None else default

def _social_enabled_from_cfg(cfg: dict) -> bool:
    """
    Combine config-level social flags with env overrides so baselines can
    cleanly flip Stage A social on or off.
    """
    top  = bool((cfg.get("social", {}) or {}).get("enabled", True))
    sims = bool(((cfg.get("sim", {}) or {}).get("social", {}) or {}).get("enabled", True))
    default = bool(top and sims)
    env_soc = _env_bool("SOCIAL_ENABLED", default)
    env_sim = _env_bool("SIM_SOCIAL_ENABLED", env_soc)
    return bool(env_soc and env_sim)

SOCIAL_ENABLED_CFG = _social_enabled_from_cfg(CFG)

def _merged_hygiene(cfg: dict) -> SimpleNamespace:
    lang = cfg.get("language", {}) or {}
    llm_hyg = (cfg.get("llm", {}) or {}).get("hygiene", {}) or {}
    top_hyg = cfg.get("hygiene", {}) or {}
    min_words = int(lang.get("min_words", llm_hyg.get("min_words", top_hyg.get("min_words", 12))))
    redo_max = int(lang.get("redo_max", llm_hyg.get("redo_max", top_hyg.get("redo_max", 1))))
    force_q  = float(lang.get("force_question_prob", llm_hyg.get("force_question_prob", top_hyg.get("force_question_prob", 0.25))))
    return SimpleNamespace(min_words=min_words, redo_max=redo_max, force_question_prob=force_q)

HYGIENE_NS = _merged_hygiene(CFG)
HYGIENE_NS_POST = HYGIENE_NS

LLM_CFG = CFG.get("llm", {}) if isinstance(CFG.get("llm"), dict) else {}
LLM_SPK_ENABLED: bool = bool(LLM_CFG.get("speaker_enabled", True))
LLM_ALPHA: float      = float(LLM_CFG.get("alpha", 0.5))

FUSION_CFG = CFG.get("fusion", {}) if isinstance(CFG.get("fusion"), dict) else {}
ALPHA_INTENT_BIAS: float = float(FUSION_CFG.get("alpha_intent_bias", FUSION_ALPHA if 'FUSION_ALPHA' in globals() else 0.5))

# Apply env overrides
USE_LANGUAGE   = _env_bool("USE_LANGUAGE", USE_LANGUAGE)
PLANNER_TOPK   = _env_int("PLANNER_TOPK", PLANNER_TOPK)
VOTE_MIX_ALPHA = _env_float("VOTE_MIX_ALPHA", VOTE_MIX_ALPHA)
LOG_DZ_TALK    = _env_bool("LOG_DZ_TALK", LOG_DZ_TALK)
LOG_DZ_KILL    = _env_bool("LOG_DZ_KILL", LOG_DZ_KILL)
ALPHA_INTENT_BIAS = _env_float("ALPHA_INTENT_BIAS", ALPHA_INTENT_BIAS)
SEED_GLOBAL = _env_int("SEED_GLOBAL", SEED_GLOBAL)
DISCUSS_TURNS = max(1, _env_int("DISCUSS_TURNS", DISCUSS_TURNS))
# Personality steering of the PLANNER (RQ5): when on, an agent's persona biases its
# talk-intent selection (e.g. low-agreeableness/high-extraversion → more accusing).
# Previously personas only affected the language surface, never the plan.
ENABLE_PERSONA_STEER = _env_bool("ENABLE_PERSONA_STEER", bool(CFG.get("ENABLE_PERSONA_STEER", False)))
PERSONA_STEER_SCALE = float(CFG.get("PERSONA_STEER_SCALE", 1.0))

# Judge availability toggle and helper
JUDGE_ENABLED = _env_bool("JUDGE_ENABLED", True)
# Judge backend provider: openai needs an API key; local providers (hf/…) do not.
JUDGE_PROVIDER = _env_str(
    "JUDGE_PROVIDER",
    str(CFG.get("JUDGE_PROVIDER", CFG.get("LLM_PROVIDER", "openai")))
).strip().lower()
def _can_use_judge() -> bool:
    if not JUDGE_ENABLED:
        return False
    # Only the OpenAI backend requires a key; a local HF judge can run offline.
    if JUDGE_PROVIDER in ("openai", "oai", "azure", "azure_openai"):
        return os.getenv("OPENAI_API_KEY", "").strip() != ""
    return True

if not USE_LANGUAGE:
    LLM_SPK_ENABLED = False

AGENTSIM_POLICY = _env_str("AGENTSIM_POLICY", POLICY).lower()
if _env_bool("RANDOM_VOTE", False) and not AGENTSIM_POLICY:
    AGENTSIM_POLICY = "random_voting"
POLICY = AGENTSIM_POLICY or POLICY
_IS_RANDOM_POLICY = POLICY in {"random_voting", "random_vote", "random"}
_IS_HEURISTIC_POLICY = POLICY in {"heuristic_voting", "heuristic"}
_IS_JEPA_ONLY_POLICY = POLICY in {"jepa_only", "jepa_random"}

screen = font = font_s = clock = None
msg_log: Deque[Tuple[str, str]] = deque(maxlen=200)

# Seed of the game currently being simulated; set at the top of simulate_game so
# that all per-row telemetry logs the actual per-game seed rather than the
# constant module default.
_CURRENT_GAME_SEED: int = SEED_GLOBAL

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)

def write_config_snapshot(cfg: dict, path: str):
    ensure_dir(path)
    with open(path, "w",) as f:
        yaml.safe_dump(cfg, f)

def append_csv_rows(path: str, rows: list[dict]):
    if not rows:
        return
    ensure_dir(path)
    p = pathlib.Path(path)
    file_exists = p.exists()
    existing_header: List[str] = []
    if file_exists:
        try:
            with open(path, "r", newline="") as fr:
                r = csv.reader(fr)
                existing_header = next(r)
        except Exception:
            existing_header = []
    if existing_header:
        header = existing_header
    else:
        keys: set = set()
        for r in rows:
            keys.update(r.keys())
        header = sorted(keys)
    mode = "a" if file_exists and existing_header else "w"
    with open(path, mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        if mode == "w":
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})

def _assert_len(name: str, obj, expected: int, *, ctx: str = ""):
    got = (len(obj) if hasattr(obj, "__len__") else int(getattr(obj, "numel", lambda: -1)()))
    assert got == expected, (
        f"[SANITY] {name} length mismatch: expected={expected} got={got}"
        + (f" | {ctx}" if ctx else "")
    )

def _check_aux(aux: dict, *, round_num: int, agent: str):
    assert isinstance(aux, dict) and "alive" in aux, f"[SANITY] aux missing alive | r{round_num} {agent}"
    _assert_len("aux['alive']", aux["alive"], NUM_AGENTS, ctx=f"r{round_num} agent={agent}")
    if "wolves" in aux:
        _assert_len("aux['wolves']", aux["wolves"], NUM_AGENTS, ctx=f"r{round_num} agent={agent}")

def _check_mask(mask: torch.Tensor, expected: int, *, kind: str, round_num: int, agent: str):
    assert isinstance(mask, torch.Tensor) and mask.dim() == 1, \
        f"[SANITY] {kind} mask must be 1-D tensor | r{round_num} {agent}"
    _assert_len(f"{kind} mask", mask, expected, ctx=f"r{round_num} agent={agent}")
    assert torch.isfinite(mask.float()).all(), f"[SANITY] {kind} mask has non-finite values | r{round_num} agent={agent}"

def emit_event(rows, *, run_id, round_num, phase_code, phase_str, agent, role,
               choice_type, payload_idx, mask_names=None, judge=None,
               dz=None, speaker_mode="", persona_norm=0.0, error: str = "", policy: str = ""):
    j = judge if (judge and not error) else None
    rows.append({
        "run_id": run_id,
        "seed": _CURRENT_GAME_SEED,
        "round": round_num,
        "phase": phase_str,
        "phase_code": phase_code,
        "agent": agent,
        "role": role,
        "choice_type": choice_type,
        "choice_payload": payload_idx,
        "mask_size": (len(mask_names) if mask_names is not None else ""),
        "judge_score": "" if not j else f"{j.get('score', 0):.4f}",
        "coh": "" if not j else f"{j.get('coherence', 0):.4f}",
        "truth": "" if not j else f"{j.get('truthfulness', 0):.4f}",
        "role_score": "" if not j else f"{j.get('role_alignment', 0):.4f}",
        "safety": "" if not j else f"{j.get('social_safety', 0):.4f}",
        "dz_l2": "" if not dz else f"{dz.get('l2',0):.6f}",
        "dz_1mcos": "" if not dz else f"{dz.get('1mcos',0):.6f}",
        "speaker_mode": speaker_mode,
        "persona_norm": persona_norm,
        "error": error or "",
        "policy": policy or "",
        "target": "",
        "target_is_wolf": "",
    })

def _annotate_vote(rows: list[dict], *, round_num: int, agent_name: str, target_name: Optional[str], target_role: Optional[str]):
    try:
        target_is_wolf = ""
        if target_role is not None:
            target_is_wolf = int(target_role == WEREWOLF)
        for row in reversed(rows):
            if row.get("round") != round_num:
                break
            if row.get("phase") == "DAY_VOTE" and row.get("agent") == agent_name and not row.get("target"):
                row["target"] = target_name or ""
                row["target_is_wolf"] = target_is_wolf
                break
    except Exception:
        pass

def _mouth_log(msg: str):
    print(f"[MOUTHPIECE] {msg}", flush=True)

def _recent_texts_of(ag, k: int = 3) -> list[str]:
    try:
        return [m for (_n, m) in list(ag.message_memory)[-k:] if m and m.strip()]
    except Exception:
        return []

def _finite(t: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(t):
        return torch.tensor([], dtype=torch.float32)
    return torch.nan_to_num(t, nan=0.0, posinf=1e6, neginf=-1e6)

def _finite_mean(xs: List[float]) -> float:
    vals = [float(v) for v in xs if np.isfinite(v)]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))

def _persona_talk_bias(ag: BaseAgent, num_cats: int, device) -> Optional[torch.Tensor]:
    """
    Persona-derived additive bias over talk-intent categories
    (0 accuse, 1 defend, 2 hedge, 3 question, 4 vote). Gated by ENABLE_PERSONA_STEER.
    Makes personality produce distinct *planning* styles, not just phrasing.
    """
    if not ENABLE_PERSONA_STEER:
        return None
    eff = getattr(ag, "persona_effects", None)
    if not isinstance(eff, dict):
        return None
    b = torch.zeros(num_cats, device=device)
    accuse_boost = (float(eff.get("accuse_bias_scale", 1.0)) - 1.0)   # ~[-0.5, +0.5]
    hedge_boost = float(eff.get("hedge_prob_boost", 0.0)) * 2.0       # ~[-0.16, +0.24]
    if num_cats > 0:
        b[0] += accuse_boost                       # accuse
    if num_cats > 4:
        b[4] += 0.5 * accuse_boost                 # explicit vote (assertiveness)
    if num_cats > 2:
        b[2] += hedge_boost                        # hedge
    return b * PERSONA_STEER_SCALE


@torch.no_grad()
def _fused_intent_for_agent(ag: BaseAgent, z_t: torch.Tensor, *, recent_texts: List[str], alpha: Optional[float] = None) -> Dict[str, torch.Tensor]:
    th_logits = None
    try:
        fp = getattr(ag, "planner_factorized", None)
        if fp is not None and hasattr(fp, "talk"):
            num_cats = int(getattr(fp.talk.net[-1], "out_features", 5))  # type: ignore
            mask = torch.ones(1, num_cats, dtype=torch.bool, device=z_t.device)
            th_logits = fp.talk(z_t.unsqueeze(0), mask=mask).squeeze(0).float().detach()
            # Personality steering of the plan (RQ5).
            pbias = _persona_talk_bias(ag, num_cats, z_t.device)
            if pbias is not None:
                th_logits = th_logits + pbias
    except Exception:
        th_logits = None

    bh_logits = None
    try:
        if isinstance(getattr(ag, "bias_head", None), LogitBiasHead):
            h = _mk_hist_feats(recent_texts).to(z_t.device)
            if h.dim() == 1:
                h = h.unsqueeze(0)
            if z_t.dim() == 1:
                z_in = z_t.unsqueeze(0)
            else:
                z_in = z_t
            bh_logits = ag.bias_head(z_in, h).squeeze(0).detach().float()
    except Exception:
        bh_logits = None

    th_p = torch.softmax(th_logits, -1) if th_logits is not None else None
    bh_p = torch.softmax(bh_logits, -1) if bh_logits is not None else None

    use_alpha = float(ALPHA_INTENT_BIAS if alpha is None else alpha) if 'ALPHA_INTENT_BIAS' in globals() else float(ALPHA_INTENT_BIAS if alpha is None else alpha)
    if th_p is None and bh_p is None:
        fused = None
    elif th_p is None:
        fused = bh_p
    elif bh_p is None:
        fused = th_p
    else:
        fused = use_alpha * th_p + (1.0 - use_alpha) * bh_p
        fused = fused / fused.sum().clamp_min(1e-6)

    if fused is not None:
        cat_id = int(torch.multinomial(fused, 1).item())
    elif th_logits is not None:
        cat_id = int(torch.argmax(th_logits).item())
        fused = torch.softmax(th_logits, -1)
    else:
        cat_id = int(getattr(ag, "talk_category_last", -1))
        if cat_id is None or cat_id < 0:
            cat_id = 0
        C = int(getattr(th_logits, "numel", lambda: 5)()) if th_logits is not None else 5
        fused = torch.full((C,), 1.0 / C)

    return {
        "th_logits": th_logits if th_logits is not None else torch.tensor([]),
        "bh_logits": bh_logits if bh_logits is not None else torch.tensor([]),
        "fused_probs": fused.detach().cpu(),
        "cat_id": torch.tensor(int(cat_id)),
    }

def _aux_with_texts(ag: BaseAgent, agents: List[BaseAgent]) -> dict:
    aux = ag.make_aux(agents)
    try:
        aux["recent_texts"] = _recent_texts_of(ag, k=3)
        aux["neighbor_texts"] = [m for (n, m) in list(ag.message_memory)[-6:] if n != ag.name and m and m.strip()]
    except Exception:
        aux["recent_texts"] = []
        aux["neighbor_texts"] = []
    return aux

def _safe_norm_probs(x: torch.Tensor) -> torch.Tensor:
    s = float(x.sum().item())
    if s <= 0.0 or not torch.isfinite(x).all():
        return torch.full_like(x, 1.0 / max(1, x.numel()))
    return x / s

def _mix_topk_scores(topk_probs_planner: List[float], judged: List[dict], alpha: float) -> int:
    if alpha <= 0.0:
        return max(range(len(judged)), key=lambda i: float(judged[i].get("score", 0.0)))
    if alpha >= 1.0:
        return int(torch.tensor(topk_probs_planner).argmax().item())
    p_pl = torch.tensor([max(0.0, float(p)) for p in topk_probs_planner], dtype=torch.float32)
    p_pl = _safe_norm_probs(p_pl)
    p_j = torch.tensor([max(0.0, float(x.get("score", 0.0))) for x in judged], dtype=torch.float32)
    p_j = _safe_norm_probs(p_j)
    mix = alpha * p_pl + (1.0 - alpha) * p_j
    return int(torch.argmax(mix).item())

def _wrap(text: str, fnt: pygame.font.Font, width: int) -> list[str]:
    out, cur = [], ""
    for word in text.split():
        test = f"{cur}{word} "
        if fnt.size(test)[0] <= width:
            cur = test
        else:
            out.append(cur)
            cur = f"{word} "
    if cur:
        out.append(cur)
    return out

def _draw_log() -> None:
    box_h = 30 * MSG_LOG_LIMIT + 10
    pygame.draw.rect(
        screen, (20, 20, 20),
        (MSG_BOX_X - 10, SCREEN_H - box_h - 10, MSG_BOX_W + 20, box_h),
    )
    y = SCREEN_H - 40
    for name, msg in list(msg_log)[-MSG_LOG_LIMIT:][::-1]:
        for line in _wrap(f"{name}: {msg}", font_s, MSG_BOX_W)[::-1]:
            screen.blit(font_s.render(line.strip(), True, (220, 220, 220)), (MSG_BOX_X, y))
            y -= 20

def draw_agents(agents: list[BaseAgent]) -> None:
    screen.fill((30, 30, 30))
    pad = 80
    spacing = (SCREEN_W - MSG_BOX_W - 2 * pad) // max(1, len(agents) - 1)
    y = SCREEN_H // 3
    for idx, ag in enumerate(agents):
        x = pad + idx * spacing
        colour = (200, 0, 0) if ag.alive else (80, 80, 80)
        pygame.draw.circle(screen, colour, (x, y), AGENT_R)
        screen.blit(font.render(ag.name, True, (255, 255, 255)), (x - AGENT_R, y + AGENT_R + 10))
    _draw_log()
    pygame.display.flip()

def _get_heads(ag: BaseAgent):
    if hasattr(ag, "planner_factorized") and ag.planner_factorized is not None:
        return ag.planner_factorized
    return ag.planner

def _vote_mask_for(ag: BaseAgent, agents: list[BaseAgent], num_agents: int) -> torch.Tensor:
    mask = torch.zeros(num_agents, dtype=torch.bool)
    for x in agents:
        if x.alive and x.name != ag.name:
            idx = int(x.name.split("_")[1])
            if 0 <= idx < num_agents:
                mask[idx] = True
    _check_mask(mask, num_agents, kind="vote", round_num=-1, agent="(build)")
    return mask

def _kill_mask_for(wolf: BaseAgent, agents: list[BaseAgent], num_agents: int) -> torch.Tensor:
    mask = torch.zeros(num_agents, dtype=torch.bool)
    for x in agents:
        if x.alive and x.name != wolf.name and x.role != WEREWOLF:
            idx = int(x.name.split("_")[1])
            if 0 <= idx < num_agents:
                mask[idx] = True
    _check_mask(mask, num_agents, kind="kill", round_num=-1, agent="(build)")
    return mask

def _talk_mask(num_cats: int) -> torch.Tensor:
    return torch.ones(num_cats, dtype=torch.bool)

def _vote_topk_for_agent(ag: BaseAgent, z_t: torch.Tensor, agents: list[BaseAgent], k: int) -> List[tuple[str, float]]:
    alive = [x for x in agents if x.alive and x.name != ag.name]
    if not alive:
        return []
    alive_idx = [int(x.name.split("_")[1]) for x in alive]
    with torch.no_grad():
        fp = _get_heads(ag)
        if hasattr(fp, "vote"):
            try:
                num_agents = int(fp.vote.net[-1].out_features)  # type: ignore[attr-defined]
            except Exception:
                num_agents = NUM_AGENTS
            if num_agents != NUM_AGENTS:
                raise AssertionError(
                    f"[SANITY] VoteHead out_features={num_agents} != NUM_AGENTS={NUM_AGENTS} | agent={ag.name}"
                )
            vmask = _vote_mask_for(ag, agents, num_agents).unsqueeze(0)
            _check_mask(vmask.squeeze(0), num_agents, kind="vote", round_num=-1, agent=ag.name)
            logits = fp.vote(z_t.unsqueeze(0), mask=vmask).squeeze(0)
        else:
            logits = ag.planner(z_t.unsqueeze(0)).squeeze(0)
        probs  = torch.softmax(logits, dim=-1)
        masked = torch.full_like(logits, float("-inf"))
        for idx in alive_idx:
            if 0 <= idx < logits.numel():
                masked[idx] = logits[idx]
        k_eff = min(k, len(alive_idx))
        _, topi = torch.topk(masked, k=k_eff)
        out: List[tuple[str, float]] = []
        for idx in topi.tolist():
            name = f"Agent_{idx}"
            out.append((name, float(probs[idx].item())))
        return out

def _agent_context_block(ag: BaseAgent, max_lines: int = 6) -> str:
    lines = []
    for n, m in list(ag.message_memory)[-max_lines:]:
        if not m.strip():
            continue
        lines.append(f"- {n}: {m.strip()}")
    return "\n".join(lines) if lines else "- (no recent messages heard)"

def _candidate_text(target_name: str) -> str:
    return f"We should vote to eliminate {target_name}."

def _talk_vote_alignment(cat_id: int) -> float:
    """DEPRECATED constant lookup (kept for back-compat). Prefer _talk_vote_align_real."""
    if cat_id == 4:   return 1.0
    if cat_id == 0:   return 0.9
    if cat_id == 3:   return 0.6
    if cat_id == 2:   return 0.4
    if cat_id == 1:   return 0.2
    return 0.5

def _talk_vote_align_real(ag: "BaseAgent", voted_idx: int) -> float:
    """
    Real talk→vote alignment: for a directed utterance (accuse/vote), 1.0 if the
    agent voted for the same target it accused/proposed, else 0.0. Undirected
    utterances (hedge/defend/question or no target) return NaN so they are excluded
    from the mean rather than counted as a fixed constant.
    """
    cat = int(getattr(ag, "talk_category_last", -1))
    tgt = int(getattr(ag, "talk_target_last_idx", -1))
    # Only accuse (0) and explicit vote (4) are directed vote-relevant intents.
    if cat not in (0, 4) or tgt < 0:
        return float("nan")
    try:
        return 1.0 if int(voted_idx) == tgt else 0.0
    except Exception:
        return float("nan")

def _intent_name_from_id(cid: int) -> str:
    try:
        return CAT_ORDER[int(cid)]
    except Exception:
        return "hedge"

def _resolve_target(idx: Optional[int], alive_agents: List[BaseAgent]) -> Optional[str]:
    try:
        if idx is None or idx < 0 or idx >= NUM_AGENTS:
            return None
        name = f"Agent_{idx}"
        return name if any(a.name == name and a.alive for a in alive_agents) else None
    except Exception:
        return None

def _pick_valid_target(ag: BaseAgent,
                       living: List[BaseAgent],
                       prefer: Optional[List[str]] = None) -> Optional[str]:
    names_alive = {a.name for a in living if a.alive}
    if prefer:
        for nm in prefer:
            if nm and nm in names_alive and nm != ag.name:
                return nm
    try:
        if getattr(ag, "message_memory", None):
            for n, _ in reversed(ag.message_memory):
                if n and n in names_alive and n != ag.name:
                    return n
    except Exception:
        pass
    candidates = [a.name for a in living if a.alive and a.name != ag.name]
    return random.choice(candidates) if candidates else None

def _log_utterance(speaker: str, text: str, *, visual: bool):
    try:
        print(f"{speaker}: {text}", flush=True)
    except Exception:
        pass
    if visual:
        msg_log.append((speaker, text))

_AGENT_MENTION_RE = re.compile(
    r"""
    (?:
        \bAgent
        (?:[_\s]*|['’]\s*)
        (\d+)
        (?:\b|(?=['’]s))
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)
_AGENT_STRAY_RE = re.compile(r"\bAgent['’]\b", re.IGNORECASE)

def _sanitize_agent_mentions(
    text: str,
    *,
    allow_specific: bool,
) -> str:
    if not text:
        return text
    s = text
    s = _AGENT_STRAY_RE.sub("someone", s)

    def _norm_or_neutral(m: re.Match) -> str:
        num = m.group(1)
        if not allow_specific:
            end = m.end()
            possessive = ""
            if end < len(s) and s[end:end+2] in ("'s", "’s"):
                possessive = "'s"
            return f"someone{possessive}"
        start, end = m.span()
        possessive = ""
        if end < len(s) and s[end:end+2] in ("'s", "’s"):
            possessive = s[end:end+2]
        return f"Agent_{num}{possessive}"

    s = s.replace("’s", "'s")
    s = _AGENT_MENTION_RE.sub(_norm_or_neutral, s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _ensure_planner_heads_match_roster(ag: BaseAgent, num_agents: int) -> None:
    gp = getattr(ag, "get_planner", None)
    if callable(gp):
        try:
            p = gp(num_agents_override=num_agents)
            v = getattr(p, "vote", None)
            assert v is not None and hasattr(v, "net") and hasattr(v.net[-1], "out_features"), \
                f"[SANITY] Planner missing vote head for {ag.name}"
            out_dim = v.net[-1].out_features
            assert out_dim == num_agents, \
                f"[SANITY] VoteHead out_features={out_dim} != NUM_AGENTS={num_agents} | agent={ag.name}"
            return
        except TypeError:
            pass
        except AssertionError as e:
            raise
        except Exception as e:
            print(f"[SANITY] get_planner override check failed for {ag.name}: {type(e).__name__}: {e}")
    try:
        p = gp() if callable(gp) else _get_heads(ag)
        v = getattr(p, "vote", None)
        if v is not None and hasattr(v, "net") and hasattr(v.net[-1], "out_features"):
            out_dim = v.net[-1].out_features
            assert out_dim == num_agents, \
                f"[SANITY] VoteHead out_features={out_dim} != NUM_AGENTS={num_agents} | agent={ag.name}"
    except Exception as e:
        print(f"[SANITY] legacy planner check failed for {ag.name}: {type(e).__name__}: {e}")

def _speaker_llm_fallback_generate(
    ag: BaseAgent,
    z_t: torch.Tensor,
    *,
    role: str,
    phase: str,
    plan: dict,
    recent_texts: List[str],
) -> tuple[str, dict]:
    try:
        tok = getattr(ag, "tokenizer", None) or _LLM_TOK
        prompt, base_kwargs = build_prompt_and_controls(
            tokenizer=tok,
            role=role or "Unknown",
            phase=phase,
            intent=str(plan.get("intent", "hedge")),
            plan=plan,
            recent_texts=recent_texts,
            dialog_state=getattr(ag, "dialog_state", None),
            self_name=ag.name,
        )
        if not (plan.get("target") or plan.get("target_idx") is not None):
            prompt += (
                "\n\n[STYLE]\n"
                "If you do not have a specific player as a target, speak generally.\n"
                "Do not name or refer to any player by 'Agent_#' or by name.\n"
                "Prefer neutral phrasing such as 'someone', 'a player', or 'the situation'."
            )
        talkhead_probs = None
        if "fused_probs" in plan and isinstance(plan["fused_probs"], list) and plan["fused_probs"]:
            talkhead_probs = torch.tensor(plan["fused_probs"], dtype=torch.float32)

        if isinstance(getattr(ag, "bias_head", None), LogitBiasHead) and tok is not None:
            fused_kwargs = with_fused_bias_generate_kwargs(
                tokenizer=tok,
                head=ag.bias_head,
                z_t=z_t,
                talkhead_probs=talkhead_probs,
                alpha=float(ALPHA_INTENT_BIAS),
                role=role or "Unknown",
                recent_texts=recent_texts,
                persona_effects=getattr(ag, "persona_effects", None),
            )
        else:
            fused_kwargs = {}

        gen_kwargs = dict(base_kwargs)
        if fused_kwargs:
            lp = list(gen_kwargs.get("logits_processor", []))
            lp.extend(fused_kwargs.get("logits_processor", []))
            gen_kwargs["logits_processor"] = lp

        llm = getattr(ag, "llm_fn", None)
        text_out = None
        if callable(llm):
            text_out = llm(prompt, generate_kwargs=gen_kwargs)
            if isinstance(text_out, tuple) and len(text_out) >= 1:
                text_out = text_out[0]
        text_out = text_out if isinstance(text_out, str) else ""
        text_guarded, meta_guard = guard_and_shape(text_out, plan, role or "Unknown", phase, HYGIENE_NS)
        text_guarded = text_guarded if (text_guarded and text_guarded.strip()) else SAFE_FALLBACK
        rp = repetition_penalty(text_guarded, n=2)
        meta = {"repetition_penalty": rp, **meta_guard}
        return text_guarded, meta
    except Exception as e:
        _mouth_log(f"{ag.name} speaker_llm fallback path failed: {type(e).__name__}: {e}")
        return SAFE_FALLBACK, {"repetition_penalty": None, "violated": False, "redo": False}

# Heuristic vote chooser for heuristic policy
def _heuristic_vote_choice(ag: BaseAgent, living: list[BaseAgent]) -> Optional[str]:
    counts = Counter()
    try:
        for n, m in list(getattr(ag, "message_memory", []))[-12:]:
            if not m:
                continue
            for other in living:
                if other.name == ag.name:
                    continue
                if other.name in m:
                    counts[other.name] += 1
    except Exception:
        pass
    if counts:
        cand, _ = counts.most_common(1)[0]
        if any(x.name == cand and x.alive for x in living):
            return cand
    try:
        z_t = _finite(ag.encode_current_belief(getattr(ag, "round_num_hint", 0), living).detach())
    except Exception:
        z_t = None
    try:
        topk = _vote_topk_for_agent(ag, z_t, living, k=PLANNER_TOPK) if z_t is not None else []
        if topk:
            return topk[0][0]
    except Exception:
        pass
    choices = [x.name for x in living if x.alive and x.name != ag.name]
    return random.choice(choices) if choices else None

def simulate_game(visual: bool = True, seed: int | None = None):
    # Resolve the per-game seed. Priority: explicit arg > GAME_SEED env > SEED_GLOBAL.
    # Threading a distinct seed per game is what makes repeated games in a run
    # independent instead of identical (previously every game reset to SEED_GLOBAL).
    if seed is None:
        game_seed = _env_int("GAME_SEED", SEED_GLOBAL)
    else:
        game_seed = int(seed)
    global _CURRENT_GAME_SEED
    _CURRENT_GAME_SEED = game_seed
    env_run_id = os.getenv("RUN_ID", "").strip()
    if env_run_id:
        run_id = env_run_id
    else:
        run_id = f"run_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    set_seed(game_seed)
    pathlib.Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    if SAVE_CFG_SNAPSHOT:
        write_config_snapshot(CFG, RUN_CFG_PATH)
    with open(RUN_META_PATH, "w") as f:
        json.dump({
            "run_id": run_id,
            "seed": game_seed,
            "timestamp": int(time.time()),
            "device": str(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")),
            "policy": POLICY,
        }, f)

    lang_writer = None    # type: ignore
    if LangMetricsWriter is not None:
        try:
            lang_writer = LangMetricsWriter(csv_path=LANG_METRICS_CSV, run_id=run_id, seed=game_seed)
        except Exception:
            lang_writer = None

    _talk_align_accum: List[float] = []

    def _lang_emit(row: dict):
        # Accumulate talk→vote alignment values for the per-game outcome summary.
        try:
            v = row.get("align_tv", None)
            if v is not None:
                fv = float(v)
                if fv == fv:  # not NaN
                    _talk_align_accum.append(fv)
        except Exception:
            pass
        if lang_writer is not None:
            if hasattr(lang_writer, "write"):
                return lang_writer.write(row)
            if hasattr(lang_writer, "add"):
                return lang_writer.add(row)
        append_csv_rows(LANG_METRICS_CSV, [row])

    if visual:
        pygame.init()
        global screen, font, font_s, clock
        screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.setCaption("JEPA Werewolf Sim")
        font = pygame.font.SysFont(None, 32)
        font_s = pygame.font.SysFont(None, 24)
        clock = pygame.time.Clock()

    agents = [BaseAgent(f"Agent_{i}") for i in range(NUM_AGENTS)]
    assert len(agents) == NUM_AGENTS, "[SANITY] agent list does not match NUM_AGENTS"
    assign_roles(agents, NUM_WEREWOLVES, seed=game_seed)
    if apply_personality is not None:
        apply_personality(agents)
    print("▶ Assigned roles:", ", ".join(f"{a.name}:{a.role}" for a in agents))
    if POLICY:
        print(f"[POLICY] Using policy='{POLICY}'")

    for ag in agents:
        ag.last_delta_social_norm = 0.0
        ag.social_enabled = bool(SOCIAL_ENABLED_CFG)

    shared_msg_encoder = MessageEncoder()
    for ag in agents:
        ag.message_encoder = shared_msg_encoder

    # Load the shared, trained JEPA belief encoder once and give every agent the
    # SAME instance (single consistent representational space, per the thesis).
    shared_belief_encoder = load_shared_belief_encoder()
    shared_belief_encoder.eval()
    for ag in agents:
        ag.encoder = shared_belief_encoder

    # Load the shared, trained social-influence module and give every agent the same
    # instance (previously each agent had a fresh, untrained one → inert corrections).
    _shared_social = load_shared_social()
    if _shared_social is not None:
        _shared_social.eval()
        for ag in agents:
            ag.social = _shared_social

    # Load the trained factorized world model + phase-action encoder + factorized
    # planner per role, so decisions run on trained networks (previously the sim
    # loaded a legacy '{role}_jepa.pt' that training never produced → untrained).
    _factorized_cache: Dict[str, tuple] = {}
    for ag in agents:
        if ag.role not in _factorized_cache:
            _factorized_cache[ag.role] = load_role_models_factorized(ag.role)
        wm, pae, fplanner = _factorized_cache[ag.role]
        ag.world_model = wm
        ag.phase_action_encoder = pae
        ag.planner_factorized = fplanner

    for ag in agents:
        _ensure_planner_heads_match_roster(ag, NUM_AGENTS)

    if not USE_LANGUAGE:
        _mouth_log("USE_LANGUAGE is False, attaching static fallback")
        for ag in agents:
            ag.attach_llm(lambda prompt, generate_kwargs=None: SAFE_FALLBACK, tokenizer=None)
    else:
        from llm_script import llm_fn_from_env
        mouth_fn = llm_fn_from_env()
        try:
            from llm_script import tok as _llm_tok
        except Exception:
            _llm_tok = None
        for ag in agents:
            ag.attach_llm(mouth_fn, tokenizer=_llm_tok)

    try:
        judge_rubric = JudgeRubric.load(RUBRIC_PATH)
    except Exception as e:
        raise RuntimeError(f"Failed to load judge rubric at {RUBRIC_PATH}: {e}")

    executor = cf.ThreadPoolExecutor(max_workers=NUM_AGENTS)
    rollout = []
    round_num = 0

    metrics_rows: list[dict] = []
    phase_log: list[dict] = []
    mask_logs: list[dict] = []
    social_stats_log: list[dict] = []

    while True:
        if visual:
            pygame.event.pump()
        round_num += 1
        living = [a for a in agents if a.alive]
        print(f"\n=== Day {round_num} ===")

        try:
            if USE_LANGUAGE:
                alive_names_gate = [a.name for a in agents if getattr(a, "alive", False)]
                _set_allowed_names(alive_names_gate)
        except Exception as e:
            _mouth_log(f"allowed-names gate (DAY) failed: {type(e).__name__}: {e}")

        z_pre_talk: Dict[str, torch.Tensor] = {}
        if LOG_DZ_TALK:
            for ag in living:
                z_pre_talk[ag.name] = _finite(ag.encode_current_belief(round_num, agents).detach())

        # Each alive player speaks once per discussion turn, for DISCUSS_TURNS turns per
        # day. Repeating `living` yields full speaking passes (previously the
        # discuss_turns config was ignored and everyone spoke exactly once).
        for ag in (living * DISCUSS_TURNS):
            try:
                z_t_discuss = _finite(ag.encode_current_belief(round_num, agents).detach())
                x_t_discuss_cur = getattr(ag, "_last_obs_x", None)  # raw obs behind z_t_discuss (for encoder training)
                recent_texts = [m for (_n, m) in list(ag.message_memory)[-3:]] if getattr(ag, "message_memory", None) else []
                fused = _fused_intent_for_agent(ag, z_t_discuss, recent_texts=recent_texts, alpha=ALPHA_INTENT_BIAS)
                cat_id = int(fused["cat_id"].item())
                fused_probs = fused["fused_probs"].tolist()

                arg_id: Optional[int] = None
                named_target: Optional[str] = None
                topk_for_ref = _vote_topk_for_agent(ag, z_t_discuss, living, k=PLANNER_TOPK)
                if cat_id in (0, 4) and topk_for_ref:
                    named_target = topk_for_ref[0][0]
                    try:
                        arg_id = int(named_target.split("_")[1])
                    except Exception:
                        arg_id = None
                elif cat_id == 1:
                    named_target = ag.name
                    try:
                        arg_id = int(ag.name.split("_")[1])
                    except Exception:
                        arg_id = None
                elif cat_id == 3:
                    try:
                        named_target = list(ag.message_memory)[-1][0] if ag.message_memory else None
                        arg_id = int(named_target.split("_")[1]) if named_target else None
                    except Exception:
                        named_target = None
                        arg_id = None

                intent_name = _intent_name_from_id(cat_id)

                if intent_name == "defend" and not named_target:
                    named_target = ag.name
                    try:
                        arg_id = int(ag.name.split("_")[1])
                    except Exception:
                        arg_id = None
                elif intent_name in ("accuse", "vote", "question") and not named_target:
                    prefer = [n for (n, _p) in topk_for_ref] if topk_for_ref else None
                    named_target = _pick_valid_target(ag, living, prefer=prefer)
                    try:
                        arg_id = int(named_target.split("_")[1]) if named_target else None
                    except Exception:
                        arg_id = None

                try:
                    plan = build_plan_tuple(
                        role=ag.role or "Unknown",
                        phase="DAY_DISCUSS",
                        intent=intent_name,
                        fused_probs=fused_probs,
                        target=named_target,
                        self_name=ag.name,
                        round_num=round_num,
                    )
                except Exception:
                    plan = {"intent": intent_name, "target": named_target, "shape": ""}

                try:
                    if arg_id is not None and isinstance(plan, dict):
                        plan["target_idx"] = int(arg_id)
                    forced = _resolve_target(plan.get("target_idx"), living)
                    if forced:
                        plan["target"] = forced
                        named_target = forced
                except Exception:
                    pass

                if not LLM_SPK_ENABLED:
                    _mouth_log(f"{ag.name} router disabled by cfg, emitting SAFE_FALLBACK")
                    text = SAFE_FALLBACK
                    gen_meta = {"repetition_penalty": None, "violated": False, "redo": False}
                    ag.speaker_mode = "disabled"
                else:
                    use_primary_ok = False
                    text = ""
                    gen_meta = {}
                    try:
                        text, gen_meta = chatgpt_llm_with_bias(
                            z_t_discuss,
                            ag,
                            named_target=plan.get("target") or named_target,
                            fusion_alpha=float(ALPHA_INTENT_BIAS),
                            alpha=float(ALPHA_INTENT_BIAS),
                            plan=plan,
                            phase="DAY_DISCUSS",
                        )
                        ag.speaker_mode = "llm+bias"
                        use_primary_ok = True
                        _mouth_log(f"{ag.name} bias-path ok intent={intent_name}, target={plan.get('target') or named_target}, alpha={ALPHA_INTENT_BIAS}")
                        _mouth_log(f"{ag.name} decoded primary len={len((text or '').split())}")
                    except Exception as e:
                        _mouth_log(f"{ag.name} primary mouthpiece failed: {type(e).__name__}: {e}")
                        if DEBUG_MODE:
                            raise
                    if not use_primary_ok:
                        try:
                            plan_fb = dict(plan)
                            plan_fb["fused_probs"] = fused_probs
                            text_raw, gen_meta_fb = _speaker_llm_fallback_generate(
                                ag,
                                z_t_discuss,
                                role=ag.role or "Unknown",
                                phase="DAY_DISCUSS",
                                plan=plan_fb,
                                recent_texts=recent_texts,
                            )
                            text_guarded, meta_guard = guard_and_shape(
                                text_raw, plan_fb, ag.role or "Unknown", "DAY_DISCUSS", HYGIENE_NS
                            )
                            text = text_guarded if (text_guarded and text_guarded.strip()) else SAFE_FALLBACK
                            gen_meta = {**gen_meta_fb, **meta_guard}
                            ag.speaker_mode = "llm"
                            _mouth_log(f"{ag.name} fallback guarded len={len((text or '').split())} violated={gen_meta.get('violated', False)} redo={gen_meta.get('redo', False)}")
                        except Exception as e2:
                            _mouth_log(f"{ag.name} fallback prompt path failed: {type(e2).__name__}: {e2}")
                            text = SAFE_FALLBACK
                            gen_meta = {"repetition_penalty": None, "violated": False, "redo": False}
                            ag.speaker_mode = "safe"

                try:
                    text = postprocess_text(text, role=ag.role or "Unknown", cfg=HYGIENE_NS)
                    if (not text) or (not text.strip()):
                        _mouth_log(f"{ag.name} postprocess emptied text, forcing SAFE_FALLBACK")
                        text = SAFE_FALLBACK
                except Exception as e:
                    _mouth_log(f"{ag.name} postprocess failed: {type(e).__name__}: {e}")
                    text = text if (text and text.strip()) else SAFE_FALLBACK

                is_fallback_text = (text.strip() == SAFE_FALLBACK)
                if is_fallback_text:
                    named_target = None
                    try:
                        if isinstance(plan, dict):
                            plan["target"] = None
                            if "target_idx" in plan:
                                plan["target_idx"] = None
                    except Exception:
                        pass

                try:
                    text = normalize_utterance(text)
                except Exception:
                    text = (text or "").strip()

                allow_specific = bool((plan.get("target") or plan.get("target_idx") is not None) and not is_fallback_text)
                try:
                    text = _sanitize_agent_mentions(text, allow_specific=allow_specific)
                except Exception:
                    pass

                ag.buffer_message(ag.name, text)

                retry_count = 1 if gen_meta.get("redo") else 0
                if getattr(ag, "msg_buffer", None):
                    if not is_fallback_text:
                        ag.msg_buffer[-1]["named_target"] = plan.get("target") or named_target
                    else:
                        ag.msg_buffer[-1]["named_target"] = None
                    ag.msg_buffer[-1]["plan"] = plan
                    ag.msg_buffer[-1]["strict_ok"] = int(not gen_meta.get("violated", False))
                    ag.msg_buffer[-1]["retry_count"] = retry_count

                _log_utterance(ag.name, text, visual=visual)
                if not text or not text.strip():
                    _mouth_log(f"{ag.name} final printed text is EMPTY AFTER CLEAN")

                if getattr(ag, "msg_buffer", None):
                    row = ag.msg_buffer[-1]
                    row.update({
                        "z": z_t_discuss.detach().cpu(),
                        "template_id": cat_id,
                        "talk_category": cat_id,
                        "arg_id": arg_id,
                        "p_intent": fused_probs,
                        "hist_feats": _mk_hist_feats(recent_texts),
                        "repetition_penalty": gen_meta.get("repetition_penalty", None),
                        "round": round_num,
                        "router_dbg": None,
                    })
                ag.talk_category_last = int(cat_id)
                # Track the directed target of this utterance (accuse/vote/defend/question)
                # so talk→vote alignment can measure real consistency, not a constant lookup.
                ag.talk_target_last_idx = arg_id if arg_id is not None else -1

                try:
                    if hasattr(ag, "dialog_state") and hasattr(ag.dialog_state, "update_from_msg"):
                        ag.dialog_state.update_from_msg(
                            round_id=round_num, speaker=ag.name, text=text,
                            inferred={"intent": plan.get("intent"), "target": plan.get("target")}
                        )
                except Exception:
                    pass

                try:
                    _lang_emit({
                        "run_id": run_id,
                        "seed": _CURRENT_GAME_SEED,
                        "phase": "DAY_DISCUSS",
                        "round": round_num,
                        "agent": ag.name,
                        "words": len((text or "").split()),
                        "has_question": int("?" in (text or "")),
                        "violated_rule": int(gen_meta.get("violated", False)),
                        "intent": plan.get("intent", ""),
                        "tone": plan.get("shape", ""),
                        "target": (plan.get("target") or ""),
                        "strict_ok": int(not gen_meta.get("violated", False)),
                        "retry_count": retry_count,
                    })
                except Exception:
                    pass

                utter_judge = None
                if _can_use_judge():
                    try:
                        ctx_block = _agent_context_block(ag, max_lines=3)
                        j_items = [{"context": ctx_block, "role": ag.role or "Unknown", "candidate": text}]
                        j_res = score_batch(j_items, judge_rubric, run_id=run_id, round_num=round_num, phase="DAY_DISCUSS", agent=ag.name)[0]
                        subs = j_res.get("subscores", {}) or {}
                        utter_judge = {
                            "score": float(j_res.get("score", 0.0)),
                            "coherence": float(subs.get("coherence", 0.0)),
                            "truthfulness": float(subs.get("truthfulness", 0.0)),
                            "role_alignment": float(subs.get("role_alignment", 0.0)),
                            "social_safety": float(subs.get("social_safety", 0.0)),
                        }
                        if getattr(ag, "msg_buffer", None):
                            ag.msg_buffer[-1]["judge_score"] = utter_judge["score"]
                            ag.msg_buffer[-1]["judge_subscores"] = {
                                "coherence": utter_judge["coherence"],
                                "truthfulness": utter_judge["truthfulness"],
                                "role_alignment": utter_judge["role_alignment"],
                                "social_safety": utter_judge["social_safety"],
                            }
                    except Exception:
                        utter_judge = None

                emit_event(
                    metrics_rows,
                    run_id=run_id, round_num=round_num,
                    phase_code=0, phase_str="DAY_DISCUSS",
                    agent=ag.name, role=ag.role,
                    choice_type="TALK_INTENT",
                    payload_idx=int(cat_id),
                    judge=utter_judge,
                    speaker_mode=getattr(ag, "speaker_mode", "") or "",
                    persona_norm=getattr(ag, "persona_norm", 0.0),
                    policy=POLICY,
                )

                if LOG_DZ_TALK and ag.name in z_pre_talk:
                    try:
                        z_post = _finite(ag.encode_current_belief(round_num, agents).detach())
                        base = _finite(z_pre_talk[ag.name])
                        dz_l2 = float(torch.norm(z_post - base).item())
                        cosv = F.cosine_similarity(z_post.unsqueeze(0), base.unsqueeze(0))
                        dz_1mcos = float(1.0 - float(cosv.item()) if torch.isfinite(cosv).all() else 1.0)
                        for row in reversed(metrics_rows):
                            if row["round"] != round_num:
                                break
                            if row["phase"] == "DAY_DISCUSS" and row["agent"] == ag.name and not row.get("dz_l2"):
                                row["dz_l2"] = f"{dz_l2:.6f}"
                                row["dz_1mcos"] = f"{dz_1mcos:.6f}"
                                break
                    except Exception:
                        pass

                try:
                    z_talk_pre = z_pre_talk.get(ag.name, z_t_discuss)
                    z_talk_post = _finite(ag.encode_current_belief(round_num, agents).detach())
                    x_next_talk_cur = getattr(ag, "_last_obs_x", None)  # raw obs behind z_talk_post
                    talk_payload_t = torch.tensor(int(cat_id))
                    talk_aux = _aux_with_texts(ag, agents)
                    _check_aux(talk_aux, round_num=round_num, agent=ag.name)
                    # Raw observations for JEPA encoder training (re-encoded at train time)
                    talk_aux["x_t"] = x_t_discuss_cur
                    talk_aux["x_next"] = x_next_talk_cur
                    rollout.append((
                        z_talk_pre,
                        torch.tensor(0),
                        talk_payload_t,
                        z_talk_post,
                        ag.role or "Unknown",
                        "TALK_INTENT",
                        talk_aux,
                    ))
                except Exception:
                    pass

            except Exception as e:
                print(f"[DISCUSS ERROR] {ag.name}: {type(e).__name__}: {e}")

            if visual:
                try:
                    draw_agents(agents)
                    clock.tick(FPS)
                    for ev in pygame.event.get():
                        if ev.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit()
                except Exception:
                    pass

        living = [a for a in agents if a.alive]

        z_pre_map: Dict[str, torch.Tensor] = {}
        x_pre_map: Dict[str, torch.Tensor] = {}  # raw obs behind each z_pre (for encoder training)
        for ag in living:
            z_pre_map[ag.name] = _finite(ag.encode_current_belief(round_num, agents).detach())
            _xp = getattr(ag, "_last_obs_x", None)
            if _xp is not None:
                x_pre_map[ag.name] = _xp

        # Uniform-trust neighbor-mean latent per agent, stored on vote rollouts so the
        # social-influence module can be trained (delta_from_inputs) at train time.
        zn_mean_map: Dict[str, torch.Tensor] = {}
        # Aggregated neighbor message embedding per agent (thesis §3.9 δ_msg pathway).
        msg_neigh_map: Dict[str, torch.Tensor] = {}
        if len(living) > 1:
            _msg_emb: Dict[str, torch.Tensor] = {}
            for n in living:
                try:
                    e = n.message_encoder(getattr(n, "last_message", "") or "")
                    _msg_emb[n.name] = _finite(e).detach().reshape(-1).float()
                except Exception:
                    pass
            for ag in living:
                others = [z_pre_map[n.name] for n in living if n.name != ag.name and n.name in z_pre_map]
                if others:
                    zn_mean_map[ag.name] = torch.stack(others, 0).mean(0).detach()
                m_others = [_msg_emb[n.name] for n in living if n.name != ag.name and n.name in _msg_emb]
                if m_others:
                    msg_neigh_map[ag.name] = torch.stack(m_others, 0).mean(0)

        per_agent_deltas: List[float] = []
        per_agent_stats: List[dict] = []
        z_post_map: Dict[str, torch.Tensor] = {}
        EPS = 1e-12

        if SOCIAL_ENABLED_CFG:
            for ag in living:
                z_self = _finite(z_pre_map[ag.name])
                info = {}
                # Use the trained, message-aware social correction (delta_from_inputs)
                # so eval matches training and carries the wolf-directed signal.
                try:
                    mu = zn_mean_map.get(ag.name)
                    mm = msg_neigh_map.get(ag.name)
                    soc = getattr(ag, "social", None)
                    if soc is not None and mu is not None and hasattr(soc, "delta_from_inputs"):
                        with torch.no_grad():
                            d = soc.delta_from_inputs(
                                z_self.reshape(1, -1),
                                mu.reshape(1, -1),
                                mm.reshape(1, -1) if mm is not None else None,
                            ).reshape(-1)
                        z_updated = z_self + _finite(d)
                        info = {"delta_norm": float(d.norm().item())}
                        ag.last_delta_social_norm = float(d.norm().item())
                    else:
                        res = ag.compute_social_update(z_self, [n for n in living if n.name != ag.name])
                        z_updated, info = (res[0], res[1] or {}) if isinstance(res, tuple) else (res, {})
                except Exception:
                    z_updated, info = z_self, {}

                if z_updated is None or not torch.is_tensor(z_updated) or not torch.isfinite(z_updated).all():
                    z_updated = z_self
                z_updated = _finite(z_updated).detach()
                z_post_map[ag.name] = z_updated

                dn_from_info = float(info.get("delta_norm", 0.0) or 0.0)
                dn_fallback = float(torch.norm(z_updated - z_self).item())
                delta_norm = dn_from_info if dn_from_info > EPS else dn_fallback

                ag.last_delta_social_norm = delta_norm
                ag.social_enabled = True
                per_agent_deltas.append(delta_norm)

                try:
                    zj = _finite(z_self)
                    if zj.numel() == 0:
                        var_pre = float("nan")
                    else:
                        v = (zj + 1e-8 * torch.randn_like(zj)).var(unbiased=False)
                        v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                        var_pre = float(v.item())
                except Exception:
                    var_pre = float("nan")

                try:
                    zj_post = _finite(z_updated)
                    if zj_post.numel() == 0:
                        var_post = float("nan")
                    else:
                        v_post = (zj_post + 1e-8 * torch.randn_like(zj_post)).var(unbiased=False)
                        v_post = torch.nan_to_num(v_post, nan=0.0, posinf=0.0, neginf=0.0)
                        var_post = float(v_post.item())
                except Exception:
                    var_post = float("nan")

                per_agent_stats.append({
                    "agent": ag.name,
                    "role": ag.role,
                    "delta_norm": delta_norm,
                    "var_pre": var_pre,
                    "var_post": var_post,
                })
        else:
            for ag in living:
                z_post_map[ag.name] = z_pre_map[ag.name]
                ag.last_delta_social_norm = 0.0
                ag.social_enabled = False
                per_agent_deltas.append(0.0)
                per_agent_stats.append({
                    "agent": ag.name,
                    "role": ag.role,
                    "delta_norm": 0.0,
                    "var_pre": float("nan"),
                    "var_post": float("nan"),
                })

        try:
            def _pop_var_mean(stack: torch.Tensor) -> float:
                if stack.numel() == 0:
                    return 0.0
                vs = (stack + 1e-8 * torch.randn_like(stack)).var(dim=0, unbiased=False)
                vs = torch.nan_to_num(vs, nan=0.0, posinf=0.0, neginf=0.0)
                return float(vs.mean().item())

            if len(living) > 0:
                pre_stack  = torch.stack([_finite(z_pre_map[a.name])  for a in living],  dim=0)
                post_stack = torch.stack([_finite(z_post_map[a.name]) for a in living],  dim=0)
                dz_var_pre  = _pop_var_mean(pre_stack)
                dz_var_post = _pop_var_mean(post_stack)
            else:
                dz_var_pre, dz_var_post = 0.0, 0.0
        except Exception:
            dz_var_pre, dz_var_post = 0.0, 0.0

        def _finite_mean_list(xs: List[float]) -> float:
            vals = [float(v) for v in xs if np.isfinite(v)]
            return float(sum(vals) / max(1, len(vals))) if vals else 0.0

        mean_delta = _finite_mean_list(per_agent_deltas)
        applied_count = int(sum(1 for d in per_agent_deltas if d > EPS))

        social_stats_log.append({
            "round": round_num,
            "per_agent": per_agent_stats,
            "dz_var_pre": dz_var_pre,
            "dz_var_post": dz_var_post,
            "mean_delta_norm": mean_delta,
            "applied_count": applied_count,
            "num_agents": len(per_agent_stats),
            "enabled": bool(SOCIAL_ENABLED_CFG),
        })

        z_map: Dict[BaseAgent, torch.Tensor] = {
            ag: z_post_map.get(ag.name, z_pre_map[ag.name]) for ag in living
        }

        pending: Dict[str, Tuple[torch.Tensor, int, torch.Tensor, str, str, dict]] = {}
        vote_map: Dict[BaseAgent, BaseAgent] = {}

        for ag in living:
            z_t = _finite(z_map[ag])
            alive_names = [x.name for x in living if x.name != ag.name]
            if not alive_names:
                emit_event(
                    metrics_rows, run_id=run_id, round_num=round_num,
                    phase_code=1, phase_str="DAY_VOTE",
                    agent=ag.name, role=ag.role, choice_type="VOTE_TARGET",
                    payload_idx=-1, mask_names=[],
                    speaker_mode=getattr(ag, "speaker_mode", "") or "",
                    persona_norm=getattr(ag, "persona_norm", 0.0),
                    policy=POLICY,
                )
                continue

            # Random voting policy
            if _IS_RANDOM_POLICY:
                target_name = random.choice(alive_names)
                target = next((x for x in living if x.name == target_name), None)
                vote_map[ag] = target
                tgt_idx = int(target_name.split('_')[1])
                aux_snap = _aux_with_texts(ag, agents)
                _check_aux(aux_snap, round_num=round_num, agent=ag.name)
                pending[ag.name] = (
                    z_t.detach(),
                    1,
                    torch.tensor(int(tgt_idx)),
                    ag.role or "Unknown",
                    "VOTE_TARGET",
                    aux_snap,
                )
                print(f"Random→ {ag.name} votes {target_name}")
                align_tv = _talk_vote_align_real(ag, tgt_idx)
                emit_event(
                    metrics_rows,
                    run_id=run_id, round_num=round_num,
                    phase_code=1, phase_str="DAY_VOTE",
                    agent=ag.name, role=ag.role,
                    choice_type="VOTE_TARGET",
                    payload_idx=tgt_idx,
                    mask_names=alive_names,
                    judge=None,
                    speaker_mode=getattr(ag, "speaker_mode", "") or "",
                    persona_norm=getattr(ag, "persona_norm", 0.0),
                    policy=POLICY,
                )
                try:
                    _lang_emit({
                        "run_id": run_id,
                        "seed": _CURRENT_GAME_SEED,
                        "phase": "DAY_VOTE",
                        "round": round_num,
                        "agent": ag.name,
                        "align_tv": float(align_tv),
                    })
                except Exception:
                    pass
                _annotate_vote(metrics_rows, round_num=round_num, agent_name=ag.name, target_name=target_name, target_role=getattr(target, "role", None))
                continue

            # Heuristic voting policy
            if _IS_HEURISTIC_POLICY:
                target_name = _heuristic_vote_choice(ag, living)
                if not target_name:
                    continue
                target = next((x for x in living if x.name == target_name), None)
                vote_map[ag] = target
                tgt_idx = int(target_name.split("_")[1])
                aux_snap = _aux_with_texts(ag, agents)
                _check_aux(aux_snap, round_num=round_num, agent=ag.name)
                pending[ag.name] = (
                    z_t.detach(),
                    1,
                    torch.tensor(int(tgt_idx)),
                    ag.role or "Unknown",
                    "VOTE_TARGET",
                    aux_snap,
                )
                print(f"Heuristic→ {ag.name} votes {target_name}")
                align_tv = _talk_vote_align_real(ag, tgt_idx)
                emit_event(
                    metrics_rows,
                    run_id=run_id, round_num=round_num,
                    phase_code=1, phase_str="DAY_VOTE",
                    agent=ag.name, role=ag.role,
                    choice_type="VOTE_TARGET",
                    payload_idx=tgt_idx,
                    mask_names=alive_names,
                    judge=None,
                    speaker_mode=getattr(ag, "speaker_mode", "") or "",
                    persona_norm=getattr(ag, "persona_norm", 0.0),
                    policy=POLICY,
                )
                try:
                    _lang_emit({
                        "run_id": run_id,
                        "seed": _CURRENT_GAME_SEED,
                        "phase": "DAY_VOTE",
                        "round": round_num,
                        "agent": ag.name,
                        "align_tv": float(align_tv),
                    })
                except Exception:
                    pass
                _annotate_vote(metrics_rows, round_num=round_num, agent_name=ag.name, target_name=target_name, target_role=getattr(target, "role", None))
                continue

            # JEPA-only branch: top-1 from JEPA/plan head, no judge mixing
            if _IS_JEPA_ONLY_POLICY:
                topk = _vote_topk_for_agent(ag, z_t, living, k=PLANNER_TOPK)
                if not topk:
                    continue
                best_name = topk[0][0]
                target = next((x for x in living if x.name == best_name), None)
                if target is None:
                    target = next((x for x in living if x.name != ag.name), living[0])
                vote_map[ag] = target
                tgt_idx = int(target.name.split("_")[1])
                aux_snap = _aux_with_texts(ag, agents)
                _check_aux(aux_snap, round_num=round_num, agent=ag.name)
                pending[ag.name] = (
                    z_t.detach(),
                    1,
                    torch.tensor(int(tgt_idx)),
                    ag.role or "Unknown",
                    "VOTE_TARGET",
                    aux_snap,
                )
                print(f"JepaOnly→ {ag.name} votes {target.name}")
                align_tv = _talk_vote_align_real(ag, tgt_idx)
                emit_event(
                    metrics_rows,
                    run_id=run_id, round_num=round_num,
                    phase_code=1, phase_str="DAY_VOTE",
                    agent=ag.name, role=ag.role,
                    choice_type="VOTE_TARGET",
                    payload_idx=tgt_idx,
                    mask_names=alive_names,
                    judge=None,
                    speaker_mode=getattr(ag, "speaker_mode", "") or "",
                    persona_norm=getattr(ag, "persona_norm", 0.0),
                    policy=POLICY,
                )
                try:
                    _lang_emit({
                        "run_id": run_id,
                        "seed": _CURRENT_GAME_SEED,
                        "phase": "DAY_VOTE",
                        "round": round_num,
                        "agent": ag.name,
                        "align_tv": float(align_tv),
                    })
                except Exception:
                    pass
                _annotate_vote(metrics_rows, round_num=round_num, agent_name=ag.name, target_name=target.name if target else "", target_role=getattr(target, "role", None))
                continue

            # Planner top-k
            topk = _vote_topk_for_agent(ag, z_t, living, PLANNER_TOPK)
            if not topk:
                continue

            assert all(n.startswith("Agent_") for n in alive_names), \
                f"[SANITY] vote mask has bad names: {alive_names} | r{round_num} {ag.name}"
            assert len(set(alive_names)) == len(alive_names), \
                f"[SANITY] vote mask has duplicates: {alive_names} | r{round_num} {ag.name}"

            mask_logs.append({"round": round_num, "phase": "DAY_VOTE", "actor": ag.name, "mask": alive_names})
            phase_log.append({"round": round_num, "phase": "DAY_VOTE"})

            # If judge is unavailable, pick planner top1 and log error
            if not _can_use_judge():
                best_name = topk[0][0]
                target = next((x for x in living if x.name == best_name), None)
                vote_map[ag] = target
                tgt_idx = int(best_name.split('_')[1])
                aux_snap = _aux_with_texts(ag, agents)
                _check_aux(aux_snap, round_num=round_num, agent=ag.name)
                pending[ag.name] = (
                    z_t.detach(),
                    1,
                    torch.tensor(int(tgt_idx)),
                    ag.role or "Unknown",
                    "VOTE_TARGET",
                    aux_snap,
                )
                print(f"PlannerNoJudge→ {ag.name} votes {best_name}")
                align_tv = _talk_vote_align_real(ag, tgt_idx)
                emit_event(
                    metrics_rows,
                    run_id=run_id, round_num=round_num,
                    phase_code=1, phase_str="DAY_VOTE",
                    agent=ag.name, role=ag.role,
                    choice_type="VOTE_TARGET",
                    payload_idx=tgt_idx,
                    mask_names=alive_names,
                    judge=None,
                    speaker_mode=getattr(ag, "speaker_mode", "") or "",
                    persona_norm=getattr(ag, "persona_norm", 0.0),
                    error="judge_disabled",
                    policy=POLICY,
                )
                try:
                    _lang_emit({
                        "run_id": run_id,
                        "seed": _CURRENT_GAME_SEED,
                        "phase": "DAY_VOTE",
                        "round": round_num,
                        "agent": ag.name,
                        "align_tv": float(align_tv),
                    })
                except Exception:
                    pass
                _annotate_vote(metrics_rows, round_num=round_num, agent_name=ag.name, target_name=best_name, target_role=getattr(target, "role", None))
                continue

            context_block = _agent_context_block(ag, max_lines=3)
            judge_items = [{
                "context": context_block,
                "role": ag.role or "Unknown",
                "candidate": _candidate_text(name),
            } for (name, _p) in topk]

            # Per-candidate talk→vote alignment for the judge: 1.0 if the candidate is
            # the target the agent accused/proposed this round, else 0.0.
            _tt = int(getattr(ag, "talk_target_last_idx", -1))
            _tcat = int(getattr(ag, "talk_category_last", -1))
            def _cand_align(nm: str) -> float:
                if _tcat not in (0, 4) or _tt < 0:
                    return 0.0
                try:
                    return 1.0 if int(nm.split("_")[1]) == _tt else 0.0
                except Exception:
                    return 0.0
            align_vec = [_cand_align(nm) for (nm, _p) in topk]

            judged = score_batch(
                judge_items, judge_rubric,
                run_id=run_id, round_num=round_num, phase="DAY_VOTE", agent=ag.name,
                alignment_values=align_vec
            )

            mix_idx = _mix_topk_scores([p for (_n, p) in topk], judged, VOTE_MIX_ALPHA)
            best_name = topk[mix_idx][0]
            best_j = judged[mix_idx]

            target = next((x for x in living if x.name == best_name), None)
            if target is None:
                target = next((x for x in living if x.name != ag.name), living[0])

            vote_map[ag] = target
            tgt_idx = int(target.name.split('_')[1])
            # Real talk→vote alignment: did the agent vote for whom it accused/proposed?
            align_tv = _talk_vote_align_real(ag, tgt_idx)
            try:
                _lang_emit({
                    "run_id": run_id,
                    "seed": _CURRENT_GAME_SEED,
                    "phase": "DAY_VOTE",
                    "round": round_num,
                    "agent": ag.name,
                    "align_tv": float(align_tv),
                })
            except Exception:
                pass
            a_idx = torch.tensor(int(tgt_idx))
            aux_snap = _aux_with_texts(ag, agents)
            _check_aux(aux_snap, round_num=round_num, agent=ag.name)
            pending[ag.name] = (
                z_t.detach(),
                1,
                a_idx,
                ag.role or "Unknown",
                "VOTE_TARGET",
                aux_snap,
            )

            if getattr(ag, "msg_buffer", None):
                ag.msg_buffer[-1]["alignment_vote"] = float(align_tv)

            subs = {}
            s_val = None
            error_tag = ""
            try:
                if isinstance(best_j, dict):
                    error_tag = str(best_j.get("error", "")).strip()
                    if "subscores" in best_j and best_j.get("subscores") is not None and not error_tag:
                        subs = best_j.get("subscores", {}) or {}
                        s_val = best_j.get("score", None)
                else:
                    error_tag = "judge_error"
            except Exception:
                error_tag = "judge_error"

            if (s_val is None) or error_tag:
                log_line = (f"Judge→ {ag.name} votes {target.name} "
                            f"[score=NA | coh=NA truth=NA role=NA safety=NA]"
                            f"{(' error='+error_tag) if error_tag else ''}")
                print(log_line)
                if visual:
                    msg_log.append(("Judge", log_line))
                emit_event(
                    metrics_rows,
                    run_id=run_id, round_num=round_num,
                    phase_code=1, phase_str="DAY_VOTE",
                    agent=ag.name, role=ag.role,
                    choice_type="VOTE_TARGET",
                    payload_idx=int(target.name.split("_")[1]) if target else -1,
                    mask_names=alive_names,
                    judge=None,
                    speaker_mode=getattr(ag, "speaker_mode", "") or "",
                    persona_norm=getattr(ag, "persona_norm", 0.0),
                    error=error_tag or "judge_error",
                    policy=POLICY,
                )
            else:
                s = float(s_val)
                log_line = (f"Judge→ {ag.name} votes {target.name} "
                            f"[score={s:.2f} | coh={subs.get('coherence',0.0):.2f} "
                            f"truth={subs.get('truthfulness',0.0):.2f} "
                            f"role={subs.get('role_alignment',0.0):.2f} "
                            f"safety={subs.get('social_safety',0.0):.2f}]")
                print(log_line)
                if visual:
                    msg_log.append(("Judge", log_line))
                emit_event(
                    metrics_rows,
                    run_id=run_id, round_num=round_num,
                    phase_code=1, phase_str="DAY_VOTE",
                    agent=ag.name, role=ag.role,
                    choice_type="VOTE_TARGET",
                    payload_idx=int(target.name.split("_")[1]) if target else -1,
                    mask_names=alive_names,
                    judge={
                        "score": s,
                        "coherence": subs.get("coherence", 0.0),
                        "truthfulness": subs.get("truthfulness", 0.0),
                        "role_alignment": subs.get("role_alignment", 0.0),
                        "social_safety": subs.get("social_safety", 0.0),
                    },
                    speaker_mode=getattr(ag, "speaker_mode", "") or "",
                    persona_norm=getattr(ag, "persona_norm", 0.0),
                    policy=POLICY,
                )

            for row in reversed(metrics_rows):
                if row["round"] != round_num:
                    break
                if row["phase"] == "DAY_VOTE" and row["agent"] == ag.name and not row.get("align_tv"):
                    row["align_tv"] = f"{align_tv:.3f}"
                    break

            _annotate_vote(metrics_rows, round_num=round_num, agent_name=ag.name, target_name=target.name if target else "", target_role=getattr(target, "role", None))

        if vote_map:
            print("✉ Votes:", ", ".join(f"{k.name}->{v.name}" for k, v in vote_map.items()))
        else:
            print("✉ Votes: (none)")

        try:
            alive_now = _alive_name_set(agents)
        except Exception:
            alive_now = [a.name for a in agents if a.alive]
        votes_dict = {k.name: v.name for k, v in vote_map.items()}
        res = resolve_votes_detailed(votes_dict, alive_names=alive_now)
        if res.get("winner") is not None:
            win_name = res["winner"]
            victim = next((a for a in agents if a.name == win_name), None)
            if victim is not None and victim.alive:
                eliminate_player(victim)
                print(f"[DAY_KILL] {win_name}")
                if visual:
                    msg_log.append(("System", f"{win_name} eliminated by day vote."))
        else:
            print("[DAY_KILL] no elimination (tie or no votes)")
            if visual:
                msg_log.append(("System", "No elimination (tie or no votes)."))

        try:
            for ag in agents:
                if getattr(ag, "msg_buffer", None) and ag.msg_buffer:
                    nt = ag.msg_buffer[-1].get("named_target", None)
                    ag.msg_buffer[-1]["elim"] = bool(res.get("winner") and nt and nt == res.get("winner"))
        except Exception:
            pass

        NCFG = (CFG.get("sim", {}).get("night_chat", {}) if isinstance(CFG.get("sim", {}).get("night_chat", {}), dict) else {})
        NCHAT_ON   = bool(NCFG.get("enabled", False))
        NCHAT_TURNS= int(NCFG.get("turns", 2))
        NCHAT_JUDGE= bool(NCFG.get("judge_on", False))
        NCHAT_DSOC = float(NCFG.get("delta_social_scale", 1.0))
        NCHAT_UI   = bool(NCFG.get("ui_preview", False))
        NCHAT_HINT = str(NCFG.get("intent_target", "top1")).lower()
        NCHAT_FUSE = bool(NCFG.get("use_bias_fusion", True))

        wolves = [a for a in agents if a.alive and a.role == WEREWOLF]
        consensus_tally: Counter[str] = Counter()

        try:
            if USE_LANGUAGE:
                alive_names_gate = [a.name for a in agents if getattr(a, "alive", False)]
                _set_allowed_names(alive_names_gate)
        except Exception as e:
            _mouth_log(f"allowed-names gate (NIGHT) failed: {type(e).__name__}: {e}")

        if NCHAT_ON and len(wolves) >= 1:
            for t in range(max(0, NCHAT_TURNS)):
                for wolf in wolves:
                    try:
                        z_t_talk = wolf.encode_current_belief(round_num, agents).detach()
                        recent_texts = _recent_texts_of(wolf, k=3)
                        fused = _fused_intent_for_agent(wolf, z_t_talk, recent_texts=recent_texts, alpha=ALPHA_INTENT_BIAS)
                        wolf.talk_category_last = int(fused["cat_id"].item())
                    except Exception:
                        wolf.talk_category_last = int(getattr(wolf, "talk_category_last", -1))

                    named_target = None
                    try:
                        if NCHAT_HINT == "self":
                            named_target = wolf.name
                        else:
                            z_tmp = wolf.encode_current_belief(round_num, agents)
                            non_wolf_alive = [a for a in agents if a.alive and a.role != WEREWOLF]
                            topk = _vote_topk_for_agent(wolf, z_tmp, non_wolf_alive, k=PLANNER_TOPK)
                            named_target = topk[0][0] if topk else None
                    except Exception:
                        named_target = None

                    try:
                        text, meta = chatgpt_llm_with_bias(
                            z_t_talk,
                            wolf,
                            named_target=named_target,
                            fusion_alpha=float(ALPHA_INTENT_BIAS),
                            alpha=float(ALPHA_INTENT_BIAS),
                        )
                    except Exception as e:
                        _mouth_log(f"{wolf.name} night primary failed: {type(e).__name__}: {e}")
                        text, meta = SAFE_FALLBACK, {"repetition_penalty": None}

                    is_fallback_text_n = (text.strip() == SAFE_FALLBACK)
                    if is_fallback_text_n:
                        named_target = None

                    try:
                        text = normalize_utterance(text)
                    except Exception:
                        text = (text or "").strip()
                    try:
                        allow_specific = bool(named_target) and not is_fallback_text_n
                        text = _sanitize_agent_mentions(text, allow_specific=allow_specific)
                    except Exception:
                        pass

                    for w2 in wolves:
                        w2.buffer_message(wolf.name, text)
                    try:
                        print(f"[NightChat] {wolf.name}: {text}", flush=True)
                    except Exception:
                        pass
                    if NCHAT_UI and visual:
                        msg_log.append(("NightChat", f"{wolf.name}: {text}"))

                    if getattr(wolf, "msg_buffer", None):
                        row = wolf.msg_buffer[-1]
                        row.setdefault("phase", "NIGHT_DISCUSS")
                        row["round"] = round_num
                        row["repetition_penalty"] = meta.get("repetition_penalty", None)
                        row["night_chat"] = True

                    try:
                        _lang_emit({
                            "run_id": run_id,
                            "seed": _CURRENT_GAME_SEED,
                            "phase": "NIGHT_DISCUSS",
                            "round": round_num,
                            "agent": wolf.name,
                            "words": len((text or "").split()),
                            "has_question": int("?" in (text or "")),
                            "violated_rule": 0,
                            "intent": _intent_name_from_id(int(getattr(wolf, "talk_category_last", -1)) if getattr(wolf, "talk_category_last", -1) is not None else -1),
                            "tone": "",
                            "target": (named_target or ""),
                            "strict_ok": 1,
                            "retry_count": 0,
                        })
                    except Exception:
                        pass

                    if named_target:
                        consensus_tally[named_target] += 1

                    utter_judge = None
                    if NCHAT_JUDGE and _can_use_judge():
                        try:
                            wolf_only = [(n, m) for (n, m) in list(wolf.message_memory)[-6:] if any(w.name == n for w in wolves)]
                            ctx_block = "\n".join(f"- {n}: {m.strip()}" for (n, m) in wolf_only[-3:]) or "- (no wolf chat yet)"
                            j_items = [{"context": ctx_block, "role": wolf.role or "Unknown", "candidate": text}]
                            j_res = score_batch(j_items, judge_rubric, run_id=run_id, round_num=round_num, phase="NIGHT_DISCUSS", agent=wolf.name)[0]
                            subs = j_res.get("subscores", {}) or {}
                            utter_judge = {
                                "score": float(j_res.get("score", 0.0)),
                                "coherence": float(subs.get("coherence", 0.0)),
                                "truthfulness": float(subs.get("truthfulness", 0.0)),
                                "role_alignment": float(subs.get("role_alignment", 0.0)),
                                "social_safety": float(subs.get("social_safety", 0.0)),
                            }
                        except Exception:
                            utter_judge = None

                    emit_event(
                        metrics_rows,
                        run_id=run_id, round_num=round_num,
                        phase_code=0, phase_str="NIGHT_DISCUSS",
                        agent=wolf.name, role=wolf.role,
                        choice_type="TALK_INTENT",
                        payload_idx=int(getattr(wolf, "talk_category_last", -1)),
                        judge=utter_judge,
                        speaker_mode=getattr(wolf, "speaker_mode", "") or "",
                        persona_norm=getattr(wolf, "persona_norm", 0.0),
                        policy=POLICY,
                    )

            total_mentions = sum(consensus_tally.values())
            chat_consensus_target = None
            consensus_frac = 0.0
            if total_mentions > 0:
                chat_consensus_target, cnt = consensus_tally.most_common(1)[0]
                consensus_frac = float(cnt) / float(total_mentions)
            for wolf in wolves:
                if getattr(wolf, "msg_buffer", None) and wolf.msg_buffer:
                    wolf.msg_buffer[-1]["night_consensus"] = consensus_frac
                    wolf.msg_buffer[-1]["consensus_target"] = chat_consensus_target
        else:
            chat_consensus_target = None
            consensus_frac = 0.0

        wolves = [a for a in agents if a.alive and a.role == WEREWOLF]
        villagers_alive = [a for a in agents if a.alive and a.role != WEREWOLF]
        if wolves and villagers_alive:
            legal_targets = [a.name for a in villagers_alive]
            assert all(n.startswith("Agent_") for n in legal_targets), \
                f"[SANITY] kill mask has bad names: {legal_targets} | r{round_num}"
            assert len(set(legal_targets)) == len(legal_targets), \
                f"[SANITY] kill mask has duplicates: {legal_targets} | r{round_num}"
            for wolf in wolves:
                mask_logs.append({"round": round_num, "phase": "NIGHT_VOTE", "actor": wolf.name, "mask": legal_targets})

            wolf_choices: Dict[str, str] = {}
            for wolf in wolves:
                target_name = None
                if _IS_RANDOM_POLICY:
                    target_name = random.choice(legal_targets)
                else:
                    fp_w = _get_heads(wolf)
                    try:
                        z_t_w = z_map.get(wolf, wolf.encode_current_belief(round_num, agents))
                    except Exception:
                        z_t_w = wolf.encode_current_belief(round_num, agents)
                    if hasattr(fp_w, "kill") and not _IS_RANDOM_POLICY:
                        try:
                            with torch.no_grad():
                                try:
                                    nA = int(fp_w.kill.net[-1].out_features)  # type: ignore[attr-defined]
                                except Exception:
                                    nA = NUM_AGENTS
                                kmask = _kill_mask_for(wolf, agents, nA).unsqueeze(0)
                                _check_mask(kmask.squeeze(0), nA, kind="kill", round_num=round_num, agent=wolf.name)
                                k_logits = fp_w.kill(z_t_w.unsqueeze(0), mask=kmask).squeeze(0)
                                tgt_idx = int(torch.argmax(k_logits).item())
                                candidate = f"Agent_{tgt_idx}"
                                if candidate in legal_targets:
                                    target_name = candidate
                        except Exception:
                            target_name = None
                    if target_name is None and not _IS_RANDOM_POLICY:
                        try:
                            topk_vill = _vote_topk_for_agent(wolf, _finite(z_t_w), villagers_alive, k=PLANNER_TOPK)
                            if topk_vill:
                                target_name = topk_vill[0][0]
                        except Exception:
                            target_name = None
                    if target_name is None:
                        if chat_consensus_target in legal_targets:
                            target_name = chat_consensus_target
                        else:
                            target_name = random.choice(legal_targets)

                wolf_choices[wolf.name] = target_name

                tgt_idx = int(target_name.split("_")[1])
                aux_snap = _aux_with_texts(wolf, agents)
                _check_aux(aux_snap, round_num=round_num, agent=wolf.name)
                pending[wolf.name] = (
                    _finite(z_map.get(wolf, wolf.encode_current_belief(round_num, agents))).detach(),
                    2,
                    torch.tensor(int(tgt_idx)),
                    wolf.role or "Unknown",
                    "KILL_TARGET",
                    aux_snap,
                )

                if _IS_RANDOM_POLICY:
                    print(f"RandomKill→ {wolf.name} votes {target_name}")
                else:
                    print(f"WolfVote→ {wolf.name} votes {target_name}")

            tally_list = list(wolf_choices.values())
            # Thesis night-kill rule: p_i ∝ exp(c_i / max_c). temperature=1.0 and
            # sampling via the seeded global RNG (no majority short-circuit).
            night_choice = consensus_target(tally_list, temperature=1.0)
            if night_choice and (night_choice in legal_targets):
                victim = next((a for a in agents if a.name == night_choice and a.alive), None)
            else:
                victim = None

            if victim is not None:
                eliminate_player(victim)
                print(f"[NIGHT_KILL] {victim.name}")
                if visual:
                    msg_log.append(("Night", f"{victim.name} slain."))
                emit_event(
                    metrics_rows,
                    run_id=run_id, round_num=round_num,
                    phase_code=2, phase_str="NIGHT_KILL",
                    agent=wolves[0].name, role=wolves[0].role,
                    choice_type="KILL_TARGET",
                    payload_idx=int(victim.name.split("_")[1]),
                    mask_names=legal_targets,
                    speaker_mode=getattr(wolves[0], "speaker_mode", "") or "",
                    persona_norm=getattr(wolves[0], "persona_norm", 0.0),
                    policy=POLICY,
                )
                phase_log.append({
                    "round": round_num,
                    "phase": "NIGHT_KILL",
                    "actor": "WOLVES",
                    "target": victim.name
                })
            else:
                print("[NIGHT_KILL] no consensus")
                if visual:
                    msg_log.append(("Night", "No kill (no consensus)."))

        z_deltas = []
        cos_deltas = []
        dz_by_agent: Dict[str, float] = {}
        cos_by_agent: Dict[str, float] = {}

        for ag in agents:
            if ag.name in pending:
                z_next = _finite(ag.encode_current_belief(round_num + 1, agents).detach())
                x_next_cur = getattr(ag, "_last_obs_x", None)  # raw obs behind z_next
                z_t, ph_code, payload_idx, role, choice_type, aux = pending[ag.name]
                _check_aux(aux, round_num=round_num, agent=ag.name)
                # Raw observations for JEPA encoder training (re-encoded at train time)
                if isinstance(aux, dict):
                    aux.setdefault("x_t", x_pre_map.get(ag.name))
                    aux["x_next"] = x_next_cur
                    # Neighbor-mean latent + message embedding for training social.
                    _znm = zn_mean_map.get(ag.name)
                    if _znm is not None:
                        aux["z_neigh_mean"] = _znm
                    _mnm = msg_neigh_map.get(ag.name)
                    if _mnm is not None:
                        aux["msg_neigh_mean"] = _mnm
                rollout.append((
                    z_t,
                    torch.tensor(int(ph_code)),
                    torch.tensor(int(payload_idx)),
                    z_next,
                    role or "Unknown",
                    choice_type,
                    aux,
                ))
                l2 = torch.norm(_finite(z_next) - _finite(z_t)).item()
                z_deltas.append(l2)
                cos_val_t = F.cosine_similarity(_finite(z_next).unsqueeze(0), _finite(z_t).unsqueeze(0))
                cos_val = float(cos_val_t.item()) if torch.isfinite(cos_val_t).all() else 0.0
                cos_deltas.append(1.0 - cos_val)
                dz_by_agent[ag.name] = l2
                cos_by_agent[ag.name] = 1.0 - cos_val

        if z_deltas:
            mean_l2 = _finite_mean(z_deltas)
            mean_1mcos = _finite_mean(cos_deltas) if cos_deltas else 0.0
            print(f"[Δz] L2={mean_l2:.4f}  (1-cos)={mean_1mcos:.4f}")

            for row in reversed(metrics_rows):
                if row["round"] != round_num:
                    break
                if row["phase"] == "DAY_VOTE":
                    ag_name = row["agent"]
                    if ag_name in dz_by_agent:
                        row["dz_l2"] = f"{dz_by_agent.get(ag_name, 0.0):.6f}"
                        row["dz_1mcos"] = f"{cos_by_agent.get(ag_name, 0.0):.6f}"

            if LOG_DZ_KILL:
                for row in reversed(metrics_rows):
                    if row["round"] != round_num:
                        break
                    if row["phase"] == "NIGHT_KILL":
                        ag_name = row["agent"]
                        if ag_name in dz_by_agent:
                            row["dz_l2"] = f"{dz_by_agent.get(ag_name, 0.0):.6f}"
                            row["dz_1mcos"] = f"{cos_by_agent.get(ag_name, 0.0):.6f}"

        wolves_alive = [a for a in agents if a.alive and a.role == WEREWOLF]
        vill_alive = [a for a in agents if a.alive and a.role != WEREWOLF]
        if not wolves_alive or len(wolves_alive) >= len(vill_alive):
            break

    # Compute game outcome after the loop
    wolves_alive = [a for a in agents if a.alive and a.role == WEREWOLF]
    vill_alive = [a for a in agents if a.alive and a.role != WEREWOLF]
    villager_win = (len(wolves_alive) == 0)
    if villager_win:
        game_outcome = 1.0
    else:
        game_outcome = -1.0

    # Attach game_outcome to every rollout row
    new_rollout = []
    for (z_t, ph_code, payload_idx, z_next, role, choice_type, aux) in rollout:
        try:
            aux["game_outcome"] = game_outcome
        except Exception:
            aux = dict(aux)
            aux["game_outcome"] = game_outcome
        new_rollout.append((z_t, ph_code, payload_idx, z_next, role, choice_type, aux))
    rollout = new_rollout

    print("\n== Game over ==")

    try:
        if lang_writer is not None:
            if hasattr(lang_writer, "flush"):
                lang_writer.flush()
            elif hasattr(lang_writer, "close"):
                lang_writer.close()
    except Exception:
        pass

    append_csv_rows(METRICS_CSV, metrics_rows)
    by_phase: Dict[str, int] = {}
    for r in metrics_rows:
        by_phase[r["phase"]] = by_phase.get(r["phase"], 0) + 1
    print("[SUMMARY] rows:", len(metrics_rows), "by_phase:", by_phase, f"policy={POLICY or '(default)'}")

    executor.shutdown(wait=True)

    _social_stats = social_stats_log if social_stats_log else [{
        "round": 0,
        "per_agent": [],
        "dz_var_pre": float("nan"),
        "dz_var_post": float("nan"),
        "mean_delta_norm": 0.0,
        "applied_count": 0,
        "num_agents": 0,
    }]

    # ---- Per-game outcome summary (consumed by the baseline-ladder / sweep runners) ----
    _wolves_final = [a for a in agents if a.alive and a.role == WEREWOLF]
    _villager_win = (len(_wolves_final) == 0)
    _vote_rows = [r for r in metrics_rows
                  if r.get("phase") == "DAY_VOTE" and r.get("role") != WEREWOLF]
    _acc = [float(r["target_is_wolf"]) for r in _vote_rows
            if str(r.get("target_is_wolf", "")).strip() not in ("", "None")]
    _vill_vote_acc = (sum(_acc) / len(_acc)) if _acc else float("nan")
    _jrows = [r for r in metrics_rows
              if str(r.get("judge_score", "")).strip() not in ("", "None")]
    _jacc = float("nan")
    if _jrows:
        try:
            _js = [float(r["judge_score"]) for r in _jrows]
            _jacc = sum(1.0 for s in _js if s >= 0.5) / len(_js)
        except Exception:
            _jacc = float("nan")
    _tv_align = (sum(_talk_align_accum) / len(_talk_align_accum)) if _talk_align_accum else float("nan")

    meta_out = {
        "rounds": round_num,
        "agents": agents,
        "run_id": run_id,
        "phases": phase_log,
        "mask_logs": mask_logs,
        "social_stats": _social_stats,
        "policy": POLICY,
        "outcome": {
            "winner": "villagers" if _villager_win else "werewolves",
            "villager_win": bool(_villager_win),
            "vill_vote_accuracy": float(_vill_vote_acc),
            "judge_accept": float(_jacc),
            "talk_vote_align": float(_tv_align),
            "rounds": int(round_num),
            "seed": int(game_seed),
        },
    }
    return rollout, meta_out

def safe_simulate_game(visual: bool = True):
    try:
        result = simulate_game(visual=visual)
    except Exception as e:
        print(f"[SAFE_SIM] simulate_game crashed: {type(e).__name__}: {e}")
        return [], {"agents": [], "social_stats": [], "policy": POLICY}

    if isinstance(result, tuple) and len(result) == 2:
        rollouts, meta = result
    else:
        rollouts, meta = [], result

    if not isinstance(meta, dict):
        meta = {"agents": meta if isinstance(meta, list) else [], "social_stats": [], "policy": POLICY}
    meta.setdefault("agents", [])
    meta.setdefault("social_stats", [])
    meta.setdefault("policy", POLICY)
    return rollouts, meta

if __name__ == "__main__":
    simulate_game(visual=True)
