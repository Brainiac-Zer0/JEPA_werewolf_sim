from __future__ import annotations
# llm_script.py  — prompt-first Mouthpiece with natural-dialogue rerank (LAZY LOAD)
# - Lazy singleton loader; no GPU/model allocation at import time
# - Env > YAML > defaults; respects LLM_SPEAKER=0
# - Exposes .tokenizer for fused-bias processors (Phase-5)
# - Callable: llm_fn(prompt, generate_kwargs={...})
# - Phase-5: chatgpt_llm_from_latent / chatgpt_llm_with_bias → single-line, non-meta text
# - Heuristic de-meta + gentle early-stop + shortline selection

import os, re, yaml, torch
from typing import List, Dict, Optional, Any, Tuple, TYPE_CHECKING
from dataclasses import dataclass
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

if TYPE_CHECKING:
    from agent import BaseAgent  # typing only

# ── Config loading
def _load_config(path: str = "config.yaml") -> dict:
    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        data = {}
    return data

CFG = _load_config()

# --------- OS ENV SHIM HELPERS ----------
def _env_bool(key: str, default: bool) -> bool:
    v = os.getenv(key)
    if v is None: return default
    return str(v).strip().lower() in ("1","true","yes","y","on")

def _env_str(key: str, default: str) -> str:
    v = os.getenv(key)
    return v if v is not None else default
# ----------------------------------------

# ── Config values (env → YAML → defaults)
MODEL_ID_DEFAULT = CFG.get("LLM_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2")
DEVICE_CFG       = str(CFG.get("LLM_DEVICE", "") or "").strip().lower()
LLM_SPEAKER_CFG  = bool(CFG.get("LLM_SPEAKER", False))

# ── Hygiene helpers
BAD_QUOTES = "“”\"'«»"
# We only stop at explicit line boundaries to avoid mid-thought clipping.
STOP_SEQS = ["\nAgent_", "\nSystem:", "\nNarrator:", "\rAgent_", "\rSystem:", "\rNarrator:"]

ALLOWED_ROLES = {"Werewolf", "Worker"}
ROLE_BLOCKLIST = [
    "Engineer", "engineer",
    "Scientist", "scientist",
    "Detective", "detective",
    "Doctor", "doctor",
    "Guard", "guard",
    "Scientist.", "Engineer.", "Detective.", "Doctor.", "Guard."
]

# Meta/instructional tokens to discourage in output
META_BANS = [
    "Use", "use",
    "Do not", "do not", "Don't", "don't", "No ", "no ",
    "Keep ", "keep ",
    "Reply", "reply", "Provide", "provide", "Follow", "follow",
    "Instruction", "instruction", "Rule", "rule",
    "single sentence", "Single sentence",
    "one sentence", "One sentence",
    "third person", "Third person",
    "grammar", "Grammar", "punctuation", "Punctuation",
    "No narration", "no narration",
]

def _bad_words_ids(tokenizer) -> List[List[int]]:
    toks = tokenizer(META_BANS, add_special_tokens=False)["input_ids"]
    return [ids for ids in toks if len(ids) > 0]

def _one_line(text: str) -> str:
    if not text: return "..."
    # Take the first non-empty line after trimming prompt continuation
    for raw in text.splitlines():
        s = raw.strip()
        if not s: continue
        s = s.replace("“","").replace("”","").replace('"',"").replace("’","'")
        s = s.lstrip(" -").rstrip(" .,!?:;")
        if s: return s[:180]
    return "..."

def _early_stop(generated: str) -> str:
    # Cut only at clear line starts of our stop tags
    cut_pos = len(generated)
    for s in STOP_SEQS:
        i = generated.find(s)
        if i != -1:
            cut_pos = min(cut_pos, i)
    return generated[:cut_pos]

def _sanitize_roles(text: str) -> str:
    for w in ROLE_BLOCKLIST:
        text = text.replace(w, "someone")
    return re.sub(r"\s{2,}", " ", text).strip()

def _is_meta_like(text: str) -> bool:
    t = (text or "").strip()
    if not t: return True
    low = t.lower()
    # soft heuristic: commands / meta-talk / ultra-short
    if any(p in low for p in ("sentence", "third person", "punctuation", "grammar",
                              "reply", "provide", "follow", "instruction", "rule")):
        return True
    if t.startswith(("Use ", "Do not", "Don't", "No ", "Keep ", "Reply", "Provide", "Follow")):
        return True
    if len(t.split()) < 2:
        return True
    return False

# ── Prompt builder (tokenizer required)
def _build_prompt(tokenizer, name: str, role_str: str, heard_block: str, z_summary: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are playing a hidden-role social deduction game. "
                "Speak like a player. Output one short, natural sentence. "
                "Do not include quotes, narration, or stage directions."
            ),
        },
        {
            "role": "user",
            "content": (
                "Examples of good replies:\n"
                "- I’m not convinced by their story.\n"
                "- Someone’s behavior feels off to me."
            ),
        },
        {"role": "assistant", "content": "I’m not convinced by their story."},
        {"role": "user", "content": "Another example."},
        {"role": "assistant", "content": "Someone’s behavior feels off to me."},
        {
            "role": "user",
            "content": (
                f"You are {name}{role_str} in a hidden-role deduction game.\n"
                f"Recent dialog:\n{heard_block}\n"
                f"Your private feeling:\n- {z_summary}\n\n"
                "Now reply in one short, natural sentence, as if you are speaking in the game."
            ),
        },
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# ─────────────────────────────────────────────────────────────────────────────
# LAZY SINGLETONS (no model/tokenizer created until first actual call)
# ─────────────────────────────────────────────────────────────────────────────
_tok = None
_model = None
_llm_pipeline = None
_MOUTHPIECE = None

def _resolve_device() -> str:
    cfg = _env_str("LLM_DEVICE", DEVICE_CFG).strip().lower()
    if cfg:
        return cfg
    return "cuda:0" if torch.cuda.is_available() else "cpu"

def _lazy_load_llm():
    """
    Build (tokenizer, model, pipeline) exactly once.
    Respects device & model env; ensures pad token; prints a one-line device banner.
    """
    global _tok, _model, _llm_pipeline
    if _llm_pipeline is not None:
        return _llm_pipeline, _tok

    model_id = _env_str("LLM_MODEL_ID", MODEL_ID_DEFAULT)
    device = _resolve_device()
    use_gpu = device.startswith("cuda")
    torch_dtype = torch.float16 if use_gpu else torch.float32

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token_id is None:
        if tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({"pad_token": "<|pad|>"})
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    if getattr(model.config, "vocab_size", None) is not None and len(tok) != model.config.vocab_size:
        try:
            model.resize_token_embeddings(len(tok))
        except Exception:
            pass
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tok.pad_token_id

    if device.startswith("cuda"):
        try:
            pipe_device = 0 if device == "cuda" else int(device.split(":")[1])
        except Exception:
            pipe_device = 0
    else:
        pipe_device = -1

    llm_pipe = pipeline("text-generation", model=model, tokenizer=tok, device=pipe_device)

    if use_gpu:
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = "CUDA"
        print(f"[INFO] LLM loaded on {device.upper()}: {gpu_name}")
    else:
        print(f"[INFO] LLM loaded on {device.upper()}")

    _tok, _model, _llm_pipeline = tok, model, llm_pipe
    return _llm_pipeline, _tok

def _lazy_mouthpiece():
    """
    Build/return the global Mouthpiece. Raises if LLM_SPEAKER is disabled.
    """
    global _MOUTHPIECE
    if _MOUTHPIECE is not None:
        return _MOUTHPIECE
    if not _env_bool("LLM_SPEAKER", LLM_SPEAKER_CFG):
        raise RuntimeError("LLM mouthpiece disabled by env/config (LLM_SPEAKER=0).")
    llm_pipe, tok = _lazy_load_llm()
    device = _resolve_device()
    _MOUTHPIECE = Mouthpiece(llm_pipe, tok, device)
    return _MOUTHPIECE

# ─────────────────────────────────────────────────────────────────────────────
# Mouthpiece: prompt-first callable with reranking + tokenizer exposure
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class _AttemptSpec:
    max_new_tokens: int
    temperature: float
    top_p: float
    repetition_penalty: float
    no_repeat_ngram_size: int

class Mouthpiece:
    """
    Callable mouthpiece used by Agent.speak:
        text = mouth(prompt, generate_kwargs={"logits_processor":[...], ...})
    Also exposes:
        .tokenizer  (required for fused bias processors)
    Guarantees: one short, non-meta, single-line utterance (SpeakerPolicy still filters).
    """
    def __init__(self, pipe, tokenizer, device, use_bias_default: bool = True):
        self.pipe = pipe
        self.tokenizer = tokenizer
        self.device = device
        self._bad_ids = _bad_words_ids(tokenizer)
        self.use_bias_default = use_bias_default

        # three slightly different sampling specs to generate candidates
        self._attempts: List[_AttemptSpec] = [
            _AttemptSpec(max_new_tokens=32, temperature=0.65, top_p=0.92, repetition_penalty=1.08, no_repeat_ngram_size=3),
            _AttemptSpec(max_new_tokens=28, temperature=0.55, top_p=0.90, repetition_penalty=1.12, no_repeat_ngram_size=4),
            _AttemptSpec(max_new_tokens=24, temperature=0.50, top_p=0.88, repetition_penalty=1.18, no_repeat_ngram_size=5),
        ]

    # ---- scoring: prefer short, natural, non-meta lines
    def _score_candidate(self, s: str) -> float:
        if not s: return -1.0
        length = len(s.split())
        # length prior: best around 6–18 words
        length_score = 1.0 - abs((length - 10) / 12.0)
        length_score = max(0.0, length_score)

        meta_pen = 1.0 if _is_meta_like(s) else 0.0
        # soft punctuation prior (period or short clause)
        punct_bonus = 0.2 if (s.endswith((".", "…", "!","?")) or (length <= 8 and "," not in s)) else 0.0

        return (1.0 - 0.8 * meta_pen) * (0.6 * length_score + 0.4 * punct_bonus)

    def _generate_one(self, prompt: str, attempt: _AttemptSpec, extra_kwargs: Dict[str, Any]) -> str:
        try:
            resp = self.pipe(
                prompt,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                bad_words_ids=self._bad_ids,
                max_new_tokens=attempt.max_new_tokens,
                temperature=attempt.temperature,
                top_p=attempt.top_p,
                repetition_penalty=attempt.repetition_penalty,
                no_repeat_ngram_size=attempt.no_repeat_ngram_size,
                **(extra_kwargs or {}),
            )[0].get("generated_text", "")
        except Exception as e:
            print(f"[LLM ERROR/generate] {e}")
            return "..."
        # strip prompt echo if present
        cont = resp[len(prompt):] if resp.startswith(prompt) else resp
        cont = _early_stop(cont)
        cont = _one_line(cont)
        cont = _sanitize_roles(cont)
        cont = cont.strip(BAD_QUOTES + " ")
        return cont or "..."

    def __call__(self, prompt: str, *, generate_kwargs: Optional[Dict[str, Any]] = None) -> str:
        # generate 2–3 candidates; pick highest score non-meta else best available
        cands: List[Tuple[str, float]] = []
        for spec in self._attempts:
            txt = self._generate_one(prompt, spec, generate_kwargs or {})
            cands.append((txt, self._score_candidate(txt)))

        # prefer non-meta with best score; else best overall
        non_meta = [(t, sc) for (t, sc) in cands if not _is_meta_like(t)]
        best = max(non_meta or cands, key=lambda x: x[1])[0]
        return best or "..."

    # ── Back-compat shim retained internally if needed
    def legacy_from_latent(self, z, agent, *, use_bias: bool = None) -> str:
        role = (agent.role or "").strip()
        role_str = f", a {role}," if role in ALLOWED_ROLES else ""
        z_summary = agent.decode_z(z)
        name = agent.name
        heard_block = (
            "\n".join(
                f"- {n} said: \"{m.strip()}\""
                for n, m in list(agent.message_memory)[-6:]
                if m.strip()
            ) or "- (no recent messages heard)"
        )
        prompt = _build_prompt(self.tokenizer, name, role_str, heard_block, z_summary)
        return self(prompt, generate_kwargs={})

# ── Try to import bias helpers (not required here; agent.speak builds them)
try:
    from speaker_llm import (
        LogitBiasHead,
        with_logit_bias_generate_kwargs,   # legacy bias-only path
        with_fused_bias_generate_kwargs,   # TalkHead × BiasHead fusion
        SPEAKER_HIST_K,                    # shared history window
    )  # noqa: F401
except Exception:
    LogitBiasHead = None
    with_logit_bias_generate_kwargs = None
    with_fused_bias_generate_kwargs = None
    SPEAKER_HIST_K = 3

# ── Helpers for Phase-5 single-line, in-scenario generation
def _latent_prompt_from_agent(tokenizer, z: torch.Tensor, agent: "BaseAgent") -> str:
    role = (agent.role or "").strip()
    role_str = f", a {role}," if role in ALLOWED_ROLES else ""
    z_summary = agent.decode_z(z)
    name = agent.name
    heard_block = (
        "\n".join(
            f"- {n} said: \"{m.strip()}\"" for n, m in list(agent.message_memory)[-6:] if m.strip()
        ) or "- (no recent messages heard)"
    )
    return _build_prompt(tokenizer, name, role_str, heard_block, z_summary)

@torch.no_grad()
def _talkhead_probs_for(agent: "BaseAgent", z: torch.Tensor) -> Optional[torch.Tensor]:
    """Return softmax over talk categories if TalkHead is available; else None."""
    try:
        fp = getattr(agent, "planner_factorized", None)
        if fp is None or not hasattr(fp, "talk"):
            return None
        # infer category count; default to 5 if unavailable
        try:
            num_cats = int(fp.talk.net[-1].out_features)  # type: ignore[attr-defined]
        except Exception:
            num_cats = 5
        dev = z.device if torch.is_tensor(z) else None
        mask = torch.ones(1, num_cats, dtype=torch.bool, device=dev)
        logits = fp.talk(z.unsqueeze(0), mask=mask).squeeze(0).float()
        return torch.softmax(logits, dim=-1).detach().cpu()
    except Exception:
        return None

def _recent_texts(agent, k: int = 3) -> List[str]:  # shadow-safe helper with default k
    if not getattr(agent, "message_memory", None):
        return []
    return [m for (_, m) in list(agent.message_memory)[-k:] if m]

def _maybe_build_bias_kwargs(z: torch.Tensor, agent: "BaseAgent") -> Dict[str, Any]:
    """
    Build per-call logits processors using fused TalkHead × BiasHead if available.
    Returns {} when bias isn’t wired or speaker_llm is missing (safe no-op).
    """
    if with_fused_bias_generate_kwargs is None:
        return {}
    bias_head = getattr(agent, "bias_head", None)
    if bias_head is None or not isinstance(bias_head, LogitBiasHead):
        return {}
    recent = _recent_texts(agent, k=SPEAKER_HIST_K)
    persona = getattr(agent, "persona_effects", None)
    th_probs = _talkhead_probs_for(agent, z)
    # We only need the tokenizer; using the lazy loader here is fine because this path
    # is reached only when LLM generation is requested.
    _, tok = _lazy_load_llm()
    try:
        return with_fused_bias_generate_kwargs(
            tokenizer=tok,
            head=bias_head,
            z_t=z,
            talkhead_probs=th_probs,          # None is OK; helper handles it
            alpha=None,                       # uses env/YAML default
            role=getattr(agent, "role", None),
            recent_texts=recent,
            persona_effects=persona,
        )
    except Exception as e:
        print(f"[LLM WARN] fused-bias kwargs failed: {e}")
        return {}

# ─────────────────────────────────────────────────────────────────────────────
# Phase-5 entrypoints: both return a single-line, non-meta utterance.
# ─────────────────────────────────────────────────────────────────────────────
def chatgpt_llm_from_latent(z: torch.Tensor, agent: "BaseAgent") -> str:
    """Return one short, natural in-game sentence (no meta/instructions)."""
    mouth = _lazy_mouthpiece()
    prompt = _latent_prompt_from_agent(mouth.tokenizer, z, agent)
    return mouth(prompt, generate_kwargs={})

def chatgpt_llm_with_bias(z: torch.Tensor, agent: "BaseAgent") -> str:
    """Return one short, natural in-game sentence; uses fused TalkHead×BiasHead when available."""
    mouth = _lazy_mouthpiece()
    prompt = _latent_prompt_from_agent(mouth.tokenizer, z, agent)
    bias_kwargs = _maybe_build_bias_kwargs(z, agent)
    return mouth(prompt, generate_kwargs=bias_kwargs)

# Optional convenience: pick mouthpiece by config/env
def llm_fn_from_env() -> Mouthpiece:
    """
    Return a callable mouthpiece. If LLM is disabled (LLM_SPEAKER=0),
    provide a lightweight stub that avoids raising during sim initialization.
    """
    if not _env_bool("LLM_SPEAKER", LLM_SPEAKER_CFG):
        class _Stub:
            tokenizer = None  # not used in template-bandit runs
            def __call__(self, prompt: str, *, generate_kwargs=None) -> str:
                # Should not be called in template-bandit mode; safe fallback.
                return "..."
        return _Stub()  # type: ignore[return-value]
    return _lazy_mouthpiece()
