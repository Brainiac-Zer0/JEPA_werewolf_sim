from __future__ import annotations
# llm_script.py  — single-device safe loader + clean one-liner
# (chat template + few-shot + anti-meta bans + safe fallbacks)
# NEW: optional trainable LLM mouthpiece via logit-bias head (speaker_llm.py)

import os, torch, re, yaml
from typing import List, Dict, Optional, Any
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# ── Config loading
def _load_config(path: str = "config.yaml") -> dict:
    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        data = {}
    return data

CFG = _load_config()

# ── Config values
MODEL_ID = CFG.get("LLM_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2")
LLM_SPEAKER = bool(CFG.get("LLM_SPEAKER", False))

_cfg_device = str(CFG.get("LLM_DEVICE", "") or "").strip().lower()
if _cfg_device:
    device = _cfg_device  # "", "cpu", "cuda", "cuda:0"
else:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

USE_GPU = device.startswith("cuda")

# ── 1) Tokenizer
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
# Ensure safe padding for decoder-only models
if tok.pad_token_id is None:
    # Prefer EOS as PAD; otherwise add a fresh pad token
    if tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    else:
        tok.add_special_tokens({"pad_token": "<|pad|>"})
tok.padding_side = "left"

# ── 2) Model
torch_dtype = torch.float16 if USE_GPU else torch.float32
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=False,
)
# If we added a PAD token, make sure embeddings are resized
if getattr(model.config, "vocab_size", None) is not None and len(tok) != model.config.vocab_size:
    try:
        model.resize_token_embeddings(len(tok))
    except Exception:
        pass
# Make sure model knows the pad token id
if getattr(model.config, "pad_token_id", None) is None:
    model.config.pad_token_id = tok.pad_token_id

model = model.to(device)

# ── 3) Pipeline
pipe_device = 0 if device.startswith("cuda") else -1
llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tok,
    device=pipe_device,
)

if USE_GPU:
    try:
        gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        gpu_name = "CUDA"
    print(f"[INFO] LLM loaded on {device.upper()}: {gpu_name}")
else:
    print(f"[INFO] LLM loaded on {device.upper()}")

# ── Hygiene helpers
BAD_QUOTES = "“”\"'«»"
STOP_SEQS  = ["\n", "\r", "Agent_", "System:", "Narrator:"]
ALLOWED_ROLES = {"Werewolf", "Worker"}
ROLE_BLOCKLIST = [
    "Engineer", "engineer",
    "Scientist", "scientist",
    "Detective", "detective",
    "Doctor", "doctor",
    "Guard", "guard",
    "Scientist.", "Engineer.", "Detective.", "Doctor.", "Guard."
]

# Meta/instructional tokens to discourage anywhere in the output
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

def _one_line(text: str) -> str:
    if not text:
        return "..."
    lines = text.splitlines()
    first = (lines[0] if lines else "").strip()
    first = first.replace("“","").replace("”","").replace('"',"").replace("’","'")
    first = first.lstrip(" -").rstrip(" .,!?:;")
    return first[:160] or "..."

def _early_stop(generated: str) -> str:
    for s in STOP_SEQS:
        i = generated.find(s)
        if i != -1:
            generated = generated[:i]
    return generated

def _sanitize_roles(text: str) -> str:
    for w in ROLE_BLOCKLIST:
        text = text.replace(w, "someone")
    return re.sub(r"\s{2,}", " ", text).strip()

def _is_meta(text: str) -> bool:
    t = text.strip()
    if not t:
        return True
    meta_phrases = [
        "sentence", "third person", "punctuation", "grammar",
        "use ", "do not", "don't", "no ", "reply", "provide", "follow", "instruction", "rule"
    ]
    if any(p in t.lower() for p in meta_phrases):
        return True
    bad_starts = ("Use ", "Do not", "Don't", "No ", "Keep ", "Reply", "Provide", "Follow")
    if t.startswith(bad_starts):
        return True
    if len(t.split()) < 2:
        return True
    return False

def _bad_words_ids():
    toks = tok(META_BANS, add_special_tokens=False)["input_ids"]
    return [ids for ids in toks if len(ids) > 0]

def _recent_texts(agent, k: int = 3) -> List[str]:
    if not getattr(agent, "message_memory", None):
        return []
    return [m for (_, m) in list(agent.message_memory)[-k:] if m]

def _build_prompt(name: str, role_str: str, heard_block: str, z_summary: str) -> str:
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
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# ─────────────────────────────────────────────────────────────────────────────
# Baseline (frozen) mouthpiece
# ─────────────────────────────────────────────────────────────────────────────
def chatgpt_llm_from_latent(z, agent) -> str:
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
    prompt = _build_prompt(name, role_str, heard_block, z_summary)
    bans = _bad_words_ids()

    attempts = [
        dict(max_new_tokens=32, temperature=0.6, top_p=0.9, repetition_penalty=1.1,
             do_sample=True, no_repeat_ngram_size=3, bad_words_ids=bans),
        dict(max_new_tokens=28, temperature=0.5, top_p=0.9, repetition_penalty=1.15,
             do_sample=True, no_repeat_ngram_size=4, bad_words_ids=bans),
        dict(max_new_tokens=24, temperature=0.45, top_p=0.85, repetition_penalty=1.2,
             do_sample=True, no_repeat_ngram_size=5, bad_words_ids=bans),
    ]

    try:
        out = "..."
        for params in attempts:
            resp = llm_pipeline(
                prompt,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
                **params,
            )[0].get("generated_text", "")
            cont = resp[len(prompt):] if resp.startswith(prompt) else resp
            cont = _early_stop(cont)
            cont = _one_line(cont)
            cont = _sanitize_roles(cont)
            cont = cont.strip(BAD_QUOTES + " ")
            if not _is_meta(cont):
                out = cont
                break
            else:
                out = cont
        return out or "..."
    except Exception as e:
        print(f"[LLM ERROR] {e}")
        return "..."

# ─────────────────────────────────────────────────────────────────────────────
# Trainable LLM mouthpiece (logit-bias head) — uses Any to avoid type issues
# ─────────────────────────────────────────────────────────────────────────────
try:
    from speaker_llm import LogitBiasHead, with_logit_bias_generate_kwargs
except Exception:
    LogitBiasHead = None  # type: ignore[assignment]
    with_logit_bias_generate_kwargs = None  # type: ignore[assignment]

_BIAS_HEADS: Dict[str, Any] = {}

def _get_or_make_bias_head(role: str, latent_dim: int = 32) -> Any:
    if LogitBiasHead is None:
        return None
    key = (role or "default").lower()
    if key not in _BIAS_HEADS:
        head = LogitBiasHead(latent_dim=latent_dim)
        head.to(device)
        _BIAS_HEADS[key] = head
        print(f"[LLM-SPK] Created bias head for role={key}")
    return _BIAS_HEADS[key]

def chatgpt_llm_with_bias(z, agent) -> str:
    if with_logit_bias_generate_kwargs is None:
        return chatgpt_llm_from_latent(z, agent)

    head = getattr(agent, "llm_bias_head", None) or _get_or_make_bias_head(getattr(agent, "role", ""))
    if head is None:
        return chatgpt_llm_from_latent(z, agent)

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
    prompt = _build_prompt(name, role_str, heard_block, z_summary)
    bans = _bad_words_ids()

    persona_effects = getattr(agent, "persona_effects", None)
    proc_kwargs = with_logit_bias_generate_kwargs(
        tokenizer=tok,
               head=head,
        z_t=z.detach() if torch.is_tensor(z) else torch.tensor(z),
        role=agent.role,
        recent_texts=_recent_texts(agent, k=3),
        persona_effects=persona_effects,
    )

    attempts = [
        dict(max_new_tokens=32, temperature=0.6, top_p=0.9, repetition_penalty=1.1,
             do_sample=True, no_repeat_ngram_size=3, bad_words_ids=bans),
        dict(max_new_tokens=28, temperature=0.5, top_p=0.9, repetition_penalty=1.15,
             do_sample=True, no_repeat_ngram_size=4, bad_words_ids=bans),
        dict(max_new_tokens=24, temperature=0.45, top_p=0.85, repetition_penalty=1.2,
             do_sample=True, no_repeat_ngram_size=5, bad_words_ids=bans),
    ]

    try:
        out = "..."
        for params in attempts:
            resp = llm_pipeline(
                prompt,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
                **proc_kwargs,  # logits processors from bias head
                **params,
            )[0].get("generated_text", "")
            cont = resp[len(prompt):] if resp.startswith(prompt) else resp
            cont = _early_stop(cont)
            cont = _one_line(cont)
            cont = _sanitize_roles(cont)
            cont = cont.strip(BAD_QUOTES + " ")
            if not _is_meta(cont):
                out = cont
                break
            else:
                out = cont
        return out or "..."
    except Exception as e:
        print(f"[LLM ERROR-bias] {e}")
        return "..."

# Optional convenience: pick mouthpiece by config
def llm_fn_from_env():
    return chatgpt_llm_with_bias if LLM_SPEAKER else chatgpt_llm_from_latent
