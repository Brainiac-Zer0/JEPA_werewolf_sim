# llm_script.py  ── single-device safe loader + clean one-liner
# (chat template + few-shot + anti-meta bans + safe fallbacks + batch helpers)

import os, torch, re
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# Override the model via env to try Llama 3.1 8B:
#   $env:LLM_MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
MODEL_ID = os.environ.get("LLM_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2")
USE_GPU  = torch.cuda.is_available()

# ── 1) Tokenizer
tok = AutoTokenizer.from_pretrained(MODEL_ID)

# ── 2) Model
device      = "cuda:0" if USE_GPU else "cpu"
torch_dtype = torch.float16 if USE_GPU else torch.float32
model       = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=False,
).to(device)

# ── 3) Pipeline
llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tok,
    device=0 if USE_GPU else -1,
)

print(f"[INFO] LLM loaded on {'GPU: '+torch.cuda.get_device_name(0) if USE_GPU else 'CPU'}")

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
    "Based on", "based on",
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

def _take_generated_text(obj):
    """
    HF text-generation outputs:
      - single call: [ { "generated_text": str } ]
      - batched call: [ [ { "generated_text": str } ], [ { ... } ], ... ]
    Safely extract the first generated_text, or "" if missing.
    """
    if isinstance(obj, dict):
        return obj.get("generated_text", "")
    if isinstance(obj, list):
        if not obj:
            return ""
        first = obj[0]
        if isinstance(first, dict):
            return first.get("generated_text", "")
        if isinstance(first, list) and first and isinstance(first[0], dict):
            return first[0].get("generated_text", "")
    return ""

def _build_prompt(name: str, role_str: str, heard_block: str, z_summary: str) -> str:
    # Chat template with few-shot in-character examples
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
                "- Someone’s behavior feels off to me.\n"
                "- Their timing doesn’t add up.\n"
                "- I’m getting mixed signals from them.\n"
                "- I think they’re hiding something important.\n"
            ),
        },
        {"role": "assistant", "content": "I’m not convinced by their story."},
        {"role": "user", "content": "Another example."},
        {"role": "assistant", "content": "Someone’s behavior feels off to me."},
        {
            "role": "user",
            "content": (
                f"You are {{name}}{role_str} in a hidden-role deduction game.\n"
                f"Recent dialog:\n{heard_block}\n"
                f"Your private feeling:\n- {z_summary}\n\n"
                "Now reply in one short, natural sentence, as if you are speaking in the game."
            ).replace("{name}", name),
        },
    ]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# ── main callable used by agents (single)
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
        dict(max_new_tokens=32, temperature=0.65, top_p=0.92, repetition_penalty=1.15,
             do_sample=True, no_repeat_ngram_size=4, bad_words_ids=bans),
        dict(max_new_tokens=28, temperature=0.5, top_p=0.9, repetition_penalty=1.15,
             do_sample=True, no_repeat_ngram_size=4, bad_words_ids=bans),
        dict(max_new_tokens=24, temperature=0.45, top_p=0.85, repetition_penalty=1.2,
             do_sample=True, no_repeat_ngram_size=5, bad_words_ids=bans),
    ]

    try:
        # Attempt 1
        raw = llm_pipeline(
            prompt,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
            **attempts[0],
        )
        resp = _take_generated_text(raw)
        cont = resp[len(prompt):] if resp.startswith(prompt) else resp
        cont = _early_stop(_one_line(_sanitize_roles(cont))).strip(BAD_QUOTES + " ")
        if not _is_meta(cont):
            return cont or "..."

        # Attempts 2..N
        out = cont
        for params in attempts[1:]:
            raw = llm_pipeline(
                prompt,
                pad_token_id=tok.eos_token_id,
                eos_token_id=tok.eos_token_id,
                **params,
            )
            resp = _take_generated_text(raw)
            cont = resp[len(prompt):] if resp.startswith(prompt) else resp
            cont = _early_stop(_one_line(_sanitize_roles(cont))).strip(BAD_QUOTES + " ")
            if not _is_meta(cont):
                return cont or "..."
            out = cont

        return out or "..."
    except Exception as e:
        print(f"[LLM ERROR] {e}")
        return "..."

# ── batch helpers ─────────────────────────────────────────────────────────────

def build_prompt_from_latent(z, agent) -> str:
    """Public helper for sim.py: build a chat-formatted prompt from z+agent."""
    role = (agent.role or "").strip()
    role_str = f", a {role}," if role in ALLOWED_ROLES else ""
    name = agent.name
    z_summary = agent.decode_z(z)
    heard_block = (
        "\n".join(
            f"- {n} said: \"{m.strip()}\""
            for n, m in list(agent.message_memory)[-6:]
            if m.strip()
        ) or "- (no recent messages heard)"
    )
    return _build_prompt(name, role_str, heard_block, z_summary)

def chatgpt_llm_batch(prompts: list[str]) -> list[str]:
    """Vectorized generation: one pipeline call for many prompts.
       Falls back per-item once if a meta-style line slips through."""
    if not prompts:
        return []

    bans = _bad_words_ids()
    params = dict(
        max_new_tokens=28, temperature=0.6, top_p=0.9,
        repetition_penalty=1.15, do_sample=True,
        no_repeat_ngram_size=4, bad_words_ids=bans,
        pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id,
    )

    # First pass (batched)
    raw_outs = llm_pipeline(prompts, **params)
    cleaned = []
    fallback_idx = []

    for idx, (p, item) in enumerate(zip(prompts, raw_outs)):
        resp = _take_generated_text(item)
        cont = resp[len(p):] if resp.startswith(p) else resp
        cont = _early_stop(_one_line(_sanitize_roles(cont))).strip(BAD_QUOTES + " ")
        if _is_meta(cont):
            fallback_idx.append(idx)
        cleaned.append(cont or "...")

    # Per-item fallback with stricter decoding if needed
    if fallback_idx:
        strict = dict(
            max_new_tokens=24, temperature=0.5, top_p=0.9,
            repetition_penalty=1.2, do_sample=True,
            no_repeat_ngram_size=5, bad_words_ids=bans,
            pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id,
        )
        retry_prompts = [prompts[i] for i in fallback_idx]
        retry_raw = llm_pipeline(retry_prompts, **strict)
        for j, i in enumerate(fallback_idx):
            resp = _take_generated_text(retry_raw[j])
            cont = resp[len(prompts[i]):] if resp.startswith(prompts[i]) else resp
            cont = _early_stop(_one_line(_sanitize_roles(cont))).strip(BAD_QUOTES + " ")
            cleaned[i] = cont or "..."

    return cleaned
