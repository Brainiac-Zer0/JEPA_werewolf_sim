# llm_script.py  ── single-device safe loader + clean one-liner

import os, torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
USE_GPU  = torch.cuda.is_available()

# ── 1. Load tokenizer first (always small)
tok = AutoTokenizer.from_pretrained(MODEL_ID)

# ── 2. Load the model *without* sharding
device      = "cuda:0" if USE_GPU else "cpu"
torch_dtype = torch.float16 if USE_GPU else torch.float32
model       = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=False,      # ➟ avoid meta placement
).to(device)

# ── 3. Wrap with pipeline (gives nice .generate defaults)
llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tok,
    device=0 if USE_GPU else -1,   # -1 means CPU
)

print(f"[INFO] LLM loaded on {'GPU: '+torch.cuda.get_device_name(0) if USE_GPU else 'CPU'}")

# ── helper: squeeze to a single tidy line
BAD_QUOTES = "“”\"'«»"
def _postprocess(text: str) -> str:
    first = text.splitlines()[0].strip()
    return first.strip(BAD_QUOTES + " ").rstrip(" .") or "..."

# ── main callable used by agents
def chatgpt_llm_from_latent(z, agent) -> str:
    z_summary = agent.decode_z(z)
    name, role = agent.name, (agent.role or "UNDEFINED").upper()

    # last ≤6 remembered utterances, with speakers
    heard_block = (
        "\n".join(f"- {n} said: \"{m.strip()}\""
                  for n, m in list(agent.message_memory)[-6:]
                  if m.strip())
        or "- (no recent messages heard)"
    )

    prompt = (
        f"You are {name}, a {role}, in a hidden-role deduction game.\n\n"
        f"Recent dialog:\n{heard_block}\n\n"
        f"Your private feeling:\n- {z_summary}\n\n"
        "Say ONE short, natural sentence in-character. Avoid narration.\n"
    )

    try:
        generated = llm_pipeline(
            prompt,
            max_new_tokens=40,
            temperature=0.85,
            do_sample=True,
            pad_token_id=tok.eos_token_id,
        )[0]["generated_text"][len(prompt):]
        return _postprocess(generated)
    except Exception as e:
        print(f"[LLM ERROR] {e}")
        return "..."