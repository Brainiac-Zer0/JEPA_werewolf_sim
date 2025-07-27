import torch
from transformers import pipeline

# Initialize LLM pipeline (Mistral preferred, fallback to distilgpt2)
try:
    llm_pipeline = pipeline(
        "text-generation",
        model="mistralai/Mistral-7B-Instruct-v0.2",
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
except Exception as e:
    print("[WARNING] Failed to load Mistral, falling back to distilgpt2")
    llm_pipeline = pipeline("text-generation", model="distilgpt2")

if torch.cuda.is_available():
    print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("[INFO] Using CPU")

def chatgpt_llm_from_latent(z, agent):
    """Generate a natural, in-character message from latent state, with recent message context."""
    try:
        z_summary = agent.decode_z(z)
        name = agent.name
        role = agent.role.upper()
    except Exception:
        z_summary = "Belief state unavailable."
        name = agent.name
        role = agent.role or "UNDEFINED"

    # Format recent messages from heard agents
    if hasattr(agent, 'heard_messages') and agent.heard_messages:
        heard_lines = [f"- {n} said: \"{msg.strip()}\"" for n, msg in agent.heard_messages.items() if msg.strip()]
        heard_block = "\n".join(heard_lines[:6])  # Limit context
    else:
        heard_block = "- (no recent messages heard)"

    prompt = f"""You are agent {name}, a {role}, in a social deduction scenario.

Here is what others have recently said:
{heard_block}

Your current internal state tells you:
- {z_summary}

Say ONE short, natural sentence aloud, in-character. Speak like a real person. Avoid narration, game terms, or commands. Just express what you think or feel, based on your internal belief. Statements should be aimed at your general situation or at specific other agents.
Only output the line of dialog — no narration, no formatting.
"""

    try:
        output = llm_pipeline(
            prompt,
            max_new_tokens=40,
            temperature=0.85,
            do_sample=True,
            pad_token_id=128001,
        )[0]["generated_text"]

        response = output[len(prompt):].strip()
        return response
    except Exception as e:
        print(f"[LLM ERROR] {e}")
        return "..."