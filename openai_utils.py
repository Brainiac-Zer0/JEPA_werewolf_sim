# openai_utils.py
# Canonical OpenAI Responses API wrapper for AgentSim
# - Verifies environment (OPENAI_API_KEY, LLM_PROVIDER, LLM_MODEL_ID)
# - Maps friendly max_tokens -> max_completion_tokens
# - Strips unsupported knobs for o4* models
# - Normalizes I/O (prompt in input, optional system in instructions)
# - Provides a uniform extract_text() accessor

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

# ───────────────────────── Env helpers & sanity ─────────────────────────
def _env_str(key: str, default: str = "") -> str:
    v = os.getenv(key)
    return v if v is not None else default


def _env_bool(key: str, default: bool = False) -> bool:
    v = os.getenv(key)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def _looks_like_openai_model(model: str) -> bool:
    if not model:
        return False
    m = model.lower()
    # Cover common families: o4*, omni-*, gpt-4o*, gpt-4.*, gpt-3.5-*
    return (
        m.startswith("o4")
        or m.startswith("omni-")
        or m.startswith("gpt-4o")
        or m.startswith("gpt-4")
        or m.startswith("gpt-3.5")
    )


def _effective_provider() -> str:
    """
    Decide the provider based on env. Defaults to 'openai' if an OpenAI key is present
    or the model name looks like an OpenAI model.
    """
    prov = _env_str("LLM_PROVIDER", "").strip().lower()
    model = _env_str("LLM_MODEL_ID", "").strip()
    if prov:
        return prov
    if os.getenv("OPENAI_API_KEY") and (_looks_like_openai_model(model) or not model):
        return "openai"
    # Fallback: honor model hint
    if _looks_like_openai_model(model):
        return "openai"
    return "hf"  # safe default if caller imports this module but intends HF path


# ───────────────────────── OpenAI client (lazy) ─────────────────────────
_client = None  # created lazily after env checks


def _require_openai_client():
    """
    Lazily construct an OpenAI client after verifying env sanity.
    """
    global _client
    if _client is not None:
        return _client

    provider = _effective_provider()
    if provider != "openai":
        raise RuntimeError(
            "Attempted to use OpenAI path but LLM_PROVIDER is not 'openai'. "
            "Set LLM_PROVIDER=openai and ensure LLM_MODEL_ID is an OpenAI model."
        )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not api_key.strip():
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it first:\n"
            "  export OPENAI_API_KEY=sk-...\n"
            "Also ensure:\n"
            "  export LLM_PROVIDER=openai\n"
            "  export LLM_MODEL_ID=o4-mini   # or your chosen OpenAI model\n"
            "  export LLM_SPEAKER=1          # to enable the LLM mouthpiece route"
        )

    try:
        # New-style SDK import (openai>=1.0)
        from openai import OpenAI  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "OpenAI Python SDK not found or incompatible. "
            "Install/upgrade with: pip install --upgrade openai"
        ) from e

    _client = OpenAI(api_key=api_key)
    return _client


# ───────────────────────── Model capability helpers ─────────────────────────
def _is_o4_model(model: str) -> bool:
    """
    Returns True if the model behaves like an o4* Responses model
    with restricted parameter surface (no temperature, penalties, top_p).
    """
    if not model:
        return False
    m = model.lower()
    # Cover common o4 variants
    return m.startswith("o4") or m.startswith("omni-")


def _has_unsupported_temperature_error(err: Exception) -> bool:
    """
    Detect if an error message from the API indicates that 'temperature'
    is unsupported for this model.
    """
    msg = getattr(err, "message", "") or ""
    if not msg and hasattr(err, "error"):
        msg = getattr(err.error, "message", "") or ""
    if not msg:
        msg = str(err)
    msg_l = msg.lower()
    return ("unsupported parameter" in msg_l or "invalid_request_error" in msg_l) and "temperature" in msg_l


def _split_supported_kwargs(
    model: str,
    *,
    temperature: Optional[float],
    presence_penalty: Optional[float],
    frequency_penalty: Optional[float],
    top_p: Optional[float],
    max_tokens: Optional[int],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Partition kwargs into (supported_for_model, stripped_for_model).
    Also maps max_tokens -> max_completion_tokens for Responses API.
    """
    supported: Dict[str, Any] = {}
    stripped: Dict[str, Any] = {}

    # Map friendly -> API-specific
    if max_tokens is not None:
        # Responses API expects max_completion_tokens (per current error message)
        supported["max_completion_tokens"] = int(max_tokens)

    if _is_o4_model(model):
        # These are unsupported on o4* (per error logs)
        if temperature is not None:
            stripped["temperature"] = temperature
        if presence_penalty is not None:
            stripped["presence_penalty"] = presence_penalty
        if frequency_penalty is not None:
            stripped["frequency_penalty"] = frequency_penalty
        if top_p is not None:
            stripped["top_p"] = top_p
    else:
        # Non-o4 models (if you ever route them through Responses) may accept these.
        # Keep conservative: only pass if not None.
        if temperature is not None:
            supported["temperature"] = float(temperature)
        if presence_penalty is not None:
            supported["presence_penalty"] = float(presence_penalty)
        if frequency_penalty is not None:
            supported["frequency_penalty"] = float(frequency_penalty)
        if top_p is not None:
            supported["top_p"] = float(top_p)

    return supported, stripped


# ───────────────────────── Public API ─────────────────────────
def call_openai_responses(
    *,
    model: str,
    prompt: str,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    top_p: Optional[float] = None,
    system: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
):
    """
    Canonical wrapper for OpenAI Responses API.

    Args:
        model: Model name, e.g., "o4-mini".
        prompt: User-facing text prompt. Will be placed under input as a user message.
        max_tokens: Friendly alias; mapped to 'max_completion_tokens'.
        temperature, presence_penalty, frequency_penalty, top_p:
            Will be stripped automatically for o4* models.
        system: Optional system/instructions string routed to 'instructions'.
        extra: Optional dict to merge into the payload (advanced use).

    Returns:
        The raw Responses API result object from the SDK.
    """
    if not model:
        # Allow falling back to env-configured model if omitted
        model = _env_str("LLM_MODEL_ID", "").strip()
        if not model:
            raise ValueError("model is required (or set LLM_MODEL_ID in the environment)")
    if prompt is None:
        raise ValueError("prompt must not be None")

    # Enforce that the OpenAI path is actually selected and sane
    if _effective_provider() != "openai":
        raise RuntimeError(
            "OpenAI call requested but provider is not 'openai'. "
            "Set: export LLM_PROVIDER=openai ; export LLM_MODEL_ID=o4-mini"
        )

    client = _require_openai_client()

    supported, _ = _split_supported_kwargs(
        model,
        temperature=temperature,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    # Build the Responses API payload
    payload: Dict[str, Any] = {
        "model": model,
        # Responses API "input" can be a list of message objects.
        # We provide a single user turn with a single text content part.
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": str(prompt)}
                ],
            }
        ],
    }

    # Optional system/instructions channel
    if system:
        payload["instructions"] = str(system)

    # Merge supported knobs
    payload.update({k: v for k, v in supported.items() if v is not None})

    # Allow advanced extensions (e.g., reasoning config) via extra
    if extra:
        # Don't let extra clobber core safety-critical fields silently
        for k in ("model", "input"):
            if k in extra and extra[k] != payload[k]:
                raise ValueError(f"extra[{k}] attempted to override core field")
        payload.update(extra)

    # Perform the call, then defensively retry once without 'temperature' if needed
    try:
        resp = client.responses.create(**payload)
    except Exception as e:
        if _has_unsupported_temperature_error(e):
            payload.pop("temperature", None)
            resp = client.responses.create(**payload)
        else:
            raise
    return resp


def extract_text(resp: Any) -> str:
    """
    Uniformly extract primary text from a Responses API result.

    Tries several SDK-compatible access patterns to be resilient across minor SDK variations.
    Returns an empty string if nothing textual is found.
    """
    # Preferred helper on newer SDKs
    text = getattr(resp, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text

    # Try traversing structured output
    try:
        # Newer Responses objects often have .output[] with content parts
        output = getattr(resp, "output", None)
        if isinstance(output, list) and output:
            # Walk the first item that contains text
            for item in output:
                # item.content may be list of parts
                content = getattr(item, "content", None)
                if isinstance(content, list):
                    for part in content:
                        # part may have .text or .content with .text
                        p_text = getattr(part, "text", None)
                        if isinstance(p_text, str) and p_text.strip():
                            return p_text
                        # Some SDKs wrap text as dicts
                        if isinstance(part, dict):
                            maybe_text = part.get("text")
                            if isinstance(maybe_text, str) and maybe_text.strip():
                                return maybe_text
    except Exception:
        pass

    # Fallback: some SDKs return a dict-like structure
    if isinstance(resp, dict):
        # Best-effort extraction
        if "output_text" in resp and isinstance(resp["output_text"], str):
            return resp["output_text"]
        out = resp.get("output")
        if isinstance(out, list):
            for item in out:
                if isinstance(item, dict):
                    content = item.get("content")
                    if isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and isinstance(part.get("text"), str):
                                t = part["text"]
                                if t.strip():
                                    return t

    return ""


__all__ = [
    "call_openai_responses",
    "extract_text",
]
