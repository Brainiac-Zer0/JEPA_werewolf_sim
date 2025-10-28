# speaker.py
# -----------------------------------------------------------------------------
# Phase-5 mouthpiece router:
#   • Primary path: LLM one-liner with optional logit-bias fusion (speaker_llm)
#   • Fallback path: local template bandit (stable, fast)
# This file intentionally *does not* import SpeakerBandit from speaker_llm to
# avoid circular imports; only the bias utilities are imported optionally.
# -----------------------------------------------------------------------------

from __future__ import annotations
import os
import json
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple, Optional

# ── Load config
with open("config.yaml", "r") as f:
    CFG = yaml.safe_load(f) or {}

# Default templates; may be overridden by config
DEFAULT_TEMPLATES = CFG.get("DEFAULT_TEMPLATES", [
    "Accuse {target}",
    "Defend {ally}",
    "Ask {target} a question",
    "Express uncertainty",
    "Propose vote on {target}",
])

# --- Hygiene helpers (LLM safety / formatting) ---
_BAD_QUOTES = "“”\"'«»"
_STOP_SEQS  = ["\n", "\r", "Agent_", "System:", "Narrator:"]
_META_BANS  = [
    "use ", "do not", "don't", "no ", "reply", "provide", "follow", "instruction", "rule",
    "sentence", "third person", "grammar", "punctuation", "no narration"
]
_ROLE_BLOCKLIST = [
    "Engineer","engineer","Scientist","scientist","Detective","detective","Doctor","doctor","Guard","guard",
    "Scientist.","Engineer.","Detective.","Doctor.","Guard."
]

def _one_line(text: str) -> str:
    if not text:
        return "..."
    first = text.splitlines()[0].strip()
    first = first.replace("“","").replace("”","").replace('"',"").replace("’","'")
    first = first.lstrip(" -").rstrip(" .,!?:;")
    return first[:160] or "..."

def _early_stop(g: str) -> str:
    for s in _STOP_SEQS:
        i = g.find(s)
        if i != -1:
            g = g[:i]
    return g

def _sanitize_roles(s: str) -> str:
    for w in _ROLE_BLOCKLIST:
        s = s.replace(w, "someone")
    return s.strip()

def _looks_meta(s: str) -> bool:
    t = s.strip().lower()
    if not t:
        return True
    if any(p in t for p in _META_BANS):
        return True
    if len(t.split()) < 2:
        return True
    return False

def _trainable_params(mod) -> List[torch.nn.Parameter]:
    """Return only trainable params for a module; [] if module is None or param-less."""
    try:
        return [p for p in mod.parameters() if p.requires_grad]
    except Exception:
        return []

# =============================================================================
# History features (lightweight context stats)
# =============================================================================
def make_hist_feats(recent_texts: List[str], phase_code: Optional[int] = None) -> torch.Tensor:
    """
    Returns a small feature vector:
      [accusation_rate, mean_len] (+ optional one-hot phase of size 3)
    """
    if not recent_texts:
        base = torch.tensor([0.0, 0.0], dtype=torch.float32)
    else:
        n = len(recent_texts)
        acc = sum(int(("accuse" in t.lower()) or ("vote" in t.lower())) for t in recent_texts) / n
        mean_len = min(1.5, sum(len(t) for t in recent_texts) / max(1, n) / 100.0)
        base = torch.tensor([acc, mean_len], dtype=torch.float32)

    if phase_code is None:
        return base

    oh = torch.zeros(3, dtype=torch.float32)
    try:
        pc = int(phase_code)
        if 0 <= pc < 3:
            oh[pc] = 1.0
    except Exception:
        pass
    return torch.cat([base, oh], dim=0)

# =============================================================================
# Local template bandit (REINFORCE) — avoids speaker_llm import cycle
# =============================================================================
class SpeakerBandit(nn.Module):
    """
    Tiny bandit over speech-act templates.
    Input: [z_t (d), role_bit (1), hist_feats (?=2 or 5)] → logits over templates.
    Training: REINFORCE on message-level reward.
    """
    def __init__(self, latent_dim: int, num_templates: int, hidden: int = 128):
        super().__init__()
        self.num_templates = num_templates
        self.temperature = 1.0
        self._latent_dim = latent_dim
        self._hidden = hidden
        self._mlp: Optional[nn.Sequential] = None  # lazy init to match feat size at runtime

    def _build(self, in_features: int):
        self._mlp = nn.Sequential(
            nn.Linear(in_features, self._hidden),
            nn.Tanh(),
            nn.Linear(self._hidden, self.num_templates),
        )

    def forward(self, z: torch.Tensor, role_bit: torch.Tensor, hist_feats: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, role_bit, hist_feats], dim=-1)
        if self._mlp is None:
            self._build(x.size(-1))
        return self._mlp(x)

    @torch.no_grad()
    def generate(
        self,
        z_t: torch.Tensor,
        role: str,
        recent_texts: List[str],
        templates: List[str],
        candidate_targets: List[str],
        self_name: str,
        persona_effects: Optional[Dict[str, Any]] = None,
        phase_code: Optional[int] = None,
        **_ignored,
    ) -> Tuple[str, Dict[str, Any]]:
        dev = next(self.parameters()).device
        if z_t.dim() == 1:
            z_t = z_t.unsqueeze(0)
        z_t = z_t.to(dev)

        role_bit = torch.tensor([[1.0 if role.lower().startswith("were") else 0.0]],
                                device=dev, dtype=z_t.dtype)
        hist_feats = make_hist_feats(recent_texts, phase_code=phase_code).to(dev).unsqueeze(0)

        logits = self.forward(z_t, role_bit, hist_feats).squeeze(0)

        # Persona biases (optional, light touch)
        if persona_effects:
            accuse_bias_scale = float(persona_effects.get("accuse_bias_scale", 1.0))
            # turn scale into additive bias for accuse/vote templates
            accuse_bias = (accuse_bias_scale - 1.0) * 0.8  # small
            if accuse_bias != 0.0:
                idx_accuse = [i for i, t in enumerate(templates)
                              if ("accuse" in t.lower()) or ("vote" in t.lower()) or ("propose" in t.lower())]
                idx_uncert = [i for i, t in enumerate(templates)
                              if ("uncertain" in t.lower()) or ("uncert" in t.lower())]
                for i in idx_accuse:
                    logits[i] = logits[i] + accuse_bias
                for i in idx_uncert:
                    logits[i] = logits[i] - 0.5 * accuse_bias

        # Persona-driven temperature scaling (safe clamp)
        temp_scale = 1.0
        if persona_effects:
            try:
                temp_scale = float(persona_effects.get("speaker_temp_scale", 1.0))
            except Exception:
                temp_scale = 1.0
        temperature = max(1e-4, float(getattr(self, "temperature", 1.0)) * temp_scale)

        probs = F.softmax(logits / temperature, dim=-1)
        tidx  = torch.multinomial(probs, 1).item()

        # Slot filling
        target = next((t for t in candidate_targets if t != self_name), None)
        target = target or (candidate_targets[0] if candidate_targets else self_name)

        text = templates[tidx].replace("{target}", target).replace("{ally}", self_name)
        text = _sanitize_roles(_one_line(text))

        meta = {
            "mode": "bandit",
            "template_id": tidx,
            "logprob": float(torch.log(probs[tidx] + 1e-8).item()),
            "z": z_t.squeeze(0).detach().cpu(),
            "role_bit": role_bit.squeeze(0).detach().cpu(),
            "hist_feats": hist_feats.squeeze(0).detach().cpu(),
            "phase_code": phase_code,
        }
        return text, meta

    def learn_step(
        self,
        batch: List[Dict[str, Any]],
        optimizer: torch.optim.Optimizer,
        entropy_bonus: float = 0.01,
        baseline: float = None,
    ) -> Dict[str, float]:
        if not batch:
            return {"loss": 0.0, "entropy": 0.0, "R_mean": 0.0}
        device = next(self.parameters()).device

        # Expect z/role_bit/hist_feats/template_id/reward
        zs = torch.stack([b["z"] for b in batch]).to(device)
        role_bits = torch.stack([b["role_bit"] for b in batch]).to(device)
        hfs = torch.stack([b["hist_feats"] for b in batch]).to(device)
        tids = torch.tensor([b["template_id"] for b in batch], dtype=torch.long, device=device)
        rewards = torch.tensor([b["reward"] for b in batch], dtype=torch.float32, device=device)

        logits = self.forward(zs, role_bits, hfs)
        logps = torch.log_softmax(logits, dim=-1)
        sel_logp = logps.gather(1, tids.unsqueeze(1)).squeeze(1)

        if baseline is not None:
            rewards = rewards - baseline

        ent = -(logps.exp() * logps).sum(dim=-1).mean()
        loss = -(rewards * sel_logp).mean() - entropy_bonus * ent

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return {"loss": float(loss.item()), "entropy": float(ent.item()), "R_mean": float(rewards.mean().item())}

# =============================================================================
# Optional LLM bias adapter (trainable light head that steers a frozen LLM)
# =============================================================================
try:
    from speaker_llm import LogitBiasHead, with_logit_bias_generate_kwargs
except Exception:
    LogitBiasHead = None
    with_logit_bias_generate_kwargs = None

class LLMBiasAdapter(nn.Module):
    """
    Small trainable head that steers a frozen LLM via logits processors.
    Exposes: get_kwargs(...) → **kwargs for your LLM generate() call.
    """
    def __init__(self, latent_dim: int = 32, device: Optional[torch.device] = None):
        super().__init__()
        if LogitBiasHead is None:
            self.head = None
        else:
            self.head = LogitBiasHead(latent_dim=latent_dim)
            self.head.to(device if device is not None else torch.device("cpu"))

    def get_kwargs(self, tokenizer, z_t: torch.Tensor, role: str,
                   recent_texts: List[str], persona_effects: Optional[Dict[str, Any]]):
        if self.head is None or with_logit_bias_generate_kwargs is None:
            return {}
        z = z_t.detach() if torch.is_tensor(z_t) else torch.tensor(z_t)
        return with_logit_bias_generate_kwargs(
            tokenizer=tokenizer,
            head=self.head,
            z_t=z,
            role=role,
            recent_texts=recent_texts[-3:],
            persona_effects=persona_effects,
        )

    def learn_step(self, batch: List[Dict[str, Any]], optimizer: torch.optim.Optimizer,
                   entropy_bonus: float = 0.0) -> Dict[str, float]:
        """
        Simple update: minimize a small regularizer when rewards are high.
        Expects batch items to contain: {"z": Tensor, "reward": float}
        """
        if self.head is None or not batch:
            return {"loss": 0.0}
        device = next(self.head.parameters()).device
        zs = torch.stack([b["z"] for b in batch]).to(device)
        R  = torch.tensor([b["reward"] for b in batch], dtype=torch.float32, device=device)

        penalties = self.head.regularizer(zs)  # shape [B], small positive
        loss = (1.0 - R).clamp(min=0.0).mean() + 0.01 * penalties.mean() - (entropy_bonus * 0.0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return {"loss": float(loss.item())}

# =============================================================================
# Unified mouthpiece: routes between LLM and Bandit; both trainable
# =============================================================================
class SpeakerPolicy(nn.Module):
    """
    Hybrid router:
      - default: LLM (natural one-liner) with optional logit-bias head
      - fallback: Bandit templates (fast, stable)
    Trainable pieces: bandit, bias head (optional)
    """
    def __init__(self,
                 latent_dim: int,
                 templates: Optional[List[str]] = None,
                 device: Optional[torch.device] = None):
        super().__init__()
        self.templates = templates or CFG.get("speaker", {}).get("templates", DEFAULT_TEMPLATES)
        self.bandit = SpeakerBandit(
            latent_dim=latent_dim,
            num_templates=len(self.templates),
            hidden=int(CFG.get("speaker", {}).get("hidden", 128))
        )
        self.bias = LLMBiasAdapter(latent_dim=latent_dim, device=device)

        # Router thresholds
        self.use_llm = bool(CFG.get("llm", {}).get("speaker_enabled", False))
        self.bad_streak = 0
        self.max_bad = 2  # after 2 filtered generations, back off to bandit temporarily

        # Optional optimizers (attach via attach_optimizers)
        self.bandit_opt: Optional[torch.optim.Optimizer] = None
        self.bias_opt: Optional[torch.optim.Optimizer] = None

        self.to(device if device is not None else torch.device("cpu"))

        # Try to import LLM helpers lazily to avoid hard dependency
        self._llm_ok = False
        try:
            from llm_script import chatgpt_llm_with_bias, chatgpt_llm_from_latent  # type: ignore
            self._llm_with_bias = chatgpt_llm_with_bias
            self._llm_from_latent = chatgpt_llm_from_latent
            self._llm_ok = True
        except Exception:
            self._llm_with_bias = None
            self._llm_from_latent = None
            self._llm_ok = False

    @torch.no_grad()
    def _llm_generate(self,
                      name: str,
                      role: str,
                      recent_texts: List[str],
                      z_t: torch.Tensor,
                      persona_effects: Optional[Dict[str, Any]],
                      *,
                      prefix: Optional[str] = None,
                      named_target_hint: Optional[str] = None,
                      phase_code: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Preferred: build a prompt via llm_script._latent_prompt_from_agent and prepend a soft prefix.
        Fallback: use existing chatgpt_llm_* helpers (prefix not applied on fallback).
        """
        # Build a minimal proxy agent for llm_script helpers
        proxy_agent = type("A", (), {
            "role": role,
            "name": name,
            "message_memory": [],  # one-liner mouthpiece; context handled by latent
            "decode_z": staticmethod(lambda _z: ""),
            "persona_effects": persona_effects,
        })()

        # Try prompt-based route first (doesn't require self._llm_ok)
        try:
            from llm_script import _latent_prompt_from_agent, llm_fn_from_env  # type: ignore
            mouth = llm_fn_from_env()  # callable LLM mouthpiece
            tok = getattr(mouth, "tokenizer", None)
            base_prompt = _latent_prompt_from_agent(tok, z_t, proxy_agent)

            # Derive prefix if only a hint was passed
            if prefix is None and isinstance(named_target_hint, str) and named_target_hint.startswith("Agent_"):
                prefix = f"I think {named_target_hint} "

            # Soft steer for DAY_DISCUSS (phase_code==0): end with a question/accusation (gentle)
            if phase_code == 0:
                steer = " End with a pointed question or a named accusation."
            else:
                steer = ""

            final_prompt = (prefix + base_prompt) if (prefix and isinstance(prefix, str)) else base_prompt
            final_prompt = final_prompt + steer

            # Preserve bias fusion via logits processors if available
            bias_kwargs = self.bias.get_kwargs(
                tokenizer=getattr(mouth, "tokenizer", None),
                z_t=z_t.squeeze(0),
                role=role,
                recent_texts=recent_texts,
                persona_effects=persona_effects,
            )
            # Speaker defaults to improve NL hygiene (P5C)
            default_gen = {
                "min_new_tokens": 14,
                "max_new_tokens": 64,
                "temperature": 0.8,
                "top_p": 0.92,
            }
            gen_kwargs = dict(default_gen)
            if isinstance(bias_kwargs, dict):
                gen_kwargs.update(bias_kwargs)  # let bias head override/extend

            text = mouth(final_prompt, generate_kwargs=gen_kwargs)

            meta = {
                "mode": "llm",
                "steer_phase": int(phase_code) if phase_code is not None else None,
                "used_bias": bool(bias_kwargs),
                "gen_kwargs": {k: gen_kwargs.get(k) for k in ("min_new_tokens","max_new_tokens","temperature","top_p")},
            }
            # surface any repetition_penalty if present (P5D visibility; judge will also log)
            if "repetition_penalty" in gen_kwargs:
                meta["repetition_penalty"] = float(gen_kwargs["repetition_penalty"])
                meta["rp_applied"] = True
            return text, meta
        except Exception:
            # Fall back to older latent→LLM helpers (require _llm_ok)
            if self._llm_ok:
                if self.bias and self.bias.head is not None and self._llm_with_bias is not None:
                    txt = self._llm_with_bias(z_t, agent=proxy_agent)
                    return txt, {"mode": "llm", "fallback": "with_bias"}
                if self._llm_from_latent is not None:
                    txt = self._llm_from_latent(z_t, agent=proxy_agent)
                    return txt, {"mode": "llm", "fallback": "from_latent"}
            raise RuntimeError("LLM backend unavailable")

    @torch.no_grad()
    def generate(self,
                 z_t: torch.Tensor,
                 role: str,
                 recent_texts: List[str],
                 candidate_targets: List[str],
                 self_name: str,
                 *,
                 phase_code: Optional[int] = None,
                 talk_prior: Optional[Dict[str, Any]] = None,
                 persona_effects: Optional[Dict[str, Any]] = None,
                 # NEW: soft steer inputs
                 prefix: Optional[str] = None,
                 named_target_hint: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        dev = next(self.parameters()).device
        if z_t.dim() == 1:
            z_t = z_t.unsqueeze(0)
        z_t = z_t.to(dev)

        # Route: try LLM unless we’re in a backoff window
        use_llm_now = self.use_llm and (self.bad_streak < self.max_bad)
        if use_llm_now:
            try:
                text, meta_llm = self._llm_generate(
                    self_name, role, recent_texts, z_t, persona_effects,
                    prefix=prefix, named_target_hint=named_target_hint, phase_code=phase_code
                )
                text = _sanitize_roles(_one_line(_early_stop(text))).strip(_BAD_QUOTES + " ")
                if _looks_meta(text):
                    # Escalate backoff and fall through to bandit
                    self.bad_streak += 1
                    raise RuntimeError("meta-like generation")
                # success
                self.bad_streak = 0
                meta = {
                    "mode": "llm",
                    "text_raw": text,
                    "z": z_t.squeeze(0).detach().cpu(),
                    "phase_code": phase_code,
                }
                # merge llm meta (includes used_bias, gen kwargs, repetition info if any)
                if isinstance(meta_llm, dict):
                    meta.update(meta_llm)

                # Emit probe-friendly stdout line (P5I)
                if os.getenv("LLM_SPK_DEBUG", "1") == "1":
                    dbg = {
                        "mode": "llm",
                        "role": role,
                        "name": self_name,
                        "phase_code": phase_code,
                        "used_bias": meta.get("used_bias", False),
                        "gen": meta.get("gen_kwargs", {}),
                        "rp": {
                            "value": meta.get("repetition_penalty", None),
                            "applied": meta.get("rp_applied", False),
                        },
                        "preview": text[:120],
                    }
                    print("[LLM-SPK]", json.dumps(dbg), flush=True)
                return text, meta
            except Exception:
                # Fall back to bandit
                pass

        # Bandit path (stable fallback)
        role_bit = torch.tensor([[1.0 if role.lower().startswith("were") else 0.0]], device=dev, dtype=z_t.dtype)
        hf = make_hist_feats(recent_texts, phase_code).to(dev).unsqueeze(0)
        logits = self.bandit(z_t, role_bit, hf).squeeze(0)
        temperature = max(1e-4, float(getattr(self.bandit, "temperature", 1.0)))
        probs = F.softmax(logits / temperature, dim=-1)
        tidx = torch.multinomial(probs, 1).item()

        target = next((t for t in candidate_targets if t != self_name), None)
        target = target or (candidate_targets[0] if candidate_targets else self_name)
        text = self.templates[tidx].replace("{target}", target).replace("{ally}", self_name)
        text = _sanitize_roles(_one_line(text))

        meta = {
            "mode": "bandit",
            "template_id": tidx,
            "logprob": float(torch.log(probs[tidx] + 1e-8).item()),
            "z": z_t.squeeze(0).detach().cpu(),
            "role_bit": role_bit.squeeze(0).detach().cpu(),
            "hist_feats": hf.squeeze(0).detach().cpu(),
            "phase_code": phase_code,
        }
        if os.getenv("LLM_SPK_DEBUG", "1") == "1":
            dbg = {
                "mode": "bandit",
                "role": role,
                "name": self_name,
                "phase_code": phase_code,
                "template_id": tidx,
                "preview": text[:120],
            }
            print("[LLM-SPK]", json.dumps(dbg), flush=True)
        return text, meta

    # === Trainability / persistence ===
    def attach_optimizers(self, bandit_lr: float = 1e-3, bias_lr: float = 1e-3):
        """
        Create optimizers only if there are trainable parameters.
        Safe for template-bandit (param-less) and bias-head-off configurations.
        """
        # Bandit optimizer (may be param-less if MLP not built yet — handle lazily)
        bandit_params = _trainable_params(getattr(self, "bandit", None)) if getattr(self, "bandit", None) is not None else []
        self.bandit_opt = torch.optim.Adam(bandit_params, lr=bandit_lr) if bandit_params else None

        # Bias head optimizer (may be disabled or None)
        if self.bias and self.bias.head is not None:
            bias_params = _trainable_params(self.bias.head)
            self.bias_opt = torch.optim.Adam(bias_params, lr=bias_lr) if bias_params else None
        else:
            self.bias_opt = None

    def learn_step(self, batch: List[Dict[str, Any]], entropy_bonus: float = 0.01, baseline: float = 0.0) -> Dict[str, float]:
        stats = {"bandit_loss": 0.0, "bias_loss": 0.0, "R_mean": 0.0}
        if not batch:
            return stats

        # Split by path
        bandit_batch = [b for b in batch if b.get("mode") == "bandit"]
        bias_batch   = [b for b in batch if b.get("mode") == "llm"]

        if bandit_batch and self.bandit_opt is not None:
            stats_b = self.bandit.learn_step(bandit_batch, self.bandit_opt, entropy_bonus=entropy_bonus, baseline=baseline)
            stats["bandit_loss"] = stats_b["loss"]
            stats["R_mean"] = stats_b["R_mean"]

        if bias_batch and self.bias_opt is not None and self.bias and self.bias.head is not None:
            # Keep only the minimal tensors the bias head needs
            mini = [{"z": b["z"], "reward": b["reward"]} for b in bias_batch if "z" in b and "reward" in b]
            if mini:
                stats_h = self.bias.learn_step(mini, self.bias_opt, entropy_bonus=0.0)
                stats["bias_loss"] = stats_h["loss"]
        return stats

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "bandit": self.bandit.state_dict(),
            "templates": self.templates,
            "router": {"use_llm": self.use_llm, "max_bad": self.max_bad},
        }
        if self.bias and self.bias.head is not None:
            payload["bias_head"] = self.bias.head.state_dict()
        torch.save(payload, path)

    def load(self, path: str, strict: bool = True):
        st = torch.load(path, map_location="cpu")
        if "bandit" in st:
            self.bandit.load_state_dict(st["bandit"], strict=strict)
        if "bias_head" in st and self.bias and self.bias.head is not None:
            self.bias.head.load_state_dict(st["bias_head"], strict=strict)
        if "templates" in st:
            self.templates = st["templates"]
        if "router" in st:
            r = st["router"]
            self.use_llm = bool(r.get("use_llm", self.use_llm))
            self.max_bad = int(r.get("max_bad", self.max_bad))

# Explicit exports
__all__ = [
    "SpeakerPolicy",
    "SpeakerBandit",
    "LLMBiasAdapter",
    "DEFAULT_TEMPLATES",
    "make_hist_feats",
]
