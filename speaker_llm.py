# speaker_llm.py
# Trainable LLM mouthpiece via logit-bias steering.
# - LogitBiasHead(z, role_bit, hist_feats) -> weights over a small set of speech-act categories
# - BiasLexicon maps categories -> token id sets (built from tokenizer + seed words)
# - JudgeRewardBiasProcessor adds a sparse bias vector to next-token logits each step
# - Use with generate(..., logits_processor=processor_list)
#
# Notes:
# * This is lightweight and safe: we DO NOT modify the LLM weights.
# * Learning: you can do REINFORCE on the head using judge rewards (we’ll wire in train.py next).
# * If you already use template SpeakerBandit, enable this instead with LLM_SPEAKER=1 (one mouthpiece at a time).

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, List, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LogitsProcessor, LogitsProcessorList, PreTrainedTokenizerBase

# Reuse history features from the template speaker (short, stable)
try:
    from speaker import make_hist_feats
except Exception:
    # Fallback if speaker.py isn't available yet
    def make_hist_feats(recent_texts: List[str]) -> torch.Tensor:
        if not recent_texts:
            return torch.tensor([0.0, 0.0])
        n = len(recent_texts)
        acc = sum(int(("accuse" in t.lower()) or ("vote" in t.lower())) for t in recent_texts) / n
        mean_len = min(1.5, sum(len(t) for t in recent_texts) / max(1, n) / 100.0)
        return torch.tensor([acc, mean_len], dtype=torch.float32)

# ───────────── config (env-overrideable) ─────────────
BIAS_CAP: float = float(os.environ.get("LLM_SPK_BIAS_CAP", "2.0"))      # max |bias| per token logit
BASE_STRENGTH: float = float(os.environ.get("LLM_SPK_BASE_STRENGTH", "1.0"))
DEBUG: bool = os.environ.get("LLM_SPK_DEBUG", "0") == "1"

# Seed lexicon for categories (can be extended per game)
DEFAULT_LEXICON = {
    "accuse":   ["accuse", "suspicious", "suspect", "lying", "deceive", "eliminate", "vote"],
    "defend":   ["defend", "innocent", "trust", "ally", "support", "clear"],
    "hedge":    ["maybe", "perhaps", "uncertain", "unsure", "might", "seems", "appears"],
    "question": ["why", "how", "what", "who", "where", "when", "?"],
    "vote":     ["vote", "eliminate", "banish", "target", "lynch"],  # keep neutral wording
}

CAT_ORDER = ["accuse", "defend", "hedge", "question", "vote"]


# ───────────── utils ─────────────
def _role_to_bit(role: Optional[str]) -> float:
    r = (role or "").lower()
    return 1.0 if r.startswith("were") else 0.0  # 1=werewolf, 0=villager/worker


def _words_to_token_ids(tok: PreTrainedTokenizerBase, words: Iterable[str]) -> List[int]:
    ids: List[int] = []
    for w in words:
        # Take the first non-special token id of the wordpiece sequence as a proxy
        enc = tok.encode(w, add_special_tokens=False)
        if enc:
            ids.append(enc[0])
    # Deduplicate and drop pad/eos if present
    ids = list({i for i in ids if i not in (getattr(tok, "pad_token_id", None), getattr(tok, "eos_token_id", None))})
    return ids


@dataclass
class BiasLexicon:
    by_cat: Dict[str, List[int]]

    @classmethod
    def build(cls, tok: PreTrainedTokenizerBase, lexicon: Dict[str, List[str]]) -> "BiasLexicon":
        by_cat = {k: _words_to_token_ids(tok, v) for k, v in lexicon.items()}
        return cls(by_cat=by_cat)


# ───────────── model: logit-bias head ─────────────
class LogitBiasHead(nn.Module):
    """
    Tiny MLP that maps context features → category weights.
    Inputs: [ z_t (d) , role_bit (1) , hist_feats (2) ] → R^{|CAT_ORDER|}
    Output is in [-1, 1] via tanh; caller scales to bias magnitude.
    """
    def __init__(self, latent_dim: int, hidden: int = 256, num_cats: int = len(CAT_ORDER)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1 + 2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, num_cats),
            nn.Tanh(),  # keep bounded
        )

    def forward(self, z_t: torch.Tensor, role_bit: torch.Tensor, hist_feats: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_t, role_bit, hist_feats], dim=-1)
        return self.net(x)  # [-1,1] per category

    @torch.no_grad()
    def predict_weights(
        self, z_t: torch.Tensor, role: Optional[str], recent_texts: List[str]
    ) -> Dict[str, float]:
        if z_t.dim() == 1:
            z_t = z_t.unsqueeze(0)
        device = next(self.parameters()).device
        z_t = z_t.to(device)
        role_bit = torch.tensor([[ _role_to_bit(role) ]], dtype=z_t.dtype, device=device)
        hist_feats = make_hist_feats(recent_texts).to(device).unsqueeze(0)
        w = self.forward(z_t, role_bit, hist_feats).squeeze(0)  # [C]
        return {cat: float(w[i].item()) for i, cat in enumerate(CAT_ORDER)}


# ───────────── logits processor ─────────────
class JudgeRewardBiasProcessor(LogitsProcessor):
    """
    Adds a sparse bias to token logits each step, based on category weights from LogitBiasHead.
    bias_map: dict[token_id] -> float (already scaled/clipped)
    """
    def __init__(self, bias_map: Dict[int, float], cap: float = BIAS_CAP):
        self.bias_map = bias_map
        self.cap = float(cap)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if not self.bias_map:
            return scores
        # scores: [batch, vocab]; we operate per batch item with the same sparse bias
        with torch.no_grad():
            for tid, b in self.bias_map.items():
                if 0 <= tid < scores.size(-1):
                    scores[:, tid] += torch.tensor(max(-self.cap, min(self.cap, b)), device=scores.device)
        return scores


# ───────────── glue: build bias map for a single decision ─────────────
def build_bias_processor(
    tokenizer: PreTrainedTokenizerBase,
    head: LogitBiasHead,
    *,
    z_t: torch.Tensor,
    role: Optional[str],
    recent_texts: List[str],
    lexicon: Dict[str, List[str]] = DEFAULT_LEXICON,
    strength: float = BASE_STRENGTH,
    persona_scale: Optional[float] = None,
) -> Tuple[LogitsProcessorList, Dict[str, float]]:
    """
    Compute a sparse bias map from head weights and lexicon, and return a LogitsProcessorList.
    Returns (processor_list, weights_by_category).
    """
    # 1) category weights from head
    w = head.predict_weights(z_t=z_t, role=role, recent_texts=recent_texts)  # [-1,1]
    if persona_scale is not None:
        strength = float(strength) * float(persona_scale)

    # 2) tokenize seeds -> token id sets
    lex = BiasLexicon.build(tokenizer, lexicon)

    # 3) aggregate: token_bias = strength * sum_c( w_c * 1_{token in cat c} / |cat_c| )
    token_bias: Dict[int, float] = {}
    for cat, tok_ids in lex.by_cat.items():
        if not tok_ids:
            continue
        w_c = float(w.get(cat, 0.0))
        if abs(w_c) < 1e-6:
            continue
        per_token = (strength * w_c) / float(len(tok_ids))
        for tid in tok_ids:
            token_bias[tid] = token_bias.get(tid, 0.0) + per_token

    if DEBUG:
        top = sorted(token_bias.items(), key=lambda kv: abs(kv[1]), reverse=True)[:10]
        print(f"[LLM-SPK] bias strength={strength:.2f} cats={w} top_ids={top}")

    processor = LogitsProcessorList([JudgeRewardBiasProcessor(token_bias, cap=BIAS_CAP)])
    return processor, w


# ───────────── convenience wrapper for generation ─────────────
@torch.no_grad()
def with_logit_bias_generate_kwargs(
    tokenizer: PreTrainedTokenizerBase,
    head: LogitBiasHead,
    *,
    z_t: torch.Tensor,
    role: Optional[str],
    recent_texts: List[str],
    persona_effects: Optional[Dict[str, float]] = None,
    lexicon: Dict[str, List[str]] = DEFAULT_LEXICON,
    base_strength: float = BASE_STRENGTH,
) -> Dict:
    """
    Build kwargs to pass into model.generate(...). Merge this with your usual generation kwargs.
    Example:
        proc_kwargs = with_logit_bias_generate_kwargs(tok, head, z_t=z, role=agent.role, recent_texts=recent)
        out = model.generate(input_ids=..., **proc_kwargs, **your_other_kwargs)
    """
    scale = float(persona_effects.get("logit_bias_scale", 1.0)) if persona_effects else 1.0
    processors, _ = build_bias_processor(
        tokenizer, head, z_t=z_t, role=role, recent_texts=recent_texts,
        lexicon=lexicon, strength=base_strength, persona_scale=scale,
    )
    return {"logits_processor": processors}


# ───────────── simple REINFORCE learner (optional) ─────────────
class BiasHeadLearner:
    """
    Optional helper to update the LogitBiasHead with REINFORCE-style updates
    using a pseudo-logprob signal derived from category activations.
    This is a heuristic; for rigorous credit assignment you'd capture token logprobs
    and importance weights. Good enough for first experiments.
    """
    def __init__(self, head: LogitBiasHead, lr: float = 1e-3, entropy_bonus: float = 0.005):
        self.head = head
        self.opt = torch.optim.Adam(head.parameters(), lr=lr)
        self.entropy_bonus = float(entropy_bonus)

    def step(
        self,
        batch: List[Dict[str, torch.Tensor]],
        baseline: float = 0.0,
    ) -> Dict[str, float]:
        """
        batch items need:
          {
            "z": Tensor[d],
            "role_bit": Tensor[1],
            "hist_feats": Tensor[2],
            "reward": float,
          }
        """
        if not batch:
            return {"loss": 0.0, "ent": 0.0, "R_mean": 0.0}
        device = next(self.head.parameters()).device

        zs         = torch.stack([b["z"] for b in batch]).to(device)
        role_bits  = torch.stack([b["role_bit"] for b in batch]).to(device)
        hfs        = torch.stack([b["hist_feats"] for b in batch]).to(device)
        rewards    = torch.tensor([float(b["reward"]) for b in batch], device=device)

        # forward: bounded activations in [-1,1]
        w = self.head(zs, role_bits, hfs)  # [B, C]
        # Proxy categorical over "positive/negative activation" magnitude
        probs = torch.softmax(torch.abs(w), dim=-1)  # [B, C]
        ent = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()

        # Surrogate policy gradient: encourage larger magnitude where reward is positive (and vice versa)
        # Note: we detach rewards to avoid leaking gradients
        loss_pg = - (rewards - baseline).unsqueeze(1) * torch.log(probs + 1e-8)
        loss = loss_pg.mean() - self.entropy_bonus * ent

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.head.parameters(), 1.0)
        self.opt.step()

        return {"loss": float(loss.item()), "ent": float(ent.item()), "R_mean": float(rewards.mean().item())}
