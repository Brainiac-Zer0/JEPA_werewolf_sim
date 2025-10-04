# speaker.py
import yaml, torch, torch.nn as nn, torch.nn.functional as F
from typing import List, Dict, Any, Tuple, Optional

# ── Load config
with open("config.yaml", "r") as f:
    CFG = yaml.safe_load(f)

# Default templates; you can override per-game
DEFAULT_TEMPLATES = CFG.get("DEFAULT_TEMPLATES", [
    "Accuse {target}",
    "Defend {ally}",
    "Ask {target} a question",
    "Express uncertainty",
    "Propose vote on {target}",
])

def make_hist_feats(recent_texts: List[str]) -> torch.Tensor:
    if not recent_texts:
        return torch.tensor([0.0, 0.0])
    n = len(recent_texts)
    acc = sum(int(("accuse" in t.lower()) or ("vote" in t.lower())) for t in recent_texts) / n
    mean_len = min(1.5, sum(len(t) for t in recent_texts) / max(1, n) / 100.0)
    return torch.tensor([acc, mean_len], dtype=torch.float32)

class SpeakerBandit(nn.Module):
    """
    Tiny bandit over speech-act templates.
    Input: [z_t (d), role_bit (1), hist_feats (2)] → logits over templates.
    Training: REINFORCE on message-level reward.
    """
    def __init__(self, latent_dim: int, num_templates: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1 + 2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, num_templates),
        )
        self.temperature = 1.0
        self.num_templates = num_templates

    def forward(self, z: torch.Tensor, role_bit: torch.Tensor, hist_feats: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, role_bit, hist_feats], dim=-1)
        return self.net(x)

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
        **_ignored,  # tolerant to future kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        # --- ensure everything is on the module's device ---
        dev = next(self.parameters()).device
        if z_t.dim() == 1:
            z_t = z_t.unsqueeze(0)
        z_t = z_t.to(dev)

        role_bit = torch.tensor([[1.0 if role.lower().startswith("were") else 0.0]],
                                device=dev, dtype=z_t.dtype)
        hist_feats = make_hist_feats(recent_texts).to(dev).unsqueeze(0)

        logits = self.forward(z_t, role_bit, hist_feats).squeeze(0)

        # Persona biases (optional, light touch)
        if persona_effects:
            accuse_bias = float(persona_effects.get("accuse_bias", 0.0))
            if accuse_bias != 0.0:
                idx_accuse = [i for i, t in enumerate(templates)
                              if ("accuse" in t.lower()) or ("vote" in t.lower()) or ("propose" in t.lower())]
                idx_uncert = [i for i, t in enumerate(templates)
                              if ("uncertain" in t.lower()) or ("uncert" in t.lower())]
                for i in idx_accuse: logits[i] = logits[i] + accuse_bias
                for i in idx_uncert: logits[i] = logits[i] - 0.5 * accuse_bias

        # Persona-driven temperature scaling (safe clamp)
        temp_scale = 1.0
        if persona_effects:
            try:
                temp_scale = float(persona_effects.get("speaker_temp_scale", 1.0))
            except Exception:
                temp_scale = 1.0
        temperature = max(1e-4, self.temperature * temp_scale)

        probs = F.softmax(logits / temperature, dim=-1)
        tidx  = torch.multinomial(probs, 1).item()

        # Slot filling
        target = next((t for t in candidate_targets if t != self_name), None)
        target = target or (candidate_targets[0] if candidate_targets else self_name)

        text = templates[tidx].replace("{target}", target).replace("{ally}", self_name)

        meta = {
            "template_id": tidx,
            "logprob": float(torch.log(probs[tidx] + 1e-8).item()),
            "role_bit": role_bit.squeeze(0).detach().cpu(),
            "hist_feats": hist_feats.squeeze(0).detach().cpu(),
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
