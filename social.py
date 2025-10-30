from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ------------------------------ helpers -------------------------------------

def _as_2d(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1:
        return x.unsqueeze(0)
    return x

def _l2_norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.sqrt(torch.clamp((x ** 2).sum(dim=-1, keepdim=True), min=eps))

def _safe_mean(xs: List[torch.Tensor]) -> Optional[torch.Tensor]:
    xs = [x for x in xs if torch.is_tensor(x)]
    if not xs:
        return None
    return torch.stack(xs, dim=0).mean(dim=0)

# ------------------------------ core module ---------------------------------

class SocialInfluence(nn.Module):
    """
    Computes a social update δ_social given self latent z_self and neighbor latents:
        δ = scale * MLP((μ_neighbors - z_self) ⊕ sim)
    with optional trust/similarity weighting of μ_neighbors.

    trust_mode:
        - "none": uniform weights
        - "cosine": weights ∝ softmax(cos(z_self, z_n)/τ)
        - "role_affinity": weighted by a 2×2 role-affinity table

    Returns (delta, info) where info holds telemetry for logging.
    """
    def __init__(
        self,
        latent_dim: int,
        *,
        enabled: bool = True,
        scale: float = 0.05,
        hidden: int = 64,
        reg_lambda: float = 1e-3,
        trust_mode: str = "none",
        tau: float = 0.5,
        role_affinity: Optional[Dict[Tuple[str, str], float]] = None,
        max_step: float = 0.25,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.enabled = bool(enabled)
        self.scale = float(scale)
        self.reg_lambda = float(reg_lambda)
        self.trust_mode = (trust_mode or "none").lower()
        self.tau = float(tau)
        self.max_step = float(max_step)
        self.device = device

        # Default 2x2 role affinity table
        self.role_affinity = role_affinity or {
            ("Villager", "Villager"): 1.0,
            ("Villager", "Werewolf"): 0.5,
            ("Werewolf", "Villager"): 0.2,
            ("Werewolf", "Werewolf"): 0.8,
        }

        # small MLP: input = latent_dim + 1 (for sim feature)
        self.net = nn.Sequential(
            nn.Linear(self.latent_dim + 1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, self.latent_dim),
        )
        self.out_ln = nn.LayerNorm(self.latent_dim)

    # ------------------------- trust/similarity -----------------------------

    @staticmethod
    def _cosine_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        a_n = a / a.norm(dim=-1, keepdim=True).clamp_min(eps)
        b_n = b / b.norm(dim=-1, keepdim=True).clamp_min(eps)
        return (a_n * b_n).sum(dim=-1, keepdim=True)  # shape (B,1)

    def _weights_uniform(self, n: int, device: torch.device) -> torch.Tensor:
        return torch.ones(n, 1, device=device) / max(1, n)  # (n,1)

    def _weights_cosine(self, z_self: torch.Tensor, neighbors: List[torch.Tensor]) -> torch.Tensor:
        """
        Batch-safe cosine trust. Returns (n,1) weights.
        """
        device = z_self.device
        if not neighbors:
            return torch.zeros(0, 1, device=device)

        # Reduce each neighbor cosine to a scalar by averaging across batch
        sims = torch.tensor(
            [self._cosine_sim(z_self, z_n).mean().item() for z_n in neighbors],
            device=device,
            dtype=torch.float32,
        )  # (n,)

        # Softmax with basic numerical stability
        logits = sims / max(1e-6, self.tau)
        logits = logits - logits.max()  # stability
        w = torch.softmax(logits, dim=0).unsqueeze(-1)  # (n,1)
        if torch.isnan(w).any() or float(w.sum().item()) == 0.0:
            # fallback to uniform if weirdness
            w = self._weights_uniform(len(neighbors), device)
        return w

    def _weights_role_affinity(
        self,
        z_self: torch.Tensor,
        neighbors: List[torch.Tensor],
        self_role: Optional[str],
        neighbor_roles: Optional[List[Optional[str]]],
        smooth_with_cosine: bool = True,
    ) -> torch.Tensor:
        device = z_self.device
        if not neighbors:
            return torch.zeros(0, 1, device=device)
        if self_role is None or neighbor_roles is None or len(neighbor_roles) != len(neighbors):
            return self._weights_uniform(len(neighbors), device)
        raw = []
        for z_n, r_n in zip(neighbors, neighbor_roles):
            base = float(self.role_affinity.get((str(self_role), str(r_n)), 0.5))
            if smooth_with_cosine:
                # Batch-safe cosine => average to scalar
                c = float(self._cosine_sim(z_self, z_n).mean().item())
                val = 0.7 * base + 0.3 * (0.5 * (c + 1.0))
            else:
                val = base
            raw.append(val)
        raw_t = torch.tensor(raw, device=device, dtype=torch.float32).clamp_min(1e-6)  # (n,)
        raw_t = raw_t / raw_t.sum()
        return raw_t.unsqueeze(-1)  # (n,1)

    # ------------------------------ forward --------------------------------

    def forward(
        self,
        z_self: torch.Tensor,
        z_neighbors: Iterable[torch.Tensor],
        *,
        self_role: Optional[str] = None,
        neighbor_roles: Optional[Iterable[Optional[str]]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            z_self: [D] or [1,D]
            z_neighbors: iterable of [D] tensors
        Returns:
            delta: [D] social update
            info: telemetry dict
        """
        z_self = _as_2d(z_self).to(self.device or z_self.device)  # (B,D)
        neigh_list = [_as_2d(z).to(z_self.device) for z in z_neighbors]  # list of (B,D)
        n = len(neigh_list)
        info = {
            "n_neighbors": float(n),
            "trust_mode": self.trust_mode,
            "scale": float(self.scale),
            "reg_lambda": float(self.reg_lambda),
            "enabled": float(self.enabled),
        }
        if n == 0:
            zero = torch.zeros_like(z_self)
            info.update({"delta_norm": 0.0, "w_entropy": 0.0})
            return zero.squeeze(0), info

        if self.trust_mode == "cosine":
            w = self._weights_cosine(z_self, neigh_list)  # (n,1)
        elif self.trust_mode == "role_affinity":
            w = self._weights_role_affinity(z_self, neigh_list, self_role, list(neighbor_roles or []))  # (n,1)
        else:
            w = self._weights_uniform(n, z_self.device)  # (n,1)

        # Aggregate neighbor mean (batchwise)
        if z_self.size(0) == 1:
            # Simple/common case: B==1
            Z = torch.stack([z for z in neigh_list], dim=0).squeeze(1)  # (n,D)
            # add tiny jitter to avoid degenerate similarities
            Z = Z + 1e-6 * torch.randn_like(Z)
            mu = (w * Z).sum(dim=0, keepdim=True)  # (1,D)
        else:
            # If B>1, broadcast weights across batch
            Z = torch.stack([z for z in neigh_list], dim=0)  # (n,B,D)
            # add tiny jitter to avoid degenerate similarities
            Z = Z + 1e-6 * torch.randn_like(Z)
            w_b = w.unsqueeze(1)  # (n,1,1)
            mu = (w_b * Z).sum(dim=0)  # (B,D)

        # Cosine similarity feature; batch-safe -> reduce to (B,1)
        sim = self._cosine_sim(z_self, mu)  # (B,1)

        # MLP input: (mu - z_self, sim)
        inp = torch.cat([mu - z_self, sim], dim=-1)  # (B, D+1)
        delta_raw = self.net(inp)                     # (B, D)
        delta = self.out_ln(delta_raw)                # (B, D)

        # Normalize step size using batch-mean norm (scalar), with clamped denom
        step_norm = _l2_norm(delta).clamp_min(1e-8)            # (B,1)
        step_norm_scalar = step_norm.mean().clamp_min(1e-6)    # scalar tensor
        # cap the per-update magnitude in latent units, then apply user scale
        scale_cap = min(self.max_step, 1.0)
        scale_fac = scale_cap / step_norm_scalar
        delta = delta * scale_fac
        delta = self.scale * delta                              # FINAL delta

        with torch.no_grad():
            # Entropy of weights (treat as categorical over neighbors)
            w_p = (w.squeeze(-1) + 1e-8)  # (n,)
            w_p = w_p / w_p.sum()
            H_w = -torch.sum(w_p * torch.log(w_p))

            # Telemetry should reflect the FINAL scaled delta magnitude
            final_norm = _l2_norm(delta).mean()  # scalar tensor

            info.update({
                "delta_norm": float(final_norm.item()),
                "w_entropy": float(H_w.item()),
                "mean_cosine_to_mu": float(self._cosine_sim(z_self, mu).mean().item()),
            })
        return delta.squeeze(0), info

    # ------------------------- loss regularizer -----------------------------

    def regularizer(self, delta: torch.Tensor) -> torch.Tensor:
        """Small penalty λ * ||δ||²."""
        if not torch.is_tensor(delta):
            delta = torch.tensor(delta, dtype=torch.float32, device=self.device or "cpu")
        return self.reg_lambda * (delta.view(-1).dot(delta.view(-1)))

# --------------------------- convenience API -------------------------------

@dataclass
class SocialConfig:
    enabled: bool = True
    scale: float = 0.05
    hidden: int = 64
    lambda_reg: float = 1e-3
    trust: str = "none"
    tau: float = 0.5
    max_step: float = 0.25

def build_from_cfg(cfg: Dict) -> SocialInfluence:
    """Construct a SocialInfluence from a simple dict (YAML section)."""
    social = cfg or {}
    latent_dim = int(social.get("latent_dim", 32))
    return SocialInfluence(
        latent_dim=latent_dim,
        enabled=bool(social.get("enabled", True)),
        scale=float(social.get("scale", 0.05)),
        hidden=int(social.get("hidden", 64)),
        reg_lambda=float(social.get("lambda_reg", 1e-3)),
        trust_mode=str(social.get("trust", "none")),
        tau=float(social.get("tau", 0.5)),
        max_step=float(social.get("max_step", 0.25)),
    )

def apply_social_update(
    social: SocialInfluence,
    z_self: torch.Tensor,
    z_neighbors: Iterable[torch.Tensor],
    *,
    self_role: Optional[str] = None,
    neighbor_roles: Optional[Iterable[Optional[str]]] = None,
) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
    """Functional wrapper: returns (z_updated, info, reg_term)."""
    # Hard-gate: if disabled, return identity update with zero regularizer.
    if not getattr(social, "enabled", True):
        info = {
            "n_neighbors": 0.0,
            "trust_mode": "none",
            "scale": float(getattr(social, "scale", 0.0)),
            "reg_lambda": float(getattr(social, "reg_lambda", 0.0)),
            "enabled": 0.0,
            "delta_norm": 0.0,
            "w_entropy": 0.0,
            "mean_cosine_to_mu": 0.0,
        }
        return z_self, info, torch.tensor(0.0, device=z_self.device)

    delta, info = social(z_self, z_neighbors, self_role=self_role, neighbor_roles=neighbor_roles)
    z_new = z_self + delta
    reg = social.regularizer(delta)
    return z_new, info, reg
