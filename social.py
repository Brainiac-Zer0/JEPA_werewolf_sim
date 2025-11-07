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

def _safe_var(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # variance over feature dim, return scalar per batch item
    if x.dim() == 1:
        x = x.unsqueeze(0)
    v = torch.var(x, dim=-1, unbiased=False)
    v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    return v  # shape (B,)

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
        clamp_coef: float = 1.0,   # extra safety multiplier on the step after normalization
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.enabled = bool(enabled)
        self.scale = float(scale)
        self.reg_lambda = float(reg_lambda)
        self.trust_mode = (trust_mode or "none").lower()
        self.tau = float(tau)
        self.max_step = float(max_step)
        self.clamp_coef = float(max(0.0, clamp_coef))
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
        s = (a_n * b_n).sum(dim=-1, keepdim=True)
        return torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=-0.0)  # shape (B,1)

    def _weights_uniform(self, n: int, device: torch.device) -> torch.Tensor:
        w = torch.ones(n, 1, device=device) / max(1, n)
        return torch.nan_to_num(w, nan=0.0)

    def _weights_cosine(self, z_self: torch.Tensor, neighbors: List[torch.Tensor]) -> torch.Tensor:
        """
        Batch-safe cosine trust. Returns (n,1) weights.
        """
        device = z_self.device
        if not neighbors:
            return torch.zeros(0, 1, device=device)

        sims = []
        for z_n in neighbors:
            c = self._cosine_sim(z_self, _as_2d(z_n))
            sims.append(float(c.mean().item()))
        sims_t = torch.tensor(sims, device=device, dtype=torch.float32)

        logits = sims_t / max(1e-6, self.tau)
        logits = logits - logits.max()
        w = torch.softmax(logits, dim=0).unsqueeze(-1)  # (n,1)
        if torch.isnan(w).any() or float(w.sum().item()) == 0.0:
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
                c = float(self._cosine_sim(z_self, _as_2d(z_n)).mean().item())
                val = 0.7 * base + 0.3 * (0.5 * (c + 1.0))
            else:
                val = base
            raw.append(val)
        raw_t = torch.tensor(raw, device=device, dtype=torch.float32).clamp_min(1e-6)
        raw_t = raw_t / raw_t.sum()
        w = raw_t.unsqueeze(-1)
        return torch.nan_to_num(w, nan=0.0)

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
            delta: [D] social update (bounded, numerically safe)
            info: telemetry dict with delta_norm, mean_delta_norm, var_before, var_after, and counts
        """
        z_self = _as_2d(torch.nan_to_num(z_self, nan=0.0)).to(self.device or z_self.device)  # (B,D)
        neigh_list = [_as_2d(torch.nan_to_num(z, nan=0.0)).to(z_self.device) for z in z_neighbors]  # list of (B,D)
        n = len(neigh_list)

        var_before = _safe_var(z_self).mean().item()

        info = {
            "n_neighbors": float(n),
            "trust_mode": self.trust_mode,
            "scale": float(self.scale),
            "reg_lambda": float(self.reg_lambda),
            "enabled": float(self.enabled),
            "var_before": float(var_before),
            "applied_count": 0.0,
        }

        if n == 0:
            # No neighbors, identity update with robust stats
            zero = torch.zeros_like(z_self)
            var_after = float(_safe_var(z_self).mean().item())
            info.update({
                "delta_norm": 0.0,
                "mean_delta_norm": 0.0,
                "w_entropy": 0.0,
                "mean_cosine_to_mu": 0.0,
                "no_neighbors": True,
                "var_after": var_after,
                # NEW: bounded pressure exported for mouthpiece
                "social_pressure": 0.0,
            })
            return zero.squeeze(0), info

        # Trust weights
        if self.trust_mode == "cosine":
            w = self._weights_cosine(z_self, neigh_list)  # (n,1)
        elif self.trust_mode == "role_affinity":
            w = self._weights_role_affinity(z_self, neigh_list, self_role, list(neighbor_roles or []))  # (n,1)
        else:
            w = self._weights_uniform(n, z_self.device)  # (n,1)

        applied_count = float((w.squeeze(-1) > (1e-6 / max(1, n))).sum().item())
        info["applied_count"] = applied_count

        # Aggregate neighbor mean
        if z_self.size(0) == 1:
            Z = torch.stack([z for z in neigh_list], dim=0).squeeze(1)  # (n,D)
            Z = torch.nan_to_num(Z + 1e-6 * torch.randn_like(Z), nan=0.0)
            mu = (w * Z).sum(dim=0, keepdim=True)  # (1,D)
        else:
            Z = torch.stack([z for z in neigh_list], dim=0)  # (n,B,D)
            Z = torch.nan_to_num(Z + 1e-6 * torch.randn_like(Z), nan=0.0)
            w_b = w.unsqueeze(1)  # (n,1,1)
            mu = (w_b * Z).sum(dim=0)  # (B,D)

        # Cosine similarity feature; batch-safe
        sim = self._cosine_sim(z_self, mu)  # (B,1)

        # MLP input: (mu - z_self, sim), with safety
        diff = torch.nan_to_num(mu - z_self, nan=0.0)
        inp = torch.cat([diff, sim], dim=-1)  # (B, D+1)
        inp = torch.nan_to_num(inp, nan=0.0)

        delta_raw = self.net(inp)                     # (B, D)
        delta = self.out_ln(delta_raw)                # (B, D)
        delta = torch.nan_to_num(delta, nan=0.0)

        # Normalize and bound step size, then apply scale and clamp_coef
        step_norm = _l2_norm(delta).clamp_min(1e-8)                 # (B,1)
        step_norm_scalar = step_norm.mean().clamp_min(1e-6)         # scalar tensor
        scale_cap = min(self.max_step, 1.0) * max(0.0, self.clamp_coef)
        scale_fac = scale_cap / step_norm_scalar
        delta = delta * scale_fac
        delta = self.scale * delta

        # Final safety clean
        delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)

        with torch.no_grad():
            # Entropy of weights
            w_p = torch.nan_to_num(w.squeeze(-1), nan=0.0)  # (n,)
            s = float(w_p.sum().item())
            if s <= 0.0:
                w_p = self._weights_uniform(n, z_self.device).squeeze(-1)
                s = float(w_p.sum().item())
            w_p = w_p / max(s, 1e-6)
            H_w = -torch.sum(w_p * torch.log(torch.clamp(w_p, min=1e-8)))

            final_norm_per_b = _l2_norm(delta).squeeze(-1)        # (B,)
            mean_delta_norm = float(final_norm_per_b.mean().item())
            var_after = _safe_var(z_self + delta).mean().item()

            # NEW: bounded social pressure for mouthpiece sampling nudges
            social_pressure = float(min(1.0, max(0.0, 4.0 * mean_delta_norm)))

            info.update({
                "delta_norm": mean_delta_norm,        # kept for backward compatibility
                "mean_delta_norm": mean_delta_norm,   # explicit name for dashboards
                "w_entropy": float(H_w.item()),
                "mean_cosine_to_mu": float(self._cosine_sim(z_self, mu).mean().item()),
                "var_after": float(var_after),
                "social_pressure": social_pressure,   # NEW
            })
        return delta.squeeze(0), info

    # ------------------------- loss regularizer -----------------------------

    def regularizer(self, delta: torch.Tensor) -> torch.Tensor:
        """Small penalty λ * ||δ||²."""
        if not torch.is_tensor(delta):
            delta = torch.tensor(delta, dtype=torch.float32, device=self.device or "cpu")
        delta = torch.nan_to_num(delta, nan=0.0)
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
    clamp_coef: float = 1.0

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
        clamp_coef=float(social.get("clamp_coef", 1.0)),
    )

def apply_social_update(
    social: SocialInfluence,
    z_self: torch.Tensor,
    z_neighbors: Iterable[torch.Tensor],
    *,
    self_role: Optional[str] = None,
    neighbor_roles: Optional[Iterable[Optional[str]]] = None,
) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
    """
    Functional wrapper: returns (z_updated, info, reg_term).

    Guarantees:
      - Numerically safe updates with NaN protections.
      - Info includes mean_delta_norm, var_before, var_after, and applied_count.
      - Never overwrites beliefs with zeros. If disabled or no neighbors, returns identity.
    """
    # Hard-gate: if disabled, return identity update with zero regularizer.
    if not getattr(social, "enabled", True):
        var_before = float(_safe_var(_as_2d(torch.nan_to_num(z_self, nan=0.0))).mean().item())
        info = {
            "n_neighbors": 0.0,
            "trust_mode": "none",
            "scale": float(getattr(social, "scale", 0.0)),
            "reg_lambda": float(getattr(social, "reg_lambda", 0.0)),
            "enabled": 0.0,
            "delta_norm": 0.0,
            "mean_delta_norm": 0.0,
            "w_entropy": 0.0,
            "mean_cosine_to_mu": 0.0,
            "applied_count": 0.0,
            "var_before": var_before,
            "var_after": var_before,
            # NEW: bounded pressure when disabled -> 0
            "social_pressure": 0.0,
        }
        return z_self, info, torch.tensor(0.0, device=z_self.device)

    delta, info = social(z_self, z_neighbors, self_role=self_role, neighbor_roles=neighbor_roles)
    # Identity fallback if something went wrong, without overwriting with zeros
    if not torch.is_tensor(delta):
        delta = torch.zeros_like(_as_2d(z_self))
    delta = torch.nan_to_num(_as_2d(delta), nan=0.0)
    z_new = _as_2d(z_self) + delta
    z_new = torch.nan_to_num(z_new, nan=0.0).squeeze(0)

    # Surface pre and post variance at this level as well
    info.setdefault("var_before", float(_safe_var(_as_2d(z_self)).mean().item()))
    info["var_after"] = float(_safe_var(_as_2d(z_new)).mean().item())

    # Ensure social_pressure is present and bounded even if upstream changed
    sp = float(min(1.0, max(0.0, 4.0 * float(info.get("mean_delta_norm", 0.0)))))
    info["social_pressure"] = sp

    reg = social.regularizer(delta)
    return z_new, info, reg
