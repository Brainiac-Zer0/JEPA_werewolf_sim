# training_utils.py ── utility helpers for JEPA training, determinism & checkpoint I/O
from __future__ import annotations

import os
import random
import math
import csv
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch, yaml
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from encoders import ActionEncoder, PlannerHead, WorldModelMLP
# NEW: imports for phase-aware and factorized planner path
from encoders import PhaseActionEncoder  # noqa: E402
try:
    from encoders import FactorizedPlanner, TalkHead, VoteHead, KillHead  # optional convenience
except Exception:
    # Fallback shim if FactorizedPlanner isn't exported; assumes Talk/Vote/Kill are available
    class FactorizedPlanner(nn.Module):
        def __init__(self, latent_dim: int, num_agents: int, num_talk_cats: int):
            super().__init__()
            self.talk = TalkHead(latent_dim, num_talk_cats)
            self.vote = VoteHead(latent_dim, num_agents)
            self.kill = KillHead(latent_dim, num_agents)

        def forward(
            self,
            z: torch.Tensor,
            *,
            talk_mask: Optional[torch.Tensor] = None,
            vote_mask: Optional[torch.Tensor] = None,
            kill_mask: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
            return {
                "talk": self.talk(z, talk_mask),
                "vote": self.vote(z, vote_mask),
                "kill": self.kill(z, kill_mask),
            }

CHECKPOINT_DIR = "checkpoints"
LOGS_DIR = "logs"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load config
with open("config.yaml", "r") as f:
    CFG = yaml.safe_load(f) or {}

# Tunables
LAMBDA_BC: float = float(CFG.get("LAMBDA_BC", 0.5))          # used for vote/kill heads (and legacy BC)
LAMBDA_TALK: float = float(CFG.get("LAMBDA_TALK", LAMBDA_BC)) # CE weight for talk head (defaults to LAMBDA_BC)
MAX_NORM: float = float(CFG.get("MAX_NORM", 1.0))

# Optional dims (used by helpers below; keep legacy defaults)
PHASE_COUNT: int = int(CFG.get("PHASE_COUNT", 3))            # DISCUSS/VOTE/NIGHT
NUM_TALK_CATS: int = int(CFG.get("NUM_TALK_CATS", 5))
NUM_AGENTS_CFG: int = int(CFG.get("NUM_AGENTS", 6))
ACTION_EMBED_DIM: int = int(CFG.get("ACTION_DIM", CFG.get("ACTION_EMBED_DIM", 8)))  # prefer ACTION_DIM if present
LATENT_DIM: int = int(CFG.get("LATENT_DIM", 32))

# =============================================================================
# Determinism & run metadata
# =============================================================================

def set_global_determinism(seed: int = 1337) -> None:
    """Make training runs reproducible (as much as PyTorch allows)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

def save_run_config(run_id: str, cfg: Dict[str, Any]) -> str:
    """Snapshot the effective config for auditability."""
    run_dir = os.path.join(LOGS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)
    out = os.path.join(run_dir, "config.snapshot.yaml")
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=True)
    return out

# =============================================================================
# Training epoch logger (CSV)
# =============================================================================

class TrainingEpochLogger:
    """Append epoch-level metrics to logs/metrics_train.csv."""
    def __init__(self, csv_path: str = os.path.join(LOGS_DIR, "metrics_train.csv")):
        self.csv_path = csv_path
        self._ensure_header()

    def _ensure_header(self):
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=[
                        "run_id","role","epoch","L_mse","L_bc",
                        "grad_norm","lr","n_batches"
                    ],
                )
                w.writeheader()

    def log(self, row: Dict[str, Any]):
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "run_id","role","epoch","L_mse","L_bc",
                    "grad_norm","lr","n_batches"
                ],
            )
            w.writerow(row)

# =============================================================================
# Rollout normalization (legacy & phase-aware)
# =============================================================================

@dataclass
class RolloutSample:
    z_t: torch.Tensor
    a_idx: Optional[torch.Tensor]
    z_next: torch.Tensor
    role: str
    # Phase-aware (optional)
    phase: Optional[int] = None              # 0..PHASE_COUNT-1
    payload_idx: Optional[int] = None        # talk_cat or target idx
    choice_type: Optional[str] = None        # "TALK_INTENT"|"VOTE_TARGET"|"KILL_TARGET"
    # Optional per-row meta for masks (alive flags, wolf team, self idx, etc.)
    aux: Optional[Dict[str, Any]] = None

def normalize_rollouts(raw: List[Tuple]) -> List[RolloutSample]:
    """
    Accepts:
      - legacy: (z_t, a_idx, z_next, role)
      - phase-aware: (z_t, phase_code, action_payload, z_next, role[, choice_type[, aux_meta_dict]])
    """
    out: List[RolloutSample] = []
    for tup in raw:
        if len(tup) == 4:
            z_t, a_idx, z_next, role = tup
            out.append(RolloutSample(z_t=z_t, a_idx=a_idx, z_next=z_next, role=role))
        else:
            # phase-aware path
            z_t, phase, payload, z_next, role = tup[:5]
            ct = tup[5] if len(tup) >= 6 else None
            aux = tup[6] if len(tup) >= 7 and isinstance(tup[6], dict) else None
            out.append(RolloutSample(
                z_t=z_t, a_idx=None, z_next=z_next, role=role,
                phase=int(phase), payload_idx=int(payload), choice_type=ct, aux=aux
            ))
    return out

# =============================================================================
# Phase-aware action embedding (non-invasive bridge for Phase 1)
# =============================================================================

def build_action_embed_phaseaware(
    phase: torch.Tensor,            # [B]
    payload: torch.Tensor,          # [B]
    *,
    num_agents: int = NUM_AGENTS_CFG,
    num_talk_cats: int = NUM_TALK_CATS,
    action_dim: int = ACTION_EMBED_DIM,
    choice_type: Optional[str] = None,
) -> torch.Tensor:
    """
    Compose a one-hot concat [phase_onehot | payload_onehot] and project to action_dim.
    Created on the fly (no persistent params) to avoid changing model signatures in Phase 1.
    """
    B = phase.shape[0]
    phase_oh = F.one_hot(
        phase.clamp(min=0, max=PHASE_COUNT-1),
        num_classes=PHASE_COUNT
    ).float()

    if choice_type == "TALK_INTENT":
        n = num_talk_cats
    else:
        n = num_agents  # vote/kill share the agent index space

    payload = payload.clamp(min=0, max=n-1)
    pay_oh = F.one_hot(payload, num_classes=n).float()

    concat = torch.cat([phase_oh, pay_oh], dim=-1)  # [B, PHASE_COUNT + n]
    proj = nn.Linear(concat.shape[-1], action_dim, bias=False).to(concat.device)
    with torch.no_grad():
        nn.init.xavier_uniform_(proj.weight)
    return proj(concat)

# =============================================================================
# Legacy trainer (unchanged behavior) + optional CSV logging
# =============================================================================

def train_jepa(
    rollout_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]],
    world_model: WorldModelMLP,
    action_encoder: ActionEncoder,
    planner: PlannerHead,
    role_name: str = "agent",
    *,
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    run_id: str = "run",
    epoch_logger: Optional[TrainingEpochLogger] = None,
) -> None:
    if not rollout_data:
        print(f"[{role_name}] No rollout data; skipping JEPA update.")
        return

    world_model.to(DEVICE)
    action_encoder.to(DEVICE)
    planner.to(DEVICE)

    optimizer = optim.Adam(
        list(world_model.parameters()) +
        list(action_encoder.parameters()) +
        list(planner.parameters()),
        lr=learning_rate,
    )
    mse_loss = nn.MSELoss()
    ce_loss  = nn.CrossEntropyLoss()

    world_model.train(), action_encoder.train(), planner.train()

    for ep in range(1, epochs + 1):
        random.shuffle(rollout_data)

        batches = [
            rollout_data[i : i + batch_size]
            for i in range(0, len(rollout_data), batch_size)
        ]

        epoch_mse = 0.0
        epoch_bc  = 0.0
        total_gn  = 0.0

        for batch in batches:
            z_t, a_idx, z_next = zip(*[(r[0], r[1], r[2]) for r in batch])
            z_t_tensor     = torch.stack(z_t).to(DEVICE)                     # [B, latent]
            a_idx_tensor   = torch.stack(a_idx).long().squeeze().to(DEVICE)  # [B]
            z_next_tensor  = torch.stack(z_next).to(DEVICE)                   # [B, latent]

            # forward
            a_embed = action_encoder(a_idx_tensor)                            # [B, a_dim]
            z_pred  = world_model(z_t_tensor, a_embed)                        # [B, latent]
            logits  = planner(z_t_tensor)                                     # [B, num_agents]

            L_mse = mse_loss(z_pred, z_next_tensor)
            L_bc  = ce_loss(logits, a_idx_tensor)
            loss  = L_mse + LAMBDA_BC * L_bc

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # NaN guard + grad clipping
            params = list(world_model.parameters()) + list(action_encoder.parameters()) + list(planner.parameters())
            for p in params:
                if p.grad is not None and torch.isnan(p.grad).any():
                    raise RuntimeError("NaN in gradients!")
            gnorm = torch.nn.utils.clip_grad_norm_(params, MAX_NORM)
            optimizer.step()

            epoch_mse += L_mse.item()
            epoch_bc  += L_bc.item()
            total_gn  += float(gnorm)

        denom = max(1, len(batches))
        mean_mse = epoch_mse/denom
        mean_bc  = epoch_bc/denom
        mean_gn  = total_gn/denom if denom else 0.0
        cur_lr   = optimizer.param_groups[0]["lr"]

        print(f"[{role_name}  Epoch {ep}/{epochs}]  MSE: {mean_mse:.4f}  BC: {mean_bc:.4f}  |grad|: {mean_gn:.3f}")

        if epoch_logger is not None:
            epoch_logger.log({
                "run_id": run_id,
                "role": role_name,
                "epoch": ep,
                "L_mse": round(mean_mse, 6),
                "L_bc": round(mean_bc, 6),
                "grad_norm": round(mean_gn, 6),
                "lr": cur_lr,
                "n_batches": len(batches),
            })

    # save
    save_path = os.path.join(CHECKPOINT_DIR, f"{role_name.lower()}_jepa.pt")
    torch.save(
        {
            "world_model": world_model.state_dict(),
            "action_encoder": action_encoder.state_dict(),
            "planner": planner.state_dict(),
        },
        save_path,
    )
    print(f"[SAVE] {role_name} models saved → {save_path}")

# =============================================================================
# Phase-aware trainer (bridge; safe to ignore until split-heads land)
# =============================================================================

def train_jepa_phaseaware(
    rollout_data_phaseaware: List[Tuple],  # (z_t, phase_code, action_payload, z_next, role[, choice_type])
    world_model: WorldModelMLP,
    planner: PlannerHead,                  # legacy head; optional BC on vote/kill
    *,
    role_name: str = "agent",
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    run_id: str = "run",
    epoch_logger: Optional[TrainingEpochLogger] = None,
    phase_action_encoder: Optional[PhaseActionEncoder] = None,  # NEW: learnable encoder
) -> None:
    rows = normalize_rollouts(rollout_data_phaseaware)
    if not rows:
        print(f"[{role_name}] No rollout data; skipping JEPA (phase-aware) update.")
        return

    # NEW: create or use provided PhaseActionEncoder (learnable, persisted)
    pae = phase_action_encoder or PhaseActionEncoder(
        action_dim=ACTION_EMBED_DIM,
        num_agents=NUM_AGENTS_CFG,
        num_talk=NUM_TALK_CATS,
    )

    world_model.to(DEVICE)
    planner.to(DEVICE)
    pae.to(DEVICE)

    optimizer = optim.Adam(
        list(world_model.parameters()) +
        list(planner.parameters()) +
        list(pae.parameters()),
        lr=learning_rate,
    )
    mse_loss = nn.MSELoss()
    ce_loss  = nn.CrossEntropyLoss()

    world_model.train(), planner.train(), pae.train()

    for ep in range(1, epochs + 1):
        random.shuffle(rows)
        batches = [rows[i:i+batch_size] for i in range(0, len(rows), batch_size)]

        epoch_mse = 0.0
        epoch_bc  = 0.0
        total_gn  = 0.0

        for batch in batches:
            z_t_tensor    = torch.stack([r.z_t for r in batch]).to(DEVICE)
            z_next_tensor = torch.stack([r.z_next for r in batch]).to(DEVICE)

            phase = torch.tensor([r.phase if r.phase is not None else 0 for r in batch], device=DEVICE, dtype=torch.long)
            payload = torch.tensor([r.payload_idx if r.payload_idx is not None else 0 for r in batch], device=DEVICE, dtype=torch.long)
            choice_types = [r.choice_type for r in batch]

            # NEW: learnable phase-aware action embedding
            # Decide is_talk per item; pack through pae in two passes for simplicity
            is_talk_mask = torch.tensor([ct == "TALK_INTENT" for ct in choice_types], device=DEVICE, dtype=torch.bool)
            a_embed = torch.zeros(z_t_tensor.size(0), ACTION_EMBED_DIM, device=DEVICE)

            if is_talk_mask.any():
                a_embed[is_talk_mask] = pae(
                    phase_code=phase[is_talk_mask],
                    payload_idx=payload[is_talk_mask],
                    is_talk=True,
                )
            if (~is_talk_mask).any():
                a_embed[~is_talk_mask] = pae(
                    phase_code=phase[~is_talk_mask],
                    payload_idx=payload[~is_talk_mask],
                    is_talk=False,
                )

            z_pred = world_model(z_t_tensor, a_embed)
            L_mse = mse_loss(z_pred, z_next_tensor)

            # Optional planner BC against vote/kill payloads (skip talk)
            mask_idx = [i for i, ct in enumerate(choice_types) if ct in (None, "VOTE_TARGET", "KILL_TARGET")]
            if mask_idx:
                idx_tensor = torch.tensor(mask_idx, device=DEVICE, dtype=torch.long)
                logits = planner(z_t_tensor[idx_tensor])
                targets = payload[idx_tensor].clamp(0, NUM_AGENTS_CFG-1)
                L_bc = ce_loss(logits, targets)
            else:
                L_bc = torch.tensor(0.0, device=DEVICE)

            loss = L_mse + LAMBDA_BC * L_bc

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            params = list(world_model.parameters()) + list(planner.parameters()) + list(pae.parameters())
            for p in params:
                if p.grad is not None and torch.isnan(p.grad).any():
                    raise RuntimeError("NaN in gradients!")
            gnorm = torch.nn.utils.clip_grad_norm_(params, MAX_NORM)
            optimizer.step()

            epoch_mse += L_mse.item()
            epoch_bc  += float(L_bc.item())
            total_gn  += float(gnorm)

        denom = max(1, len(batches))
        mean_mse = epoch_mse/denom
        mean_bc  = epoch_bc/denom
        mean_gn  = total_gn/denom if denom else 0.0
        cur_lr   = optimizer.param_groups[0]["lr"]

        print(f"[{role_name} (phase) Epoch {ep}/{epochs}]  MSE: {mean_mse:.4f}  BC: {mean_bc:.4f}  |grad|: {mean_gn:.3f}")

        if epoch_logger is not None:
            epoch_logger.log({
                "run_id": run_id,
                "role": f"{role_name}-phase",
                "epoch": ep,
                "L_mse": round(mean_mse, 6),
                "L_bc": round(mean_bc, 6),
                "grad_norm": round(mean_gn, 6),
                "lr": cur_lr,
                "n_batches": len(batches),
            })

    save_path = os.path.join(CHECKPOINT_DIR, f"{role_name.lower()}_jepa_phase.pt")
    torch.save(
        {
            "world_model": world_model.state_dict(),
            "planner": planner.state_dict(),
            "phase_action_encoder": pae.state_dict(),  # NEW
        },
        save_path,
    )
    print(f"[SAVE] {role_name} (phase-aware) models saved → {save_path}")

# =============================================================================
# Factorized/Masked heads trainer (TalkHead, VoteHead, KillHead)
# =============================================================================

def _mask_all_true(shape: Tuple[int, int]) -> torch.Tensor:
    return torch.ones(*shape, dtype=torch.bool, device=DEVICE)

def build_talk_mask(batch_size: int, num_cats: int) -> torch.Tensor:
    """Default: allow all talk categories."""
    return _mask_all_true((batch_size, num_cats))

def build_vote_mask_from_aux(batch: List[RolloutSample], num_agents: int) -> torch.Tensor:
    """
    Build per-row vote mask using optional aux:
      aux.alive: list[bool] length=num_agents
      aux.self_idx: int
    If aux missing, allow all.
    """
    B = len(batch)
    mask = _mask_all_true((B, num_agents))
    for i, r in enumerate(batch):
        aux = r.aux or {}
        alive = aux.get("alive", None)
        self_idx = aux.get("self_idx", None)
        if isinstance(alive, list) and len(alive) == num_agents:
            alive_t = torch.tensor(alive, dtype=torch.bool, device=DEVICE)
            mask[i] &= alive_t
        if isinstance(self_idx, int) and 0 <= self_idx < num_agents:
            mask[i, self_idx] = False
    return mask

def build_kill_mask_from_aux(batch: List[RolloutSample], num_agents: int) -> torch.Tensor:
    """
    Build per-row kill mask using optional aux:
      aux.alive: list[bool]
      aux.self_idx: int
      aux.wolves: list[bool]  (team membership)
    Behavior:
      - wolf actor: can kill alive non-wolves (and not self, though self is filtered by non-wolf rule anyway)
      - non-wolf actor: mask all False (no-ops); if aux missing, we conservatively allow all.
    """
    B = len(batch)
    mask = _mask_all_true((B, num_agents))  # default permissive if aux missing
    for i, r in enumerate(batch):
        aux = r.aux or {}
        alive = aux.get("alive", None)
        wolves = aux.get("wolves", None)
        self_idx = aux.get("self_idx", None)

        # If we don't have meta, leave permissive mask (training won't explode).
        if not (isinstance(alive, list) and len(alive) == num_agents):
            continue

        alive_t = torch.tensor(alive, dtype=torch.bool, device=DEVICE)

        if isinstance(wolves, list) and len(wolves) == num_agents and isinstance(self_idx, int):
            wolves_t = torch.tensor(wolves, dtype=torch.bool, device=DEVICE)
            is_wolf_actor = bool(wolves[self_idx])
            if is_wolf_actor:
                # wolves can only kill alive non-wolves
                mask[i] &= alive_t & (~wolves_t)
            else:
                # non-wolves do not issue kill; mask all false
                mask[i, :] = False
        else:
            # Missing team info; at least don't kill dead
            mask[i] &= alive_t
    return mask

def train_jepa_factorized(
    rollout_data_phaseaware: List[Tuple],
    world_model: WorldModelMLP,
    phase_action_encoder: Optional[PhaseActionEncoder],
    planner_factorized: FactorizedPlanner,
    *,
    role_name: str = "agent",
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    run_id: str = "run",
    epoch_logger: Optional[TrainingEpochLogger] = None,
) -> None:
    """
    Full phase-aware + factorized-head trainer.
    Expects per-row choice_type ∈ {"TALK_INTENT","VOTE_TARGET","KILL_TARGET"} (or None).
    Optionally consumes per-row aux dicts to build action masks.
    """
    rows = normalize_rollouts(rollout_data_phaseaware)
    if not rows:
        print(f"[{role_name}] No rollout data; skipping JEPA (factorized) update.")
        return

    pae = phase_action_encoder or PhaseActionEncoder(
        action_dim=ACTION_EMBED_DIM,
        num_agents=NUM_AGENTS_CFG,
        num_talk=NUM_TALK_CATS,
    )

    world_model.to(DEVICE)
    planner_factorized.to(DEVICE)
    pae.to(DEVICE)

    optimizer = optim.Adam(
        list(world_model.parameters()) +
        list(planner_factorized.parameters()) +
        list(pae.parameters()),
        lr=learning_rate,
    )
    mse_loss = nn.MSELoss()
    ce_loss  = nn.CrossEntropyLoss()

    world_model.train()
    planner_factorized.train()
    pae.train()

    for ep in range(1, epochs + 1):
        random.shuffle(rows)
        batches = [rows[i:i+batch_size] for i in range(0, len(rows), batch_size)]

        epoch_mse = 0.0
        epoch_talk = 0.0
        epoch_vote = 0.0
        epoch_kill = 0.0
        total_gn  = 0.0

        for batch in batches:
            B = len(batch)
            z_t_tensor    = torch.stack([r.z_t for r in batch]).to(DEVICE)
            z_next_tensor = torch.stack([r.z_next for r in batch]).to(DEVICE)
            phase = torch.tensor([r.phase if r.phase is not None else 0 for r in batch], device=DEVICE, dtype=torch.long)
            payload = torch.tensor([r.payload_idx if r.payload_idx is not None else 0 for r in batch], device=DEVICE, dtype=torch.long)
            choice_types = [r.choice_type for r in batch]

            # === Action embeddings (learned)
            is_talk_mask = torch.tensor([ct == "TALK_INTENT" for ct in choice_types], device=DEVICE, dtype=torch.bool)
            a_embed = torch.zeros(B, ACTION_EMBED_DIM, device=DEVICE)
            if is_talk_mask.any():
                a_embed[is_talk_mask] = pae(phase_code=phase[is_talk_mask], payload_idx=payload[is_talk_mask], is_talk=True)
            if (~is_talk_mask).any():
                a_embed[~is_talk_mask] = pae(phase_code=phase[~is_talk_mask], payload_idx=payload[~is_talk_mask], is_talk=False)

            # === World model loss (JEPA)
            z_pred = world_model(z_t_tensor, a_embed)
            L_mse = mse_loss(z_pred, z_next_tensor)

            # === Build masks for the factorized heads
            talk_mask = build_talk_mask(B, NUM_TALK_CATS)
            vote_mask = build_vote_mask_from_aux(batch, NUM_AGENTS_CFG)
            kill_mask = build_kill_mask_from_aux(batch, NUM_AGENTS_CFG)

            # === Factorized planner forward
            logits = planner_factorized(
                z_t_tensor,
                talk_mask=talk_mask,
                vote_mask=vote_mask,
                kill_mask=kill_mask
            )

            # === Per-head CE losses (only where that head applies)
            is_vote_mask = torch.tensor([ct == "VOTE_TARGET" for ct in choice_types], device=DEVICE, dtype=torch.bool)
            is_kill_mask = torch.tensor([ct == "KILL_TARGET" for ct in choice_types], device=DEVICE, dtype=torch.bool)

            L_talk = torch.tensor(0.0, device=DEVICE)
            L_vote = torch.tensor(0.0, device=DEVICE)
            L_kill = torch.tensor(0.0, device=DEVICE)

            if is_talk_mask.any():
                L_talk = ce_loss(logits["talk"][is_talk_mask], payload[is_talk_mask])
            if is_vote_mask.any():
                L_vote = ce_loss(logits["vote"][is_vote_mask], payload[is_vote_mask])
            if is_kill_mask.any():
                L_kill = ce_loss(logits["kill"][is_kill_mask], payload[is_kill_mask])

            loss = L_mse + (LAMBDA_TALK * L_talk) + (LAMBDA_BC * (L_vote + L_kill))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            params = list(world_model.parameters()) + list(planner_factorized.parameters()) + list(pae.parameters())
            for p in params:
                if p.grad is not None and torch.isnan(p.grad).any():
                    raise RuntimeError("NaN in gradients!")
            gnorm = torch.nn.utils.clip_grad_norm_(params, MAX_NORM)
            optimizer.step()

            epoch_mse  += float(L_mse.item())
            epoch_talk += float(L_talk.item())
            epoch_vote += float(L_vote.item())
            epoch_kill += float(L_kill.item())
            total_gn   += float(gnorm)

        denom = max(1, len(batches))
        mean_mse  = epoch_mse/denom
        mean_talk = epoch_talk/denom
        mean_vote = epoch_vote/denom
        mean_kill = epoch_kill/denom
        mean_gn   = total_gn/denom if denom else 0.0
        cur_lr    = optimizer.param_groups[0]["lr"]

        print(
            f"[{role_name} (factorized) Epoch {ep}/{epochs}]  "
            f"MSE: {mean_mse:.4f}  TALK: {mean_talk:.4f}  VOTE: {mean_vote:.4f}  KILL: {mean_kill:.4f}  "
            f"|grad|: {mean_gn:.3f}"
        )

        if epoch_logger is not None:
            # Keep CSV compact: push (vote+kill+talk) under L_bc for compatibility
            epoch_logger.log({
                "run_id": run_id,
                "role": f"{role_name}-factorized",
                "epoch": ep,
                "L_mse": round(mean_mse, 6),
                "L_bc": round(mean_talk + mean_vote + mean_kill, 6),
                "grad_norm": round(mean_gn, 6),
                "lr": cur_lr,
                "n_batches": len(batches),
            })

    save_path = os.path.join(CHECKPOINT_DIR, f"{role_name.lower()}_jepa_factorized.pt")
    torch.save(
        {
            "world_model": world_model.state_dict(),
            "phase_action_encoder": pae.state_dict(),
            "factorized_planner": planner_factorized.state_dict(),
        },
        save_path,
    )
    print(f"[SAVE] {role_name} (factorized) models saved → {save_path}")

# =============================================================================
# I/O for checkpoints (uses CFG dims if present)
# =============================================================================

def load_role_models(role: str) -> Tuple[WorldModelMLP, ActionEncoder, PlannerHead]:
    """
    Legacy loader: returns (WM, ActionEncoder, PlannerHead) and loads {role}_jepa.pt if present.
    """
    wm = WorldModelMLP(latent_dim=LATENT_DIM, action_dim=ACTION_EMBED_DIM)
    ae = ActionEncoder(num_actions=NUM_AGENTS_CFG, action_dim=ACTION_EMBED_DIM)
    planner = PlannerHead(latent_dim=LATENT_DIM, num_agents=NUM_AGENTS_CFG)

    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{role.lower()}_jepa.pt")
    if os.path.exists(ckpt_path):
        print(f"[LOAD] {role} checkpoint ← {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu")
        wm.load_state_dict(state["world_model"])
        ae.load_state_dict(state["action_encoder"])
        planner.load_state_dict(state["planner"])
    else:
        print(f"[INIT] No checkpoint for {role}. Starting fresh.")

    return wm, ae, planner

# NEW: phase-aware loader (separate to keep legacy API unchanged)
def load_role_models_phase(role: str) -> Tuple[WorldModelMLP, PhaseActionEncoder, PlannerHead]:
    """
    Phase-aware loader: returns (WM, PhaseActionEncoder, PlannerHead) and loads {role}_jepa_phase.pt if present.
    """
    wm = WorldModelMLP(latent_dim=LATENT_DIM, action_dim=ACTION_EMBED_DIM)
    pae = PhaseActionEncoder(action_dim=ACTION_EMBED_DIM, num_agents=NUM_AGENTS_CFG, num_talk=NUM_TALK_CATS)
    planner = PlannerHead(latent_dim=LATENT_DIM, num_agents=NUM_AGENTS_CFG)

    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{role.lower()}_jepa_phase.pt")
    if os.path.exists(ckpt_path):
        print(f"[LOAD] {role} phase checkpoint ← {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu")
        wm.load_state_dict(state["world_model"])
        if "phase_action_encoder" in state:
            pae.load_state_dict(state["phase_action_encoder"])
        else:
            print(f"[WARN] Phase checkpoint missing 'phase_action_encoder'; starting it fresh.")
        # Planner present for optional BC
        if "planner" in state:
            planner.load_state_dict(state["planner"])
    else:
        print(f"[INIT] No phase checkpoint for {role}. Starting fresh.")

    return wm, pae, planner

# NEW: factorized loader
def load_role_models_factorized(role: str) -> Tuple[WorldModelMLP, PhaseActionEncoder, FactorizedPlanner]:
    """
    Factorized loader: returns (WM, PhaseActionEncoder, FactorizedPlanner) and loads {role}_jepa_factorized.pt if present.
    """
    wm = WorldModelMLP(latent_dim=LATENT_DIM, action_dim=ACTION_EMBED_DIM)
    pae = PhaseActionEncoder(action_dim=ACTION_EMBED_DIM, num_agents=NUM_AGENTS_CFG, num_talk=NUM_TALK_CATS)
    fplanner = FactorizedPlanner(latent_dim=LATENT_DIM, num_agents=NUM_AGENTS_CFG, num_talk_cats=NUM_TALK_CATS)

    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{role.lower()}_jepa_factorized.pt")
    if os.path.exists(ckpt_path):
        print(f"[LOAD] {role} factorized checkpoint ← {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu")
        if "world_model" in state:
            wm.load_state_dict(state["world_model"])
        else:
            print("[WARN] factorized ckpt missing 'world_model'; starting WM fresh.")
        if "phase_action_encoder" in state:
            pae.load_state_dict(state["phase_action_encoder"])
        else:
            print("[WARN] factorized ckpt missing 'phase_action_encoder'; starting PAE fresh.")
        if "factorized_planner" in state:
            fplanner.load_state_dict(state["factorized_planner"])
        else:
            print("[WARN] factorized ckpt missing 'factorized_planner'; starting planner fresh.")
    else:
        print(f"[INIT] No factorized checkpoint for {role}. Starting fresh.")

    return wm, pae, fplanner

# =============================================================================
# Sim runner shim (unchanged)
# =============================================================================

def run_sim_and_collect_rollouts(visual: bool = False):
    """Normalize sim output to (rollouts, meta) so train.py can use meta['agents'].""" 
    from sim import simulate_game
    ret = simulate_game(visual=visual)
    if isinstance(ret, tuple) and len(ret) == 2:
        return ret
    return ret, {}
