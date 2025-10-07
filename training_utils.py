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
# NEW: import learnable phase-aware encoder
from encoders import PhaseActionEncoder  # noqa: E402

CHECKPOINT_DIR = "checkpoints"
LOGS_DIR = "logs"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load config
with open("config.yaml", "r") as f:
    CFG = yaml.safe_load(f) or {}

# Tunables
LAMBDA_BC: float = float(CFG.get("LAMBDA_BC", 0.5))
MAX_NORM: float = float(CFG.get("MAX_NORM", 1.0))

# Optional dims (used by helpers below; keep legacy defaults)
PHASE_COUNT: int = int(CFG.get("PHASE_COUNT", 3))            # DISCUSS/VOTE/NIGHT
NUM_TALK_CATS: int = int(CFG.get("NUM_TALK_CATS", 5))
NUM_AGENTS_CFG: int = int(CFG.get("NUM_AGENTS", 6))
ACTION_EMBED_DIM: int = int(CFG.get("ACTION_EMBED_DIM", 8))
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

def normalize_rollouts(raw: List[Tuple]) -> List[RolloutSample]:
    """
    Accepts:
      - legacy: (z_t, a_idx, z_next, role)
      - phase-aware: (z_t, phase_code, action_payload, z_next, role[, choice_type])
    """
    out: List[RolloutSample] = []
    for tup in raw:
        if len(tup) == 4:
            z_t, a_idx, z_next, role = tup
            out.append(RolloutSample(z_t=z_t, a_idx=a_idx, z_next=z_next, role=role))
        elif len(tup) >= 5:
            z_t, phase, payload, z_next, role = tup[:5]
            ct = tup[5] if len(tup) >= 6 else None
            out.append(RolloutSample(
                z_t=z_t, a_idx=None, z_next=z_next, role=role,
                phase=int(phase), payload_idx=int(payload), choice_type=ct
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
