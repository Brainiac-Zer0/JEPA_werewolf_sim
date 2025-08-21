# training_utils.py ── utility helpers for JEPA training & checkpoint I/O
from __future__ import annotations

import os
import random
import csv
from datetime import datetime
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from encoders import ActionEncoder, PlannerHead, WorldModelMLP

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── tunables
LAMBDA_BC: float = 0.75
MAX_NORM: float = 1.0
PREDICT_DELTA: bool = True  # when True, train world model on Δz = z_next - z_t

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
METRICS_CSV = os.path.join(LOG_DIR, "metrics.csv")

def _append_metrics_row(row: dict):
    header = [
        "ts", "role", "epoch", "epochs",
        "mse", "bc",
        "learning_rate", "lambda_bc", "batch_size", "dataset_size",
        "predict_delta",
    ]
    file_exists = os.path.exists(METRICS_CSV)
    with open(METRICS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            w.writeheader()
        w.writerow(row)

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
) -> None:
    world_model.to(DEVICE)
    action_encoder.to(DEVICE)
    planner.to(DEVICE)

    optimizer = optim.Adam(
        list(world_model.parameters())
        + list(action_encoder.parameters())
        + list(planner.parameters()),
        lr=learning_rate,
    )
    mse_loss = nn.MSELoss()
    ce_loss  = nn.CrossEntropyLoss()

    world_model.train(), action_encoder.train(), planner.train()

    dataset_size = len(rollout_data)

    for ep in range(1, epochs + 1):
        random.shuffle(rollout_data)

        batches = [
            rollout_data[i : i + batch_size]
            for i in range(0, len(rollout_data), batch_size)
        ]

        epoch_mse = 0.0
        epoch_bc  = 0.0
        for batch in batches:
            z_t, a_idx, z_next = zip(*[(r[0], r[1], r[2]) for r in batch])

            z_t_tensor     = torch.stack(z_t).to(DEVICE)                 # [B, latent]
            a_idx_tensor   = torch.stack(a_idx).long().squeeze().to(DEVICE)  # [B]
            z_next_tensor  = torch.stack(z_next).to(DEVICE)               # [B, latent]

            # forward
            a_embed = action_encoder(a_idx_tensor)                        # [B, a_dim]
            z_pred  = world_model(z_t_tensor, a_embed)                    # [B, latent]
            logits  = planner(z_t_tensor)                                 # [B, num_agents]

            # Δ-mode target vs absolute
            target = (z_next_tensor - z_t_tensor) if PREDICT_DELTA else z_next_tensor
            L_mse  = mse_loss(z_pred, target)
            L_bc   = ce_loss(logits, a_idx_tensor)
            loss   = L_mse + LAMBDA_BC * L_bc

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # NaN guard + grad clipping
            for p in list(world_model.parameters()) + list(action_encoder.parameters()) + list(planner.parameters()):
                if p.grad is not None and torch.isnan(p.grad).any():
                    raise RuntimeError("NaN in gradients!")
            torch.nn.utils.clip_grad_norm_(
                list(world_model.parameters()) + list(action_encoder.parameters()) + list(planner.parameters()),
                MAX_NORM
            )
            optimizer.step()

            epoch_mse += L_mse.item()
            epoch_bc  += L_bc.item()

        denom = max(1, len(batches))
        mode = "Δ-mode" if PREDICT_DELTA else "abs"
        mean_mse = epoch_mse / denom
        mean_bc  = epoch_bc  / denom
        print(f"[{role_name}  Epoch {ep}/{epochs}  ({mode})]  MSE: {mean_mse:.4f}  BC: {mean_bc:.4f}")

        _append_metrics_row({
            "ts": datetime.now().isoformat(timespec="seconds"),
            "role": role_name,
            "epoch": ep,
            "epochs": epochs,
            "mse": f"{mean_mse:.6f}",
            "bc": f"{mean_bc:.6f}",
            "learning_rate": learning_rate,
            "lambda_bc": LAMBDA_BC,
            "batch_size": batch_size,
            "dataset_size": dataset_size,
            "predict_delta": int(PREDICT_DELTA),
        })

    # save
    save_path = os.path.join(CHECKPOINT_DIR, f"{role_name.lower()}_jepa.pt")
    torch.save(
        {
            "world_model": world_model.state_dict(),
            "action_encoder": action_encoder.state_dict(),
            "planner": planner.state_dict(),
            "predict_delta": PREDICT_DELTA,
        },
        save_path,
    )
    print(f"[SAVE] {role_name} models saved → {save_path}")

def load_role_models(role: str) -> Tuple[WorldModelMLP, ActionEncoder, PlannerHead]:
    wm = WorldModelMLP(latent_dim=32, action_dim=8)
    ae = ActionEncoder(num_actions=6, action_dim=8)
    planner = PlannerHead(latent_dim=32, num_agents=6)

    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{role.lower()}_jepa.pt")
    if os.path.exists(ckpt_path):
        print(f"[LOAD] {role} checkpoint ← {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        wm.load_state_dict(state["world_model"])
        ae.load_state_dict(state["action_encoder"])
        planner.load_state_dict(state["planner"])
    else:
        print(f"[INIT] No checkpoint for {role}. Starting fresh.")

    return wm, ae, planner

def run_sim_and_collect_rollouts(visual: bool = False):
    from sim import simulate_game
    return simulate_game(visual=visual)
