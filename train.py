# train.py
import sys
import os
import random
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from roles import ROLE_PRIORS

# Training hyperparameters
LATENT_DIM = 32
ACTION_DIM = 8
NUM_ACTIONS = 6
LEARNING_RATE = 1e-3
TRAIN_EPOCHS = 10
BATCH_SIZE = 16
KL_WEIGHT = 0.1
ENTROPY_WEIGHT = 0.05

def compute_kl_divergence(z_pred, z_prior):
    return torch.mean((z_pred - z_prior).pow(2))

def compute_entropy(z):
    return z.var(dim=0).mean()

def compute_expected_free_energy(z_pred, z_goal, beta=ENTROPY_WEIGHT):
    prediction_error = torch.mean((z_pred - z_goal) ** 2, dim=1)
    entropy = compute_entropy(z_pred)
    efe = prediction_error + beta * entropy
    return efe.mean()

def train_jepa(rollout_data, world_model, action_encoder):
    if not rollout_data:
        return

    optimizer = optim.Adam(
        list(world_model.parameters()) + list(action_encoder.parameters()),
        lr=LEARNING_RATE
    )
    criterion = nn.MSELoss()

    for epoch in range(TRAIN_EPOCHS):
        total_loss = 0.0
        random.shuffle(rollout_data)

        for i in range(0, len(rollout_data), BATCH_SIZE):
            batch = rollout_data[i:i + BATCH_SIZE]

            z_batch = torch.stack([triplet[0] for triplet in batch])
            a_batch = torch.stack([triplet[1] for triplet in batch]).long().squeeze()
            z_next_batch = torch.stack([triplet[2] for triplet in batch])
            roles = [triplet[3] for triplet in batch]

            a_embed = action_encoder(a_batch)
            z_pred = world_model(z_batch, a_embed)

            mse_loss = criterion(z_pred, z_next_batch)
            prior_z = z_batch.detach()
            kl_loss = compute_kl_divergence(z_pred, prior_z)
            z_goal = torch.stack([ROLE_PRIORS[r] for r in roles])
            efe_loss = compute_expected_free_energy(z_pred, z_goal=z_goal)

            total_batch_loss = mse_loss + KL_WEIGHT * kl_loss + efe_loss

            optimizer.zero_grad()
            total_batch_loss.backward()
            optimizer.step()

            total_loss += total_batch_loss.item()

        avg_loss = total_loss / max(1, len(rollout_data) // BATCH_SIZE)
        print(f"[JEPA TRAINING] Epoch {epoch + 1}/{TRAIN_EPOCHS}, Avg Loss: {avg_loss:.4f}")

def main():
    from sim import run_sim_and_collect_rollouts

    print("[SIM] Running simulation and collecting rollout data...")
    rollout_data, _ = run_sim_and_collect_rollouts()

    from encoders import WorldModelMLP, ActionEncoder
    print("[JEPA] Initializing models...")
    world_model = WorldModelMLP(latent_dim=LATENT_DIM, action_dim=ACTION_DIM)
    action_encoder = ActionEncoder(num_actions=NUM_ACTIONS, action_dim=ACTION_DIM)

    print("[JEPA] Starting training...")
    train_jepa(rollout_data, world_model, action_encoder)

if __name__ == "__main__":
    main()
