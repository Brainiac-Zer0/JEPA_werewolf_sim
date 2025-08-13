# train.py ── offline JEPA role-conditioned training loop
# ---------------------------------------------------------------------------
# For each role (Werewolf / Villager):
#   a) load (or freshly create) that role’s JEPA sub‑modules
#   b) run N_GAMES simulations collecting only that role’s roll‑outs
#   c) print dataset acceptance stats (mean ||Δz|| and (1−cos))
#   d) call train_jepa(...) to update the three sub‑modules
# Models are saved by train_jepa.
# ---------------------------------------------------------------------------

import os
import sys
import argparse
import random
from typing import List, Tuple

import torch
import torch.nn.functional as F

# Make project root importable when train.py is launched directly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from roles import WEREWOLF, VILLAGER  # noqa: E402
from training_utils import load_role_models, run_sim_and_collect_rollouts, train_jepa  # noqa: E402

# ─────────────────────────────── defaults (CLI can override)
DEFAULT_N_GAMES: int = 50
MIN_ROLLOUTS: int = 64  # guardrail: skip training if we have less than this


def collect_rollouts_for_role(
    role: str,
    n_games: int,
    use_language_env: str | None = None,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]]:
    """
    Run `n_games` simulations and grab only the rollout tuples whose *actor* has `role == role`.
    Optionally set USE_LANGUAGE env for the sim subprocess context (applies within-process too).
    """
    if use_language_env is not None:
        os.environ["USE_LANGUAGE"] = use_language_env

    all_rollouts: list = []

    for _ in range(n_games):
        sim_ret = run_sim_and_collect_rollouts(visual=False)
        rollouts = sim_ret[0] if isinstance(sim_ret, tuple) else sim_ret
        all_rollouts.extend(r for r in rollouts if r[3] == role)

    # one global shuffle
    random.shuffle(all_rollouts)
    return all_rollouts


def dataset_acceptance_stats(
    rollouts: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]]
) -> tuple[float, float]:
    """Compute mean L2 ||z_next − z_t|| and mean (1 − cosine) across the dataset."""
    if not rollouts:
        return 0.0, 0.0
    l2s, one_minus_cos = [], []
    for z_t, _a_idx, z_next, _role in rollouts:
        dz = z_next - z_t
        l2s.append(torch.norm(dz).item())
        cos = F.cosine_similarity(z_next.unsqueeze(0), z_t.unsqueeze(0)).item()
        one_minus_cos.append(1.0 - cos)
    return sum(l2s) / len(l2s), sum(one_minus_cos) / len(one_minus_cos)


def main() -> None:
    parser = argparse.ArgumentParser(description="JEPA role-conditioned training")
    parser.add_argument("--games", type=int, default=DEFAULT_N_GAMES, help="simulated games per role")
    parser.add_argument(
        "--language",
        type=int,
        choices=[0, 1],
        default=None,
        help="override USE_LANGUAGE (1=on, 0=off) during data collection only",
    )
    args = parser.parse_args()

    lang_override = None if args.language is None else ("1" if args.language == 1 else "0")

    for role_name in (WEREWOLF, VILLAGER):
        # 1) Load / init models for this role
        world_model, action_encoder, planner = load_role_models(role_name)

        # 2) Simulate games and collect data
        print(f"[JEPA] Simulating {args.games} games for role: {role_name}")
        role_rollouts = collect_rollouts_for_role(role_name, args.games, use_language_env=lang_override)
        print(f"[JEPA] Collected {len(role_rollouts)} roll-outs for role {role_name}")

        if len(role_rollouts) < MIN_ROLLOUTS:
            print(f"[JEPA][WARN] Only {len(role_rollouts)} roll-outs for {role_name} (<{MIN_ROLLOUTS}); skipping training.")
            continue

        # 2.5) Acceptance stats on dataset
        mean_l2, mean_1mcos = dataset_acceptance_stats(role_rollouts)
        print(f"[JEPA][DATA] {role_name}: mean ||Δz|| = {mean_l2:.4f} ; mean (1−cos) = {mean_1mcos:.4f}")

        # 3) Train
        print(f"[JEPA] Training JEPA modules for role: {role_name}")
        train_jepa(
            rollout_data=role_rollouts,
            world_model=world_model,
            action_encoder=action_encoder,
            planner=planner,
            role_name=role_name,
        )

    print("\n[JEPA] All roles trained and checkpoints updated.")


if __name__ == "__main__":
    main()
