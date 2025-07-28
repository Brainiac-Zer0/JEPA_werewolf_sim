# train.py ── offline JEPA role-conditioned training loop
# ---------------------------------------------------------------------------
# 1. For each role (Werewolf / Villager) we
#       a) load (or freshly create) that role’s JEPA sub-modules
#       b) run N_GAMES simulations collecting only that role’s roll-outs
#       c) call train_jepa(...) to update the three sub-modules
#
# 2. The simulation itself is delegated to training_utils.run_sim_and_collect_rollouts
#    which returns either  (rollouts, meta_dict)  or just rollouts.
#
# 3. Each rollout tuple is:
#       ( z_t , a_t_idx , z_next , role_name )
#
# 4. Models are **saved by train_jepa**; we do not manage checkpoints here.
# ---------------------------------------------------------------------------

import os
import sys
import random                         # noqa: F401  (kept for future sampling needs)
from typing import List, Tuple        # type hints

import torch

# Make project root importable when train.py is launched directly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from roles            import WEREWOLF, VILLAGER                     # noqa: E402
from training_utils   import load_role_models, run_sim_and_collect_rollouts, train_jepa  # noqa: E402


# ─────────────────────────────── hyper-params
N_GAMES: int = 50                    # number of simulated games per role
LATENT_DIM: int = 32                 # kept for reference; not used directly here
ACTION_DIM: int = 8
NUM_ACTIONS: int = 6

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ─────────────────────────────── helpers
def collect_rollouts_for_role(
    role: str,
    n_games: int,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]]:
    """
    Run `n_games` simulations and grab only the rollout tuples whose
    *actor* has `role == role`.
    """
    all_rollouts: list = []

    for _ in range(n_games):
        sim_ret = run_sim_and_collect_rollouts(visual=False)

        # run_sim_and_collect_rollouts may return (rollouts, meta) or rollouts
        rollouts = sim_ret[0] if isinstance(sim_ret, tuple) else sim_ret

        # filter by role so we train each role’s JEPA on its own data
        all_rollouts.extend(r for r in rollouts if r[3] == role)

    return all_rollouts


# ─────────────────────────────── main training routine
def main() -> None:
    for role_name in (WEREWOLF, VILLAGER):
        # 1) Load / init models for this role
        world_model, action_encoder, planner = load_role_models(role_name)

        # 2) Simulate games and collect data
        print(f"[JEPA] Simulating {N_GAMES} games for role: {role_name}")
        role_rollouts = collect_rollouts_for_role(role_name, N_GAMES)
        print(f"[JEPA] Collected {len(role_rollouts)} roll-outs for role {role_name}")

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


# ─────────────────────────────── CLI entry
if __name__ == "__main__":
    main()
