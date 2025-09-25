# train.py ── offline JEPA + optional speaker learning (role-conditioned)
# -----------------------------------------------------------------------------
# JEPA (unchanged):
#   • For each role, simulate N_GAMES, collect rollouts (z_t, a_idx, z_next, role)
#   • Call train_jepa(...) to update world_model, action_encoder, planner
#
# Speaker learning (optional; requires SPEAKER_ENABLED=1 and agents in meta):
#   • After each simulated game, gather agents’ msg_buffer entries
#   • Score messages with LLM-as-Judge (rubric)
#   • Compute role-conditioned rewards (villager vs werewolf) + tiny persona nudge
#   • Run REINFORCE update per agent via agent.speaker.learn_step(...)
# -----------------------------------------------------------------------------

import os
import sys
from typing import List, Tuple, Any

import torch

# Make project root importable when train.py is launched directly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from roles import WEREWOLF, VILLAGER                            # noqa: E402
from training_utils import load_role_models, run_sim_and_collect_rollouts, train_jepa  # noqa: E402
from judge import score_batch, JudgeRubric                      # noqa: E402

# ─────────────────────────────── hyper-params
N_GAMES: int = int(os.environ.get("N_GAMES", "50"))  # per-role
LATENT_DIM: int = 32
ACTION_DIM: int = 8
NUM_ACTIONS: int = 6

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Speaker toggle + judge config
SPEAKER_ENABLED = os.environ.get("SPEAKER_ENABLED", "0") == "1"
JUDGE_RUBRIC_PATH = os.environ.get("JUDGE_RUBRIC", "judge_rubric.yaml")

# Role-conditioned reward weights (tweak as needed)
VILLAGER_W = {"truthfulness": 0.5, "coherence": 0.3, "social_safety": 0.2}
WEREWOLF_W = {"truthfulness": -0.3, "coherence": 0.5, "social_safety": 0.2}


# ─────────────────────────────── helpers: speaker reward / training
def _role_reward(subs: dict, role: str, persona_effects: dict | None = None) -> float:
    """Compute a simple role-conditioned reward from judge subscores + tiny persona nudge."""
    role_l = (role or "").lower()
    is_vill = role_l.startswith("vill") or role_l.startswith("work")
    W = VILLAGER_W if is_vill else WEREWOLF_W

    r = 0.0
    for k, w in W.items():
        r += float(w) * float(subs.get(k, 0.0))

    # Tiny coherence bias from persona (kept subtle so learning dominates)
    if persona_effects:
        r += 0.1 * float(persona_effects.get("coherence_weight_bonus", 0.0)) * float(subs.get("coherence", 0.0))

    # clamp for stability
    return max(-1.0, min(1.0, r))


def _train_speakers_from_agents(agents: List[Any], rubric: JudgeRubric) -> None:
    """
    Pull pending messages from each agent.msg_buffer, score with judge, assign rewards,
    and run a REINFORCE step per agent speaker.
    """
    if not SPEAKER_ENABLED or not agents:
        return

    # Gather messages pending reward
    items, ptrs = [], []  # ptrs keep (agent, idx_in_buffer)
    for ag in agents:
        if not hasattr(ag, "msg_buffer") or not ag.msg_buffer:
            continue
        for i, m in enumerate(ag.msg_buffer):
            if m.get("reward", None) is None:
                items.append({"context": "", "role": ag.role or "Unknown", "candidate": m.get("text", "")})
                ptrs.append((ag, i))

    if not items:
        return

    # Judge scoring (batched)
    results = score_batch(items, rubric)

    # Assign rewards
    for (ag, i), res in zip(ptrs, results):
        subs = res.get("subscores", {}) if isinstance(res, dict) else {}
        persona_effects = getattr(ag, "persona_effects", None)
        R = _role_reward(subs, ag.role or "Unknown", persona_effects)
        ag.msg_buffer[i]["reward"] = float(R)

    # Train each agent's speaker on its batch
    for ag in agents:
        if not getattr(ag, "speaker", None) or not getattr(ag, "speaker_opt", None):
            continue
        batch = [m for m in ag.msg_buffer if m.get("reward") is not None]
        if not batch:
            continue
        stats = ag.speaker.learn_step(batch, ag.speaker_opt, entropy_bonus=0.01, baseline=0.0)
        # clear buffer after update (or keep a sliding window if you prefer)
        ag.msg_buffer.clear()
        print(f"[SPEAKER] {ag.name} loss={stats['loss']:.4f} ent={stats['entropy']:.3f} R={stats['R_mean']:.3f}")


# ─────────────────────────────── rollout collection (unchanged API)
def collect_rollouts_for_role(
    role: str,
    n_games: int,
    rubric: JudgeRubric | None = None,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]]:
    """
    Run `n_games` simulations and grab only the rollout tuples whose *actor*
    has `role == role`. If the simulator returns agents in meta, also trains speakers.
    """
    all_rollouts: list = []

    for _ in range(n_games):
        sim_ret = run_sim_and_collect_rollouts(visual=False)

        # May return (rollouts, meta) or just rollouts
        if isinstance(sim_ret, tuple):
            rollouts, meta = sim_ret
        else:
            rollouts, meta = sim_ret, {}

        # Speaker learning pass (if agents provided by the simulator)
        if SPEAKER_ENABLED and rubric is not None:
            agents = meta.get("agents") if isinstance(meta, dict) else None
            if agents:
                _train_speakers_from_agents(agents, rubric)

        # Filter by role so we train each role’s JEPA on its own data
        all_rollouts.extend(r for r in rollouts if r[3] == role)

    return all_rollouts


# ─────────────────────────────── main training routine
def main() -> None:
    # Prepare rubric once if speaker training is enabled
    rubric = None
    if SPEAKER_ENABLED:
        try:
            rubric = JudgeRubric.load(JUDGE_RUBRIC_PATH)
            print(f"[SPEAKER] Loaded judge rubric: {JUDGE_RUBRIC_PATH}")
        except Exception as e:
            print(f"[SPEAKER] WARNING: failed to load rubric ({e}); speaker learning will be skipped.")
            rubric = None

    for role_name in (WEREWOLF, VILLAGER):
        # 1) Load / init models for this role
        world_model, action_encoder, planner = load_role_models(role_name)

        # 2) Simulate games and collect data (+ optional speaker training)
        print(f"[JEPA] Simulating {N_GAMES} games for role: {role_name}")
        role_rollouts = collect_rollouts_for_role(role_name, N_GAMES, rubric=rubric)
        print(f"[JEPA] Collected {len(role_rollouts)} roll-outs for role {role_name}")

        # 3) Train JEPA modules for this role
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
