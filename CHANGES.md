# Correctness & Reproducibility Changes

This pass fixes bugs found while preparing the codebase for publication and
re-running the experiments. Items are grouped by severity. File references point
to the current tree.

## HIGH — results-invalidating

### Train → eval were disconnected (the sim ran untrained networks)
Training saves `checkpoints/{role}_jepa_factorized.pt`, but `sim.py` loaded the
legacy `load_role_models()` which looks for `{role}_jepa.pt` — a file the
factorized trainer never writes. Every agent therefore started from random
weights at evaluation, and the belief encoder was created fresh per agent.
**Fix:** `sim.py` now loads `load_role_models_factorized()` (world model +
phase-action encoder + factorized planner) and a single shared belief encoder
(`load_shared_belief_encoder()`) for all agents.

### JEPA belief encoder was never trained
Latents were `.detach()`-ed at collection, the encoder was excluded from the
optimizer, and it was never checkpointed — so the central JEPA claim ("gradients
reach the context encoder") was false. **Fix:** rollouts now carry the raw
observation (`aux["x_t"]`, `aux["x_next"]`); the factorized trainer re-encodes
them so `z_t = enc(x_t)` (grad) and the target `z_next = enc(x_next).detach()`
(stop-grad, preventing collapse). The encoder is added to the optimizer, shared
across agents/roles, and saved to `checkpoints/belief_encoder.pt`.

### Repeated games were identical
`simulate_game()` reset the RNG to a constant `SEED_GLOBAL` every game and role
assignment used a fixed seed, so "N games" collapsed to one trajectory repeated
N times. **Fix:** each game takes a distinct, reproducible seed
(`train.py:_game_seed_for`), threaded through `simulate_game(seed=…)` and
`assign_roles(seed=…)`; per-row telemetry logs the actual per-game seed.

### Kill head never trained
`make_aux` derives the wolf mask from `agent.is_wolf`, but `assign_roles` only
set `agent.role`, leaving `is_wolf=False` for everyone. The reconstructed kill
mask then marked the wolf a non-wolf actor and masked every target, so the kill
cross-entropy was `inf`/`NaN`. **Fix:** `assign_roles` sets `agent.is_wolf`;
the trainer also filters any row whose target logit is non-finite. Kill loss now
trains (kill accuracy > 0).

## MED — behavior/metric mismatches with the thesis

- **REINFORCE baseline** (`speaker.py`, `train.py`): the bandit used a hardcoded
  `baseline=0.0`. Added an EMA reward baseline (`R − b`) as the thesis specifies.
- **Night-kill softmax** (`world.py`, `sim.py`): removed the injected
  `temperature=0.7`, the >50 % majority short-circuit, and the fixed `Random(0)`
  tie-break. Now samples `p_i ∝ exp(c_i/max_c)` with the seeded global RNG.
- **Discussion turns** (`sim.py`): `discuss_turns` was ignored (one pass). Now
  each alive player speaks once per turn for `DISCUSS_TURNS` turns.
- **Talk→vote alignment metric** (`sim.py`): was a constant lookup on the talk
  category. Now measures real consistency — did the agent vote for the target it
  accused/proposed (NaN for undirected utterances, excluded from the mean).
- **Intent/bias fusion** (`speaker_llm.py`): combined softmax *probabilities*;
  now combines in logit space `b = α·b_planner + (1−α)·b_bias` then softmax, per
  the thesis.

## LOW — hygiene

- **Social role-affinity keys** (`social.py`): were `"Villager"/"Werewolf"` while
  the configured villager role is `"Worker"`, so every lookup missed. Now keyed by
  the configured role names.
- **Offline judge** (`sim.py`): `_can_use_judge()` required `OPENAI_API_KEY` even
  for a local HF judge; now only the OpenAI backend requires a key.
- **Speaker/judge provider split** (`llm_script.py`): the speaker ignored config
  and defaulted to HF while the judge read config (OpenAI). The speaker now falls
  back to the config provider.
- **Windows console**: set `PYTHONUTF8=1` to avoid cp1252 errors on the `▶`/`Δ`
  prints (documented in the README).

## New reproducibility infrastructure

- `run_baseline_ladder.py` — runs the B0..B6 ladder (subprocess per baseline,
  toggle mapping documented) and writes Table 1 with 95 % bootstrap CIs.
- `run_sweeps.py` — recreates the S1/S2 sweep harness (the original was never
  committed) with the existing `logs/sweeps/**` output schema.
- `simulate_game` now returns a per-game `outcome` dict (winner, vote accuracy,
  judge acceptance, talk→vote alignment) so runners need no CSV parsing.
- `requirements.txt`, `README.md`.

## Post-review improvements (make promised components actually function)

A component-level review found several thesis claims that were present but inert.
Fixed in phases (each offline-validated, with regression tests in test_phase1.py /
test_phase2.py):

- **Phase 1a — FEP planning (RQ2).** `talk_entropy_w`/`talk_kl_uniform_w` were
  declared but never consumed. Now added to the factorized planner loss (entropy
  penalty + KL-to-uniform prior); verified the penalty lowers planner talk entropy.
- **Phase 1b — Personality steering (RQ5).** Personas affected only phrasing.
  Added a persona→talk-intent bias in the planner (`ENABLE_PERSONA_STEER`); verified
  a high-accuse persona measurably raises P(accuse).
- **Phase 2 — Social influence (RQ6).** The eval-time module was a fresh, untrained,
  latent-only correction that never changed a vote (even at 5×‖z‖). Rebuilt so it is:
  (a) **trained** — wolf-supervised so δ shifts villager votes toward the actual
  wolves; (b) **message-aware** (§3.9) — a message-embedding projection feeds δ
  (content signal activates with language on); (c) **relatively scaled** — ‖δ‖ is a
  fraction of ‖z_self‖ (robust to the encoder's norm; a fixed magnitude was ~2.6% and
  inert); (d) **checkpointed/loaded** as a shared instance (mirrors the encoder).
  Verified offline: social now changes 37% of games and **improves villager vote
  accuracy (+3 pts, B1 > B2)**. Key subtlety fixed: training δ inside the main BC
  loss taught it to be inert (reproduce the unshifted vote); δ is now trained only by
  the wolf-supervision signal.

## Known-stale (not yet fixed)
- `test_roles.py` imports `DEFECTIVE/WORKER` (pre-rename API); `test_agent.py`
  calls `encode_current_belief()` with no args; `test_llm_script.py` requires the
  OpenAI backend. These predate the current API.
