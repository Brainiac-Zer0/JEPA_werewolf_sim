# AgentSim — Predictive Agents for Social Deduction (Werewolf)

Reference implementation for the thesis *"Learning to Deceive and Persuade:
Predictive Agents with Language and Personality Diversity in Social Deduction
Games."* A JEPA belief encoder + world model, a phase-aware factorized planner
(talk / vote / kill heads), a two-stage language mouthpiece (SpeakerBandit +
LogitBiasHead), an LLM-as-judge, and a social-influence latent-correction module,
evaluated in a 9-player Werewolf variant (7 villagers "Worker" + 2 werewolves).

---

## 1. Setup

```bash
pip install -r requirements.txt
```

- Python 3.14, validated on CPU. For GPU training install the CUDA `torch` build.
- First run downloads `sentence-transformers/all-MiniLM-L6-v2` (~90 MB) for the
  message encoder.
- The OpenAI speaker/judge backend additionally needs `pip install openai` and
  `OPENAI_API_KEY`. Offline runs use the HF backend (`LLM_PROVIDER=hf`) or disable
  language entirely (`USE_LANGUAGE=0`).

Roles are configured in `config.yaml` (`WEREWOLF: Werewolf`, `VILLAGER: Worker`).

---

## 2. Component toggles (environment variables)

`sim.py` reads these at import time, so set them before launching (the ladder /
sweep runners do this per condition in subprocesses):

| Variable | Meaning |
|---|---|
| `USE_LANGUAGE` | `1`/`0` — enable the LLM speaker + language pathway |
| `JUDGE_ENABLED` | `1`/`0` — enable the LLM-as-judge (vote re-ranking + rewards) |
| `JUDGE_PROVIDER` | `openai` (needs key) or `hf` (local, offline) |
| `LLM_PROVIDER` | `openai` or `hf` — speaker/judge backend |
| `SOCIAL_ENABLED` | `1`/`0` — social-influence latent correction |
| `AGENTSIM_POLICY` | `""` (planner), `jepa_only`, `heuristic`, `random_voting` |
| `DISCUSS_TURNS` | discussion turns per day (each alive player speaks once/turn) |
| `SEED_GLOBAL` / `GAME_SEED` | seeds for single-game runs |

---

## 3. Quick offline smoke test (CPU, no API key)

```bash
# One headless game + one JEPA training epoch (trains encoder+world model+planner)
USE_LANGUAGE=0 LLM_PROVIDER=hf JUDGE_ENABLED=0 \
python train.py --mode factorized --games_per_cycle 2 --outer_cycles 1 --epochs 2 --speaker 0 --seed 1337

# Mini baseline ladder (Table 1 format) + mini sweep (Table 2 format)
LLM_PROVIDER=hf python run_baseline_ladder.py --games 6 --seeds 1337 --baselines B6_random,B2_jepa_planner,B0_full
LLM_PROVIDER=hf python run_sweeps.py --games 6 --seeds 1337 --only lambda_reg
```

On Windows PowerShell, set env vars with `$env:USE_LANGUAGE="0"` etc., and prefer
`$env:PYTHONUTF8="1"` so console prints don't hit cp1252 errors.

---

## 4. Full reproduction (GPU + OpenAI) — regenerating Table 1 & Table 2

### 4a. Train (thesis scale: 5 cycles × 200 games/role = 1000/role)

```bash
export OPENAI_API_KEY=...            # for the o4-mini speaker/judge
LLM_PROVIDER=openai USE_LANGUAGE=1 JUDGE_ENABLED=1 \
python train.py --mode factorized \
  --outer_cycles 5 --games_per_cycle 200 --epochs 5 \
  --speaker 1 --seed 1337
```

Writes `checkpoints/{werewolf,worker}_jepa_factorized.pt`,
`checkpoints/belief_encoder.pt` (shared encoder), and mouthpieces.

### 4b. Baseline ladder → Table 1 (450 games × 3 seeds)

```bash
LLM_PROVIDER=openai python run_baseline_ladder.py --games 450 --seeds 1337,2718,3141
# -> logs/baselines/baseline_summary.csv  (win rate, vote acc, judge acc, Δz MSE, 95% CIs)
```

Baseline → toggle mapping (`run_baseline_ladder.py`):

| Baseline | policy | language | judge | social |
|---|---|---|---|---|
| B6 Random | random_voting | off | off | off |
| B5 Heuristic | heuristic | off | off | off |
| B4 LLM-Only | random_voting | on | off | off |
| B3 JEPA-Only | jepa_only | off | off | off |
| B2 JEPA+Planner | planner | off | off | off |
| B1 +Social | planner | off | off | on |
| B0 Full | planner | on | on | on |

### 4c. Sweeps → Table 2 (300 games × 2 seeds)

```bash
LLM_PROVIDER=openai python run_sweeps.py --games 300 --seeds 1337,2718
# -> logs/sweeps/sweep_summary.csv
```

---

## 5. What changed in this correctness pass

These fixes were required before results can be trusted (see `CHANGES.md` for the
full list with rationale):

- **Train→eval linkage.** The simulator previously loaded a legacy checkpoint that
  training never produced, so *every evaluation ran untrained networks*. The sim
  now loads the factorized world model + planner and the shared belief encoder.
- **JEPA encoder now trains.** The belief encoder was frozen (detached latents,
  excluded from the optimizer, never checkpointed). It is now trained via a JEPA
  objective with stop-grad next-state targets and shared across agents.
- **Independent games.** Every game was re-seeded to the same constant, making
  repeated games identical; each game now uses a distinct, reproducible seed.
- **Kill head trains.** Ground-truth wolf flags were missing from the training
  mask metadata, so the kill cross-entropy was degenerate; wolves now learn to
  target.
- **REINFORCE baseline, night-kill softmax, discussion turns, talk→vote alignment
  metric, and intent/bias fusion** were corrected to match the thesis.

---

## 6. Caveats

- `run_sweeps.py`'s `lambda_reg` family varies the social-correction scale
  (`SOC_SCALE`); the thesis's λ_reg is a training-time regularizer with no
  eval-time effect. Confirm `SOC_SCALE` is honored by the social module for a
  non-flat sweep, or sweep it as a training parameter instead.
- The `alpha_fusion` sweep requires a working language backend (OpenAI key or a
  real local HF model via `LLM_MODEL_ID`); it is not meaningful with language off.
- `test_roles.py` / `test_agent.py` / `test_llm_script.py` are stale relative to
  the current API (see `CHANGES.md`).
