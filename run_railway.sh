#!/usr/bin/env bash
# AgentSim validation entrypoint for Railway (CPU + OpenAI).
# Persists checkpoints/logs/HF-cache to the mounted Volume and prints the results
# table to the deploy logs (the reliable way to read results on Railway).
set -uo pipefail
cd /app

export PYTHONUTF8=1 HF_HUB_DISABLE_TELEMETRY=1 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
export LLM_PROVIDER=openai USE_LANGUAGE=1 JUDGE_ENABLED=1
# Speaker + judge both default to the cheap non-reasoning model (override via
# SPEAKER_MODEL / JUDGE_MODEL env if you want strict o4-mini fidelity).
export LLM_MODEL_ID="${SPEAKER_MODEL:-gpt-4o-mini}"
export JUDGE_MODEL_ID="${JUDGE_MODEL:-gpt-4o-mini}"

# ---- fail fast on missing key ----
if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "[FATAL] OPENAI_API_KEY is not set. Add it in Railway → Variables, then redeploy."
  exit 1
fi

# ---- persistence: redirect checkpoints/logs/HF-cache to the Volume ----
DATA="${DATA_DIR:-/data}"
mkdir -p "$DATA/checkpoints" "$DATA/logs" "$DATA/hf_cache"
export HF_HOME="$DATA/hf_cache"
rm -rf checkpoints logs
ln -s "$DATA/checkpoints" checkpoints
ln -s "$DATA/logs" logs

MODE="${RUN_MODE:-pilot}"
echo "=================================================================="
echo " AgentSim run | MODE=$MODE | speaker=$LLM_MODEL_ID judge=$JUDGE_MODEL_ID"
echo "=================================================================="

# --retrain = retrain-per-condition: each rung trains its OWN model (namespaced
# checkpoint dir) with exactly the components it is evaluated with, then is scored
# against that model. B0/B4 train with language on → the judge/speaker (OpenAI) are
# exercised during TRAINING too, so the planner actually sees the social+LLM inputs.
set -e
if [ "$MODE" = "full" ]; then
  # Thesis scale (expensive). ~thousands of games with judge/language; hours-to-days.
  python run_baseline_ladder.py --retrain \
      --train-games 200 --train-cycles 5 --train-epochs 5 \
      --games 450 --seeds 1337,2718,3141
  # Sweeps probe the FULL trained system's sensitivity → point them at ck_full
  # (retrain-per-condition writes there, not to the default checkpoints/).
  CHECKPOINT_DIR=checkpoints_ablation/ck_full python run_sweeps.py --games 300 --seeds 1337,2718
elif [ "$MODE" = "publish" ]; then
  # Publication run, cost- AND memory-optimized. B0-B1 is already significant at
  # medium training, so we do NOT retrain the LLM models at full scale:
  #   * free variants (ck_base/ck_social): 40x8x6 — MORE total training than medium
  #     to firm up planner (B2-B6) and social (B1-B2), but games_per_cycle stays at
  #     40, the memory-safe peak. (200/cycle OOM-killed the container: train.py's
  #     collect_rollouts_for_role holds a whole cycle of rollouts in RAM.) $0 API.
  #   * language variants (ck_llm/ck_full): kept at medium (40x2x4) — the costly
  #     OpenAI-in-the-loop training, held down because B0-B1 is already P=1.00.
  # Eval FREE rungs at 450x3 (tight CIs, $0); LANGUAGE rungs B0/B4 at 150x3 (already
  # significant — saves API + wall-clock). Then persona/social sweeps (Table 2).
  python run_baseline_ladder.py --retrain \
      --train-games 40 --train-cycles 8 --train-epochs 6 \
      --lang-train-games 40 --lang-train-cycles 2 --lang-train-epochs 4 \
      --games 450 --lang-games 150 --seeds 1337,2718,3141
  # Sweeps probe the FULL trained system's sensitivity → point them at ck_full
  # (retrain-per-condition writes there, not to the default checkpoints/).
  CHECKPOINT_DIR=checkpoints_ablation/ck_full python run_sweeps.py --games 300 --seeds 1337,2718
elif [ "$MODE" = "medium" ]; then
  # Statistically-meaningful but affordable: resolves the contrasts (win-rate CI
  # ~±0.09) across the full 7-rung ladder. No sweeps (Table 2) — run those in full.
  python run_baseline_ladder.py --retrain \
      --train-games 40 --train-cycles 2 --train-epochs 4 \
      --games 100 --seeds 1337,2718,3141
else
  # Cheap pilot: confirm B0 (full) separates from the baselines before scaling up.
  python run_baseline_ladder.py --retrain \
      --train-games 20 --train-cycles 1 --train-epochs 3 \
      --games 20 --seeds 1337 --baselines B6_random,B2_jepa_planner,B0_full
fi
python assess_ladder.py || true
set +e

echo ""
echo "==================== RESULTS (Table 1) ==========================="
cat "$DATA/logs/baselines/baseline_summary.csv" 2>/dev/null || echo "(no baseline_summary.csv)"
echo ""
echo "==================== RESULTS (sweeps) ============================"
cat "$DATA/logs/sweeps/sweep_summary.csv" 2>/dev/null || echo "(no sweeps in this mode)"
echo "=================================================================="
echo "DONE. Full CSVs persisted under the Volume ($DATA/logs/…)."

# Keep the container alive only if you want to exec in / re-read the volume.
if [ "${KEEP_ALIVE:-0}" = "1" ]; then
  echo "KEEP_ALIVE=1 → sleeping so the volume stays mounted for retrieval."
  sleep infinity
fi
