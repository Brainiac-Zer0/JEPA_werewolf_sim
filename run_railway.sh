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

set -e
if [ "$MODE" = "full" ]; then
  # Thesis scale (expensive). ~thousands of games with judge/language.
  python train.py --mode factorized --outer_cycles 5 --games_per_cycle 200 --epochs 5 --speaker 1 --seed 1337
  python run_baseline_ladder.py --games 450 --seeds 1337,2718,3141
  python run_sweeps.py --games 300 --seeds 1337,2718
else
  # Cheap pilot: confirm B0 (full) separates from the baselines before the full run.
  python train.py --mode factorized --outer_cycles 1 --games_per_cycle 20 --epochs 3 --speaker 1 --seed 1337
  python run_baseline_ladder.py --games 20 --seeds 1337 --baselines B6_random,B2_jepa_planner,B0_full
fi
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
