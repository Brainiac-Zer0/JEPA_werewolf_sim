#!/usr/bin/env bash
# AgentSim validation on a GPU box (RunPod / Vast / any CUDA machine) using a LOCAL
# HuggingFace model for speaker + judge — $0 API. A ~7B judge is capable enough to
# emit the rubric JSON; on GPU it's fast. Speaker/judge auto-load on cuda (fp16).
#
#   SETUP=1 bash run_local.sh     # first run on a fresh PyTorch pod (installs deps)
#   RUN_MODE=pilot bash run_local.sh
#   RUN_MODE=full  bash run_local.sh
set -uo pipefail
cd "$(dirname "$0")"

# Optional dependency install (skip torch — the PyTorch pod already has CUDA torch).
if [ "${SETUP:-0}" = "1" ]; then
  pip install -q transformers sentence-transformers pyyaml scipy scikit-learn pandas pygame-ce accelerate
fi

export PYTHONUTF8=1 HF_HUB_DISABLE_TELEMETRY=1 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
# $0-API: local HF models on the GPU for BOTH speaker and judge.
export LLM_PROVIDER=hf JUDGE_PROVIDER=hf USE_LANGUAGE=1 JUDGE_ENABLED=1
export LLM_DEVICE=cuda JUDGE_DEVICE=cuda
# 7B is a capable judge and fits in ~16-24 GB (fp16). Use a 3B model on smaller GPUs.
export LLM_MODEL_ID="${SPEAKER_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
export JUDGE_MODEL_ID="${JUDGE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"

DATA="${DATA_DIR:-$(pwd)}"
export HF_HOME="${HF_HOME:-$DATA/hf_cache}"; mkdir -p "$HF_HOME"

python - <<'PY'
import torch
ok = torch.cuda.is_available()
print(f"[GPU] cuda_available={ok} " + (torch.cuda.get_device_name(0) if ok else "(NO GPU — this will be very slow; use a GPU pod)"))
PY

MODE="${RUN_MODE:-pilot}"
echo "=================================================================="
echo " AgentSim LOCAL run | MODE=$MODE | speaker=$LLM_MODEL_ID | judge=$JUDGE_MODEL_ID | \$0 API"
echo "=================================================================="

set -e
if [ "$MODE" = "full" ]; then
  python train.py --mode factorized --outer_cycles 5 --games_per_cycle 200 --epochs 5 --speaker 1 --seed 1337
  python run_baseline_ladder.py --games 450 --seeds 1337,2718,3141
  python run_sweeps.py --games 300 --seeds 1337,2718
elif [ "$MODE" = "medium" ]; then
  # Resolves B0-vs-B6 across the full ladder at a fraction of full's cost (no sweeps).
  python train.py --mode factorized --outer_cycles 2 --games_per_cycle 40 --epochs 4 --speaker 1 --seed 1337
  python run_baseline_ladder.py --games 100 --seeds 1337,2718,3141
else
  python train.py --mode factorized --outer_cycles 1 --games_per_cycle 20 --epochs 3 --speaker 1 --seed 1337
  python run_baseline_ladder.py --games 20 --seeds 1337 --baselines B6_random,B2_jepa_planner,B0_full
fi
set +e

echo ""
echo "==================== RESULTS (Table 1) ==========================="
cat logs/baselines/baseline_summary.csv 2>/dev/null || echo "(no baseline_summary.csv)"
echo ""
echo "==================== RESULTS (sweeps) ============================"
cat logs/sweeps/sweep_summary.csv 2>/dev/null || echo "(no sweeps in this mode)"
echo "=================================================================="
echo "DONE. Full CSVs under logs/. \$0 spent on API."
