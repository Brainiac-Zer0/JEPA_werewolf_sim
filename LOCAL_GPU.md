# $0-API validation on a rented GPU (RunPod / Vast.ai)

Run the language pipeline with a **local** HuggingFace model for speaker + judge —
**no OpenAI cost**. A ~7B judge is capable enough to emit the rubric JSON, and on a
GPU it's fast. You pay only for GPU time (~$0.30–0.50/hr), so a pilot is ~$2–10.

> Not on Railway — Railway has no GPU. Use RunPod or Vast.ai. For the OpenAI path
> instead, see `RAILWAY.md`.

## Hardware
- **GPU: 24 GB** (e.g. RTX 4090 / A5000 / L4) comfortably runs `Qwen2.5-7B-Instruct`
  in fp16 (~15 GB). On a **16 GB** card use a 3B model: `SPEAKER_MODEL=Qwen/Qwen2.5-3B-Instruct JUDGE_MODEL=Qwen/Qwen2.5-3B-Instruct`.
- Disk: ~30 GB (model + checkpoints).

## Path A — RunPod/Vast PyTorch pod (simplest)
1. Launch a pod from a **PyTorch** template (CUDA torch preinstalled), 24 GB GPU.
2. In the pod terminal:
   ```bash
   git clone -b rework-for-publication <your-repo-url> agentsim && cd agentsim
   SETUP=1 RUN_MODE=pilot bash run_local.sh        # installs deps + runs the pilot
   ```
3. Watch the logs. You should see `[GPU] cuda_available=True …` and
   `[INFO] LLM loaded on CUDA: …`. Results (Table 1) print at the end and land in `logs/`.
4. If the pilot looks right (B0 beats B6, social/persona effects show), run the full:
   ```bash
   RUN_MODE=full bash run_local.sh
   ```

## Path B — containerized (Dockerfile.gpu)
On any CUDA host with the NVIDIA container runtime:
```bash
docker build -f Dockerfile.gpu -t agentsim-gpu .
docker run --gpus all -e RUN_MODE=pilot -v $PWD/out:/app/logs agentsim-gpu
```

## Env knobs (both paths)
| Var | Default | Meaning |
|---|---|---|
| `RUN_MODE` | `pilot` | `pilot` (cheap sanity) or `full` (thesis scale) |
| `SPEAKER_MODEL` / `JUDGE_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | local HF model ids |
| `SETUP` | `0` | `1` = pip-install deps first (fresh pod) |

## Notes
- The JEPA nets + MiniLM auto-use the GPU (`DEVICE=auto`); the LLM auto-loads on
  `cuda:0` in fp16 and prints its device. If it ever loads on CPU, set
  `LLM_DEVICE=cuda` / `JUDGE_DEVICE=cuda` (the run script already does).
- This produces a **real** end-to-end result (unlike a CPU-small model), because a
  7B judge can actually score the rubric. It still won't match the thesis's `o4-mini`
  numbers exactly — different judge — but it's a valid, publishable-quality local run.
- **Destroy the pod when done** so it stops billing.
