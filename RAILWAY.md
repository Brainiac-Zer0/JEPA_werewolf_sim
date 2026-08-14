# Running the AgentSim validation on Railway

Railway is **CPU-only** (no GPU), so this runs the **OpenAI-backed** path
(`gpt-4o-mini` speaker + judge). It won't run a local LLM judge — for a $0-API run
you need a GPU host (RunPod/Vast) instead.

The models here are tiny; the only real cost is the **OpenAI API**. Railway compute
for a CPU job is a few dollars at most.

## Files (already in the repo)
- `Dockerfile` — CPU image with pinned deps + `openai`.
- `railway.json` — Dockerfile build, `restartPolicyType: NEVER` (batch job, don't loop).
- `run_railway.sh` — entrypoint: redirects checkpoints/logs/HF-cache to the Volume,
  runs train → ladder (→ sweeps in `full`), and **prints the results table to the logs**.
- `.dockerignore` — keeps the image small.

## Deploy (once)
1. Push the `rework-for-publication` branch to a GitHub repo you own.
2. Railway → **New Project → Deploy from GitHub repo** → pick the repo/branch.
   Railway auto-detects the `Dockerfile`.
3. **Variables** (Settings → Variables):
   - `OPENAI_API_KEY` = your key  *(required; the run exits fast without it)*
   - `RUN_MODE` = `pilot` (default), `medium` (resolves B0-vs-B6 affordably), or `full`
   - optional: `SPEAKER_MODEL` / `JUDGE_MODEL` (default `gpt-4o-mini`),
     `KEEP_ALIVE=1` (keep the container up after finishing so you can exec in).
4. **Volume** (required so results/checkpoints survive): add a Volume mounted at
   **`/data`**. The run writes `checkpoints/`, `logs/`, and the HF cache there.
   (If you mount elsewhere, set `DATA_DIR` to that path.)
5. Deploy. Watch **Deploy Logs**.

## Getting results
- The **Table 1 / sweep summaries print to the deploy logs** at the end — that's the
  simplest way to read them on Railway.
- Full per-game CSVs persist on the Volume under `/data/logs/…`. To pull them, set
  `KEEP_ALIVE=1` and use Railway's shell (`railway run` / service shell) to `cat` or
  copy them out, or add a step that uploads them somewhere you control.

## Cost / time notes
- `pilot` = 1 training cycle (20 games) + a 3-rung ladder (20 games). Enough to see
  whether **B0 (full) separates from B6 (random)**. ~a few $ of API, runs in well
  under an hour of API-bound wall-clock.
- `full` = thesis scale (5×200 training, 450×3 ladder, 300×2 sweeps). Hundreds of $
  of API and many hours — only run after the pilot looks right.
- Do the **pilot first**. If B0 clearly beats B6 and social/persona effects appear,
  switch `RUN_MODE=full`.

## If the image build fails on a dependency pin
The pins in `requirements.txt` were captured on Python 3.14; on the image's Python a
specific `torch`/`transformers` pin may be unavailable. Relax that one line (drop the
`==x.y.z`) and redeploy — the code isn't sensitive to exact minor versions.
