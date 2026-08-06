# AgentSim — Railway/container image (CPU). Runs the OpenAI-backed validation.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    # headless: sim imports pygame; no display in a container
    SDL_VIDEODRIVER=dummy \
    SDL_AUDIODRIVER=dummy

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# CPU torch wheels + pinned deps + the OpenAI backend. If a pin is unavailable for
# this Python, relax it in requirements.txt — the code is not sensitive to exact
# minor versions (see RAILWAY.md).
RUN pip install --upgrade pip \
    && pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt openai

COPY . .
RUN chmod +x run_railway.sh

CMD ["bash", "run_railway.sh"]
