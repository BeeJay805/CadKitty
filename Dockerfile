# CadKitty backend (Fly.io). Wraps the forge_mind pipeline in a FastAPI server.
# Heavy because build123d bundles OpenCASCADE (OCP); the apt libs below are what OCP needs
# to import even headless. Build: `fly deploy` (from repo root) or `docker build .`.
FROM python:3.11-slim

# OpenCASCADE / build123d runtime libs (headless) + OpenMP for numpy/OCP.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 \
      libglu1-mesa \
      libxrender1 \
      libxext6 \
      libsm6 \
      libx11-6 \
      libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# FORGE_BUILD_CACHE enables the content-hash build cache: an identical (prompt + settings) request
# returns the prior result instantly at $0 (no LLM, no sandbox). The key includes the cheatsheet +
# helper-source hashes, so any code change invalidates it -- never stale geometry. On /tmp it is
# per-machine and wiped on redeploy; point it at a Fly volume to make it durable across deploys.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    FORGE_WORK_DIR=/tmp/forge-jobs \
    FORGE_BUILD_CACHE=/tmp/forge-cache

WORKDIR /app

# Only what the package build needs — the rest of the repo is excluded via .dockerignore.
COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install ".[server,fea]"

EXPOSE 8000
# --workers 2: an agentic generation blocks its worker for minutes; a second worker keeps serving
# /health (so Fly doesn't health-kill the machine mid-build) and a second user. Scale workers with
# the VM memory in fly.toml (each worker can spawn a ~2 GB sandbox subprocess).
CMD ["uvicorn", "forge_mind.server.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
