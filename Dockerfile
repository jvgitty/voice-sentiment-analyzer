FROM python:3.12-slim

# Base image pinned to 3.12: nemo_toolkit[asr] supports 3.10-3.12 cleanly.
# Local dev venv runs 3.13 (NeMo installs there too as of 2.7), but 3.12
# is the safe production target.

WORKDIR /app

# System libs for audio I/O and NeMo's preprocessing:
#   - ffmpeg, libsndfile1: audio decode / WAV I/O.
#   - libsox-fmt-all, sox:  NeMo audio preprocessing.
#   - git: NeMo pulls a couple of git-only requirements at install time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        sox \
        libsox-fmt-all \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src/ ./src/

RUN pip install --no-cache-dir .

# Lazy-download Parakeet at runtime instead of baking it into the image.
#
# Fly Machines cap rootfs at 8 GB uncompressed across every VM size.
# The Python deps (torch + NeMo + faster-whisper) already push the
# image to ~3 GB; adding Parakeet's ~2 GB on top puts us within
# margin of that cap.
#
# The compromise: ship only the dependencies in the image, and let
# the runtime first-request pull Parakeet on demand. This is one-time-
# per-Machine cost: Fly's local disk persists across auto-suspend
# cycles, so the model is only re-fetched if the Machine is destroyed
# and recreated (e.g. on redeploy or region change).
#
# First-request latency after a fresh deploy: ~30-60s while Parakeet
# downloads. Acceptable for the webhook-callback architecture.
#
# HF_HOME stays set so the cache lives at a well-known location,
# surviving suspend/resume on the Machine's local disk.
#
# When the LLM extraction layer lands (Phase 2/3 of the pivot), the
# next Dockerfile revision will either bake the Qwen3.5-9B-Instruct
# GGUF in or lazy-pull it the same way Parakeet does today, depending
# on the final image-size math.
ENV HF_HOME=/opt/hf-cache

EXPOSE 8080

CMD ["uvicorn", "vsa.api:app", "--host", "0.0.0.0", "--port", "8080"]
