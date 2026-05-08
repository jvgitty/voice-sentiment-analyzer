FROM python:3.12-slim

# Slice 3 pins the base image to 3.12: nemo_toolkit[asr] supports 3.10–3.12
# cleanly. The local dev venv runs 3.13 (NeMo installs there too as of 2.7),
# but 3.12 is the safe production target.

WORKDIR /app

# System libs needed by audio/ML deps:
#   - ffmpeg, libsndfile1: librosa / soundfile audio backends.
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
# composites.yaml is the editable spec for confidence/engagement/calmness
# formulas (Slice 6). Pipeline.__init__ loads it at request time via
# CompositeScorer.from_yaml; without it in the image, every request 500s
# at pipeline construction.
COPY composites.yaml ./

RUN pip install --no-cache-dir .

# Lazy-download models at runtime instead of baking them into the image.
#
# The PRD originally called for baking model weights to keep cold starts
# fast. We tried, but Fly Machines cap rootfs at 8 GB uncompressed across
# every VM size, and the three baked models (~4 GB together) push the
# image over that limit ("Not enough space to unpack image").
#
# The compromise: ship only the dependencies in the image (already ~3 GB
# from torch/NeMo/transformers/speechbrain), and let the runtime first-
# request pull each model on demand. This is one-time-per-Machine cost:
# Fly's local disk persists across auto-suspend cycles, so models are
# only re-fetched if the Machine is destroyed and recreated (e.g. on
# redeploy or region change).
#
# First-request latency after a fresh deploy: roughly 3-5 minutes total
# while Parakeet (~2 GB), audeering wav2vec2 (~1 GB), and SpeechBrain
# IEMOCAP (~1 GB) download in sequence on demand. Acceptable for the
# webhook-callback architecture (the caller is fire-and-forget anyway).
#
# HF_HOME stays set so all three libraries cache to the same well-known
# location, surviving suspend/resume on the Machine's local disk.
ENV HF_HOME=/opt/hf-cache

EXPOSE 8080

CMD ["uvicorn", "vsa.api:app", "--host", "0.0.0.0", "--port", "8080"]
