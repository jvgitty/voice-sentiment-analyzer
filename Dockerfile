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

# Bake Parakeet TDT 0.6B into the image at build time (~2 GB).
#
# v0.1.1's lazy-load-on-first-request behavior turned into an operational
# cliff: every fresh Machine paid a ~30-60s cold-start hit while the
# weights pulled from HuggingFace. Baking eliminates that cliff for the
# transcription stage entirely — Parakeet is on disk the instant uvicorn
# binds.
#
# We bake by downloading via huggingface-hub (cheap, no model load) rather
# than calling NeMo's from_pretrained (which would spin up the full ASR
# loader and burn 30s of build time on every layer rebuild).
#
# HF_HOME at /opt/hf-cache is where huggingface-hub caches. NeMo / faster-
# whisper / huggingface-hub all honor this path, so subsequent loads at
# runtime hit the baked cache rather than re-downloading.
ENV HF_HOME=/opt/hf-cache

RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('nvidia/parakeet-tdt-0.6b-v2')"

# Qwen3.5-9B-Instruct GGUF (~5.5 GB) is NOT baked.
#
# Fly's rootfs cap is 8 GiB uncompressed. Baking both Parakeet (~2 GB) and
# Qwen (~5.5 GB) on top of our ~2 GB of Python deps puts us at ~9.5 GB,
# over the cap. We bake the smaller and 100%-of-requests-needed model
# (Parakeet) and let the larger one lazy-pull on first /analyze.
#
# The lazy-pull path is implemented in vsa.extraction.llm._resolve_gguf_path:
# if LLM_MODEL_PATH points at an existing file, use it; otherwise download
# from LLM_GGUF_REPO / LLM_GGUF_FILE via huggingface_hub.hf_hub_download
# (which caches under HF_HOME, so subsequent loads on the same Machine
# skip the network call).
#
# Operators who want both baked (e.g. running on a non-Fly host with a
# bigger rootfs) can set:
#   RUN python -c "from huggingface_hub import hf_hub_download; \
#       hf_hub_download(\
#           repo_id='bartowski/Qwen_Qwen3.5-9B-Instruct-GGUF', \
#           filename='Qwen_Qwen3.5-9B-Instruct-Q4_K_M.gguf')"
# Image size will grow by ~5.5 GB.
ENV LLM_MODEL_PATH=/opt/models/qwen3.5-9b-instruct-q4_k_m.gguf
ENV LLM_GGUF_REPO=bartowski/Qwen_Qwen3.5-9B-Instruct-GGUF
ENV LLM_GGUF_FILE=Qwen_Qwen3.5-9B-Instruct-Q4_K_M.gguf

# Pre-create the /opt/models directory so an operator (or a future
# Phase-3 follow-up) can drop a GGUF file there without futzing with
# permissions. Empty by default; the lazy-pull path takes over when
# the file isn't found.
RUN mkdir -p /opt/models

EXPOSE 8080

CMD ["uvicorn", "vsa.api:app", "--host", "0.0.0.0", "--port", "8080"]
