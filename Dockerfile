FROM python:3.12-slim

# Base image pinned to 3.12: nemo_toolkit[asr] supports 3.10-3.12 cleanly.
# Local dev venv runs 3.13 (NeMo installs there too as of 2.7), but 3.12
# is the safe production target.
#
# GPU support note: we use a plain python:3.12-slim base rather than an
# nvidia/cuda:* image. PyTorch's CUDA wheels ship their own bundled CUDA
# runtime libs and llama-cpp-python's CUDA wheels do the same, so we
# don't need the full nvidia/cuda base image (~2 GB of extra layers).
# Fly's GPU machines pass through the NVIDIA driver from the host; the
# wheels' bundled userspace libs talk to that driver directly.

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

# Install CUDA-enabled PyTorch BEFORE the rest of the dependencies.
# pip's resolver will see torch already satisfied with the CUDA build
# and won't try to pull the default (CPU-only) wheel from PyPI when
# `pip install .` runs the project's dependency graph next.
#
# CUDA 12.1 chosen for compatibility with Fly's A10 GPU driver (which
# supports CUDA 12.x). Bumping to a newer CUDA index URL is a one-line
# change here when Fly's driver line catches up.
RUN pip install --no-cache-dir torch \
    --index-url https://download.pytorch.org/whl/cu121

# Install CUDA-enabled llama-cpp-python from abetlen's prebuilt wheel
# index. This is critical — the default PyPI wheel is CPU-only and
# silently no-ops the n_gpu_layers parameter, leaving Qwen running on
# CPU even though a GPU is available.
RUN pip install --no-cache-dir llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

COPY pyproject.toml ./
COPY src/ ./src/

RUN pip install --no-cache-dir .

# HF cache location, used by huggingface-hub for both Parakeet (lazy-
# loaded by NeMo at first /analyze) and Qwen (lazy-loaded by our
# extractor's _resolve_gguf_path helper).
ENV HF_HOME=/opt/hf-cache

# We do NOT bake Parakeet into the image on the GPU build. Image size
# math: python base + torch CUDA wheel + llama-cpp CUDA wheel + NeMo
# deps already lands around 6-7 GB, and Fly's 8 GB rootfs cap doesn't
# leave room for the +2 GB Parakeet bake the CPU build does.
#
# First-request cost for downloading Parakeet on a fresh Machine is
# ~30-60 seconds at HuggingFace's typical bandwidth — negligible
# compared to the multi-minute CPU times we used to fight, and the
# download caches under HF_HOME for the lifetime of the Machine.

# Lazy-pull config for Qwen3.5-9B-Instruct GGUF. Same env vars as the
# CPU build; _resolve_gguf_path in vsa.extraction.llm uses them.
ENV LLM_MODEL_PATH=/opt/models/qwen3.5-9b-instruct-q4_k_m.gguf
ENV LLM_GGUF_REPO=Smoffyy/Qwen3.5-9B-Instruct-Pure-GGUF
ENV LLM_GGUF_FILE=Qwen3.5-9B-Q4_K_M.gguf

# Offload all model layers to GPU. -1 = use as many as fit in VRAM,
# which on the A10's 24 GB is all of them for a Q4_K_M 9B model.
# An operator who wants partial CPU offload (e.g. to free VRAM for
# concurrent requests in the future) can override this env var.
ENV LLM_N_GPU_LAYERS=-1

# Pre-create the /opt/models directory so an operator (or a future
# follow-up) can drop a GGUF file there. Empty by default; the
# lazy-pull path takes over when the file isn't found.
RUN mkdir -p /opt/models

EXPOSE 8080

CMD ["uvicorn", "vsa.api:app", "--host", "0.0.0.0", "--port", "8080"]
