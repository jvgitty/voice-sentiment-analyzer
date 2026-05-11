"""Modal deployment for voice-note-transcription.

Wraps the FastAPI app in a Modal serverless function with GPU support.
The rest of the codebase (Pipeline, transcriber, extractor, schema)
is unchanged — Modal is purely the deployment runtime.

Why Modal
---------
The original Fly deployment hit a hard wall: shared-CPU machines can't
hold Parakeet's chunked-inference memory growth on long audio, and
Fly's GPU offering isn't available to new customers as of 2026-05.
Modal solves both: a real GPU on a per-second pricing model that
auto-scales to zero when idle.

Why T4 specifically
-------------------
T4 has 16 GB VRAM, which is the smallest GPU that comfortably fits
both Parakeet (~1.5 GB fp16 weights + ~3 GB per-chunk activations)
and Qwen3.5-9B-Instruct Q4_K_M (~5.5 GB weights + ~1.5 GB KV cache).
At ~$0.59/hr while running and ~$0 idle, a typical 100-req/day
workload costs roughly $15-45/month — mostly absorbed by Modal
Starter's $30/month free credits during early customer beta.

Deploying
---------
Prerequisites (one-time):
  1. Sign up at modal.com (free, no credit card needed).
  2. ``pip install modal``
  3. ``modal token new`` to authenticate.
  4. Create the secret holding API_KEY (and any other env-var
     overrides):
       modal secret create voice-note-transcription-secrets API_KEY=<value>

Deploy:
  modal deploy modal_app.py

Modal builds the GPU image (~10 min the first time as the CUDA torch
wheel is ~3 GB), provisions a container behind an HTTPS endpoint, and
prints the URL. Subsequent deploys reuse the image layers and are
fast.

HIPAA / BAA note
----------------
Modal Starter (the free tier this file targets) does NOT carry a BAA.
For HIPAA workloads (signing actual hospitals or law firms as
customers), upgrade the Modal organization to Enterprise — the BAA is
part of that tier. The code in this file does not change between
Starter and Enterprise; only the org-level plan does.
"""

from __future__ import annotations

import modal

APP_NAME = "voice-note-transcription"

# ---------------------------------------------------------------------------
# Image definition
# ---------------------------------------------------------------------------
# Base: NVIDIA's CUDA 12.1 runtime image with cuDNN 8.
#
# Why not debian_slim: an earlier attempt used debian_slim plus the
# CUDA-enabled torch wheel. That worked for torch (its CUDA wheel
# bundles its own runtime libs) but llama-cpp-python's CUDA build
# uses ctypes.dlopen to find ``libcudart.so.12``, which checks
# standard system paths (LD_LIBRARY_PATH + /usr/lib + /usr/local/cuda),
# not torch's bundled libs. The deploy succeeded but the extractor
# crashed on first inference with:
#
#   Failed to load shared library 'libllama.so':
#     libcudart.so.12: cannot open shared object file
#
# Switching to ``nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04``
# puts libcudart in /usr/local/cuda/lib64 where llama-cpp-python's
# loader finds it. cuDNN is included for completeness; some torch
# ops depend on it transitively.
#
# Order matters: CUDA-enabled torch and llama-cpp-python must be
# installed BEFORE the project's PyPI deps so pip's resolver sees
# them already satisfied and doesn't pull the CPU-only PyPI wheels.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04",
        add_python="3.12",
    )
    # System libs for audio I/O and NeMo's preprocessing path.
    .apt_install(
        "ffmpeg",
        "libsndfile1",
        "sox",
        "libsox-fmt-all",
        "git",
    )
    # CUDA-enabled PyTorch. The CUDA runtime is bundled inside the
    # wheel so we don't need an nvidia/cuda base image. CUDA 12.1
    # chosen for compatibility with Modal's T4 driver.
    .pip_install(
        "torch",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    # CUDA-enabled llama-cpp-python from abetlen's prebuilt wheel
    # index. Critical: the default PyPI wheel is CPU-only and
    # silently no-ops the ``n_gpu_layers`` arg even when a GPU is
    # present, leaving Qwen running on CPU.
    .pip_install(
        "llama-cpp-python>=0.2.90",
        extra_index_url="https://abetlen.github.io/llama-cpp-python/whl/cu121",
    )
    # Project deps mirrored from pyproject.toml. Listed explicitly
    # here (rather than via pip_install_from_pyproject) so the image
    # definition is self-documenting and doesn't depend on Modal's
    # pyproject parsing API staying stable.
    .pip_install(
        "fastapi>=0.110",
        "uvicorn[standard]>=0.27",
        "pydantic>=2.5",
        "httpx>=0.27",
        "numpy>=1.24",
        "scipy>=1.11",
        "typer>=0.12",
        "nemo_toolkit[asr]>=2.0",
        "faster-whisper>=1.0",
        "huggingface-hub>=0.24",
    )
    # Default env vars. Operators can override any of these via the
    # Modal secret attached to the function below; values here are
    # the production defaults. NOTE: must come BEFORE add_local_dir
    # — Modal rejects any build step after a local-file step unless
    # we opt in to copy=True (which forces a slow image rebuild on
    # every source edit).
    .env(
        {
            "HF_HOME": "/opt/hf-cache",
            "LLM_MODEL_PATH": "/opt/models/qwen3.5-9b-instruct-q4_k_m.gguf",
            "LLM_GGUF_REPO": "Smoffyy/Qwen3.5-9B-Instruct-Pure-GGUF",
            "LLM_GGUF_FILE": "Qwen3.5-9B-q4_k_m.gguf",
            "LLM_N_GPU_LAYERS": "-1",
        }
    )
    # Our source code. The ``vsa`` package lives at ``src/vsa/`` in
    # the repo. We use ``add_local_dir`` (rather than
    # ``add_local_python_source``) so Modal copies the files directly
    # without needing the package to be pip-installed in the local
    # Python — the latter would require ``pip install -e .`` against
    # the full dep tree (NeMo + torch + llama-cpp-python) just to
    # deploy, which is painful on Windows / non-CUDA hosts.
    #
    # Modal containers run from ``/root`` and have it on ``sys.path``
    # by default, so copying to ``/root/vsa`` makes ``import vsa``
    # work inside the container with no extra PYTHONPATH wiring.
    #
    # MUST be the last step in the image build — Modal optimizes
    # local-file additions to happen on container start rather than
    # at image-build time, which means no further build steps can
    # follow.
    .add_local_dir("src/vsa", remote_path="/root/vsa")
)

# ---------------------------------------------------------------------------
# Persistent storage for the HuggingFace cache
# ---------------------------------------------------------------------------
# Without this volume, every cold-started container re-downloads
# Parakeet (~2 GB) and Qwen3.5-9B Q4_K_M (~5.5 GB) from HuggingFace.
# That's ~3 min of wasted cold-start cost per container restart.
#
# With this volume mounted at /opt/hf-cache (matching the HF_HOME env
# var above), the model files persist across container restarts and
# across deploys. Containers cold-started after the first download
# skip the network entirely — they only pay model-load-into-GPU time.
hf_cache_volume = modal.Volume.from_name(
    "voice-note-hf-cache",
    create_if_missing=True,
)

# ---------------------------------------------------------------------------
# Modal App
# ---------------------------------------------------------------------------
app = modal.App(APP_NAME, image=image)


@app.function(
    # T4 GPU: 16 GB VRAM. Cheapest realistic option that fits both
    # models with headroom. See module docstring for the math.
    gpu="T4",
    # Generous per-request timeout. 20-min audio + first-time model
    # downloads can take several minutes; 15 min gives margin without
    # masking real hangs (Modal kills the container at this limit).
    timeout=900,
    # API key + any other env-var overrides. Create the secret with:
    #   modal secret create voice-note-transcription-secrets API_KEY=...
    # Additional values inside the same secret (e.g. LLM_GGUF_REPO
    # for a different uploader) override the image's .env() defaults.
    secrets=[
        modal.Secret.from_name("voice-note-transcription-secrets"),
    ],
    # Persistent HF cache mount. Same path as the HF_HOME env var.
    volumes={"/opt/hf-cache": hf_cache_volume},
    # Cap how many parallel containers Modal spins up under burst
    # load. Matches the Fly pattern of "burst to 3" — three
    # concurrent voice-note submissions don't queue, they fan out to
    # three GPU containers.
    max_containers=3,
)
@modal.asgi_app()
def web():
    """ASGI entrypoint for the FastAPI app.

    Modal wraps the returned FastAPI instance in an HTTPS endpoint
    and routes incoming requests to it. All actual request handling,
    auth, audio fetching, pipeline orchestration, and webhook
    callback logic lives in :mod:`vsa.api` — completely unchanged
    from the Fly deployment.

    The first request after a fresh container start pays:
      * Container cold-start: ~10-30s
      * Parakeet download (if not yet cached on the volume): ~30-60s
      * Qwen GGUF download (if not yet cached on the volume): ~1-2 min
      * Model load onto GPU: ~10-20s
      * Actual inference: ~30-60s

    Total first-request: ~3-5 min on a fresh deploy, ~30-60s after
    the volume is warmed.
    """
    from vsa.api import app as fastapi_app

    return fastapi_app
