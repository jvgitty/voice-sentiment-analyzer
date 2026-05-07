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

RUN pip install --no-cache-dir .

# Pre-download the Parakeet TDT 0.6B v2 weights into the image so cold
# starts on Fly don't pay 2GB of egress, and so air-gapped containers can
# still serve traffic. The model lands in HF_HOME's hub cache; we point
# HF_HOME at /opt/hf-cache so it survives across layers and is the same
# path NeMo will look at at runtime.
ENV HF_HOME=/opt/hf-cache
RUN python -c "import nemo.collections.asr as nemo_asr; \
    nemo_asr.models.ASRModel.from_pretrained('nvidia/parakeet-tdt-0.6b-v2')"

EXPOSE 8080

CMD ["uvicorn", "vsa.api:app", "--host", "0.0.0.0", "--port", "8080"]
