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

# Pre-download model weights into the image so cold starts on Fly don't
# pay several GB of egress, and so air-gapped containers can still serve
# traffic. Everything lands under HF_HOME's hub cache (/opt/hf-cache),
# which both NeMo, transformers, and SpeechBrain look at at runtime.
ENV HF_HOME=/opt/hf-cache
RUN python -c "import nemo.collections.asr as nemo_asr; \
    nemo_asr.models.ASRModel.from_pretrained('nvidia/parakeet-tdt-0.6b-v2')"

# Slice 5: bake both emotion-recognition models into the image.
#
# Dimensional model (audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim):
# pulled via huggingface_hub.snapshot_download — the model uses a custom
# Wav2Vec2 regression head subclass at runtime (see vsa.features.emotion),
# so we just need the weights & processor on disk, not loaded into a
# specific architecture.
#
# Categorical model (speechbrain/emotion-recognition-wav2vec2-IEMOCAP):
# pulled via SpeechBrain's foreign_class. SpeechBrain copies the chosen
# files into a savedir under HF_HOME so the runtime classifier picks them
# up without re-downloading. We force LocalStrategy.COPY because the
# default symlink strategy fails on filesystems without symlink support.
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim')"

RUN python -c "import vsa.features.emotion as e; \
    e._patch_speechbrain_lazy_imports(); \
    from speechbrain.inference.interfaces import foreign_class; \
    from speechbrain.utils.fetching import LocalStrategy; \
    foreign_class(source='speechbrain/emotion-recognition-wav2vec2-IEMOCAP', \
        pymodule_file='custom_interface.py', \
        classname='CustomEncoderWav2vec2Classifier', \
        savedir='/opt/hf-cache/hub/speechbrain-iemocap', \
        local_strategy=LocalStrategy.COPY)"

EXPOSE 8080

CMD ["uvicorn", "vsa.api:app", "--host", "0.0.0.0", "--port", "8080"]
