# Voice Note Transcription

Local voice-note transcription + LLM extraction service. POST a signed
audio URL, get back a transcript with word-level timestamps PLUS a
structured JSON extraction (title, summary, type, tags, entities,
tasks). GPU-accelerated, stateless, auto-scales to zero. Audio lives
in `/tmp` during processing and is deleted when the request completes
— the service never stores anything.

> **v0.2 status.** The May 2026 pivot from voice sentiment analysis
> (archived under `v0.1.1-archived-sentiment` tag) is complete. The
> service now runs on Modal's T4 GPU with Parakeet TDT 0.6B for
> transcription and Qwen3.5-9B-Instruct for structured extraction.
> See [`docs/ROADMAP.md`](docs/ROADMAP.md) for what's planned for v2
> (multi-tenant client config, BAA paths, model tiers).

The same Python core runs two ways: a FastAPI HTTP service deployed
on Modal (the production interface) and a `vsa analyze` CLI for
local one-shot use.

---

## Sample request and response

```bash
curl -X POST https://your-app.fly.dev/analyze \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "audio_url": "https://example-storage.com/signed/voice-note.wav",
    "callback_url": "https://your-edge-function.example.com/webhook",
    "callback_secret": "0123456789abcdef0123456789abcdef",
    "metadata": {"note_id": "abc-123", "user_id": "u-42"},
    "request_id": "req-2026-05-10-0001"
  }'
```

The HTTP response (and webhook body) is shaped like this:

```json
{
  "schema_version": "2.0",
  "audio": {"duration_seconds": 92.4, "sample_rate": 16000, "channels": 1},
  "transcription": {
    "engine": "parakeet-tdt-0.6b-v2",
    "language": "en",
    "text": "so today I want to talk about ...",
    "words": [
      {"w": "so", "start": 0.12, "end": 0.31, "conf": 0.98},
      "..."
    ]
  },
  "processing": {
    "started_at": "2026-05-10T18:14:02Z",
    "completed_at": "2026-05-10T18:15:47Z",
    "library_versions": {"vsa": "0.2.0"},
    "errors": []
  }
}
```

The webhook fires `POST <callback_url>` with the same JSON wrapped in
`{request_id, status, metadata, result}`, signed with
`X-Signature-256: sha256=<HMAC-SHA256 of the body using callback_secret>`.

When transcription returns no text (silent or non-speech audio, hard
model failure), the callback uses `status: "failed"` with an `error`
field instead of `status: "completed"`. Downstream consumers can take
a single failure code path on this status.

---

## Deploy to Modal in 5 minutes

The shipped `modal_app.py` deploys to a **T4 GPU container** on Modal
(~$0.59/hr while running, $0 when idle, auto-scales to zero). The
[Modal Starter plan](https://modal.com/pricing) is free with $30/month
in compute credits — that covers tens of requests/day during early
customer beta.

**Prerequisites (one-time):**

```bash
pip install modal
modal token new        # opens browser to authenticate

# Create the secret that holds API_KEY (and any other env-var overrides).
# Generate a random key:
modal secret create voice-note-transcription-secrets \
    API_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
```

**Deploy:**

```bash
modal deploy modal_app.py
```

First build takes ~10 minutes (the CUDA torch wheel is ~3 GB).
Subsequent deploys reuse cached image layers and finish in seconds.
Modal prints an HTTPS URL when deployment is complete — that's your
service endpoint.

**Smoke test:**

```bash
curl -X POST https://<your-modal-app-url>.modal.run/analyze \
  -H "Authorization: Bearer <the-API_KEY-you-set>" \
  -H "Content-Type: application/json" \
  -d '{
    "audio_url": "https://<some-public-audio.wav>",
    "callback_url": "https://webhook.site/<your-test-token>",
    "callback_secret": "0123456789abcdef0123456789abcdef",
    "metadata": {"note_id": "smoke-test"},
    "request_id": "smoke-1"
  }'
```

The first request after a fresh deploy pays a one-time download
cost (~3 min) while Parakeet and Qwen3.5-9B fetch from HuggingFace
onto Modal's persistent volume. Subsequent requests reuse the cached
weights and complete in ~30–60s including GPU inference.

### HIPAA / BAA path

Modal Starter does **not** carry a BAA. For HIPAA workloads (signing
hospitals or law firms as actual customers), upgrade your Modal
organization to Enterprise — the BAA is part of that tier. **The code
in this repo does not change** between Starter and Enterprise; only
the org-level plan does.

---

## Run locally (CLI)

For one-shot local use against an audio file on disk (no HTTP server,
no GPU needed for the CLI path itself since the LLM extractor's
`LLM_N_GPU_LAYERS=0` falls back to CPU when no GPU is present):

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"

# Print result JSON to stdout.
vsa analyze path/to/recording.wav

# Or write to a file.
vsa analyze path/to/recording.wav --out result.json
```

For production-equivalent HTTP API testing, deploy to Modal — that's
the supported path.

---

## Long audio: how chunking works

Voice notes up to ~20 minutes are supported. The Parakeet TDT 0.6B
encoder uses self-attention whose memory is quadratic in sequence
length, which on CPU caps reliable single-pass transcription at roughly
60–90 seconds of audio. For longer files the transcriber automatically
splits the input into non-overlapping chunks of at most
`PARAKEET_CHUNK_SECONDS` (default 60s), runs each chunk through
`model.transcribe()` sequentially, and merges the resulting text and
word-level timestamps. Memory is forcibly released between chunks via
`gc.collect()` + `malloc_trim` (glibc) and arena fragmentation is
minimized via `MALLOC_ARENA_MAX=2`.

Short audio (≤ chunk size) takes the original single-pass code path
with zero overhead.

---

## Configuration

| Env var | Default | Purpose |
|---------|---------|---------|
| `API_KEY` | — (required) | Bearer token for the `/analyze` endpoint |
| `TRANSCRIBER_ENGINE` | `parakeet` | `parakeet` or `whisper` |
| `WHISPER_MODEL` | `small` | Whisper model size when `TRANSCRIBER_ENGINE=whisper` |
| `PARAKEET_CHUNK_SECONDS` | `60` | Chunk length for long-audio transcription |
| `MAX_AUDIO_BYTES` | `52428800` (50 MB) | Reject oversized audio uploads early |
| `HF_HOME` | `/opt/hf-cache` | Model cache location (Parakeet + lazy-pulled GGUFs) |
| `LLM_MODEL_PATH` | `/opt/models/qwen3.5-9b-instruct-q4_k_m.gguf` | Local GGUF path. If the file exists, used directly; otherwise the lazy-download path below kicks in. |
| `LLM_GGUF_REPO` | `Smoffyy/Qwen3.5-9B-Instruct-Pure-GGUF` | HuggingFace repo for the lazy-download fallback. |
| `LLM_GGUF_FILE` | `Qwen3.5-9B-q4_k_m.gguf` | GGUF filename inside the repo (note: lowercase — Smoffyy's convention). |
| `LLM_CONTEXT_SIZE` | `8192` | LLM context window in tokens. |
| `LLM_THREADS` | `0` (auto) | n_threads for CPU inference. |
| `LLM_TEMPERATURE` | `0.2` | Sampling temperature. Low for consistent extraction. |
| `LLM_MAX_OUTPUT_TOKENS` | `2048` | Cap on JSON output length. |
| `LLM_N_GPU_LAYERS` | `-1` (all) | LLM layers offloaded to GPU. Set to `0` for CPU-only inference. Requires the CUDA build of llama-cpp-python (shipped in our Docker image). |

## Where the models live

| Model | Source | Loaded |
|-------|--------|--------|
| **Parakeet TDT 0.6B** (transcription) | Lazy-downloaded from HuggingFace on first request (~2 GB). | First request after a fresh Machine pays a ~30–60s download. NeMo then loads it onto the GPU. Subsequent requests reuse the cache. |
| **Qwen3.5-9B-Instruct Q4_K_M** (extraction) | Lazy-downloaded from HuggingFace on first request (~5.5 GB). | First request after a fresh Machine pays a ~1–2 min download. llama-cpp-python then loads it onto the GPU. Subsequent requests reuse the cache. |

Both models are lazy-pulled rather than baked into the Modal image. The CUDA-enabled PyTorch and llama-cpp-python wheels plus the NVIDIA CUDA base image already account for ~7 GB of image weight, and baking the models on top would slow every deploy without buying anything (Modal Volumes give us the same per-Machine persistence story with less image-rebuild cost). Once downloaded, the model weights persist on the `voice-note-hf-cache` Modal Volume across container restarts and deploys.

---

## Tests

```bash
pip install -e ".[dev]"
pytest
```

Slow tests that touch real model loads (Parakeet, Whisper) live under
`tests/test_transcription.py` and `tests/test_whisper.py`. The rest of
the suite uses fake transcribers and runs in seconds.
