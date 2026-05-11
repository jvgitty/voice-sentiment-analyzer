# Voice Note Transcription

Local voice-note transcription service. POST a signed audio URL, get
back a transcript with word-level timestamps. CPU-only, stateless,
scale-to-zero on Fly.io. Audio lives in `/tmp` during processing and is
deleted when the request completes — the service never stores anything.

> **v0.2 status.** This service pivoted in May 2026 from voice
> sentiment analysis (the v0.1.1 archive, retrievable via the
> `v0.1.1-archived-sentiment` git tag) to transcription-only. The next
> phase adds a local LLM extraction layer (tags, entities, tasks,
> summary) using Qwen3.5-9B-Instruct. See [`docs/ROADMAP.md`](docs/ROADMAP.md)
> for the v2 plan.

The same Python core runs three ways: a FastAPI HTTP service (the
production interface), a `vsa analyze` CLI for local debugging, and
`from vsa import Pipeline` for tests and library use.

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

## Deploy to Fly.io in 5 minutes

You'll need a Fly.io account with a payment method on file and **GPU
access enabled** (Fly gates GPU machines behind a billing-verified
flag). Install the `flyctl` CLI from
[fly.io/docs/hands-on/install-flyctl](https://fly.io/docs/hands-on/install-flyctl/).

The shipped `fly.toml` deploys to an **A10 GPU machine** (24 GB VRAM,
~$1.50/hr while running, auto-suspends to ~$0 when idle). GPU is
required: an earlier CPU-only deployment hit the wall on long-audio
transcription regardless of memory tuning. The full rationale is in
the `fly.toml` comment block.

GPU machines are only available in a subset of Fly regions. As of
2026-05, common GPU regions are `iad` (Ashburn), `ord` (Chicago),
`lhr` (London), `nrt` (Tokyo), and `syd` (Sydney). Verify availability
for your chosen region with `fly platform vm-sizes`.

From the repo root:

```bash
# 1. Authenticate (opens a browser).
fly auth login

# 2. Create the app from the shipped fly.toml.
fly launch --no-deploy --name <your-unique-app-name> --region iad

# 3. Set the inbound API key. Generate a random one:
fly secrets set API_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")

# (Optional) override transcription engine. Default is Parakeet.
# fly secrets set TRANSCRIBER_ENGINE=whisper WHISPER_MODEL=small

# 4. Deploy.
fly deploy

# 5. Smoke test:
curl -X POST https://<your-app>.fly.dev/analyze \
  -H "Authorization: Bearer <the-API_KEY-you-just-set>" \
  -H "Content-Type: application/json" \
  -d '{
    "audio_url": "https://<some-public-audio.wav>",
    "callback_url": "https://webhook.site/<your-test-token>",
    "callback_secret": "0123456789abcdef0123456789abcdef",
    "metadata": {"note_id": "smoke-test"},
    "request_id": "smoke-1"
  }'
```

The first request after a deploy or a long idle period pays a cold-start
latency hit (Parakeet downloads on first use, ~30–60s). Subsequent
requests reuse the cached model on the Machine's local disk.

The shipped `fly.toml` is configured for scale-to-zero with auto-suspend
on idle and a hard concurrency limit of 1 in-flight request per Machine
(Fly's proxy wakes additional Machines for concurrent requests, up to 3
total). See the comments in `fly.toml` for tunable knobs.

---

## Run locally

For local development without Fly, use the CLI directly against a local
audio file:

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"

# Print result JSON to stdout.
vsa analyze path/to/recording.wav

# Or write to a file.
vsa analyze path/to/recording.wav --out result.json
```

Or run the same Docker image used in production:

```bash
cp .env.example .env
# Edit .env to set API_KEY to a real value.

docker compose up --build

# In another terminal, serve a sample WAV so the container can fetch it:
cd /path/to/folder/with/sample.wav
python -m http.server 9000

curl -X POST http://localhost:8080/analyze \
  -H "Authorization: Bearer $(grep ^API_KEY= .env | cut -d= -f2)" \
  -H "Content-Type: application/json" \
  -d '{
    "audio_url": "http://host.docker.internal:9000/sample.wav",
    "callback_url": "http://host.docker.internal:9001/cb",
    "callback_secret": "0123456789abcdef0123456789abcdef",
    "metadata": {"note_id": "local-dev"},
    "request_id": "local-1"
  }'
```

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
| `LLM_GGUF_REPO` | `bartowski/Qwen_Qwen3.5-9B-Instruct-GGUF` | HuggingFace repo for the lazy-download fallback. |
| `LLM_GGUF_FILE` | `Qwen_Qwen3.5-9B-Instruct-Q4_K_M.gguf` | GGUF filename inside the repo. |
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

Both models are lazy-pulled rather than baked into the Docker image: the CUDA-enabled PyTorch and llama-cpp-python wheels already account for ~6 GB of image weight, and baking either model on top would push us over Fly's 8 GB rootfs cap. Once downloaded, the model weights persist on the Machine's local disk across auto-suspend/resume cycles, so the download cost is one-time per Machine instance (typically per deploy or per burst-scale event).

---

## Tests

```bash
pip install -e ".[dev]"
pytest
```

Slow tests that touch real model loads (Parakeet, Whisper) live under
`tests/test_transcription.py` and `tests/test_whisper.py`. The rest of
the suite uses fake transcribers and runs in seconds.
