# Voice Sentiment Analyzer

Open-source voice sentiment analysis service. POST a signed audio URL,
get back a JSON describing *how the speaker sounded* — pitch, jitter,
shimmer, prosody, dimensional emotion (arousal / valence / dominance),
and three composite scores (`confidence`, `engagement`, `calmness`)
each with a transparent `_components` breakdown plus a time-windowed
view of how the metrics shift inside a single recording. CPU-only,
stateless, scale-to-zero on Fly.io. Audio lives in `/tmp` during
analysis and is deleted when the request completes — the analyzer
never stores anything.

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
    "callback_url": "https://your-edge-function.example.com/vsa-webhook",
    "callback_secret": "0123456789abcdef0123456789abcdef",
    "metadata": {"note_id": "abc-123", "user_id": "u-42"},
    "request_id": "req-2026-05-08-0001"
  }'
```

The HTTP response (and webhook body) is shaped like this — abridged for
readability:

```json
{
  "schema_version": "1.0",
  "audio": {"duration_seconds": 92.4, "sample_rate": 16000, "channels": 1},
  "transcription": {
    "engine": "parakeet-tdt-0.6b-v2",
    "language": "en",
    "text": "so today I want to talk about ...",
    "words": [{"w": "so", "start": 0.12, "end": 0.31, "conf": 0.98}, "..."]
  },
  "acoustic": {
    "pitch":   {"mean_hz": 142.3, "std_hz": 28.1, "range_hz": 117.4, "...": "..."},
    "loudness": {"mean_db": -22.1, "std_db": 4.6, "rms_mean": 0.071},
    "voice_quality": {"jitter_local": 0.012, "shimmer_local": 0.058, "hnr_db": 14.2, "voiced_unvoiced_ratio": 0.71},
    "spectral": {"centroid_mean": 1820.4, "rolloff_mean": 3540.2, "bandwidth_mean": 1920.1, "mfcc_means": ["..."]}
  },
  "prosody": {
    "speaking_rate_wpm": 148.0,
    "speaking_rate_sps": 4.2,
    "pause_count": 6,
    "pause_total_seconds": 7.3,
    "pause_mean_seconds": 1.2,
    "filler_rate": 0.04
  },
  "emotion": {
    "dimensional": {"model": "audeering/...-msp-dim", "arousal": 0.62, "valence": 0.58, "dominance": 0.71},
    "categorical": {"model": "speechbrain/...-IEMOCAP", "label": "neu", "scores": {"hap": 0.18, "neu": 0.55, "ang": 0.12, "sad": 0.15}}
  },
  "composite": {
    "confidence": 0.74,
    "engagement": 0.68,
    "calmness":   0.51,
    "_components": {
      "confidence": {
        "jitter_steadiness":  0.130,
        "shimmer_steadiness": 0.085,
        "energy_steadiness":  0.180,
        "low_filler_rate":    0.075,
        "pace_steadiness":    0.100,
        "dominance":          0.142
      },
      "engagement": {"...": "..."},
      "calmness":   {"...": "..."}
    },
    "_formulas": {
      "confidence": ["1 - normalize(acoustic.voice_quality.jitter_local, 0.005, 0.025)", "..."],
      "engagement": ["..."],
      "calmness":   ["..."]
    }
  },
  "windows": [
    {"start_sec": 0.0,  "end_sec": 30.0, "pitch_mean_hz": 138.4, "loudness_mean_db": -21.4, "arousal": 0.59, "valence": 0.60, "confidence": 0.71, "engagement": 0.65, "calmness": 0.55},
    {"start_sec": 30.0, "end_sec": 60.0, "...": "..."}
  ],
  "processing": {
    "started_at": "2026-05-08T18:14:02Z",
    "completed_at": "2026-05-08T18:15:47Z",
    "library_versions": {"vsa": "0.1.0"},
    "errors": []
  }
}
```

The webhook fires `POST <callback_url>` with the same JSON wrapped in
`{request_id, status, metadata, result}`, signed with
`X-Signature-256: sha256=<HMAC-SHA256 of the body using callback_secret>`.

---

## Deploy to Fly.io in 5 minutes

You will need a Fly.io account with a payment method on file (Fly's
free Machines tier covers the idle case but you still need a card to
launch). Install the `flyctl` CLI from
[fly.io/docs/hands-on/install-flyctl](https://fly.io/docs/hands-on/install-flyctl/).

From the repo root:

```bash
# 1. Authenticate (opens a browser).
fly auth login

# 2. Create the app from the shipped fly.toml. --no-deploy lets you
#    set secrets first; pick any region from `fly platform regions`.
#    The default in fly.toml is iad (Ashburn, VA).
fly launch --no-deploy --name <your-unique-app-name> --region iad

# 3. Set the inbound API key. Generate a random one:
fly secrets set API_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")

# (Optional) override transcription engine / window size / max audio
# size — defaults are fine for a journaling use case.
# fly secrets set TRANSCRIBER_ENGINE=whisper WHISPER_MODEL=small WINDOW_SECONDS=30

# 4. Deploy. First build is slow (~5–7 GB image with model weights
#    baked in); subsequent deploys reuse cached layers.
fly deploy

# 5. Smoke test from outside the deploying machine. Replace
#    <your-app> and <some-public-audio.wav> with real values:
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

The first request after a deploy or after the Machine has been idle
will pay a cold-start latency hit (model load is several seconds);
subsequent requests reuse the loaded models.

The shipped `fly.toml` is configured for pattern-alpha: scale to zero
when idle, burst up to 3 concurrent Machines under load, with one
in-flight request per Machine. See the comments in `fly.toml` for
tunable knobs (region, CPU/RAM, max Machines).

---

## Run locally with Docker Compose

For development you can run the same Docker image used in production
against your own audio files, no Fly account needed:

```bash
# 1. Copy the example env file and edit it. At minimum set API_KEY.
cp .env.example .env
# (open .env in your editor, replace API_KEY=replace-me-... with a real value)

# 2. Build and start. First build takes a while — the image bakes
#    in NeMo + audeering wav2vec2 + SpeechBrain IEMOCAP weights.
docker compose up --build

# 3. In another terminal, serve a sample WAV over HTTP so the
#    container can fetch it:
cd /path/to/folder/with/sample.wav
python -m http.server 9000

# 4. POST an analysis job. host.docker.internal resolves to the
#    host machine from inside the container.
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

You can run `nc -l 9001` (or any tiny HTTP listener) in a third
terminal if you want to see the webhook callback.

For a fully local one-shot without the HTTP service, use the CLI:

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"

# Print result JSON to stdout.
vsa analyze path/to/recording.wav

# Or write to a file.
vsa analyze path/to/recording.wav --out result.json
```

---

## How the composite scores work

The analyzer ships three composite scores, each in `[0, 1]`, each a
weighted sum of normalized subscores. Plain English:

- **`confidence`** — does the speaker sound steady and self-assured?
  Combines low jitter and shimmer (steady vocal fold vibration), low
  filler-word rate, steady energy across the recording, steady pace,
  and high model-detected dominance.
- **`engagement`** — does the speaker sound animated, expressive, alive?
  Combines wider pitch range, more loudness variation, higher arousal,
  and an unhurried-but-not-stalling pace with low pause ratio.
- **`calmness`** — does the speaker sound relaxed and unhurried?
  Combines low pitch variation, low jitter, low arousal, slight
  positive-valence bias, and a slow speaking rate.

Each composite is computed as `sum(weight_i * subscore_i)` where every
subscore is in `[0, 1]`. The weights for each composite sum to 1.0 so
the output stays in `[0, 1]`. Each subscore that needs raw acoustic
data uses `normalize(value, lo, hi)` — a min-max clip where `lo` maps
to 0 and `hi` maps to 1. Where one input is missing (e.g. the emotion
model failed) the composite skips that component and re-normalizes the
remaining weights, so a single flaky model can't wipe out the whole
section.

Every weight, every clinical threshold, and every formula lives in
[`composites.yaml`](composites.yaml) — that file is the source of
truth. Fork it, tune it, disagree publicly. The output JSON includes
a `_components` map showing each subscore's weighted contribution
(they sum to the composite within float tolerance) so you can see
exactly *why* today's `confidence` dropped — was it the fillers, the
jitter, or the dominance? — without re-deriving the math.

`composites.yaml` calls out which thresholds carry actual research
backing (jitter and shimmer ranges from Boone, McFarlane et al.'s
*The Voice and Voice Therapy*; arousal/valence/dominance from the
audeering MSP-Podcast model card) and which are pragmatic defaults
the author calibrated for a self-journaling use case and explicitly
marks `"tunable, not clinical."` Calibrate those against your own
baseline data once you have a few weeks of recordings.

---

## Configuration reference

Every runtime knob is an environment variable. Set them via Fly secrets
(`fly secrets set NAME=value`) in production or via the `.env` file
in local Docker Compose dev.

| Variable | Default | What it controls | When to change it |
| --- | --- | --- | --- |
| `API_KEY` | *(none, required)* | Inbound `Authorization: Bearer` token. Requests without it get 401. | Required. Generate something random and unguessable; rotate if you suspect a leak. |
| `TRANSCRIBER_ENGINE` | `parakeet` | Which transcription model to use. `parakeet` (NeMo Parakeet TDT 0.6B v2, English-only, default) or `whisper` (faster-whisper, multilingual). | Set to `whisper` if you record in a non-English language or want a smaller/cheaper model. |
| `WHISPER_MODEL` | `small` | When `TRANSCRIBER_ENGINE=whisper`, which faster-whisper size to load. One of `tiny`, `base`, `small`, `medium`, `large-v3`. Ignored when using parakeet. | Larger = slower + more accurate. `small` is a good middle ground. |
| `WINDOW_SECONDS` | `30` | Width of each window in the time-windowed analysis section. The whole audio is tiled into non-overlapping windows of this size. | Drop to 10–15s if you want finer-grained trend within short clips; raise to 60s+ for longer recordings where 30s is too noisy. |
| `MAX_AUDIO_BYTES` | `52428800` (50 MiB) | Hard cap on audio size the `AudioFetcher` will accept. Oversized payloads are rejected before download. | Raise if you record long sessions; lower for tighter cost protection. |
| `HF_HOME` | `/opt/hf-cache` (set in Dockerfile) | Hugging Face cache root. Models are baked into this path at image build time. | Don't touch unless you're rebuilding the image with a different cache layout. |

---

## Architectural notes

- **Stateless.** No database, no Redis, no persistent volumes. The
  service is pure-function: signed URL in, sentiment JSON out. Audio
  is fetched to `/tmp` (RAM-backed tmpfs on Fly Machines) during
  analysis and deleted when the request completes — success or
  failure. Result JSONs are not stored on the analyzer; they are
  returned in the HTTP response and POSTed (HMAC-signed) to the
  caller's `callback_url`. **The caller's database is the system of
  record.** Operations is "git push, fly deploy" — no migrations, no
  state recovery.
- **Pattern alpha (long-held HTTP).** Each `/analyze` request stays
  in flight on the Machine for the duration of the analysis. The
  caller (e.g. a Supabase edge function with a 150s timeout) is
  fire-and-forget: it disconnects after its own timeout, but the
  analyzer keeps running because Fly's proxy still considers the
  Machine busy — auto-suspend won't fire mid-job. On completion the
  analyzer fires the HMAC-signed webhook to `callback_url`. No queue,
  no Redis, no checkpoints. Tradeoff explicitly accepted: a hard
  Machine crash mid-job loses that one job; the caller can re-trigger.
- **Privacy by default.** Audio never persists. The analyzer reads
  the signed URL, downloads to `/tmp`, runs the analysis, deletes the
  file, returns the JSON. No copies, no caches, no re-uploads. Result
  JSONs are not stored on the analyzer either — only the caller's
  database holds them. The image bakes in model weights so the
  analyzer never phones home for downloads at runtime.
- **Three interfaces, one core.** `vsa.Pipeline` (the Python class)
  is the substance. `vsa.api` (FastAPI) and `vsa.cli` (Click) are
  ~100-line wrappers around it. Tests target `Pipeline` directly via
  `from vsa import Pipeline`; the wrapper layers are thin enough that
  the integration test on `Pipeline` covers them implicitly.
- **Partial-success contract.** Every analyzer (transcription,
  acoustic, prosody, emotion, composite, windowed) is independent. A
  failure in one section nulls *that* section in the output and
  appends a string to `processing.errors`; everything else still
  runs. The composite scorer re-normalizes around missing inputs so
  e.g. an emotion-model crash still produces a valid `confidence`
  score from the remaining components.
- **CPU-only.** All inference runs on shared-CPU Fly Machines. No
  GPU dependency, no CUDA. Latency is generous (multi-second to
  multi-minute per recording) — fine for an asynchronous webhook
  workflow, not fine for live streaming.

---

## Status

Slices 1–9 shipped: scaffolding, auth, audio fetcher, acoustic
features, prosody, emotion, composite scoring, time-windowed
analysis, faster-whisper alternative transcriber. Slice 10 (this
slice) adds the deploy artifacts and this README. See open issues for
roadmap; PRs welcome.
