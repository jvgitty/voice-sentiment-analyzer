# Voice sentiment analysis (v0.1.x)

> **Status:** Archived May 2026. The code described below is NOT in
> the current codebase. It lives in the `v0.1.1-archived-sentiment`
> git tag. This document exists so the work isn't forgotten — it is
> reference-only, not a guide for current implementation.

## What this was

A speech-feature stack that ran alongside transcription and tried to
infer "how the speaker sounded" — pitch, voice quality, emotion,
speaking rate — and roll those features into three composite scores
(`confidence`, `engagement`, `calmness`) per recording, plus a
time-tiled view of how those scores shifted within a single audio.

The original product hypothesis: voice notes carry emotional and
acoustic information that text alone misses, and clients (initially
imagined as personal-productivity / journaling users) would value
that as a differentiator.

## The full pipeline

The v0.1.1 `Pipeline.analyze()` ran six feature stages after
ffmpeg normalization to 16 kHz mono PCM WAV:

1. **Parakeet transcription** (kept in v0.2)
2. **AcousticAnalyzer** — Praat (parselmouth) + librosa
3. **EmotionAnalyzer** — two wav2vec2 models in parallel
4. **ProsodyAnalyzer** — text-derived rate-of-speech, pauses, fillers
5. **WindowedAnalyzer** — time-tiled re-runs of acoustic + dimensional
   emotion at ~30s window granularity
6. **CompositeScorer** — YAML-driven formulas combining outputs of
   #2-#4 into three named scores

The `AnalyzeResult` JSON exposed every intermediate output. The
service was open-source and the schema was intended to let callers
build their own composite formulas on top of the raw features.

## The Python libraries

### Acoustic features

| Library | Purpose | Approx size |
|---------|---------|-------------|
| `praat-parselmouth` | Pitch, jitter, shimmer, HNR via Praat | ~50 MB |
| `librosa` | Spectral centroid, rolloff, bandwidth, MFCCs, audio I/O | ~30 MB + heavy transitive deps |
| `scipy` | numerical underpinnings | ~80 MB |
| `numpy` | array operations | ~30 MB |

Outputs (per recording):
- `pitch`: `mean_hz`, `median_hz`, `std_hz`, `min_hz`, `max_hz`, `range_hz`
- `loudness`: `mean_db`, `std_db`, `rms_mean`
- `voice_quality`: `jitter_local`, `shimmer_local`, `hnr_db`, `voiced_unvoiced_ratio`
- `spectral`: `centroid_mean`, `rolloff_mean`, `bandwidth_mean`, `mfcc_means` (list of 13)

### Emotion (dimensional)

- **Model:** `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim`
- **Library:** `transformers` (HuggingFace)
- **Size:** ~1.3 GB resident
- **Architecture:** Wav2Vec2 backbone + custom 3-output regression head
- **License:** Custom non-commercial — verify before any commercial use.
- **Outputs:** continuous `arousal`, `valence`, `dominance` scalars
  approximately in `[0, 1]`.
- **Quirk:** ships with a non-standard head, requires a custom subclass
  of `Wav2Vec2PreTrainedModel` per the model card.

### Emotion (categorical)

- **Model:** `speechbrain/emotion-recognition-wav2vec2-IEMOCAP`
- **Library:** `speechbrain`
- **Size:** ~1.3 GB resident
- **Architecture:** Wav2Vec2 backbone + classifier head trained on
  IEMOCAP's 4-class label set
- **License:** Apache 2.0
- **Outputs:** label ∈ `{neutral, angry, happy, sad}` plus per-class
  softmax probabilities.
- **Quirk:** required a Windows-only workaround for SpeechBrain's
  `LazyModule.ensure_module` path-separator bug (Python 3.13 + Windows
  + SpeechBrain 1.1.0). See the patched `_patch_speechbrain_lazy_imports`
  function in `src/vsa/features/emotion.py` at the archive tag.

### Prosody (text-derived, no models)

Pure-function module. No model load. Used the transcript's word-level
timestamps to compute:

- `speaking_rate_wpm`, `speaking_rate_sps`
- `pause_count`, `pause_total_seconds`, `pause_mean_seconds`
- `filler_rate` — based on a fixed lookup of English filler words
  (`um`, `uh`, `like`, `you know`, `i mean`, `sort of`, `basically`,
  `literally`)

### Windowed analysis

Tiled `[0, audio_duration)` with non-overlapping ~30s windows. Each
window re-ran the (cheap) AcousticAnalyzer and the (expensive)
dimensional EmotionAnalyzer. Categorical emotion was skipped per
window because it added cost without contributing to headline metrics.

### Composite scoring

A YAML file (`composites.yaml`) defined three composite scores as
weighted sums of normalized subscores. The Python code maintained a
closed registry of callable subscore implementations keyed by name;
the YAML was the editable spec, the code was the source of truth on
execution. Mismatches between spec and registry surfaced as
`processing.errors`.

The three composites:

- **`confidence`** — weighted sum of: low jitter, low shimmer, energy
  steadiness, low filler rate, pace steadiness, model-detected
  dominance.
- **`engagement`** — weighted sum of: pitch variation, valence,
  arousal, speaking rate, energy mean.
- **`calmness`** — weighted sum of: low arousal, low pitch variation,
  steady pace, high HNR, low filler rate.

Each composite's output JSON carried a `_components` breakdown
showing each subscore's contribution and a `_formulas` echo of the
YAML's source strings, so callers could verify the math.

## Why it was removed

Three reasons, weighted roughly equally:

### 1. CPU memory wall

The three wav2vec2-class models plus Parakeet plus per-window
re-inference saturated 8 GB CPU machines on Fly. We chased this for
four PRs:

- **#33** evicted Parakeet and SpeechBrain caches between phases.
  Right optimization, wrong bottleneck — the peak happened *inside*
  Parakeet's `model.transcribe()` call on long audio, before eviction
  could fire.
- **#35** chunked Parakeet inference into 60s segments. Confirmed the
  per-chunk peak was small.
- **#36** added `malloc_trim(0)` between chunks and `MALLOC_ARENA_MAX=2`
  to fight glibc's reluctance to return free pages. Got total-VM from
  17.5 GB → 16.2 GB across 11 chunks — measurable but insufficient.
- **#37** bumped to 16 GB shared-cpu-8x. Still OOM'd at 16 GB resident
  on the 13-min smoke test.

The pattern was: memory grew monotonically with audio length
regardless of cap. The mitigations slowed the growth; nothing
stopped it. This was the trigger for the platform pivot to GPU.

### 2. Questionable product value

When we sat with the actual outputs, the dimensional emotion scores
(`arousal: 0.62, valence: 0.58, dominance: 0.71`) and acoustic
composites (`confidence: 0.74`) were research-grade numbers, not
product-grade insights. Compare to what an LLM extracts from the
transcript:

| Sentiment stack | LLM extraction |
|---|---|
| `arousal: 0.62` | `mood: "energized and contemplative"` |
| `confidence: 0.74` | summary mentioning the speaker's certainty |
| `categorical: {happy: 0.18, neutral: 0.55, ...}` | extracted `themes` and `tasks` |

For the target audiences we were imagining (attorneys, doctors,
individual professionals taking voice notes), the right-column
outputs are directly actionable. The left-column outputs would have
required the customer to build their own interpretation layer on top.

### 3. Operational cost

Three heavy ML models meant three sets of HuggingFace downloads on
every cold-start, three sets of model-load delays, three potential
failure modes per request, and three library dep trees to keep in
sync. Replacing the sentiment trio with one general-purpose LLM
shrank the failure surface dramatically.

## Things we tried that didn't move the needle enough

Captured for the next person who thinks they can make this work on CPU:

- **Cache eviction between pipeline phases.** Helped post-transcription;
  didn't touch the transcription-time peak.
- **`malloc_trim(0)` + `MALLOC_ARENA_MAX=2`.** Useful glibc tuning for
  any long-running ML worker. Bought us one extra chunk before OOM.
- **Bumping VM size 4 GB → 8 GB → 16 GB.** Linear progression of
  OOMs at exactly the new cap. Memory growth was monotonic.
- **Switching transcription engine to faster-whisper as a control
  test.** Same accumulation pattern; rules out NeMo-specific leaks.
- **Subprocess-per-chunk transcription** (designed but not built).
  Would have worked but at the cost of multi-minute extra wall clock
  per request from per-chunk model reloads.

The thing that would have actually worked on CPU: dropping the
categorical emotion model entirely (saving ~1.3 GB) and using
quantized int8 versions of the remaining models. We never got that
far because the pivot decision made it unnecessary.

## When to reconsider

Sentiment analysis would be worth re-adding if any of these become
true:

1. **A regulated medical use case validates acoustic emotion markers.**
   E.g., a depression-screening or cognitive-decline-tracking product
   where the dimensional scores have peer-reviewed clinical
   correlations. This is a substantively different product line and
   would need its own FDA / clinical-validation posture.
2. **A high-volume customer specifically asks for it** and is willing
   to pay enough to cover the operational overhead. Coaching apps and
   sales-call-QA platforms occasionally want voice stress / engagement
   metrics. Note: voice stress detection is heavily contested in
   legal contexts and courts often reject it.
3. **GPU costs collapse enough** that running the trio alongside the
   LLM is operationally trivial. Already partially true on Modal's
   T4; the question becomes whether anyone wants the outputs.
4. **We discover that LLM-extracted "mood" misses something** that
   acoustic features capture cleanly — e.g., genuine vs. performative
   emotional content. This would need user-research evidence, not a
   gut feeling.

If reconsidering: start by checking out `v0.1.1-archived-sentiment`,
running the existing pipeline on real customer audio, and showing
the outputs side-by-side with the current v0.2 LLM extraction to a
real prospective user. Get a clear "yes I'd pay for this column"
before re-integrating any of it.

## References

- **Git tag:** `v0.1.1-archived-sentiment` — full pre-pivot code.
- **Phase 1 strip PR:** `voice-note-transcription#39` — what got
  removed and why, with full diff stats.
- **Phase 5 Modal deploy PR:** `voice-note-transcription#43` — the
  platform decision that replaced the sentiment-era Fly deployment.
- **Original PRD discussion:** referenced inline in the v0.1.1
  schema docstrings (see archive tag).
- **Acoustic feature library docs:**
  [parselmouth](https://parselmouth.readthedocs.io/),
  [librosa](https://librosa.org/).
- **Emotion model cards:**
  [audeering wav2vec2 MSP](https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim),
  [SpeechBrain IEMOCAP](https://huggingface.co/speechbrain/emotion-recognition-wav2vec2-IEMOCAP).
