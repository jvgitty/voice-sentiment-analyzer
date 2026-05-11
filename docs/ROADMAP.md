# Roadmap

This file tracks scope explicitly deferred from v1. Anything captured here
was discussed during planning, judged worth doing, but kept out of v1 to
ship something working sooner.

When picking up an item, open a GitHub issue referencing the relevant
section so cross-references survive the eventual repo rename / split.

---

## v1 — current scope

**Goal:** fully-local transcription + LLM extraction service. HIPAA-friendly
deployment on Fly.io shared-CPU.

**Pipeline:**

1. ffmpeg-based audio normalization (16 kHz mono PCM WAV)
2. NeMo Parakeet transcription with chunked inference and explicit memory
   release between chunks
3. Local LLM extraction (Qwen3.5-9B-Instruct via llama-cpp-python, GBNF
   grammar-constrained JSON sampling)

**Response shape:** transcript + structured extraction (title, summary,
type, mood, tags, themes, people, locations, projects, businesses,
tech_stack, tasks).

**Default voice-note types:** `idea`, `journal`, `task`, `meditation`,
`other` (fallback).

---

## v2 — multi-tenant client configuration

The service is intended to be sold to clients (attorneys, doctors, and
other professionals who need HIPAA-style local processing). v2 makes the
service multi-tenant rather than single-config.

- **Per-client API keys** instead of the v1 shared bearer token. Keys
  carry their config bundle in a server-side lookup table.
- **Per-client type catalog**: each client defines their own list of
  voice-note types with descriptions. The default `idea/journal/task/
  meditation/other` catalog only applies if no client override is set.
- **Per-client system prompt overlays**: append client-specific
  instructions (tone, terminology preferences, must-extract fields) to
  the base prompt without forking the prompt entirely.
- **Per-client extraction schema variants**: legal, medical, general.
  Different domains care about different entity types. v2 ships at least
  these three templates; clients can extend.
- **Per-client domain glossaries**: lists of terms the LLM should
  recognize verbatim (drug names, case identifiers, internal project
  codenames). Injected into the prompt context.
- **Per-client rate limiting and audit logging**: usage attribution,
  per-request audit trails for HIPAA-relevant clients.
- **Per-client model tier selection**: a `fast` tier (smaller LLM, lower
  cost, faster wall clock) versus a `quality` tier (larger LLM, better
  extraction). See the **Model tiers** section below.

API shape sketch (not committed; just for planning):

```http
POST /analyze
Authorization: Bearer <client-api-key>
Content-Type: application/json

{
  "audio_url": "...",
  "callback_url": "...",
  "callback_secret": "...",
  "request_id": "...",
  "metadata": { ... },

  // Optional per-request overrides — fall back to the client's
  // configured defaults if omitted.
  "voice_note_types": [{ "name": "...", "description": "..." }],
  "domain": "legal" | "medical" | "general",
  "model_tier": "fast" | "quality"
}
```

---

## v2 — extraction quality and prompt evolution

- **Few-shot examples in the system prompt**: one or two `(transcript →
  expected JSON)` pairs anchor the model's output more consistently than
  schema-only instructions. ~300-token cost per request, worth it.
- **`enable_thinking=False` validation** for Qwen3.5: confirm the
  "thinking" chain isn't being emitted before the final answer (kills
  latency if it leaks through).
- **Self-consistency check**: optionally run extraction twice and
  reconcile divergences, for clients on the `quality` tier who can
  absorb the latency.
- **Schema versioning in the response**: include `extraction_schema_version`
  so downstream consumers can branch on shape changes without breaking.

---

## Model tiers

v1 ships with a single model (Qwen3.5-9B-Instruct, Q4_K_M). v2 should
let operators choose:

- **`fast` tier**: smaller model, e.g. a 4B-class with at least 16K
  context. Roughly 3–5× faster extraction wall clock, lower per-request
  cost, marginally lower quality on edge cases. Good for high-volume
  clients or non-critical workloads.
- **`quality` tier**: larger model, e.g. Qwen3.6-35B-A3B or comparable.
  Requires a bigger Fly VM (`performance-4x` or higher) and accepts
  longer wall clock in exchange for materially better extraction on
  complex transcripts. Best paired with GPU instances if that becomes
  an option.

Both tiers should expose the same JSON schema so client integrations
don't have to branch on the model.

---

## Operational improvements

- **Cold-start latency**: bake the LLM GGUF and Parakeet weights into
  the Docker image rather than re-downloading from HuggingFace on every
  machine wake. The current "~4 GB pulled on each cold start" cliff
  noted in the v0.1.1 handoff goes away once we own the bytes. Already
  planned for v1 Phase 3 — flagged here so the same pattern applies if
  we ever add additional models.
- **Fly edge proxy timeout**: the v0.1.1 smoke test repeatedly hit the
  default ~60s edge timeout on cold-started machines, returning 502 to
  curl even when the upstream eventually delivered the webhook. Add
  `[http_service.http_options.idle_timeout]` and/or `request_timeout`
  in `fly.toml` so the proxy waits long enough for first-request model
  load to complete. Small one-line PR.
- **Health checks block** in `fly.toml` so deploys roll only after the
  app is actually ready, eliminating the ~1-second race between proxy
  retries and uvicorn binding.
- **Subprocess-per-chunk transcription**: a leftover idea from the
  sentiment-analyzer OOM saga. Spawning a fresh worker process per
  transcription chunk guarantees full memory reclamation (no glibc
  arena fragmentation, no NeMo internal state buildup). Trade-off is
  the per-chunk model reload cost. Not needed at v1 since chunking +
  malloc_trim + the 16 GB VM together cleared the OOM, but worth
  keeping in our back pocket if any future model proves leakier than
  Parakeet.
- **Metrics + tracing**: per-phase wall-clock and peak RSS logged at
  the end of every `/analyze` call (env-gated). Lets us see real
  production memory and latency without re-running smoke tests.

---

## Platform decision: Modal (post-Fly pivot, 2026-05)

The v0.2 pivot from sentiment analysis to transcription + LLM
extraction was initially deployed to Fly.io's shared-CPU machines.
That didn't work: Parakeet's chunked-inference memory grows
monotonically with audio length on CPU, and we OOM'd at every memory
cap we set (4 GB → 8 GB → 16 GB). We then tried Fly's GPU offering
and discovered that Fly is no longer onboarding new GPU customers
(see [Fly's blog post](https://fly.io/blog/wrong-about-gpu/)).

Moved to **Modal** for the GPU runtime:

- T4 GPU at ~$0.59/hr, scale-to-zero, $30/month free credits on
  Starter plan covers early-customer beta volume.
- Modal Enterprise tier offers BAA for HIPAA workloads — the upgrade
  path when actually signing regulated customers.
- Python-native deployment (`modal deploy modal_app.py`) avoids the
  Dockerfile + fly.toml + secrets-set ceremony that bit us repeatedly
  on Fly.

The Fly artifacts (`fly.toml`, `Dockerfile`) are preserved in the repo
as documentation of the failed path. Eventually they should move to
a `legacy/` directory or be deleted.

When to revisit:
- If Modal's pricing changes meaningfully, or
- If a workload-shape change (e.g. always-on inference, batch
  processing) makes RunPod / Hetzner / bare metal genuinely
  cheaper at our scale, or
- If we want a multi-region active-active deployment Modal can't
  easily express.

---

## Lessons captured from v0.1.1 (sentiment analyzer)

Kept here so the same mistakes don't recur in v2 / v3 architecture
decisions. Full retrospective belongs in a separate document if the
team wants one; this is the bullet list.

- **Measure peak RSS during local dev before deploying.** The
  sentiment OOM saga (4 GB → 8 GB → 16 GB) was preventable with one
  `tracemalloc` measurement against the real-world test audio. Cheap
  upfront, much cheaper than four PRs of remediation.
- **Verify hypotheses against actual OOM traces before writing
  optimization code.** PR #33 (cache eviction) was based on a
  plausible reading of the code but didn't match the actual OOM
  trace. PR #35 (chunking) was the real fix.
- **MoE-on-CPU is an anti-pattern** for production webhook services.
  All experts have to be in RAM, the router causes cache misses, and
  the architecture was designed for GPU. Stick to dense models on
  shared-CPU Fly tiers.
- **Multimodal models for text-only workloads are fine** in modern
  architectures (unified vision-language training, not bolt-on
  adapters). The "capacity split hurts text quality" concern is
  largely outdated. Cost is ~10–15% wasted RAM for the vision
  encoder.
- **glibc holds freed pages on its own free list** rather than
  returning them to the kernel. `malloc_trim(0)` + `MALLOC_ARENA_MAX=2`
  helps but isn't a complete fix. Process-level isolation (one process
  per heavy computation) is the only fully reliable answer for
  long-running ML workers.

---

## How to use this file

When work begins on an item:

1. Open a GitHub issue with the section heading as the issue title.
2. Edit this file to add a `→ tracked in #<issue-number>` link next to
   the bullet so anyone landing here from search can find the active
   work.
3. Don't delete entries on completion — strike them through and link
   to the closing PR so this file doubles as a "what got built and
   when" record.
