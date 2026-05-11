# Deprecated: archived reference material

This folder contains documentation about features and architectural
experiments that **are NOT part of the current codebase**. They were
either removed during a pivot or never shipped, but the work and the
lessons are preserved here so a future maintainer (or the project
owner) can evaluate whether to bring them back.

## How to use this folder

- **Read these documents when explicitly asked.** Don't pull them into
  context for ordinary work on the current codebase — they describe
  code paths that no longer exist and will confuse decisions about
  what's actually deployed today.
- **The code itself is not here.** It lives in `v0.1.1-archived-sentiment`
  and earlier git tags. To inspect the implementation: `git checkout
  v0.1.1-archived-sentiment`.

## Contents

- [`sentiment-analysis.md`](sentiment-analysis.md) — The v0.1.x voice
  sentiment analysis stack: acoustic features (Praat / librosa),
  dimensional + categorical emotion (audeering wav2vec2 + SpeechBrain
  IEMOCAP), prosody, windowed time-series, composite scoring. Why it
  was built, why it was removed, what we tried to make it work on
  CPU, and when it would be worth reconsidering.
