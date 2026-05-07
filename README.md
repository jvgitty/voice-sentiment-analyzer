# Voice Sentiment Analyzer

Open-source voice sentiment analysis service. Takes an audio file and returns a structured JSON of acoustic features, emotion scores, and composite metrics (confidence, engagement, calmness).

Status: **early scaffolding** -- see open issues for the design PRD.

## Local development

Install the package in editable mode (this also registers the `vsa` console
script via `[project.scripts]`):

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"
```

Then run `vsa analyze` against a local audio file:

```bash
# Print the result JSON to stdout.
vsa analyze path/to/recording.wav

# Write the result JSON to a file instead of stdout.
vsa analyze path/to/recording.wav --out result.json

# Override the transcription engine or the time-window size.
# NOTE: --engine and --window-seconds are accepted today but their effect
# (wiring through to TRANSCRIBER_ENGINE / WINDOW_SECONDS) lands in later
# slices; the values are currently passed through but not yet read.
vsa analyze path/to/recording.wav --engine whisper --window-seconds 15
```

The CLI exits 0 on success and non-zero with an error message on stderr
when the audio path is missing or unreadable.
