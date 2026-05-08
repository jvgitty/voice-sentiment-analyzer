"""CLI entrypoint for the voice sentiment analyzer.

Wraps `Pipeline.analyze` directly (no HTTP) so users can run analysis on a
local audio file from the command line.
"""

import asyncio
from enum import Enum
from pathlib import Path
from typing import Optional

import typer

from vsa.pipeline import Pipeline


class Engine(str, Enum):
    parakeet = "parakeet"
    whisper = "whisper"


app = typer.Typer(help="Voice Sentiment Analyzer CLI.", no_args_is_help=True)


@app.callback()
def _main() -> None:
    """Voice Sentiment Analyzer CLI."""


@app.command()
def analyze(
    audio_path: Path,
    out: Optional[Path] = typer.Option(
        None,
        "--out",
        help="Path to write the result JSON. If omitted, JSON is printed to stdout.",
    ),
    engine: Optional[Engine] = typer.Option(
        None,
        "--engine",
        help=(
            "Transcription engine override (parakeet|whisper). Stub for now: "
            "the env var this maps to (TRANSCRIBER_ENGINE) is wired in a "
            "later slice; the value is currently accepted but not used."
        ),
    ),
    window_seconds: Optional[int] = typer.Option(
        None,
        "--window-seconds",
        help=(
            "Override window size for the time-series view. Stub for now: "
            "the env var this maps to (WINDOW_SECONDS) is wired in a "
            "later slice; the value is currently accepted but not used."
        ),
    ),
) -> None:
    """Analyze a local audio file and emit the result JSON."""
    # NOTE: --engine and --window-seconds are accepted but currently
    # only "passed through" -- the env vars they map to (TRANSCRIBER_ENGINE,
    # WINDOW_SECONDS) are introduced in later slices (Slice 7 and Slice 9).
    _ = engine
    _ = window_seconds

    if not audio_path.exists():
        typer.echo(f"error: audio file not found: {audio_path}", err=True)
        raise typer.Exit(code=1)

    pipeline = Pipeline()
    try:
        result = asyncio.run(pipeline.analyze(audio_path))
    except Exception as exc:  # noqa: BLE001 - surface any failure to the user
        typer.echo(f"error: failed to analyze {audio_path}: {exc}", err=True)
        raise typer.Exit(code=1)

    payload = result.model_dump_json()

    if out is not None:
        out.write_text(payload)
    else:
        typer.echo(payload)


# Allow ``python -m vsa.cli ...`` in addition to the ``vsa`` console script
# entry point. Without this block, runpy imports this module but never invokes
# the Typer app, leading to silent zero-output exits.
if __name__ == "__main__":
    app()
