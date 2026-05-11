"""CLI entrypoint for the voice note transcription service.

Wraps :class:`vsa.pipeline.Pipeline` directly (no HTTP) so users can
transcribe a local audio file from the command line.
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


app = typer.Typer(
    help="Voice note transcription CLI.", no_args_is_help=True
)


@app.callback()
def _main() -> None:
    """Voice note transcription CLI."""


@app.command()
def analyze(
    audio_path: Path,
    out: Optional[Path] = typer.Option(
        None,
        "--out",
        help=(
            "Path to write the result JSON. If omitted, JSON is printed "
            "to stdout."
        ),
    ),
    engine: Optional[Engine] = typer.Option(
        None,
        "--engine",
        help=(
            "Transcription engine override (parakeet|whisper). Currently "
            "accepted for forward-compatibility; the actual engine is "
            "selected by the TRANSCRIBER_ENGINE env var."
        ),
    ),
) -> None:
    """Transcribe a local audio file and emit the result JSON."""
    # ``--engine`` is accepted but not wired to a runtime override yet —
    # TRANSCRIBER_ENGINE remains the single source of truth so the CLI
    # and the FastAPI handler stay in agreement.
    _ = engine

    if not audio_path.exists():
        typer.echo(f"error: audio file not found: {audio_path}", err=True)
        raise typer.Exit(code=1)

    pipeline = Pipeline()
    try:
        result = asyncio.run(pipeline.analyze(audio_path))
    except Exception as exc:  # noqa: BLE001 -- surface any failure to user
        typer.echo(
            f"error: failed to analyze {audio_path}: {exc}", err=True
        )
        raise typer.Exit(code=1)

    payload = result.model_dump_json()

    if out is not None:
        out.write_text(payload)
    else:
        typer.echo(payload)


# Allow ``python -m vsa.cli ...`` in addition to the ``vsa`` console
# script entry point. Without this block, runpy imports this module but
# never invokes the Typer app, leading to silent zero-output exits.
if __name__ == "__main__":
    app()
