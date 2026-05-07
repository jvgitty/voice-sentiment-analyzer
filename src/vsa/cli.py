"""CLI entrypoint for the voice sentiment analyzer.

Wraps `Pipeline.analyze` directly (no HTTP) so users can run analysis on a
local audio file from the command line.
"""

import asyncio
from pathlib import Path
from typing import Optional

import typer

from vsa.pipeline import Pipeline


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
) -> None:
    """Analyze a local audio file and emit the result JSON."""
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
