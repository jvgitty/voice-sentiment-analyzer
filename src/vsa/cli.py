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
    pipeline = Pipeline()
    result = asyncio.run(pipeline.analyze(audio_path))
    payload = result.model_dump_json()

    if out is not None:
        out.write_text(payload)
    else:
        typer.echo(payload)
