"""CLI entrypoint for the voice sentiment analyzer.

Wraps `Pipeline.analyze` directly (no HTTP) so users can run analysis on a
local audio file from the command line.
"""

import asyncio
from pathlib import Path

import typer

from vsa.pipeline import Pipeline


app = typer.Typer(help="Voice Sentiment Analyzer CLI.", no_args_is_help=True)


@app.callback()
def _main() -> None:
    """Voice Sentiment Analyzer CLI."""


@app.command()
def analyze(audio_path: Path) -> None:
    """Analyze a local audio file and print the result JSON to stdout."""
    pipeline = Pipeline()
    result = asyncio.run(pipeline.analyze(audio_path))
    typer.echo(result.model_dump_json())
