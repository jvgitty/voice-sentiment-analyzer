"""Tests for the vsa CLI."""

import json
from pathlib import Path

from typer.testing import CliRunner

from vsa.cli import app


runner = CliRunner()


class TestCliAnalyze:
    def test_analyze_prints_json_to_stdout(self, fixture_wav_path: Path) -> None:
        result = runner.invoke(app, ["analyze", str(fixture_wav_path)])

        assert result.exit_code == 0, result.stdout
        payload = json.loads(result.stdout)
        assert payload["schema_version"] == "1.0"
