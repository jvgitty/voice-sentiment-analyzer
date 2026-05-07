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

    def test_analyze_writes_json_to_out_path(
        self, fixture_wav_path: Path, tmp_path: Path
    ) -> None:
        out_path = tmp_path / "result.json"
        result = runner.invoke(
            app, ["analyze", str(fixture_wav_path), "--out", str(out_path)]
        )

        assert result.exit_code == 0, result.stdout
        assert out_path.exists()
        payload = json.loads(out_path.read_text())
        assert payload["schema_version"] == "1.0"
        # When writing to file, stdout should not contain the JSON payload.
        assert "schema_version" not in result.stdout
