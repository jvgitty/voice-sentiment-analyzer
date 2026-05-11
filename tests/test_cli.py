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
        assert payload["schema_version"] == "2.0"

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
        assert payload["schema_version"] == "2.0"
        # When writing to file, stdout should not contain the JSON payload.
        assert "schema_version" not in result.stdout

    def test_analyze_missing_file_exits_nonzero_with_stderr(
        self, tmp_path: Path
    ) -> None:
        missing = tmp_path / "does_not_exist.wav"
        result = runner.invoke(app, ["analyze", str(missing)])

        assert result.exit_code != 0
        # CliRunner exposes combined output via `result.output`; on newer
        # Click versions stderr is also separated. Either should mention
        # the bad path or that it doesn't exist.
        combined = (result.output or "") + (
            result.stderr if result.stderr_bytes is not None else ""
        )
        assert (
            "does_not_exist.wav" in combined
            or "not found" in combined.lower()
            or "exist" in combined.lower()
        )

    def test_analyze_help_lists_all_flags(self) -> None:
        result = runner.invoke(app, ["analyze", "--help"])

        assert result.exit_code == 0, result.output
        assert "--out" in result.output
        assert "--engine" in result.output
