"""Smoke tests — verify the package imports and CLI entry point work."""

import json
from pathlib import Path
from click.testing import CliRunner
from context_profiler.cli import main

FIXTURES = Path(__file__).parent / "fixtures"


def test_import():
    import context_profiler
    assert hasattr(context_profiler, "__version__")


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "context-profiler" in result.output


def test_analyze_snapshot(tmp_path):
    snapshot = FIXTURES / "repeated_tool_calls.json"
    runner = CliRunner()
    result = runner.invoke(main, ["analyze", str(snapshot)])
    assert result.exit_code == 0
    assert "Token Distribution" in result.output


def test_analyze_json_output(tmp_path):
    snapshot = FIXTURES / "repeated_tool_calls.json"
    out = tmp_path / "report.json"
    runner = CliRunner()
    result = runner.invoke(main, ["analyze", str(snapshot), "-o", str(out)])
    assert result.exit_code == 0
    assert out.exists()
    data = json.loads(out.read_text())
    assert "analyzers" in data


def test_analyze_html_output(tmp_path):
    snapshot = FIXTURES / "repeated_tool_calls.json"
    out = tmp_path / "report.html"
    runner = CliRunner()
    result = runner.invoke(main, ["analyze", str(snapshot), "--html", str(out)])
    assert result.exit_code == 0
    assert out.exists()
    assert "<html" in out.read_text().lower()
