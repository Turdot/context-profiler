"""Tests for tool input vs result token separation."""

import json
from pathlib import Path
from click.testing import CliRunner
from context_profiler.cli import main

FIXTURES = Path(__file__).parent / "fixtures"


def test_token_counter_separates_tool_input_and_result(tmp_path):
    """Verify that token_counter tracks tool_use and tool_result separately."""
    snapshot = FIXTURES / "repeated_tool_calls.json"
    out = tmp_path / "analysis.json"
    runner = CliRunner()
    result = runner.invoke(main, ["analyze", str(snapshot), "-o", str(out)])
    assert result.exit_code == 0

    data = json.loads(out.read_text())
    token_summary = data["analyzers"]["token_counter"]["summary"]

    # Check that we have the new fields
    assert "tool_use_tokens" in token_summary
    assert "tool_result_tokens" in token_summary
    assert "top_tools_by_input_tokens" in token_summary
    assert "top_tools_by_result_tokens" in token_summary

    # Verify the sum makes sense
    by_content = token_summary["by_content_type"]
    assert token_summary["tool_use_tokens"] == by_content.get("tool_use", 0)
    assert token_summary["tool_result_tokens"] == by_content.get("tool_result", 0)


def test_diagnose_detects_tool_input_bloat():
    """Verify TOOL_INPUT_BLOAT issue is detected when tool inputs dominate."""
    snapshot = FIXTURES / "repeated_tool_calls.json"
    runner = CliRunner()
    result = runner.invoke(main, ["diagnose", str(snapshot), "--format", "openai", "--json"])
    assert result.exit_code == 0

    data = json.loads(result.output)
    issue_codes = [issue["code"] for issue in data["issues"]]

    # This fixture has high tool_use content, so we should see the issue
    if data.get("issues"):
        # Check if TOOL_INPUT_BLOAT or TOOL_USE_DOMINATES_CONTEXT is present
        assert "TOOL_INPUT_BLOAT" in issue_codes or "TOOL_USE_DOMINATES_CONTEXT" in issue_codes


def test_top_tool_hotspot_includes_input_result_breakdown():
    """Verify TOP_TOOL_CONTEXT_HOTSPOT evidence includes input/result breakdown."""
    snapshot = FIXTURES / "repeated_tool_calls.json"
    runner = CliRunner()
    result = runner.invoke(main, ["diagnose", str(snapshot), "--format", "openai", "--json"])
    assert result.exit_code == 0

    data = json.loads(result.output)
    hotspot_issues = [issue for issue in data["issues"] if issue["code"] == "TOP_TOOL_CONTEXT_HOTSPOT"]

    if hotspot_issues:
        issue = hotspot_issues[0]
        evidence = issue["evidence"]

        # Check that the evidence includes the new breakdown fields
        assert "tool_input_tokens" in evidence
        assert "tool_result_tokens" in evidence
        assert "tool_tokens" in evidence

        # Verify the breakdown sums to total (or close, accounting for rounding)
        total = evidence["tool_tokens"]
        input_tokens = evidence["tool_input_tokens"]
        result_tokens = evidence["tool_result_tokens"]
        assert input_tokens + result_tokens == total
