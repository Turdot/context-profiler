"""Smoke tests — verify the package imports and CLI entry point work."""

import json
from pathlib import Path
from click.testing import CliRunner
from context_profiler.cli import main
from context_profiler.context_diff import _artifact_from_text

FIXTURES = Path(__file__).parent / "fixtures"


def _langfuse_trace(input_payload):
    return {
        "id": "trace-test",
        "projectId": "project-test",
        "name": "Claude Code - Turn 1",
        "timestamp": "2026-05-14T03:22:15.000Z",
        "observations": [
            {
                "id": "generation-test",
                "type": "GENERATION",
                "name": "Claude Response",
                "startTime": "2026-05-14T03:22:15.000Z",
                "input": input_payload,
                "model": "claude",
            }
        ],
    }


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


def test_analyze_snapshot_auto_format(tmp_path):
    snapshot = FIXTURES / "repeated_tool_calls.json"
    runner = CliRunner()
    result = runner.invoke(main, ["analyze", str(snapshot), "--format", "auto"])
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


def test_formats_list_json():
    runner = CliRunner()
    result = runner.invoke(main, ["formats", "list", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "formats" in data
    names = [f["name"] for f in data["formats"]]
    assert "openai" in names
    assert "cursor-jsonl" in names
    assert "claude-code-jsonl" in names
    assert "agent-trace" in names
    assert "agent-trajectories" in names
    assert "swe-agent-traj" in names
    assert "toolathlon" not in names


def test_formats_describe_json():
    runner = CliRunner()
    result = runner.invoke(main, ["formats", "describe", "openai", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["name"] == "openai"
    assert "required_signals" in data
    assert data["input_kind"] == "provider-request"
    assert data["confidence"] == "exact"
    assert "limitations" in data
    assert "agent_conversion_guidance" in data


def test_schema_trace_json():
    runner = CliRunner()
    result = runner.invoke(main, ["schema", "trace", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["title"] == "ContextTrace"
    assert data["type"] == "object"


def test_validate_known_fixture_json():
    snapshot = FIXTURES / "repeated_tool_calls.json"
    runner = CliRunner()
    result = runner.invoke(main, ["validate", str(snapshot), "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["valid"] is True
    assert data["detected_format"] in {"openai", "anthropic", "langfuse"}


def test_validate_unknown_shape_guides_agent(tmp_path):
    unsupported = tmp_path / "unsupported.json"
    unsupported.write_text(json.dumps({"unexpected": "shape"}), encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(main, ["validate", str(unsupported), "--json"])
    assert result.exit_code == 1
    data = json.loads(result.output)
    assert data["valid"] is False
    assert data["errors"][0]["code"] == "UNSUPPORTED_SHAPE"
    assert "agent_action" in data["errors"][0]
    assert any("schema trace" in step for step in data["next_steps"])


def test_validate_langfuse_without_analyzable_generation_is_invalid(tmp_path):
    trace = tmp_path / "unsupported_langfuse.json"
    trace.write_text(json.dumps(_langfuse_trace({"unexpected": "shape"})), encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(main, ["validate", str(trace), "--format", "langfuse", "--json"])
    assert result.exit_code == 1
    data = json.loads(result.output)
    assert data["valid"] is False
    assert data["errors"][0]["code"] == "NO_ANALYZABLE_GENERATIONS"


def test_normalize_known_fixture_json():
    snapshot = FIXTURES / "repeated_tool_calls.json"
    runner = CliRunner()
    result = runner.invoke(main, ["normalize", str(snapshot), "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["schema_version"] == "0.1"
    assert data["runs"]


def test_diagnose_json_contains_issues():
    snapshot = FIXTURES / "repeated_tool_calls.json"
    runner = CliRunner()
    result = runner.invoke(main, ["diagnose", str(snapshot), "--format", "auto", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["schema_version"] == "0.1"
    assert "issues" in data
    assert "summary" in data


def test_diagnose_langfuse_simple_generation_input_from_stdin():
    trace = _langfuse_trace({"role": "user", "content": "hello"})
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["diagnose", "-", "--format", "langfuse", "--json"],
        input=json.dumps(trace),
    )
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["mode"] == "session"
    assert data["analysis_scope"]["format"] == "langfuse"
    assert data["analysis_scope"]["input_kind"] == "observability-trace"


def test_analyze_langfuse_from_stdin():
    trace = _langfuse_trace({"messages": [{"role": "user", "content": "hello"}]})
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["analyze", "-", "--format", "langfuse"],
        input=json.dumps(trace),
    )
    assert result.exit_code == 0
    assert "Token Distribution" in result.output


def test_analyze_cursor_transcript_html(tmp_path):
    transcript = FIXTURES / "cursor_transcript.jsonl"
    out = tmp_path / "cursor-report.html"
    runner = CliRunner()
    result = runner.invoke(main, ["analyze", str(transcript), "--format", "cursor-jsonl", "--html", str(out)])
    assert result.exit_code == 0
    assert out.exists()
    assert "<html" in out.read_text().lower()


def test_diagnose_claude_code_transcript_json():
    transcript = FIXTURES / "claude_code_transcript.jsonl"
    runner = CliRunner()
    result = runner.invoke(main, ["diagnose", str(transcript), "--format", "claude-code-jsonl", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["schema_version"] == "0.1"
    assert data["mode"] == "session"
    assert data["analysis_scope"]["input_kind"] == "agent-transcript"
    assert data["analysis_scope"]["confidence"] == "partial"
    assert data["analysis_scope"]["limitations"]


def test_normalize_cursor_transcript_json():
    transcript = FIXTURES / "cursor_transcript.jsonl"
    runner = CliRunner()
    result = runner.invoke(main, ["normalize", str(transcript), "--from", "cursor-jsonl", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["runs"][0]["source_format"] == "cursor-jsonl"
    assert len(data["runs"][0]["events"]) >= 2


def test_diagnose_includes_context_diff_summary():
    transcript = FIXTURES / "cursor_transcript.jsonl"
    runner = CliRunner()
    result = runner.invoke(main, ["diagnose", str(transcript), "--format", "cursor-jsonl", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "diff_summary" in data
    assert data["diff_summary"]["transition_count"] > 0
    assert data["diff_summary"]["max_added_tokens"] > 0
    assert "diff_hints" in data


def test_diagnose_hints_possible_artifact_churn():
    transcript = FIXTURES / "artifact_churn_transcript.jsonl"
    runner = CliRunner()
    result = runner.invoke(main, ["diagnose", str(transcript), "--format", "cursor-jsonl", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    hint_types = [hint["type"] for hint in data["diff_hints"]]
    assert "possible_artifact_churn" in hint_types
    churn_hint = next(hint for hint in data["diff_hints"] if hint["type"] == "possible_artifact_churn")
    assert churn_hint["evidence"]["artifact_key"] == "src/Button.tsx"


def test_diagnose_reports_tool_hotspots():
    snapshot = FIXTURES / "repeated_tool_calls.json"
    runner = CliRunner()
    result = runner.invoke(main, ["diagnose", str(snapshot), "--format", "openai", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    issue_codes = [issue["code"] for issue in data["issues"]]
    assert "TOOL_USE_DOMINATES_CONTEXT" in issue_codes
    assert "TOP_TOOL_CONTEXT_HOTSPOT" in issue_codes


def test_artifact_extraction_keeps_jsonl_extension():
    text = "/tmp/context-profiler-current-chat.jsonl"
    assert _artifact_from_text(text) == "/tmp/context-profiler-current-chat.jsonl"


def test_skill_distribution_manifests():
    root = Path(__file__).parents[1]
    skill = root / "skills" / "analyze-agent-context" / "SKILL.md"
    open_plugin = root / ".plugin" / "plugin.json"
    claude_marketplace = root / ".claude-plugin" / "marketplace.json"

    assert skill.exists()
    assert "name: analyze-agent-context" in skill.read_text()

    plugin_data = json.loads(open_plugin.read_text())
    assert plugin_data["name"] == "context-profiler"
    assert plugin_data["skills"] == ["./skills/analyze-agent-context"]

    marketplace_data = json.loads(claude_marketplace.read_text())
    assert marketplace_data["plugins"][0]["name"] == "context-profiler"
    assert marketplace_data["plugins"][0]["skills"] == ["./skills/analyze-agent-context"]


def test_diagnose_agent_trace_sample_json():
    sample = Path(__file__).parents[1] / "examples" / "agent-trace" / "sample.json"
    runner = CliRunner()
    result = runner.invoke(main, ["diagnose", str(sample), "--format", "agent-trace", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["analysis_scope"]["format"] == "agent-trace"
    assert data["mode"] == "session"
    assert data["diff_summary"]["transition_count"] > 0
    assert data["diff_hints"]


def test_analyze_agent_trace_sample_html(tmp_path):
    sample = Path(__file__).parents[1] / "examples" / "agent-trace" / "sample.json"
    out = tmp_path / "agent-trace-report.html"
    runner = CliRunner()
    result = runner.invoke(main, ["analyze", str(sample), "--format", "agent-trace", "--html", str(out)])
    assert result.exit_code == 0
    assert out.exists()
    assert "<html" in out.read_text().lower()
