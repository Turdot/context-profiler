"""Smoke tests — verify the package imports and CLI entry point work."""

import json
from pathlib import Path
from click.testing import CliRunner
from context_profiler.cli import main
from context_profiler.context_diff import _artifact_from_text
from context_profiler.profiler import load_session

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


def test_analyze_html_escapes_embedded_script_tags(tmp_path):
    trace = _langfuse_trace({
        "messages": [
            {
                "role": "user",
                "content": "<html><script>console.log('x')</script>\\n\\nAfter script</html>",
            }
        ]
    })
    input_path = tmp_path / "langfuse-html-payload.json"
    input_path.write_text(json.dumps(trace), encoding="utf-8")
    out = tmp_path / "report.html"

    runner = CliRunner()
    result = runner.invoke(main, ["analyze", str(input_path), "--format", "langfuse", "--html", str(out)])

    assert result.exit_code == 0
    html = out.read_text()
    assert html.count("</script>") == 1
    assert "<\\/script>" in html


def test_analyze_html_includes_findings_drawer(tmp_path):
    snapshot = FIXTURES / "repeated_tool_calls.json"
    out = tmp_path / "report.html"

    runner = CliRunner()
    result = runner.invoke(main, ["analyze", str(snapshot), "--format", "openai", "--html", str(out)])

    assert result.exit_code == 0
    html = out.read_text()
    assert 'id="findings-btn"' in html
    assert 'id="findings-panel"' in html
    assert '"diagnosis"' in html
    assert "TOOL_USE_DOMINATES_CONTEXT" in html


def test_findings_drawer_groups_findings_by_code():
    template = (Path(__file__).parents[1] / "src" / "context_profiler" / "templates" / "report.html").read_text()
    assert "function groupFindings" in template
    assert "finding-group-header" in template
    assert "findings-close-icon" in template


def test_html_includes_persistence_view():
    template = (Path(__file__).parents[1] / "src" / "context_profiler" / "templates" / "report.html").read_text()
    assert 'data-view="persistence"' in template
    assert 'id="persistence-pane"' in template
    assert "function renderPersistenceView" in template
    assert "persistence_blocks" in template
    assert "persistenceScrollY" in template
    assert "ROLE_COLORS_P" in template
    assert "session_insights" in template


def test_html_flow_view_renders_spawn_fanout_to_real_blocks():
    template = (Path(__file__).parents[1] / "src" / "context_profiler" / "templates" / "report.html").read_text()
    selected_body = template.split("function selectedPropagationLinks() {", 1)[1].split("function renderPropagationSankey()", 1)[0]
    render_body = template.split("function renderPropagationSankey() {", 1)[1].split("function flowNodeHitTest", 1)[0]

    assert "const FLOW_MAX_LINKS = 36" in template
    assert "link.type === 'spawn'" in selected_body
    assert "link.type === 'accumulation'" not in selected_body
    assert ".slice(0, 28)" in selected_body
    assert "return [...spawnLinks, ...carryLinks].slice(0, FLOW_MAX_LINKS)" in selected_body
    assert "function buildFlowGroups" in template
    assert "const flowGroups = buildFlowGroups(visibleLinks)" in render_body
    assert "const targetKey = `target:${group.source}:${link.target}`" in template
    assert "sourceY" in render_body
    assert "targetY" in render_body
    assert "Spawn" in template


def test_html_includes_artifacts_view_for_duplication_insights():
    template = (Path(__file__).parents[1] / "src" / "context_profiler" / "templates" / "report.html").read_text()
    assert 'id="artifacts-pane"' in template
    assert "function renderArtifactsView" in template
    assert "artifact_duplications" in template


def test_html_artifact_left_cards_have_scannable_metrics_and_tool_badges():
    template = (Path(__file__).parents[1] / "src" / "context_profiler" / "templates" / "report.html").read_text()

    assert "function artifactKeyParts" in template
    assert "function renderArtifactToolBadges" in template
    assert "artifact-prefix" in template
    assert "artifact-id" in template
    assert "artifact-metric-chip" in template
    assert "artifact-tool-badge" in template
    assert ".artifact-item.active::before" in template


def test_html_report_has_consistent_hover_and_focus_polish():
    template = (Path(__file__).parents[1] / "src" / "context_profiler" / "templates" / "report.html").read_text()

    assert ":focus-visible" in template
    assert "translateX(2px)" in template
    assert "box-shadow" in template


def test_html_artifact_timeline_cells_truncate_before_next_column():
    template = (Path(__file__).parents[1] / "src" / "context_profiler" / "templates" / "report.html").read_text()

    assert ".artifact-timeline-row > span" in template
    assert "text-overflow: ellipsis" in template
    assert "min-width: 0" in template


def test_html_artifacts_panel_scrollbar_and_splitter_match_other_panes():
    template = (Path(__file__).parents[1] / "src" / "context_profiler" / "templates" / "report.html").read_text()

    assert ".artifacts-left::-webkit-scrollbar" in template
    assert ".artifacts-right::-webkit-scrollbar" in template
    assert 'id="artifacts-resize-handle"' in template
    assert "const artifactsResizeHandle" in template
    assert "artifactsResizeHandle.addEventListener('mousedown'" in template
    assert "artifactsLeft.style.width" in template


def test_html_json_tree_preserves_multiline_strings():
    template = (Path(__file__).parents[1] / "src" / "context_profiler" / "templates" / "report.html").read_text()
    assert ".replace(/\\n/g, '\\\\n')" not in template
    assert "json-multiline-str" in template


def test_html_round_navigation_does_not_auto_open_detail():
    template = (Path(__file__).parents[1] / "src" / "context_profiler" / "templates" / "report.html").read_text()
    switch_body = template.split("function switchToRound(index) {", 1)[1].split("function switchToSnapshot()", 1)[0]
    assert "showRoundDiffDetail(currentRound)" not in switch_body


def test_html_icicle_view_hides_artifacts_pane_when_switching_back():
    template = (Path(__file__).parents[1] / "src" / "context_profiler" / "templates" / "report.html").read_text()
    icicle_branch = template.split("if (view === 'icicle') {", 1)[1].split("} else if (view === 'tools')", 1)[0]

    assert "artifactsPane.classList.remove('visible')" in icicle_branch


def test_html_resizing_detail_panel_rerenders_icicle_during_drag():
    template = (Path(__file__).parents[1] / "src" / "context_profiler" / "templates" / "report.html").read_text()
    detail_drag = template.split("if (!resizeDragging) return;", 1)[1].split("document.addEventListener('mouseup'", 1)[0]

    assert "detailPanel.style.height = detailHeight + 'px';" in detail_drag
    assert "requestRender();" in detail_drag


def test_html_tool_invocation_arrow_is_centered_css_indicator():
    template = (Path(__file__).parents[1] / "src" / "context_profiler" / "templates" / "report.html").read_text()

    assert ".tools-detail-html .td-inv-header .arrow::before" in template
    assert "justify-self: end" in template
    assert "border-left: 5px solid" in template
    assert '<span class="arrow" aria-hidden="true"></span>' in template
    assert '<span class="arrow">▶</span>' not in template


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


def test_load_langfuse_session_json_preserves_trace_boundaries(tmp_path):
    session_export = {
        "id": "session-test",
        "projectId": "project-test",
        "createdAt": "2026-05-14T03:20:00.000Z",
        "traces": [
            _langfuse_trace({"messages": [{"role": "user", "content": "first"}]}),
            _langfuse_trace({"messages": [{"role": "user", "content": "second"}]}),
        ],
    }
    session_export["traces"][0]["id"] = "trace-1"
    session_export["traces"][0]["timestamp"] = "2026-05-14T03:21:00.000Z"
    session_export["traces"][1]["id"] = "trace-2"
    session_export["traces"][1]["timestamp"] = "2026-05-14T03:22:00.000Z"

    path = tmp_path / "langfuse-session.json"
    path.write_text(json.dumps(session_export), encoding="utf-8")

    session = load_session(path, format_hint="langfuse")

    assert len(session.requests) == 2
    assert [req.trace_index for req in session.requests] == [0, 1]
    assert session.metadata["session_id"] == "session-test"
    assert session.metadata["num_traces"] == 2
    assert session.metadata["turn_boundaries"] == [0, 1]


def test_load_langfuse_session_without_observations_guides_fetching(tmp_path):
    session_export = {
        "id": "session-test",
        "projectId": "project-test",
        "traces": [
            {
                "id": "trace-1",
                "projectId": "project-test",
                "timestamp": "2026-05-14T03:21:00.000Z",
            },
            {
                "id": "trace-2",
                "projectId": "project-test",
                "timestamp": "2026-05-14T03:22:00.000Z",
            },
        ],
    }
    path = tmp_path / "langfuse-session-no-observations.json"
    path.write_text(json.dumps(session_export), encoding="utf-8")

    session = load_session(path, format_hint="langfuse")

    assert session.requests == []
    assert session.metadata["warnings"] == [
        "Langfuse session contains traces, but none include embedded analyzable GENERATION observations. "
        "Fetch observations for each trace and include them in the trace objects before analysis."
    ]


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
    claude_plugin = root / ".claude-plugin" / "plugin.json"

    assert skill.exists()
    assert "name: analyze-agent-context" in skill.read_text()

    plugin_data = json.loads(open_plugin.read_text())
    assert plugin_data["name"] == "context-profiler"
    assert plugin_data["skills"] == ["./skills/analyze-agent-context"]

    claude_data = json.loads(claude_plugin.read_text())
    assert claude_data["name"] == "context-profiler"


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
