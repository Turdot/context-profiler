# CLI Harness MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first agent-readable CLI harness layer for `context-profiler`: format discovery, schema output, validation, normalization, and structured diagnosis.

**Architecture:** Keep the existing request/session profiler intact. Add small focused modules for format metadata, canonical schema, validation/normalization helpers, and diagnosis rendering. The first implementation should reuse existing adapters/analyzers rather than introducing the full Context Event Graph.

**Tech Stack:** Python 3.10+, Click, Rich, dataclasses, existing `context_profiler` adapters/analyzers/reporters, pytest via current smoke tests.

---

## File Structure

- Modify `src/context_profiler/cli.py`
  - Add `formats`, `schema`, `validate`, `normalize`, and `diagnose` commands.
  - Keep `analyze` backward compatible.
- Modify `src/context_profiler/profiler.py`
  - Register existing `ContentRepeatAnalyzer` and `FieldRepeatAnalyzer`.
  - Add reusable load helpers if needed.
- Create `src/context_profiler/formats.py`
  - Own format registry and `formats describe` payloads.
- Create `src/context_profiler/schemas.py`
  - Own canonical `trace` and `diagnosis` JSON Schema dictionaries.
- Create `src/context_profiler/validation.py`
  - Own validation result model and auto-detection error reporting.
- Create `src/context_profiler/diagnostics.py`
  - Convert `ProfileResult` into issue-code based diagnosis JSON.
- Modify `tests/test_smoke.py`
  - Add CLI tests for new commands.
- Modify `README.md`
  - Document the first agent-friendly CLI workflow.

## Task 1: Format Discovery and Schema Commands

**Files:**
- Create: `src/context_profiler/formats.py`
- Create: `src/context_profiler/schemas.py`
- Modify: `src/context_profiler/cli.py`
- Test: `tests/test_smoke.py`

- [ ] **Step 1: Add tests for `formats` and `schema` commands**

Add these tests to `tests/test_smoke.py`:

```python
def test_formats_list_json():
    runner = CliRunner()
    result = runner.invoke(main, ["formats", "list", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "formats" in data
    assert "openai" in [f["name"] for f in data["formats"]]


def test_formats_describe_json():
    runner = CliRunner()
    result = runner.invoke(main, ["formats", "describe", "openai", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["name"] == "openai"
    assert "required_signals" in data


def test_schema_trace_json():
    runner = CliRunner()
    result = runner.invoke(main, ["schema", "trace", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["title"] == "ContextTrace"
    assert data["type"] == "object"
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest tests/test_smoke.py::test_formats_list_json tests/test_smoke.py::test_formats_describe_json tests/test_smoke.py::test_schema_trace_json -v
```

Expected: FAIL because commands do not exist.

- [ ] **Step 3: Implement format registry**

Create `src/context_profiler/formats.py`:

```python
"""Known input format metadata for agent-readable CLI discovery."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class FormatSpec:
    name: str
    description: str
    status: str
    required_signals: list[str]
    common_sources: list[str]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


FORMAT_REGISTRY: dict[str, FormatSpec] = {
    "openai": FormatSpec(
        name="openai",
        description="OpenAI-compatible chat completion request with messages and optional tools.",
        status="supported",
        required_signals=["messages[] with role/content", "optional tools[] with function schemas"],
        common_sources=["OpenAI API logs", "Azure OpenAI logs", "OpenAI-compatible gateways"],
        notes=["Tool calls are read from assistant message tool_calls."],
    ),
    "anthropic": FormatSpec(
        name="anthropic",
        description="Anthropic Messages API request with typed content blocks.",
        status="supported",
        required_signals=["messages[]", "content blocks with type", "optional tools[] with input_schema"],
        common_sources=["Anthropic API logs", "Claude-compatible request captures"],
        notes=["System prompt may be top-level system string or blocks."],
    ),
    "langfuse": FormatSpec(
        name="langfuse",
        description="Langfuse trace export containing observations and GENERATION inputs.",
        status="supported",
        required_signals=["observations[]", "GENERATION observations with input.messages"],
        common_sources=["Langfuse UI export", "Langfuse API", "langfuse-cli"],
        notes=["Current adapter extracts generation inputs and delegates to OpenAI parsing."],
    ),
    "otel": FormatSpec(
        name="otel",
        description="OpenTelemetry or OpenInference span tree for LLM and agent operations.",
        status="planned",
        required_signals=["trace_id", "span_id", "parent span relation", "span attributes"],
        common_sources=["Phoenix", "Braintrust", "CrewAI", "Pydantic AI", "OpenAI Agents SDK exporters"],
        notes=["Initial version should focus on schema documentation before full span ingestion."],
    ),
    "agent-trace": FormatSpec(
        name="agent-trace",
        description="Multi-turn agent traces with llm_steps and tool spans.",
        status="planned",
        required_signals=["llm_steps[]", "spans[]", "tool_input/tool_output"],
        common_sources=["pagarsky/agent-trace"],
        notes=["Strong candidate for Context Event Graph demos because LLM steps and tool spans are explicit."],
    ),
    "agent-trajectories": FormatSpec(
        name="agent-trajectories",
        description="Large multi-turn academic agent trajectory records across multiple benchmarks.",
        status="planned",
        required_signals=["conversation messages", "benchmark metadata", "reward/evaluation fields"],
        common_sources=["cx-cmu/agent_trajectories"],
        notes=["Best fit for turn-to-turn context evolution analysis across long trajectories."],
    ),
    "swe-agent-traj": FormatSpec(
        name="swe-agent-traj",
        description="SWE-agent trajectory files with thought/action/observation steps and LM queries.",
        status="planned",
        required_signals=["trajectory steps", "query/messages", "action", "observation"],
        common_sources=["SWE-agent .traj files", "nebius/SWE-agent-trajectories"],
        notes=["Useful for coding-agent patch churn, terminal output, and test feedback loop analysis."],
    ),
}


def list_formats() -> list[dict[str, Any]]:
    return [spec.to_dict() for spec in FORMAT_REGISTRY.values()]


def describe_format(name: str) -> dict[str, Any]:
    try:
        return FORMAT_REGISTRY[name].to_dict()
    except KeyError as exc:
        available = ", ".join(sorted(FORMAT_REGISTRY))
        raise ValueError(f"Unknown format '{name}'. Available formats: {available}") from exc
```

- [ ] **Step 4: Implement schema registry**

Create `src/context_profiler/schemas.py`:

```python
"""JSON Schemas exposed by the CLI for agent-readable contracts."""

from __future__ import annotations

from typing import Any


TRACE_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "ContextTrace",
    "type": "object",
    "required": ["schema_version", "runs"],
    "properties": {
        "schema_version": {"type": "string", "const": "0.1"},
        "runs": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["run_id", "events"],
                "properties": {
                    "run_id": {"type": "string"},
                    "source_format": {"type": "string"},
                    "events": {"type": "array", "items": {"type": "object"}},
                },
            },
        },
    },
}


DIAGNOSIS_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "ContextDiagnosis",
    "type": "object",
    "required": ["schema_version", "issues", "summary"],
    "properties": {
        "schema_version": {"type": "string", "const": "0.1"},
        "summary": {"type": "object"},
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["code", "severity", "message"],
                "properties": {
                    "code": {"type": "string"},
                    "severity": {"type": "string", "enum": ["info", "warning", "critical"]},
                    "message": {"type": "string"},
                    "evidence": {"type": "object"},
                    "recommendation": {"type": "string"},
                },
            },
        },
    },
}


def get_schema(name: str) -> dict[str, Any]:
    if name == "trace":
        return TRACE_SCHEMA
    if name == "diagnosis":
        return DIAGNOSIS_SCHEMA
    raise ValueError("Unknown schema. Available schemas: trace, diagnosis")
```

- [ ] **Step 5: Add CLI commands**

Modify `src/context_profiler/cli.py`:

```python
import json
```

Add below `main()`:

```python
@main.group()
def formats():
    """Discover supported input formats."""
    pass


@formats.command("list")
@click.option("--json", "json_output", is_flag=True, help="Output machine-readable JSON")
def formats_list(json_output: bool):
    """List supported and planned input formats."""
    from context_profiler.formats import list_formats

    data = {"formats": list_formats()}
    if json_output:
        click.echo(json.dumps(data, indent=2, ensure_ascii=False))
        return

    console = Console()
    for fmt in data["formats"]:
        console.print(f"[bold]{fmt['name']}[/bold] ({fmt['status']}) - {fmt['description']}")


@formats.command("describe")
@click.argument("name")
@click.option("--json", "json_output", is_flag=True, help="Output machine-readable JSON")
def formats_describe(name: str, json_output: bool):
    """Describe a supported input format."""
    from context_profiler.formats import describe_format

    try:
        data = describe_format(name)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    if json_output:
        click.echo(json.dumps(data, indent=2, ensure_ascii=False))
        return

    console = Console()
    console.print(f"[bold]{data['name']}[/bold] ({data['status']})")
    console.print(data["description"])
    console.print("Required signals:")
    for signal in data["required_signals"]:
        console.print(f"  - {signal}")


@main.command("schema")
@click.argument("name", type=click.Choice(["trace", "diagnosis"]))
@click.option("--json", "json_output", is_flag=True, help="Output machine-readable JSON")
def schema_command(name: str, json_output: bool):
    """Print JSON Schema for context-profiler contracts."""
    from context_profiler.schemas import get_schema

    data = get_schema(name)
    if json_output:
        click.echo(json.dumps(data, indent=2, ensure_ascii=False))
        return
    click.echo(json.dumps(data, indent=2, ensure_ascii=False))
```

- [ ] **Step 6: Run tests**

Run:

```bash
pytest tests/test_smoke.py::test_formats_list_json tests/test_smoke.py::test_formats_describe_json tests/test_smoke.py::test_schema_trace_json -v
```

Expected: PASS.

## Task 2: Validation and Normalization Commands

**Files:**
- Create: `src/context_profiler/validation.py`
- Modify: `src/context_profiler/cli.py`
- Test: `tests/test_smoke.py`

- [ ] **Step 1: Add tests**

Add to `tests/test_smoke.py`:

```python
def test_validate_known_fixture_json():
    snapshot = FIXTURES / "repeated_tool_calls.json"
    runner = CliRunner()
    result = runner.invoke(main, ["validate", str(snapshot), "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["valid"] is True
    assert data["detected_format"] in {"openai", "anthropic", "langfuse"}


def test_normalize_known_fixture_json():
    snapshot = FIXTURES / "repeated_tool_calls.json"
    runner = CliRunner()
    result = runner.invoke(main, ["normalize", str(snapshot), "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["schema_version"] == "0.1"
    assert data["runs"]
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest tests/test_smoke.py::test_validate_known_fixture_json tests/test_smoke.py::test_normalize_known_fixture_json -v
```

Expected: FAIL because commands do not exist.

- [ ] **Step 3: Implement validation helpers**

Create `src/context_profiler/validation.py`:

```python
"""Validation and canonical normalization helpers for CLI commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from context_profiler.adapters.auto_detect import detect_adapter
from context_profiler.adapters.langfuse_adapter import is_langfuse_trace, parse_langfuse_trace
from context_profiler.models import APIRequest, Session
from context_profiler.profiler import load_session


def load_json_input(path: str) -> Any:
    if path == "-":
        import sys

        return json.loads(sys.stdin.read())
    with open(Path(path), encoding="utf-8") as f:
        return json.load(f)


def validate_input(path: str, format_hint: str | None = None) -> dict[str, Any]:
    try:
        data = load_json_input(path)
        if isinstance(data, dict) and data.get("schema_version") == "0.1" and "runs" in data:
            return {"valid": True, "detected_format": "context-trace", "errors": []}
        if isinstance(data, dict) and is_langfuse_trace(data):
            return {"valid": True, "detected_format": "langfuse", "errors": []}
        if isinstance(data, dict):
            adapter = detect_adapter(data)
            return {"valid": True, "detected_format": adapter.name, "errors": []}
        return {"valid": False, "detected_format": None, "errors": ["Input must be a JSON object"]}
    except Exception as exc:
        return {"valid": False, "detected_format": None, "errors": [str(exc)]}


def request_to_event(request: APIRequest) -> dict[str, Any]:
    return {
        "event_type": "llm_request",
        "request_index": request.request_index,
        "trace_index": request.trace_index,
        "source_format": request.source_format,
        "model": request.model,
        "total_input_tokens": request.total_input_tokens,
        "message_count": len(request.messages),
        "tool_count": len(request.tools),
    }


def normalize_input(path: str, format_hint: str | None = None) -> dict[str, Any]:
    data = load_json_input(path)
    if isinstance(data, dict) and data.get("schema_version") == "0.1" and "runs" in data:
        return data

    if isinstance(data, dict) and is_langfuse_trace(data):
        session = parse_langfuse_trace(data)
    elif path != "-":
        session = load_session(Path(path), format_hint=format_hint)
    else:
        adapter = detect_adapter(data)
        session = Session(requests=[adapter.parse(data)])

    return {
        "schema_version": "0.1",
        "runs": [
            {
                "run_id": session.metadata.get("trace_id") or "run-0",
                "source_format": session.metadata.get("source_format") or (
                    session.requests[0].source_format if session.requests else "unknown"
                ),
                "metadata": session.metadata,
                "events": [request_to_event(req) for req in session.requests],
            }
        ],
    }
```

- [ ] **Step 4: Add CLI commands**

Modify `src/context_profiler/cli.py`:

```python
@main.command("validate")
@click.argument("path", type=click.Path(exists=False), required=True)
@click.option("--format", "fmt", default=None, help="Input format hint")
@click.option("--json", "json_output", is_flag=True, help="Output machine-readable JSON")
def validate_command(path: str, fmt: str | None, json_output: bool):
    """Validate an input trace or request."""
    from context_profiler.validation import validate_input

    data = validate_input(path, format_hint=fmt)
    if json_output:
        click.echo(json.dumps(data, indent=2, ensure_ascii=False))
        return
    console = Console()
    color = "green" if data["valid"] else "red"
    console.print(f"[{color}]valid={data['valid']} format={data['detected_format']}[/{color}]")
    for error in data["errors"]:
        console.print(f"[red]{error}[/red]")
    if not data["valid"]:
        raise SystemExit(1)


@main.command("normalize")
@click.argument("path", type=click.Path(exists=False), required=True)
@click.option("--from", "from_format", default=None, help="Input format hint")
@click.option("--json", "json_output", is_flag=True, help="Output machine-readable JSON")
def normalize_command(path: str, from_format: str | None, json_output: bool):
    """Normalize input into ContextTrace JSON."""
    from context_profiler.validation import normalize_input

    data = normalize_input(path, format_hint=from_format)
    click.echo(json.dumps(data, indent=2, ensure_ascii=False))
```

- [ ] **Step 5: Run tests**

Run:

```bash
pytest tests/test_smoke.py::test_validate_known_fixture_json tests/test_smoke.py::test_normalize_known_fixture_json -v
```

Expected: PASS.

## Task 3: Diagnosis Command and Analyzer Registration

**Files:**
- Create: `src/context_profiler/diagnostics.py`
- Modify: `src/context_profiler/profiler.py`
- Modify: `src/context_profiler/cli.py`
- Test: `tests/test_smoke.py`

- [ ] **Step 1: Add tests**

Add to `tests/test_smoke.py`:

```python
def test_diagnose_json_contains_issues():
    snapshot = FIXTURES / "repeated_tool_calls.json"
    runner = CliRunner()
    result = runner.invoke(main, ["diagnose", str(snapshot), "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["schema_version"] == "0.1"
    assert "issues" in data
    assert "summary" in data
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
pytest tests/test_smoke.py::test_diagnose_json_contains_issues -v
```

Expected: FAIL because command does not exist.

- [ ] **Step 3: Register existing analyzers**

Modify `src/context_profiler/profiler.py` imports:

```python
from context_profiler.analyzers.content_repeat import ContentRepeatAnalyzer
from context_profiler.analyzers.field_repeat import FieldRepeatAnalyzer
```

Change `ALL_ANALYZERS`:

```python
ALL_ANALYZERS: list[BaseAnalyzer] = [
    TokenCounterAnalyzer(),
    ContentRepeatAnalyzer(),
    FieldRepeatAnalyzer(),
]
```

- [ ] **Step 4: Implement diagnosis renderer**

Create `src/context_profiler/diagnostics.py`:

```python
"""Convert profile results into stable agent-readable diagnosis reports."""

from __future__ import annotations

from typing import Any

from context_profiler.profiler import ProfileResult


def _severity_for_ratio(ratio: float) -> str:
    if ratio >= 0.3:
        return "critical"
    if ratio >= 0.1:
        return "warning"
    return "info"


def diagnose_result(result: ProfileResult) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []

    token = result.analyzer_results.get("token_counter")
    if token:
        summary = token.summary
        total = summary.get("total_input_tokens", 0) or 0
        tool_defs = summary.get("tool_definition_tokens", 0) or 0
        if total and tool_defs / total > 0.3:
            issues.append({
                "code": "STATIC_CONTEXT_BLOAT",
                "severity": _severity_for_ratio(tool_defs / total),
                "message": "Tool definitions consume a large share of input context.",
                "evidence": {"tool_definition_tokens": tool_defs, "total_input_tokens": total},
                "recommendation": "Audit unused or oversized tool schemas and consider smaller tool descriptions.",
            })

    content = result.analyzer_results.get("content_repeat")
    if content:
        wasted = content.summary.get("total_wasted_tokens", 0)
        ratio = content.summary.get("waste_ratio", 0.0)
        if wasted > 0:
            issues.append({
                "code": "REPEATED_CONTENT_BLOCK",
                "severity": _severity_for_ratio(ratio),
                "message": "Repeated content blocks waste context tokens.",
                "evidence": {"total_wasted_tokens": wasted, "waste_ratio": ratio},
                "recommendation": "Replace repeated long content with references or summaries.",
            })

    field = result.analyzer_results.get("field_repeat")
    if field:
        wasted = field.summary.get("total_wasted_tokens", 0)
        ratio = field.summary.get("waste_ratio", 0.0)
        if wasted > 0:
            issues.append({
                "code": "REPEATED_TOOL_INPUT",
                "severity": _severity_for_ratio(ratio),
                "message": "Tool input fields repeat large similar values across calls.",
                "evidence": {
                    "total_wasted_tokens": wasted,
                    "waste_ratio": ratio,
                    "top_offenders": field.summary.get("top_offenders", []),
                },
                "recommendation": "Move stable repeated arguments into references or shorter identifiers.",
            })

    return {
        "schema_version": "0.1",
        "source": result.source,
        "mode": result.mode,
        "summary": {
            "issue_count": len(issues),
            "warnings": result.all_warnings,
        },
        "issues": issues,
    }
```

- [ ] **Step 5: Add CLI command**

Modify `src/context_profiler/cli.py`:

```python
@main.command("diagnose")
@click.argument("path", type=click.Path(exists=True), required=True)
@click.option("--format", "fmt", default=None, help="Input format hint")
@click.option("--json", "json_output", is_flag=True, help="Output machine-readable JSON")
def diagnose_command(path: str, fmt: str | None, json_output: bool):
    """Diagnose context pathologies in an input trace or request."""
    from context_profiler.diagnostics import diagnose_result

    input_path = Path(path)
    session = try_load_langfuse(input_path)
    if session is not None or input_path.is_dir() or input_path.suffix == ".jsonl" or fmt == "langfuse":
        session = session or load_session(input_path, format_hint=fmt)
        result = profile_session(session, source=str(input_path))
    else:
        request = load_request(input_path, format_hint=fmt)
        result = profile_request(request, source=str(input_path))

    data = diagnose_result(result)
    if json_output:
        click.echo(json.dumps(data, indent=2, ensure_ascii=False))
        return

    console = Console()
    console.print(f"[bold]Issues:[/bold] {data['summary']['issue_count']}")
    for issue in data["issues"]:
        console.print(f"[{issue['severity']}]{issue['code']}[/{issue['severity']}] {issue['message']}")
```

- [ ] **Step 6: Run tests**

Run:

```bash
pytest tests/test_smoke.py::test_diagnose_json_contains_issues -v
```

Expected: PASS.

- [ ] **Step 7: Run full smoke tests**

Run:

```bash
pytest tests/test_smoke.py -v
```

Expected: PASS.

## Task 4: README Update

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add CLI harness section**

Add after Quick Start examples:

```markdown
## Agent-Friendly CLI Harness

context-profiler exposes discovery and validation commands so coding agents can inspect supported inputs before analyzing traces:

```bash
# Discover supported raw formats
context-profiler formats list --json
context-profiler formats describe langfuse --json

# Discover the canonical trace and diagnosis schemas
context-profiler schema trace --json
context-profiler schema diagnosis --json

# Validate and normalize arbitrary trace input
context-profiler validate trace.json --format auto --json
context-profiler normalize trace.json --from auto --json

# Produce an agent-readable diagnosis
context-profiler diagnose trace.json --format auto --json
```

The intended workflow is: bring any trace or trajectory, let an agent or adapter normalize it, then run context-profiler to diagnose context growth, repeated tool inputs, verbose tool outputs, and stale or repeated content.
```

- [ ] **Step 2: Run help command**

Run:

```bash
context-profiler --help
```

Expected: New commands appear in help.

- [ ] **Step 3: Final verification**

Run:

```bash
pytest tests/test_smoke.py -v
```

Expected: PASS.

## Self-Review

- Spec coverage: This MVP covers CLI discovery, schema, validate, normalize, diagnosis, and first use of existing repeat analyzers. It does not implement full Context Event Graph or subagent graph yet; those remain later phases.
- Placeholder scan: No placeholder tasks remain. Later phases are explicitly out of MVP scope.
- Type consistency: `formats`, `schema`, `validate`, `normalize`, and `diagnose` command names match the CLI design spec.
