# CLAUDE.md

This file guides AI coding agents working on `context-profiler`.

## Project Overview

`context-profiler` is a trace-source agnostic context analysis harness for LLM agents. It accepts raw provider requests, observability traces, local coding-agent transcripts, and future benchmark trajectories, then reports how context grows, repeats, and concentrates around tools/artifacts.

Core boundary: this project analyzes traces. It does not fetch traces, replay agent loops, or execute third-party tools.

## Common Commands

```bash
# Run the smoke suite
PYTHONPATH=src uv run --with pytest pytest tests/test_smoke.py -v

# Check CLI entry point
PYTHONPATH=src uv run context-profiler --help

# Analyze a raw request
PYTHONPATH=src uv run context-profiler analyze tests/fixtures/repeated_tool_calls.json --format openai

# Diagnose a transcript
PYTHONPATH=src uv run context-profiler diagnose tests/fixtures/cursor_transcript.jsonl --format cursor-jsonl --json

# Generate HTML
PYTHONPATH=src uv run context-profiler analyze tests/fixtures/cursor_transcript.jsonl --format cursor-jsonl --html /tmp/context-profiler-report.html
```

## Architecture

- `src/context_profiler/cli.py` — Click CLI entry point.
- `src/context_profiler/formats.py` — format registry exposed to agents.
- `src/context_profiler/schemas.py` — JSON Schemas for agent-readable contracts.
- `src/context_profiler/validation.py` — validation and canonical normalization helpers.
- `src/context_profiler/profiler.py` — session/request loading and analyzer orchestration.
- `src/context_profiler/analyzers/` — token, repeat, and field analyzers.
- `src/context_profiler/context_diff.py` — turn-to-turn diff evidence and hints.
- `src/context_profiler/diagnostics.py` — stable issue-code diagnosis JSON.
- `src/context_profiler/reporters/` — CLI, JSON, and HTML reports.
- `src/context_profiler/templates/report.html` — self-contained report UI.
- `skills/` — public Agent Skills distribution.

## Format and Analyzer Rules

- Add synthetic fixtures under `tests/fixtures/`; do not commit real private traces.
- Every supported format must have metadata in `formats.py` with `input_kind`, `confidence`, `analysis_scope`, `limitations`, and `agent_conversion_guidance`.
- Keep analyzer outputs evidence-first: issue code, severity, token counts, request indices, tool names, artifact keys, and recommendation.
- For heuristic findings, use hint language such as `possible_*` and include confidence.
- Do not overclaim exactness for `agent-transcript`; transcripts may omit hidden prompts, tool definitions, rules, and provider compaction.

## Product Boundaries

Do not add provider-specific fetch clients to core. Agents can use Langfuse CLI, SDKs, Hugging Face tooling, or local transcript discovery to obtain data, then pass files or stdin to `context-profiler`.

Do not make Toolathlon a first-class multi-turn demo. Prefer genuinely multi-turn datasets such as `pagarsky/agent-trace`, `cx-cmu/agent_trajectories`, and SWE-agent trajectories for future research examples.

## Skill Distribution

Canonical public skills live in `skills/`.

Do not put product-distribution skills in `.agents/skills/` or `.claude/skills/`; those directories are for project-local agent behavior.

If skill paths change, update:

- `.plugin/plugin.json`
- `.claude-plugin/marketplace.json`
- README skill distribution section
- tests covering manifests
