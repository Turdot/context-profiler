# CLAUDE.md

This file guides AI coding agents working on `context-profiler`.

## Project Overview

`context-profiler` is the evidence layer for context engineering. It accepts raw provider requests, observability traces, and agent trajectories, then reports how context grows, repeats, and concentrates around tools/artifacts — so humans and agents know what to compact and where it's safe to cut.

Core boundary: the library and CLI analyze trace files or stdin. Agents may fetch user-requested trace exports with tools such as `curl` before invoking `context-profiler`, but core should not replay agent loops or execute third-party tools.

## Common Commands

```bash
# Run the smoke suite
PYTHONPATH=src uv run --with pytest pytest tests/test_smoke.py -v

# Check CLI entry point
PYTHONPATH=src uv run context-profiler --help

# Analyze a multi-turn session (canonical demo)
PYTHONPATH=src uv run context-profiler analyze examples/swe_agent/session.jsonl --format openai --html /tmp/report.html

# Diagnose for agent consumption
PYTHONPATH=src uv run context-profiler diagnose examples/swe_agent/session.jsonl --format openai --json

# Analyze a raw request
PYTHONPATH=src uv run context-profiler analyze tests/fixtures/repeated_tool_calls.json --format openai
```

## Architecture

- `src/context_profiler/cli.py` — Click CLI entry point.
- `src/context_profiler/models.py` — core data model (Session, APIRequest, Message, ContentBlock, BlockType, Role).
- `src/context_profiler/adapters/` — format-specific input adapters (OpenAI, Anthropic, Langfuse, agent-trace, transcript).
- `src/context_profiler/formats.py` — format registry exposed to agents.
- `src/context_profiler/schemas.py` — JSON Schemas for agent-readable contracts.
- `src/context_profiler/validation.py` — validation and canonical normalization helpers.
- `src/context_profiler/profiler.py` — session/request loading and analyzer orchestration.
- `src/context_profiler/analyzers/` — token, repeat, and field analyzers.
- `src/context_profiler/context_diff.py` — turn-to-turn diff evidence and hints.
- `src/context_profiler/session_insights.py` — session-level carryover, budget pressure, artifact lifecycle, and propagation analysis.
- `src/context_profiler/diagnostics.py` — stable issue-code diagnosis JSON.
- `src/context_profiler/reporters/` — CLI, JSON, and HTML reports.
- `src/context_profiler/templates/report.html` — self-contained report UI (views: Icicle, Tools, Persistence).
- `skills/` — public Agent Skills distribution.
- `examples/` — multi-turn demo datasets with conversion scripts.

## HTML Report Views

The interactive HTML report has three main views:

- **Icicle** — token distribution per request, timeline navigation, semantic/diff color modes.
- **Tools** — tool token hotspots, invocation detail, schema size.
- **Persistence** — heatmap of content blocks × requests showing what persists across turns (replaces the old Flow view).

## Format Strategy

Core adapters (tested, maintained): `openai`, `anthropic`, `langfuse`, `cursor-jsonl`, `claude-code-jsonl`, `agent-trace`.

For new benchmark trajectory formats (SWE-agent, T1, tau-bench, etc.), write conversion scripts in `examples/<dataset>/convert.py` that output OpenAI-format JSONL. Do not add new core adapters for every dataset.

## Format and Analyzer Rules

- Add synthetic fixtures under `tests/fixtures/`; do not commit real private traces.
- Every supported format must have metadata in `formats.py` with `input_kind`, `confidence`, `analysis_scope`, `limitations`, and `agent_conversion_guidance`.
- Keep analyzer outputs evidence-first: issue code, severity, token counts, request indices, tool names, artifact keys, and recommendation.
- For heuristic findings, use hint language such as `possible_*` and include confidence.
- Do not overclaim exactness for `agent-transcript`; transcripts may omit hidden prompts, tool definitions, rules, and provider compaction.

## Product Boundaries

Do not add provider-specific fetch clients to core. Agents can use Langfuse public API exports via `curl`, SDKs, Hugging Face tooling, or local transcript discovery to obtain data when the user asks, then pass files or stdin to `context-profiler`. Prefer direct Langfuse public API exports over `langfuse-cli` for trace analysis because the CLI can omit fields needed for nested observations and generations.

Canonical multi-turn demo datasets are in `examples/`: SWE-agent trajectories (nebius), lmcache-agentic-traces (sammshen), and OpenHands trajectories (nvidia). For new datasets, write conversion scripts in `examples/<dataset>/convert.py` rather than adding core adapters.

## Skill Distribution

Canonical public skills live in `skills/`.

Do not put product-distribution skills in `.agents/skills/` or `.claude/skills/`; those directories are for project-local agent behavior.

If skill paths change, update:

- `.plugin/plugin.json`
- `.claude-plugin/plugin.json`
- README skill distribution section
- tests covering manifests
