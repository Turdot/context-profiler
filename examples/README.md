# Examples

Runnable examples for context analysis harness workflows.

```bash
PYTHONPATH=src uv run context-profiler formats list --json
```

## SWE-agent Trajectory (recommended multi-turn demo)

A real 63-step coding agent trajectory from [nebius/SWE-agent-trajectories](https://huggingface.co/datasets/nebius/SWE-agent-trajectories) (CC-BY-4.0). See [`swe_agent/README.md`](swe_agent/README.md) for details.

```bash
PYTHONPATH=src uv run context-profiler analyze examples/swe_agent/session.jsonl --format openai --html /tmp/swe-agent-report.html
PYTHONPATH=src uv run context-profiler diagnose examples/swe_agent/session.jsonl --format openai --json
```

Findings: 26.9% content duplication, artifact churn on `reproduce.py`, monotonic context growth from 2.1K to 27.1K tokens.

## Raw Provider Request

```bash
PYTHONPATH=src uv run context-profiler diagnose tests/fixtures/repeated_tool_calls.json --format openai --json
PYTHONPATH=src uv run context-profiler analyze tests/fixtures/repeated_tool_calls.json --format openai --html /tmp/context-profiler-openai.html
```

Findings: `TOOL_USE_DOMINATES_CONTEXT`, `TOP_TOOL_CONTEXT_HOTSPOT`, `REPEATED_CONTENT_BLOCK`.

## Academic AgentTrace

A multi-step MBPP trajectory from [`pagarsky/agent-trace`](https://huggingface.co/datasets/pagarsky/agent-trace):

```bash
PYTHONPATH=src uv run context-profiler analyze examples/agent-trace/sample.json --format agent-trace --html /tmp/agent-trace-report.html
```

## Langfuse Export

```bash
context-profiler validate trace.json --format langfuse --json
context-profiler diagnose trace.json --format langfuse --json
context-profiler analyze trace.json --format langfuse --html /tmp/langfuse-report.html
```

## Agent Transcripts (Cursor / Claude Code)

```bash
PYTHONPATH=src uv run context-profiler analyze tests/fixtures/cursor_transcript.jsonl --format cursor-jsonl --html /tmp/cursor-report.html
PYTHONPATH=src uv run context-profiler analyze tests/fixtures/claude_code_transcript.jsonl --format claude-code-jsonl --html /tmp/claude-report.html
```

Analysis is partial — transcripts may omit hidden prompts, tool definitions, and provider compaction.

## Converting Other Datasets

For datasets not natively supported, write a conversion script in `examples/<dataset>/convert.py` that outputs OpenAI-format JSONL. See `examples/swe_agent/convert.py` as a reference.

Candidate datasets for future examples:
- [T1 (CapitalOne)](https://huggingface.co/datasets/capitalone/T1) — multi-domain tool planning
- [tau-bench (Sierra)](https://github.com/sierra-research/tau-bench) — retail/airline agent conversations
- [neulab/agent-data-collection](https://huggingface.co/datasets/neulab/agent-data-collection) — 18 environments, standardized format
