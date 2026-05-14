# context-profiler

[![PyPI version](https://img.shields.io/pypi/v/context-profiler)](https://pypi.org/project/context-profiler/)
[![Python](https://img.shields.io/pypi/pyversions/context-profiler)](https://pypi.org/project/context-profiler/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Trace-source agnostic context analysis harness for LLM agents.

Bring any agent trace, loop, transcript, or raw provider request. `context-profiler` explains how the context grows, which tools dominate it, what repeats, and where turn-to-turn changes happen.

![Icicle view — token distribution breakdown](assets/demo-snapshot.gif)

## Why

Agent systems fail quietly when context grows too large, repeats stale observations, carries verbose tool payloads, or churns on the same artifact for many turns. Observability tools tell you what happened. `context-profiler` focuses on what happened to the context.

It is designed for both humans and agents:

- Humans get an interactive HTML report with timeline, icicle, diff, and tool views.
- Agents get stable JSON from `validate`, `diagnose`, and `schema` commands.
- Skills teach Cursor, Claude Code, and other Agent Skills compatible tools when to call the CLI.

![Session mode — timeline and diff](assets/demo-session.gif)

## What It Finds

- **Tool bloat**: tool inputs/results dominate the visible context.
- **Context growth spikes**: individual turns that add large payloads.
- **Repeated content**: exact or near-duplicate blocks retained in context.
- **Repeated tool inputs**: large repeated tool arguments.
- **Artifact churn**: the same file/component/path appears across many tool calls.
- **Partial-scope warnings**: transcripts are useful, but not raw provider requests.

## Research Context

`context-profiler` is motivated by recent work showing that long-horizon agents are constrained not only by model quality, but also by how their working context is retained, compressed, and reused across turns.

Related work:

- **ByteDance Seed — _Scaling Long-Horizon LLM Agent via Context-Folding_**  
  Studies context management for long-horizon agents through folding and summarizing intermediate sub-trajectories. This motivates `context-profiler`'s focus on turn-to-turn context diffs, retained observations, and compression/pruning evidence.
- **SWE-agent — _Agent-Computer Interfaces Enable Automated Software Engineering_**  
  Shows the importance of the agent-computer interface for software-engineering agents, motivating analysis of tool calls, terminal output, and artifact churn.
- **WebArena — _A Realistic Web Environment for Building Autonomous Agents_**  
  Demonstrates the value of realistic multi-step agent trajectories, motivating support for loop/transcript analysis rather than only single prompt snapshots.

## Install

```bash
pip install context-profiler
```

Or install from source:

```bash
git clone https://github.com/Turdot/context-profiler.git
cd context-profiler
pip install -e .
```

## Quick Start

Analyze a raw provider request:

```bash
context-profiler analyze request.json --format auto
context-profiler diagnose request.json --format auto --json
```

Analyze a coding-agent transcript:

```bash
context-profiler diagnose cursor-transcript.jsonl --format cursor-jsonl --json
context-profiler analyze claude-code-session.jsonl --format claude-code-jsonl --html report.html
```

Analyze a Langfuse export:

```bash
context-profiler validate trace.json --format langfuse --json
context-profiler diagnose trace.json --format langfuse --json
context-profiler analyze trace.json --format langfuse --html report.html
```

Generate an interactive report:

```bash
context-profiler analyze session.jsonl --html report.html
```

## Agent-Friendly CLI Harness

`context-profiler` is strict about supported formats but helpful when input does not match. Agents can discover contracts and adapt unsupported traces without asking users to reshape data manually.

```bash
# Discover supported formats
context-profiler formats list --json
context-profiler formats describe cursor-jsonl --json

# Discover canonical contracts
context-profiler schema trace --json
context-profiler schema diagnosis --json

# Validate and normalize
context-profiler validate trace.json --format auto --json
context-profiler normalize trace.json --from auto --json

# Diagnose for agent consumption
context-profiler diagnose trace.json --format auto --json
```

If validation fails, the JSON response includes `errors[].agent_action` and `next_steps` so the agent can convert the trace into `ContextTrace`.

## Agent Skill Distribution

This repository ships an `analyze-agent-context` skill for Cursor, Claude Code, and other Agent Skills / Open Plugins compatible tools.

The skill does not fetch traces. It teaches agents to use `context-profiler` whenever the user asks to analyze a trace, loop, transcript, agent run, context growth, stale context, or tool bloat.

Canonical skill:

```text
skills/analyze-agent-context/SKILL.md
```

Plugin manifests:

```text
.plugin/plugin.json
.claude-plugin/marketplace.json
```

## Supported Inputs

Use `context-profiler formats list --json` for the current machine-readable registry.

| Kind | Formats | Confidence |
|------|---------|------------|
| Provider request | OpenAI, Anthropic | exact |
| Observability trace | Langfuse, planned OTel/OpenInference | high |
| Agent transcript | Cursor JSONL, Claude Code JSONL | partial |
| Benchmark trajectory | planned agent-trace, agent_trajectories, SWE-agent | dataset-dependent |

For `agent-transcript`, analysis is intentionally marked `partial`: hidden system prompts, rules, tool definitions, MCP schemas, and provider compaction may not be present.

## HTML Report

The HTML report is self-contained and keeps the existing profiler style:

- **Icicle view**: hierarchical token breakdown, zoom, breadcrumb navigation.
- **Context timeline**: growth over turns, with small markers for large additions/tool-input spikes.
- **Diff mode**: unchanged / added / removed content between requests.
- **Tools view**: per-tool token table and invocation details.
- **Detail panel**: selected node or turn-level diff evidence.

## Example Diagnosis

```json
{
  "issues": [
    {
      "code": "TOOL_USE_DOMINATES_CONTEXT",
      "severity": "critical",
      "message": "Tool inputs dominate the visible context."
    },
    {
      "code": "TOP_TOOL_CONTEXT_HOTSPOT",
      "message": "ApplyPatch is the largest visible tool context hotspot."
    }
  ],
  "diff_hints": [
    {
      "type": "large_addition",
      "request_index": 76,
      "evidence": {
        "added_tokens": 7473,
        "top_added_tool": "ApplyPatch"
      }
    }
  ]
}
```

## Examples

See [`examples/README.md`](examples/README.md) for runnable fixtures and conversion patterns.

Recommended demo order:

1. Raw OpenAI/Anthropic request.
2. Cursor or Claude Code transcript.
3. Langfuse trace export.
4. Multi-turn academic trajectories such as `pagarsky/agent-trace`, `cx-cmu/agent_trajectories`, or SWE-agent traces.

## Docs

- [CLI harness design](docs/design/cli-harness.md)
- [Roadmap](docs/roadmap.md)

## What It Does Not Do

- It does not fetch traces from Langfuse, Hugging Face, Cursor, or Claude Code.
- It does not replay agent loops.
- It does not execute tools.
- It does not replace observability platforms.
- It does not pretend agent transcripts are exact raw provider requests.

## Development

```bash
PYTHONPATH=src uv run --with pytest pytest tests/test_smoke.py -v
```

## Acknowledgements

This project is inspired by and learned from:

- [context-lens](https://github.com/larsderidder/context-lens) — local proxy for capturing and visualizing LLM API calls
- [ContextFlame](https://github.com/jcgs2503/contextflame) — flamegraph-based token profiling for Claude Code
- [speedscope](https://www.speedscope.app/) — the icicle / flamegraph UI design is inspired by speedscope's interactive visualization

## License

[MIT](LICENSE)
