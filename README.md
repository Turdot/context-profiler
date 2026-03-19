# context-profiler

[![PyPI version](https://img.shields.io/pypi/v/context-profiler.svg)](https://pypi.org/project/context-profiler/)
[![Python](https://img.shields.io/pypi/pyversions/context-profiler.svg)](https://pypi.org/project/context-profiler/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Framework-agnostic profiler for LLM agent context windows. Parses raw API request JSON and visualizes **where your tokens go** — no SDK instrumentation needed.

![Icicle view — token distribution breakdown](assets/demo-snapshot.gif)

## Why

LLM agent frameworks accumulate tool definitions, system prompts, and conversation history in the context window. You can't optimize what you can't see.

context-profiler gives you:

- **Token distribution** — breakdown by role, content type, and tool name
- **Icicle visualization** — [speedscope](https://www.speedscope.app/)-style interactive view, zoom into any node
- **Context growth timeline** — stacked area chart across a session, see the inflection point
- **Diff visualization** — what's new vs. what's history vs. what got pruned, per request

![Session mode — timeline and diff](assets/demo-session.gif)

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

```bash
# Analyze a single API request (snapshot mode)
context-profiler analyze request.json

# Analyze context growth over multiple requests (session mode)
context-profiler analyze session.jsonl
context-profiler analyze requests_dir/

# Generate an interactive HTML report
context-profiler analyze session.jsonl --html report.html

# Export JSON report
context-profiler analyze request.json -o report.json

# Specify format explicitly
context-profiler analyze request.json --format openai
context-profiler analyze request.json --format anthropic

# Analyze Langfuse traces
context-profiler analyze trace.json --format langfuse

# Multi-trace session (multiple Langfuse exports)
context-profiler analyze trace1.json trace2.json trace3.json --html report.html
```

## Supported Formats

Auto-detected from JSON structure:

| Format | Input | Mode |
|--------|-------|------|
| **OpenAI** | `{messages, tools}` | snapshot |
| **Anthropic** | `{messages, tools}` with content blocks | snapshot |
| **Langfuse trace** | `{observations: [{type: "GENERATION", ...}]}` | session |
| **JSONL** | One request per line | session |
| **Directory** | Folder of `.json` files | session |

## HTML Report Features

The HTML report is a self-contained file with no external dependencies:

- **Icicle view** — hierarchical token breakdown, click to zoom, breadcrumb navigation
- **Tools view** — per-tool token table with stacked bars, sortable columns
- **Timeline** — stacked area chart (system / tool defs / messages), click to select request
- **Color modes** — Semantic (by role) or Diff (unchanged / added / removed)
- **Role filters** — toggle visibility by role (system, user, assistant, tool)
- **Detail panel** — content preview and JSON tree for any selected node

## CLI Output

```
⚠ Warnings
  • Tool definitions consume 15.2K tokens (35.4% of total)

Token Distribution
  Category                  Tokens    % of Total
  Total Input                42.9K          100%
    System Prompt              1.2K          2.8%
    Tool Definitions          15.2K         35.4%
    Messages (assistant)       8.4K         19.6%
    Messages (tool)           14.1K         32.9%
    Messages (user)            4.0K          9.3%

Top Tools by Token Usage
  Tool                                Tokens    Calls
  playwright_with_chunk_browser_snap  12.4K        8
  filesystem-read_file                 3.2K        5
  local-search_in_turn                 1.1K        3
```

## Examples

The `examples/` directory contains a complete demo using [Toolathlon](https://huggingface.co/datasets/hkust-nlp/Toolathlon-Trajectories) trajectories:

```bash
cd examples/

# Convert Toolathlon data to context-profiler input
python convert_toolathlon.py toolathlon_raw.json --mode snapshot -o snapshot.json
python convert_toolathlon.py toolathlon_raw.json --mode session -o session.jsonl

# Analyze
context-profiler analyze snapshot.json
context-profiler analyze session.jsonl --html report.html
```

See [`examples/README.md`](examples/README.md) for supported formats and conversion patterns.

## Acknowledgements

This project is inspired by and learned from:

- [context-lens](https://github.com/larsderidder/context-lens) — local proxy for capturing and visualizing LLM API calls
- [ContextFlame](https://github.com/jcgs2503/contextflame) — flamegraph-based token profiling for Claude Code
- [speedscope](https://www.speedscope.app/) — the icicle / flamegraph UI design is inspired by speedscope's interactive visualization

## License

[MIT](LICENSE)
