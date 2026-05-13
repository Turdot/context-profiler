# Roadmap

This roadmap focuses on making `context-profiler` a top-tier open source context analysis harness for LLM agents.

## Current Focus

### 1. Context Diff Engine

Build reliable turn-to-turn evidence before making strong stale-content claims.

Current and near-term outputs:

- added tokens
- removed tokens
- retained tokens
- top added blocks
- top removed blocks
- top tool additions
- artifact keys
- possible artifact churn

### 2. Tool Context Diagnosis

Make tool-driven context pressure obvious.

Current issue codes:

- `TOOL_USE_DOMINATES_CONTEXT`
- `TOP_TOOL_CONTEXT_HOTSPOT`
- `REPEATED_CONTENT_BLOCK`
- `REPEATED_TOOL_INPUT`

Near-term improvements:

- distinguish tool input vs tool result pressure
- suppress low-value repeated-field findings
- improve artifact key extraction
- add stronger evidence for repeated modification loops

### 3. Agent Skill Distribution

Ship a complete `analyze-agent-context` skill for Cursor, Claude Code, and Open Plugins compatible tools.

The skill should teach agents:

- do not fetch traces unless asked
- validate any trace/loop/transcript before analysis
- use `diagnose --json` for machine-readable findings
- generate HTML only when useful for the user
- explain confidence and limitations

## Planned Format Support

### Observability Traces

- OpenTelemetry / OpenInference spans
- LangSmith run trees
- richer Langfuse observation exports

### Coding Agent Trajectories

- SWE-agent `.traj` files
- mini-SWE-agent output files
- richer Claude Code subagent linkage

### Academic Multi-Turn Datasets

Prefer datasets with real turn-to-turn evolution:

- `pagarsky/agent-trace`
- `cx-cmu/agent_trajectories`
- SWE-agent trajectories

Toolathlon is not a priority first-class format because its multi-turn structure is less natural for context evolution analysis.

## Later

### MCP Server

Expose context-profiler as MCP tools after the CLI and skill workflow are stable.

Candidate tools:

- `validate_trace`
- `diagnose_trace`
- `generate_html_report`
- `describe_format`
- `get_schema`

### Context Event Graph

Evolve deterministic diff evidence into a graph model:

- repeated content edges
- artifact modification chains
- superseded context hints
- orphaned context hints
- subagent leakage hints

### Release Automation

Use GitHub Releases and PyPI Trusted Publishing for versioned releases.

Release checklist:

1. update `pyproject.toml`
2. update `CHANGELOG.md`
3. run smoke tests
4. build package
5. tag release
6. publish through CI
