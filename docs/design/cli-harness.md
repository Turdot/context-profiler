# CLI Harness Design

`context-profiler` is a trace-source agnostic context analysis harness for LLM agents.

The CLI is designed for two users at once:

- humans who want a readable report
- coding agents that need discoverable commands, JSON contracts, and actionable validation errors

## Product Boundary

`context-profiler` analyzes traces. It does not fetch them.

Good upstream sources include:

- Langfuse CLI or API exports
- OpenTelemetry / OpenInference span exports
- raw OpenAI or Anthropic request logs
- Cursor or Claude Code local transcripts
- academic trajectory datasets

The agent or user brings the data. `context-profiler` validates, normalizes, diagnoses, and reports.

## Command Model

The CLI follows a discover/validate/analyze flow:

```bash
context-profiler formats list --json
context-profiler formats describe <format> --json

context-profiler schema trace --json
context-profiler schema diagnosis --json

context-profiler validate <file|-> --format auto --json
context-profiler normalize <file|-> --from auto --json

context-profiler diagnose <file|-> --format auto --json
context-profiler analyze <file|-> --format auto --html report.html
```

The style is inspired by mature agent-friendly CLIs:

- Langfuse CLI: schema-driven API access
- kubectl: resource discovery and explainability
- Terraform: validate/plan-style machine output
- GitHub CLI: human commands plus machine-readable JSON
- Repomix: AI-context oriented CLI and MCP distribution

## Input Kinds

Every format is classified by input kind and confidence.

| Input kind | Examples | Confidence |
| --- | --- | --- |
| `provider-request` | OpenAI, Anthropic | exact |
| `observability-trace` | Langfuse, OTel/OpenInference | high |
| `agent-transcript` | Cursor JSONL, Claude Code JSONL | partial |
| `benchmark-trajectory` | agent-trace, agent_trajectories, SWE-agent | dataset-dependent |

`agent-transcript` inputs are useful for visible loop analysis but are not exact raw provider requests. They may omit hidden prompts, tool definitions, rules, MCP schemas, and provider-side compaction.

## Validation Contract

Validation should be strict but helpful.

Unknown input should not be silently guessed. Instead, `validate --json` returns:

- `valid: false`
- stable error code
- expected shape
- `agent_action`
- `next_steps`

This lets Cursor, Claude Code, or another agent inspect `schema trace --json`, adapt the input into `ContextTrace`, and retry without asking the user to manually reshape data.

## Diagnosis Contract

`diagnose --json` is the primary agent-facing output.

It returns:

- `analysis_scope`: input kind, confidence, limitations
- `issues`: stable issue codes with evidence and recommendations
- `diff_summary`: turn-to-turn added/removed/retained token facts
- `diff_hints`: conservative hints such as large additions, high tool-use additions, and possible artifact churn

Issue codes should be evidence-first. Heuristic findings should use `possible_*` naming and include confidence.

## HTML Report

The HTML report is the human-facing view.

It should keep the existing visual language:

- dark, compact, monospace interface
- timeline + icicle as the main layout
- detail panel reused for selected nodes or turns
- no heavy diagnosis dashboard unless the UI direction changes intentionally

Timeline markers should reflect evidence from `context_diff`, such as large additions and high tool-use turns. Clicking a marked turn should show the diff facts in the existing detail panel.

## Future Graph Layer

The long-term model is a Context Event Graph:

- `Run`
- `Turn`
- `Span`
- `Message`
- `Artifact`
- `ContentBlock`
- `Edge`

Initial graph-like evidence should remain deterministic:

- content hash
- role and block type
- tool name
- tool call id
- artifact key
- request index

Avoid semantic stale-content claims until there is enough evidence. Prefer diff evidence plus conservative hints.
