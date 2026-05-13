---
name: analyze-agent-context
description: Analyze LLM agent traces, loops, transcripts, Langfuse exports, OpenTelemetry spans, or raw provider requests with context-profiler. Use when the user asks to analyze/debug/explain a trace, agent run, context growth, stale context, tool bloat, or a recently fetched trace from another CLI.
---

# Analyze Agent Context

Use `context-profiler` as the trace-source agnostic context analysis harness.

## Core Rule

Do not fetch traces unless the user asks you to. If another tool already fetched trace data, use that local file or recent JSON output. The source can be Langfuse CLI, Claude Code, Cursor, OpenTelemetry, raw OpenAI/Anthropic requests, or an academic trajectory dataset.

## Workflow

1. Identify the available trace or loop data:
   - File path supplied by user.
   - Recent JSON output from another CLI, such as `langfuse-cli`.
   - Current Cursor transcript under `~/.cursor/projects/**/agent-transcripts/**/*.jsonl`.
   - Current Claude Code transcript under `~/.claude/projects/**/*.jsonl`.

2. Discover support when unsure:
   ```bash
   context-profiler formats list --json
   context-profiler formats describe <format> --json
   context-profiler schema trace --json
   ```

3. Validate before analyzing:
   ```bash
   context-profiler validate <file-or-stdin> --format auto --json
   ```

4. If validation fails:
   - Read `next_steps` and `errors[].agent_action`.
   - Use `context-profiler schema trace --json`.
   - Convert the data into `ContextTrace` yourself, then pipe it back into `context-profiler`.
   - Users should not manually reshape traces.

5. Run machine-readable diagnosis:
   ```bash
   context-profiler diagnose <file-or-stdin> --format auto --json
   ```

6. If useful, generate the existing HTML report:
   ```bash
   context-profiler analyze <file-or-stdin> --format auto --html /tmp/context-profiler-report.html
   ```

7. Explain results with scope:
   - Always mention `analysis_scope.input_kind` and `analysis_scope.confidence`.
   - For `agent-transcript`, say analysis is partial because hidden system prompts, tool definitions, rules, and provider compaction may be absent.
   - Prioritize concrete issue codes, diff hints, top tools, and artifact churn evidence.

## What To Look For

- `TOOL_USE_DOMINATES_CONTEXT`: tool inputs are the main visible context pressure.
- `TOP_TOOL_CONTEXT_HOTSPOT`: one tool accounts for a large share of visible context.
- `REPEATED_CONTENT_BLOCK`: exact or near-duplicate content is retained.
- `REPEATED_TOOL_INPUT`: large repeated tool arguments.
- `large_addition`: turn-to-turn context spike.
- `high_tool_use_addition`: spike mostly caused by tool input.
- `possible_artifact_churn`: same artifact appears across multiple tool inputs.

## Output Style

Keep the user-facing summary short:

- State what was analyzed.
- State confidence/limitations.
- List the top 2-4 findings with evidence.
- Link the HTML report path if generated.
