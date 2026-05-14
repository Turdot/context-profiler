---
name: analyze-agent-context
description: Analyze LLM agent traces, loops, transcripts, Langfuse exports, OpenTelemetry spans, or raw provider requests with context-profiler. Use when the user asks to analyze/debug/explain a trace, agent run, context growth, stale context, tool bloat, or a recently fetched trace from another CLI.
---

# Analyze Agent Context

Use `context-profiler` as the trace-source agnostic context analysis harness.

## Core Rule

Do not fetch traces unless the user asks you to. If the user provides a Langfuse trace id and asks to inspect, debug, or analyze it, fetch that trace through the Langfuse public API with `curl`, then hand the fetched JSON to `context-profiler`. Do not use `langfuse-cli` for trace fetching because it may omit fields needed for complete trace analysis. Do not manually summarize the raw trace before running `context-profiler`.

If another tool already fetched trace data, use that local file or recent JSON output. The source can be Langfuse public API JSON, Claude Code, Cursor, OpenTelemetry, raw OpenAI/Anthropic requests, or an academic trajectory dataset.

Before analysis, verify the CLI is callable:

```bash
context-profiler --version
```

If the command is missing or its entry point is broken, ask the user to install it with `pipx install context-profiler` or `uv tool install context-profiler`.
If `which -a context-profiler` shows a stale broken executable before the `pipx`/`uv tool` executable, use the working executable path or fix `PATH` before continuing.

## Workflow

1. Identify the available trace or loop data:
   - File path supplied by user.
   - Recent JSON output from another tool.
   - Langfuse trace id supplied by user. Fetch it with `curl` from the Langfuse public API first:
     ```bash
     TRACE_ID="<trace-id>"
     HOST="${LANGFUSE_HOST%/}"
     OUT="/tmp/langfuse-trace-${TRACE_ID}"
     mkdir -p "$OUT"

     for v in LANGFUSE_HOST LANGFUSE_PUBLIC_KEY LANGFUSE_SECRET_KEY; do
       if [ -n "${!v}" ]; then echo "$v=set"; else echo "$v=missing"; fi
     done

     curl -fsS \
       -u "$LANGFUSE_PUBLIC_KEY:$LANGFUSE_SECRET_KEY" \
       "$HOST/api/public/traces/$TRACE_ID" \
       -o "$OUT/trace.json"

     curl -fsS \
       -u "$LANGFUSE_PUBLIC_KEY:$LANGFUSE_SECRET_KEY" \
       "$HOST/api/public/observations?traceId=$TRACE_ID&limit=100&page=1" \
       -o "$OUT/observations-page-1.json"

     context-profiler diagnose "$OUT/trace.json" --format langfuse --json
     ```
     Do not print or inline API keys. If any required environment variable is missing, ask the user to set it without pasting secrets into chat. For traces with more than 100 observations, paginate the observations endpoint before analysis.
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
