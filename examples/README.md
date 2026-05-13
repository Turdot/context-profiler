# Examples

Runnable examples for context analysis harness workflows.

The easiest way to explore the current fixtures is from the repository root:

```bash
PYTHONPATH=src uv run context-profiler formats list --json
```

## Raw Provider Request

Use the smoke-test OpenAI-style request fixture:

```bash
PYTHONPATH=src uv run context-profiler diagnose tests/fixtures/repeated_tool_calls.json --format openai --json
PYTHONPATH=src uv run context-profiler analyze tests/fixtures/repeated_tool_calls.json --format openai --html /tmp/context-profiler-openai.html
```

Expected findings:

- `TOOL_USE_DOMINATES_CONTEXT`
- `TOP_TOOL_CONTEXT_HOTSPOT`
- `REPEATED_CONTENT_BLOCK`
- Large repeated `generate_canvas_component.requirements` tool inputs

## Cursor Transcript

Use the synthetic Cursor transcript fixture:

```bash
PYTHONPATH=src uv run context-profiler diagnose tests/fixtures/cursor_transcript.jsonl --format cursor-jsonl --json
PYTHONPATH=src uv run context-profiler analyze tests/fixtures/cursor_transcript.jsonl --format cursor-jsonl --html /tmp/context-profiler-cursor.html
```

This exercises the `agent-transcript` path. The analysis is partial because transcripts may omit hidden prompts, tool definitions, rules, and provider compaction.

## Claude Code Transcript

Use the synthetic Claude Code transcript fixture:

```bash
PYTHONPATH=src uv run context-profiler diagnose tests/fixtures/claude_code_transcript.jsonl --format claude-code-jsonl --json
PYTHONPATH=src uv run context-profiler analyze tests/fixtures/claude_code_transcript.jsonl --format claude-code-jsonl --html /tmp/context-profiler-claude.html
```

## Artifact Churn

Use the artifact churn fixture to see turn-to-turn diff hints:

```bash
PYTHONPATH=src uv run context-profiler diagnose tests/fixtures/artifact_churn_transcript.jsonl --format cursor-jsonl --json
```

Expected hint:

```json
{
  "type": "possible_artifact_churn",
  "evidence": {
    "artifact_key": "src/Button.tsx"
  }
}
```

## Langfuse Export

If another tool has already fetched a Langfuse trace export, pass it directly:

```bash
context-profiler validate trace.json --format langfuse --json
context-profiler diagnose trace.json --format langfuse --json
context-profiler analyze trace.json --format langfuse --html /tmp/context-profiler-langfuse.html
```

`context-profiler` does not fetch Langfuse data. Use Langfuse's own CLI/API or any other tool to obtain the trace first.

## Adapting Other Formats

Agents should prefer the built-in format registry and schema:

```bash
context-profiler formats describe agent-trace --json
context-profiler schema trace --json
```

Good future public trajectory demos:

- `pagarsky/agent-trace`: `llm_steps[]` plus tool `spans[]`.
- `cx-cmu/agent_trajectories`: large multi-turn trajectories across several benchmarks.
- SWE-agent trajectories: coding-agent `thought/action/observation` loops and LM `query` messages.
