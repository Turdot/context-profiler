# Changelog

## 0.2.0 - Unreleased

### Added

- Agent-friendly CLI harness commands:
  - `formats list`
  - `formats describe`
  - `schema`
  - `validate`
  - `normalize`
  - `diagnose`
- Cursor and Claude Code transcript JSONL ingestion.
- `pagarsky/agent-trace` sample ingestion via `--format agent-trace`.
- Agent-readable diagnosis reports with input scope, confidence, limitations, issue codes, and recommendations.
- Context diff evidence:
  - turn-to-turn added/removed/retained token summary
  - large addition hints
  - high tool-use addition hints
  - possible artifact churn hints
- Tool context hotspot diagnosis:
  - `TOOL_USE_DOMINATES_CONTEXT`
  - `TOP_TOOL_CONTEXT_HOTSPOT`
- Agent Skill distribution:
  - `skills/analyze-agent-context/SKILL.md`
  - `.plugin/plugin.json`
  - `.claude-plugin/marketplace.json`
- HTML timeline markers for large additions and high tool-use turns.

### Changed

- Default analysis now includes content repeat and field repeat analyzers.
- README repositioned the project as a trace-source agnostic context analysis harness.
- Examples focus on raw provider requests, coding-agent transcripts, Langfuse exports, and multi-turn trajectory adapters.
