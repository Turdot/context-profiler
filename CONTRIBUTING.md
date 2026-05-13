# Contributing

Thanks for helping improve `context-profiler`.

## Development Setup

```bash
git clone https://github.com/Turdot/context-profiler.git
cd context-profiler
pip install -e .
```

For tests, this repository uses `pytest`:

```bash
PYTHONPATH=src uv run --with pytest pytest tests/test_smoke.py -v
```

## Project Boundaries

`context-profiler` analyzes traces; it does not fetch or replay them.

Prefer changes that improve one of these surfaces:

- supported input formats
- validation and schema guidance for agents
- context/token analyzers
- machine-readable diagnosis output
- HTML report clarity
- Agent Skills / plugin distribution

Avoid adding provider-specific fetch clients unless there is a strong reason. Agents can use existing CLIs, SDKs, or APIs to fetch traces and pass the result to `context-profiler`.

## Adding a Format

When adding a new input format:

1. Add format metadata in `src/context_profiler/formats.py`.
2. Include `input_kind`, `confidence`, `analysis_scope`, `limitations`, and `agent_conversion_guidance`.
3. Add parser/normalizer code in `src/context_profiler/adapters/` or an adjacent focused module.
4. Add a small synthetic fixture under `tests/fixtures/`.
5. Add smoke tests for `validate`, `normalize`, `diagnose`, and `analyze --html` if applicable.

## Adding an Analyzer

Analyzers should produce evidence, not overconfident claims.

Good analyzer outputs:

- stable issue codes
- token counts
- request/message indices
- tool names
- artifact keys
- confidence and limitations when heuristic

Avoid vague warnings without evidence.

## Agent Skill Changes

Canonical public skills live under `skills/`.

Do not put product-distribution skills under `.agents/skills/` or `.claude/skills/`; those paths are for project-local agent behavior.

If you add or rename skills, update:

- `.plugin/plugin.json`
- `.claude-plugin/marketplace.json`
- `README.md`
- tests covering the distribution files
