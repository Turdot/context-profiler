## Summary

- 

## Test Plan

- [ ] `PYTHONPATH=src uv run --with pytest pytest tests/test_smoke.py -v`
- [ ] `PYTHONPATH=src uv run context-profiler --help`

## Trace Data Safety

- [ ] This PR does not commit private traces, credentials, customer data, or production prompts.
- [ ] New fixtures are synthetic or clearly redacted.

## Format / Analyzer Checklist

If this PR adds a format:

- [ ] Format metadata was added to `src/context_profiler/formats.py`.
- [ ] Validation and/or normalization behavior is covered by tests.
- [ ] A synthetic fixture was added under `tests/fixtures/`.

If this PR adds an analyzer:

- [ ] Findings include stable issue codes and concrete evidence.
- [ ] Heuristic findings use hint language and confidence.
