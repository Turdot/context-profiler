# lmcache-agentic-traces Example

Agent traces collected for KV-cache research from [sammshen/lmcache-agentic-traces](https://huggingface.co/datasets/sammshen/lmcache-agentic-traces).

## Source

- **Dataset**: sammshen/lmcache-agentic-traces
- **Session**: swebench__astropy__astropy-13033__minimax
- **Model**: minimax-m2.5 (OpenHands agent)
- **Requests**: 35 (growing context from 3 to 68 messages)

## Usage

```bash
PYTHONPATH=src uv run context-profiler analyze examples/lmcache/session.jsonl --format openai --html /tmp/lmcache-report.html
PYTHONPATH=src uv run context-profiler diagnose examples/lmcache/session.jsonl --format openai --json
```

## Findings

- 36.5K total tokens, 1.4% exact redundancy
- 403K tokens carried across 20 persistent blocks
- Context grows monotonically (no compaction observed)
- System prompt + initial task carried all 35 turns
