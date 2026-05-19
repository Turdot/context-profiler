# OpenHands (NVIDIA SWE-Zero) Example

Agent trajectory from [nvidia/SWE-Zero-openhands-trajectories](https://huggingface.co/datasets/nvidia/SWE-Zero-openhands-trajectories).

## Source

- **Dataset**: nvidia/SWE-Zero-openhands-trajectories
- **Instance**: keras-team__keras-nlp-385
- **Agent**: OpenHands
- **Messages**: 69 (34 assistant turns)

## Usage

```bash
PYTHONPATH=src uv run context-profiler analyze examples/openhands/session.jsonl --format openai --html /tmp/openhands-report.html
PYTHONPATH=src uv run context-profiler diagnose examples/openhands/session.jsonl --format openai --json
```

## Findings

- 23.9K total tokens, tool messages = 68.5% of context
- 382K tokens carried across 20 persistent blocks
- 5 artifact churn instances detected
- Very low exact redundancy (0.2%) but high structural repetition in tool results
