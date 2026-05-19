# SWE-agent Trajectory Example

This example demonstrates context-profiler on a real multi-turn coding agent trajectory from [nebius/SWE-agent-trajectories](https://huggingface.co/datasets/nebius/SWE-agent-trajectories) (CC-BY-4.0).

## Source

- **Dataset**: nebius/SWE-agent-trajectories
- **Instance**: `Melevir__cognitive_complexity-15`
- **Model**: swe-agent-llama-70b
- **Steps**: 63 (31 assistant turns)
- **Task**: Fix incorrect counting for sequences of binary logical operators

## Usage

```bash
# Analyze the pre-converted session:
PYTHONPATH=src uv run context-profiler analyze examples/swe_agent/session.jsonl --format openai --html /tmp/swe-agent-report.html

# Or convert a fresh sample yourself:
python3 examples/swe_agent/convert.py examples/swe_agent/sample_raw.json /tmp/session.jsonl
PYTHONPATH=src uv run context-profiler analyze /tmp/session.jsonl --format openai --html /tmp/report.html
```

## What the profiler finds

- **26.9% content duplication** (7,277 redundant tokens out of 27.1K total)
- **Artifact churn**: `reproduce.py` read 6 times across turns
- **Context growth**: 2.1K → 27.1K over 31 requests (monotonic, no compaction)
- **Persistence**: system prompt + issue text carried all 31 turns (KV-cache friendly); file observations carried 10+ turns (compact candidates)

## Downloading more samples

```bash
# Fetch a single trajectory via HuggingFace API:
curl -sS "https://datasets-server.huggingface.co/rows?dataset=nebius%2FSWE-agent-trajectories&config=default&split=train&offset=50&length=1" \
  | python3 -c "import json,sys; print(json.dumps(json.load(sys.stdin)['rows'][0]['row']))" \
  > new_sample.json

# Convert and analyze:
python3 examples/swe_agent/convert.py new_sample.json session.jsonl
PYTHONPATH=src uv run context-profiler analyze session.jsonl --format openai --html report.html
```
