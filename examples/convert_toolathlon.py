#!/usr/bin/env python3
"""Convert a Toolathlon trajectory record into context-profiler input formats.

Toolathlon (https://huggingface.co/datasets/hkust-nlp/Toolathlon-Trajectories)
stores agent trajectories as flat message arrays. This script converts them
into the formats our profiler accepts:

  1. Snapshot  — single OpenAI request JSON (the final API call's input)
  2. Session   — JSONL of per-call snapshots (reconstructed from the flat array)

Usage:
    # Generate snapshot (last API call's full context)
    python convert_toolathlon.py toolathlon_raw.json --mode snapshot -o snapshot_output.json

    # Generate session (one snapshot per LLM call, showing context growth)
    python convert_toolathlon.py toolathlon_raw.json --mode session -o session_output.jsonl

    # Then analyze with context-profiler:
    context-profiler analyze snapshot_output.json
    context-profiler analyze session_output.jsonl --html report.html
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_toolathlon_record(data: dict) -> tuple[list[dict], list[dict]]:
    """Extract messages and tools from a raw Toolathlon record.

    Toolathlon stores these fields as JSON strings for HuggingFace compatibility,
    so we deserialize them if needed. The system prompt lives in
    config.system_prompts.agent and is prepended as a system message.
    """
    messages = data.get("messages", [])
    if isinstance(messages, str):
        messages = json.loads(messages)

    tool_calls_raw = data.get("tool_calls", {})
    if isinstance(tool_calls_raw, str):
        tool_calls_raw = json.loads(tool_calls_raw)
    tools = tool_calls_raw.get("tools", [])

    config = data.get("config", {})
    if isinstance(config, str):
        config = json.loads(config)
    system_prompt = config.get("system_prompts", {}).get("agent", "")
    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}] + messages

    return messages, tools


def to_snapshot(messages: list[dict], tools: list[dict]) -> dict:
    """Build a single OpenAI-format snapshot from the full message history.

    This represents what the final API call looked like — the complete
    accumulated context window. Field order matches Langfuse GENERATION input:
    tools first, then messages.
    """
    return {"tools": tools, "messages": messages}


def to_session(messages: list[dict], tools: list[dict]) -> list[dict]:
    """Reconstruct per-LLM-call snapshots from a flat message array.

    In an agent loop, each assistant message marks the output of one LLM call.
    The input to that call is all messages before it. By splitting on assistant
    boundaries we recover the growing context window at each step.

    Returns a list of OpenAI-format request dicts (one per LLM call).
    """
    snapshots = []
    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant":
            input_messages = messages[:i]
            snapshots.append({
                "tools": tools,
                "messages": input_messages,
            })
    return snapshots


def main():
    parser = argparse.ArgumentParser(
        description="Convert Toolathlon trajectory to context-profiler input"
    )
    parser.add_argument("input", help="Path to raw Toolathlon JSON record")
    parser.add_argument(
        "--mode",
        choices=["snapshot", "session"],
        default="snapshot",
        help="Output mode: snapshot (single request) or session (JSONL of per-call snapshots)",
    )
    parser.add_argument("-o", "--output", help="Output file path (default: stdout)")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    messages, tools = parse_toolathlon_record(data)

    if not messages:
        print("Error: no messages found in input", file=sys.stderr)
        sys.exit(1)

    if args.mode == "snapshot":
        result = to_snapshot(messages, tools)
        output_text = json.dumps(result, indent=2, ensure_ascii=False)
    else:
        snapshots = to_session(messages, tools)
        output_text = "\n".join(
            json.dumps(s, ensure_ascii=False) for s in snapshots
        )

    if args.output:
        Path(args.output).write_text(output_text + "\n", encoding="utf-8")
        n = len(snapshots) if args.mode == "session" else 1
        print(f"Wrote {args.mode} ({n} request(s)) to {args.output}")
    else:
        print(output_text)


if __name__ == "__main__":
    main()
