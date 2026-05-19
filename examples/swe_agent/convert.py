"""Convert nebius/SWE-agent-trajectories samples to OpenAI JSONL for context-profiler.

Usage:
    # Download a sample first:
    curl -sS "https://datasets-server.huggingface.co/rows?dataset=nebius%2FSWE-agent-trajectories&config=default&split=train&offset=50&length=1" \
      | python3 -c "import json,sys; print(json.dumps(json.load(sys.stdin)['rows'][0]['row']))" \
      > sample.json

    # Convert to session JSONL:
    python convert.py sample.json session.jsonl

    # Analyze with context-profiler:
    PYTHONPATH=src context-profiler analyze session.jsonl --format openai --html report.html
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def convert(input_path: Path, output_path: Path) -> None:
    with open(input_path) as f:
        data = json.load(f)

    trajectory = data["trajectory"]
    instance_id = data.get("instance_id", "unknown")

    system_prompt = ""
    messages: list[dict] = []

    for entry in trajectory:
        role = entry.get("role")
        text = entry.get("text") or ""
        sys_prompt = entry.get("system_prompt") or ""

        if role == "system":
            system_prompt = sys_prompt
        elif role == "user":
            messages.append({"role": "user", "content": text})
        elif role == "ai":
            tool_call, reasoning = _extract_tool_call(text)
            if tool_call:
                messages.append({
                    "role": "assistant",
                    "content": reasoning,
                    "tool_calls": [tool_call],
                })
            else:
                messages.append({"role": "assistant", "content": text})

    requests = _build_session_requests(system_prompt, messages, instance_id)

    with open(output_path, "w") as f:
        for req in requests:
            f.write(json.dumps(req, ensure_ascii=False) + "\n")

    print(f"Converted {instance_id}: {len(trajectory)} steps -> {len(requests)} requests")
    print(f"Written to {output_path}")


def _extract_tool_call(text: str) -> tuple[dict | None, str]:
    """Extract SWE-agent command blocks as tool calls."""
    lines = text.split("\n")
    reasoning_lines = []
    command_lines = []
    in_command = False

    for line in lines:
        if line.strip().startswith("```"):
            if in_command:
                in_command = False
            else:
                in_command = True
            continue
        if in_command:
            command_lines.append(line)
        else:
            reasoning_lines.append(line)

    if not command_lines:
        return None, text

    command = "\n".join(command_lines).strip()
    tool_name = command.split()[0] if command.split() else "bash"

    known_tools = {
        "find_file", "open_file", "goto", "scroll_down", "scroll_up",
        "create", "edit", "search_dir", "search_file", "find",
        "submit", "exit_cost",
    }
    if tool_name not in known_tools:
        tool_name = "bash"

    reasoning = "\n".join(reasoning_lines).strip()
    tool_call = {
        "id": f"call_{hash(command) % 10**8:08d}",
        "type": "function",
        "function": {
            "name": tool_name,
            "arguments": json.dumps({"command": command}),
        },
    }
    return tool_call, reasoning


def _build_session_requests(
    system_prompt: str,
    messages: list[dict],
    instance_id: str,
) -> list[dict]:
    """Build growing-context session: each assistant turn = one request snapshot."""
    requests = []
    accumulated: list[dict] = []

    for msg in messages:
        accumulated.append(msg)
        if msg["role"] == "assistant":
            req: dict = {
                "model": "swe-agent",
                "messages": [],
            }
            if system_prompt:
                req["messages"].append({"role": "system", "content": system_prompt})
            req["messages"].extend(accumulated)
            req["metadata"] = {
                "instance_id": instance_id,
                "request_index": len(requests),
            }
            requests.append(req)

    return requests


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input.json> <output.jsonl>")
        sys.exit(1)
    convert(Path(sys.argv[1]), Path(sys.argv[2]))
