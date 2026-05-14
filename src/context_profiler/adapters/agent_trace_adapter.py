"""Adapter for pagarsky/agent-trace style academic trajectories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from context_profiler.models import APIRequest, BlockType, ContentBlock, Message, Role, Session
from context_profiler.token_utils import count_tokens


def is_agent_trace(data: dict[str, Any]) -> bool:
    return (
        ("llm_steps" in data or "llm_steps_json" in data)
        and ("spans" in data or "spans_json" in data)
    )


def load_agent_trace_session(path: Path) -> Session:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return parse_agent_trace(data, source=str(path))


def parse_agent_trace(data: dict[str, Any], source: str = "") -> Session:
    llm_steps = _json_field(data, "llm_steps")
    spans = _json_field(data, "spans")
    metadata = _json_field(data, "metadata") if ("metadata" in data or "metadata_json" in data) else {}

    messages: list[Message] = []
    requests: list[APIRequest] = []

    prompt = data.get("prompt")
    if prompt:
        messages.append(Message(
            role=Role.USER,
            blocks=[_text_block(prompt)],
            index=len(messages),
        ))

    span_idx = 0
    for step in llm_steps:
        assistant_blocks = _assistant_blocks(step)
        if assistant_blocks:
            messages.append(Message(role=Role.ASSISTANT, blocks=assistant_blocks, index=len(messages)))

        tool_calls = step.get("tool_calls") or []
        for call in tool_calls:
            span = spans[span_idx] if span_idx < len(spans) else None
            if span is not None:
                span_idx += 1
                messages.append(_tool_result_message(span, len(messages)))

        requests.append(_snapshot(messages, len(requests), data))

    return Session(
        requests=requests,
        metadata={
            "source": source,
            "source_format": "agent-trace",
            "trace_id": data.get("trace_id"),
            "dataset_name": data.get("dataset_name") or metadata.get("dataset_name"),
            "task_id": data.get("task_id") or metadata.get("task_id"),
            "model": data.get("model") or metadata.get("model_family"),
            "llm_step_count": len(llm_steps),
            "tool_span_count": len(spans),
        },
    )


def _json_field(data: dict[str, Any], name: str) -> Any:
    if name in data:
        return data[name]
    raw = data.get(f"{name}_json")
    if raw is None:
        return [] if name in {"llm_steps", "spans"} else {}
    return json.loads(raw)


def _assistant_blocks(step: dict[str, Any]) -> list[ContentBlock]:
    blocks = []
    reasoning = step.get("reasoning_content")
    model_output = step.get("model_output")
    text_parts = [part for part in (reasoning, model_output) if part]
    if text_parts:
        blocks.append(_text_block("\n\n".join(text_parts)))

    for idx, call in enumerate(step.get("tool_calls") or []):
        name = call.get("name") or "unknown"
        arguments = call.get("arguments") or {}
        text = json.dumps(arguments, ensure_ascii=False)
        blocks.append(ContentBlock(
            block_type=BlockType.TOOL_USE,
            text=text,
            token_count=count_tokens(text),
            tool_name=name,
            tool_call_id=f"{step.get('step_id', 'step')}:tool:{idx}",
            tool_input=arguments if isinstance(arguments, dict) else {"value": arguments},
        ))
    return blocks


def _tool_result_message(span: dict[str, Any], index: int) -> Message:
    output = span.get("tool_output")
    if not isinstance(output, str):
        output = json.dumps(output, ensure_ascii=False)
    block = ContentBlock(
        block_type=BlockType.TOOL_RESULT,
        text=output or "",
        token_count=count_tokens(output or ""),
        tool_name=span.get("tool_name"),
        tool_call_id=span.get("span_id"),
    )
    return Message(role=Role.TOOL, blocks=[block], index=index)


def _text_block(text: str) -> ContentBlock:
    return ContentBlock(
        block_type=BlockType.TEXT,
        text=text,
        token_count=count_tokens(text),
    )


def _snapshot(messages: list[Message], request_index: int, data: dict[str, Any]) -> APIRequest:
    return APIRequest(
        messages=list(messages),
        tools=[],
        model=data.get("model", "unknown"),
        request_index=request_index,
        source_format="agent-trace",
        raw_input={
            "model": data.get("model", "unknown"),
            "messages": [_message_to_raw(message) for message in messages],
            "tools": [],
            "source_format": "agent-trace",
        },
    )


def _message_to_raw(message: Message) -> dict[str, Any]:
    content = []
    tool_calls = []
    for block in message.blocks:
        if block.block_type == BlockType.TEXT:
            content.append({"type": "text", "text": block.text})
        elif block.block_type == BlockType.TOOL_USE:
            tool_calls.append({
                "id": block.tool_call_id,
                "type": "function",
                "function": {
                    "name": block.tool_name or "unknown",
                    "arguments": block.text,
                },
            })
        elif block.block_type == BlockType.TOOL_RESULT:
            content.append({
                "type": "tool_result",
                "tool_use_id": block.tool_call_id,
                "content": block.text,
            })

    raw: dict[str, Any] = {"role": message.role.value, "content": content}
    if len(content) == 1 and content[0].get("type") == "text":
        raw["content"] = content[0]["text"]
    if tool_calls:
        raw["tool_calls"] = tool_calls
    return raw
