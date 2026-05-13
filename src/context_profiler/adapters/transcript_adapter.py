"""Adapters for local coding-agent transcript JSONL files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from context_profiler.models import APIRequest, BlockType, ContentBlock, Message, Role, Session
from context_profiler.token_utils import count_tokens

TRANSCRIPT_FORMATS = {"cursor-jsonl", "claude-code-jsonl"}


def is_cursor_transcript_event(event: dict[str, Any]) -> bool:
    return "role" in event and isinstance(event.get("message"), dict)


def is_claude_code_transcript_event(event: dict[str, Any]) -> bool:
    return event.get("type") in {"user", "assistant"} and isinstance(event.get("message"), dict)


def looks_like_transcript_event(event: dict[str, Any]) -> bool:
    return is_cursor_transcript_event(event) or is_claude_code_transcript_event(event)


def load_transcript_session(path: Path, source_format: str) -> Session:
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return parse_transcript_events(events, source_format=source_format, source=str(path))


def parse_transcript_events(
    events: list[dict[str, Any]],
    source_format: str,
    source: str = "",
) -> Session:
    messages: list[Message] = []
    requests: list[APIRequest] = []

    for event in events:
        msg = _message_from_event(event, index=len(messages))
        if msg is None:
            continue
        messages.append(msg)
        requests.append(_request_snapshot(messages, len(requests), source_format))

    return Session(
        requests=requests,
        metadata={
            "source": source,
            "source_format": source_format,
            "event_count": len(events),
            "message_count": len(messages),
        },
    )


def _request_snapshot(
    messages: list[Message],
    request_index: int,
    source_format: str,
) -> APIRequest:
    raw_messages = [_message_to_raw(m) for m in messages]
    return APIRequest(
        messages=list(messages),
        tools=[],
        model="unknown",
        request_index=request_index,
        source_format=source_format,
        raw_input={"messages": raw_messages, "tools": [], "source_format": source_format},
    )


def _message_from_event(event: dict[str, Any], index: int) -> Message | None:
    payload = event.get("message")
    if not isinstance(payload, dict):
        return None

    role = _parse_role(event.get("role") or payload.get("role") or event.get("type"))
    blocks = _blocks_from_content(payload.get("content"))
    if not blocks:
        return None

    if all(block.block_type == BlockType.TOOL_RESULT for block in blocks):
        role = Role.TOOL

    return Message(role=role, blocks=blocks, index=index)


def _parse_role(role: str | None) -> Role:
    if role == "assistant":
        return Role.ASSISTANT
    if role == "system":
        return Role.SYSTEM
    if role == "tool":
        return Role.TOOL
    return Role.USER


def _blocks_from_content(content: Any) -> list[ContentBlock]:
    if isinstance(content, str):
        return [_text_block(content)] if content else []

    if not isinstance(content, list):
        return []

    blocks: list[ContentBlock] = []
    for item in content:
        if isinstance(item, str):
            blocks.append(_text_block(item))
            continue
        if not isinstance(item, dict):
            continue

        item_type = item.get("type")
        if item_type == "text":
            blocks.append(_text_block(item.get("text", "")))
        elif item_type == "thinking":
            blocks.append(_text_block(item.get("thinking") or item.get("text", "")))
        elif item_type == "tool_use":
            tool_input = item.get("input", {})
            text = json.dumps(tool_input, ensure_ascii=False)
            blocks.append(ContentBlock(
                block_type=BlockType.TOOL_USE,
                text=text,
                token_count=count_tokens(text),
                tool_name=item.get("name"),
                tool_call_id=item.get("id"),
                tool_input=tool_input if isinstance(tool_input, dict) else {"value": tool_input},
            ))
        elif item_type == "tool_result":
            text = _tool_result_text(item.get("content", ""))
            blocks.append(ContentBlock(
                block_type=BlockType.TOOL_RESULT,
                text=text,
                token_count=count_tokens(text),
                tool_call_id=item.get("tool_use_id"),
            ))

    return [block for block in blocks if block.text or block.block_type != BlockType.TEXT]


def _text_block(text: str) -> ContentBlock:
    return ContentBlock(
        block_type=BlockType.TEXT,
        text=text,
        token_count=count_tokens(text),
    )


def _tool_result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return json.dumps(content, ensure_ascii=False)


def _message_to_raw(message: Message) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "role": message.role.value,
        "content": [],
    }
    tool_calls = []

    for block in message.blocks:
        if block.block_type == BlockType.TEXT:
            raw["content"].append({"type": "text", "text": block.text})
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
            raw["content"].append({
                "type": "tool_result",
                "tool_use_id": block.tool_call_id,
                "content": block.text,
            })

    if tool_calls:
        raw["tool_calls"] = tool_calls
    if len(raw["content"]) == 1 and raw["content"][0].get("type") == "text":
        raw["content"] = raw["content"][0]["text"]
    return raw
