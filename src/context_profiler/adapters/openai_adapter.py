"""Adapter for OpenAI-compatible message format.

Handles the format used by OpenAI, Azure OpenAI, and many compatible APIs.
Structure: { tools: [...], messages: [{role, content, tool_calls, tool_call_id}], model }
"""

from __future__ import annotations

import json
from typing import Any

from context_profiler.adapters.base import BaseAdapter
from context_profiler.models import (
    APIRequest,
    BlockType,
    ContentBlock,
    Message,
    Role,
    ToolDefinition,
)
from context_profiler.token_utils import count_tokens


def _parse_role(role_str: str) -> Role:
    mapping = {
        "system": Role.SYSTEM,
        "user": Role.USER,
        "assistant": Role.ASSISTANT,
        "tool": Role.TOOL,
    }
    return mapping.get(role_str, Role.USER)


class OpenAIAdapter(BaseAdapter):
    """Parse OpenAI chat completion request format."""

    def can_handle(self, data: dict[str, Any]) -> bool:
        if "messages" not in data:
            return False
        messages = data["messages"]
        if not isinstance(messages, list) or not messages:
            return False
        first = messages[0]
        # OpenAI format: messages have 'role' and 'content' as top-level keys
        # tool_calls is a list of {id, type, function: {name, arguments}}
        if "role" not in first:
            return False
        # Distinguish from Anthropic: OpenAI tool_calls use 'function' key
        if "tools" in data and isinstance(data["tools"], list) and data["tools"]:
            tool = data["tools"][0]
            if isinstance(tool, dict) and "function" in tool:
                return True
        # Also handle when no tools defined but messages use OpenAI style
        # (content is string, not list of content blocks)
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, str):
                return True
            if msg.get("tool_calls"):
                tc = msg["tool_calls"][0]
                if isinstance(tc, dict) and "function" in tc:
                    return True
        return True

    def parse(self, data: dict[str, Any]) -> APIRequest:
        tools = self._parse_tools(data.get("tools", []))
        messages = self._parse_messages(data.get("messages", []))
        model = data.get("model", "unknown")

        return APIRequest(
            messages=messages,
            tools=tools,
            model=model,
            source_format="openai",
            raw_input=data,
        )

    def _parse_tools(self, raw_tools: list[dict]) -> list[ToolDefinition]:
        result = []
        for tool in raw_tools:
            func = tool.get("function", tool)
            name = func.get("name", "unknown")
            raw_json = json.dumps(tool, ensure_ascii=False)
            token_count = count_tokens(raw_json)
            result.append(ToolDefinition(
                name=name,
                raw_json=raw_json,
                token_count=token_count,
            ))
        return result

    def _parse_messages(self, raw_messages: list[dict]) -> list[Message]:
        result = []
        for idx, msg in enumerate(raw_messages):
            role = _parse_role(msg.get("role", "user"))
            blocks = self._extract_blocks(msg)
            result.append(Message(role=role, blocks=blocks, index=idx))
        return result

    def _extract_blocks(self, msg: dict) -> list[ContentBlock]:
        blocks: list[ContentBlock] = []
        content = msg.get("content")
        tool_calls = msg.get("tool_calls", [])
        tool_call_id = msg.get("tool_call_id")

        # Text content
        if isinstance(content, str) and content:
            blocks.append(ContentBlock(
                block_type=BlockType.TEXT,
                text=content,
                token_count=count_tokens(content),
            ))
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        text = part.get("text", "")
                        blocks.append(ContentBlock(
                            block_type=BlockType.TEXT,
                            text=text,
                            token_count=count_tokens(text),
                        ))

        # Tool result (role=tool in OpenAI format)
        if tool_call_id and content:
            text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            if not blocks:
                blocks.append(ContentBlock(
                    block_type=BlockType.TOOL_RESULT,
                    text=text,
                    token_count=count_tokens(text),
                    tool_call_id=tool_call_id,
                    tool_name=msg.get("name"),
                ))
            else:
                blocks[0].block_type = BlockType.TOOL_RESULT
                blocks[0].tool_call_id = tool_call_id
                blocks[0].tool_name = msg.get("name")

        # Tool calls from assistant
        for tc in tool_calls or []:
            func = tc.get("function", {})
            name = func.get("name", "unknown")
            args_str = func.get("arguments", "{}")
            try:
                args_dict = json.loads(args_str) if isinstance(args_str, str) else args_str
            except json.JSONDecodeError:
                args_dict = {"_raw": args_str}

            args_text = args_str if isinstance(args_str, str) else json.dumps(args_str, ensure_ascii=False)

            blocks.append(ContentBlock(
                block_type=BlockType.TOOL_USE,
                text=args_text,
                token_count=count_tokens(args_text),
                tool_name=name,
                tool_call_id=tc.get("id"),
                tool_input=args_dict,
            ))

        return blocks
