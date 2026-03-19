"""Adapter for Anthropic message format.

Structure: { tools: [...], system, messages: [{role, content: [{type, ...}]}], model }
Anthropic uses content blocks: text, tool_use, tool_result as typed dicts within content arrays.
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


class AnthropicAdapter(BaseAdapter):
    """Parse Anthropic Messages API request format."""

    def can_handle(self, data: dict[str, Any]) -> bool:
        if "messages" not in data:
            return False
        messages = data["messages"]
        if not isinstance(messages, list) or not messages:
            return False
        # Anthropic signature: content is a list of typed blocks
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "type" in block:
                        block_type = block["type"]
                        if block_type in ("tool_use", "tool_result"):
                            return True
        # Anthropic tools use input_schema (not parameters)
        tools = data.get("tools", [])
        if tools and isinstance(tools, list):
            t = tools[0]
            if isinstance(t, dict) and "input_schema" in t:
                return True
        return False

    def parse(self, data: dict[str, Any]) -> APIRequest:
        messages: list[Message] = []
        idx = 0

        # System prompt — can be string or list of content blocks
        system = data.get("system")
        if system:
            if isinstance(system, str):
                text = system
            elif isinstance(system, list):
                text = "\n".join(
                    b.get("text", "") for b in system
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            else:
                text = str(system)
            blocks = [ContentBlock(
                block_type=BlockType.TEXT,
                text=text,
                token_count=count_tokens(text),
            )]
            messages.append(Message(role=Role.SYSTEM, blocks=blocks, index=idx))
            idx += 1

        for raw_msg in data.get("messages", []):
            role = Role.USER if raw_msg.get("role") == "user" else Role.ASSISTANT
            blocks = self._extract_blocks(raw_msg.get("content", []))
            messages.append(Message(role=role, blocks=blocks, index=idx))
            idx += 1

        tools = self._parse_tools(data.get("tools", []))
        model = data.get("model", "unknown")

        return APIRequest(
            messages=messages,
            tools=tools,
            model=model,
            source_format="anthropic",
        )

    def _parse_tools(self, raw_tools: list[dict]) -> list[ToolDefinition]:
        result = []
        for tool in raw_tools:
            name = tool.get("name", "unknown")
            raw_json = json.dumps(tool, ensure_ascii=False)
            result.append(ToolDefinition(
                name=name,
                raw_json=raw_json,
                token_count=count_tokens(raw_json),
            ))
        return result

    def _extract_blocks(self, content: Any) -> list[ContentBlock]:
        blocks: list[ContentBlock] = []

        if isinstance(content, str):
            blocks.append(ContentBlock(
                block_type=BlockType.TEXT,
                text=content,
                token_count=count_tokens(content),
            ))
            return blocks

        if not isinstance(content, list):
            return blocks

        for item in content:
            if isinstance(item, str):
                blocks.append(ContentBlock(
                    block_type=BlockType.TEXT,
                    text=item,
                    token_count=count_tokens(item),
                ))
                continue

            if not isinstance(item, dict):
                continue

            block_type = item.get("type", "text")

            if block_type == "text":
                text = item.get("text", "")
                blocks.append(ContentBlock(
                    block_type=BlockType.TEXT,
                    text=text,
                    token_count=count_tokens(text),
                ))

            elif block_type == "tool_use":
                tool_input = item.get("input", {})
                text = json.dumps(tool_input, ensure_ascii=False)
                blocks.append(ContentBlock(
                    block_type=BlockType.TOOL_USE,
                    text=text,
                    token_count=count_tokens(text),
                    tool_name=item.get("name"),
                    tool_call_id=item.get("id"),
                    tool_input=tool_input,
                ))

            elif block_type == "tool_result":
                result_content = item.get("content", "")
                if isinstance(result_content, list):
                    text = "\n".join(
                        b.get("text", "") for b in result_content
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                elif isinstance(result_content, str):
                    text = result_content
                else:
                    text = json.dumps(result_content, ensure_ascii=False)

                blocks.append(ContentBlock(
                    block_type=BlockType.TOOL_RESULT,
                    text=text,
                    token_count=count_tokens(text),
                    tool_name=None,
                    tool_call_id=item.get("tool_use_id"),
                ))

        return blocks
