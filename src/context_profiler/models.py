"""Canonical data models for context profiling.

All input adapters normalize into these types. Analyzers only work with these.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class BlockType(Enum):
    TEXT = "text"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"


class Role(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ContentBlock:
    """One piece of content within a message."""

    block_type: BlockType
    text: str
    token_count: int = 0

    tool_name: str | None = None
    tool_call_id: str | None = None
    tool_input: dict[str, Any] | None = None


@dataclass
class ToolDefinition:
    """A tool/function definition from the tools array."""

    name: str
    raw_json: str
    token_count: int = 0


@dataclass
class Message:
    """One element in the messages array."""

    role: Role
    blocks: list[ContentBlock]
    index: int = 0

    @property
    def total_tokens(self) -> int:
        return sum(b.token_count for b in self.blocks)

    @property
    def text_content(self) -> str:
        return "\n".join(b.text for b in self.blocks if b.text)


@dataclass
class APIRequest:
    """One LLM API call — the primary unit of analysis.

    In snapshot mode we analyze a single request.
    In session mode we analyze a sequence of requests.
    """

    messages: list[Message]
    tools: list[ToolDefinition] = field(default_factory=list)
    model: str = "unknown"
    request_index: int = 0
    trace_index: int = 0
    source_format: str = "unknown"
    raw_input: dict[str, Any] | None = None

    @property
    def total_input_tokens(self) -> int:
        msg_tokens = sum(m.total_tokens for m in self.messages)
        tool_def_tokens = sum(t.token_count for t in self.tools)
        return msg_tokens + tool_def_tokens

    @property
    def tool_definition_tokens(self) -> int:
        return sum(t.token_count for t in self.tools)

    @property
    def system_prompt_tokens(self) -> int:
        return sum(
            m.total_tokens for m in self.messages if m.role == Role.SYSTEM
        )


@dataclass
class Session:
    """An ordered list of API requests from a single agent task execution."""

    requests: list[APIRequest]
    metadata: dict[str, Any] = field(default_factory=dict)
