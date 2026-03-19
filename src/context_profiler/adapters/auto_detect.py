"""Auto-detect input format and return the appropriate adapter."""

from __future__ import annotations

from typing import Any

from context_profiler.adapters.anthropic_adapter import AnthropicAdapter
from context_profiler.adapters.base import BaseAdapter
from context_profiler.adapters.openai_adapter import OpenAIAdapter

_ADAPTERS: list[BaseAdapter] = [
    AnthropicAdapter(),
    OpenAIAdapter(),
]


def detect_adapter(data: dict[str, Any]) -> BaseAdapter:
    """Try each adapter and return the first one that can handle the data.

    Anthropic is checked first because it has a more specific signature
    (content blocks with 'type' field, tools with 'input_schema').
    """
    for adapter in _ADAPTERS:
        if adapter.can_handle(data):
            return adapter
    raise ValueError(
        "Unable to detect input format. "
        "Expected OpenAI or Anthropic message format with 'messages' key."
    )
