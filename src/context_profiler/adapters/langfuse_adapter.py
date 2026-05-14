"""Adapter for Langfuse trace export format.

Parses a Langfuse trace JSON (exported from UI or API) and extracts analyzable
GENERATION observations as a Session of APIRequests.
"""

from __future__ import annotations

import json
from typing import Any

from context_profiler.adapters.openai_adapter import OpenAIAdapter
from context_profiler.models import APIRequest, Session


_openai = OpenAIAdapter()


def is_langfuse_trace(data: dict[str, Any]) -> bool:
    """Check if the data is a Langfuse trace export."""
    return (
        "observations" in data
        and isinstance(data["observations"], list)
        and "id" in data
        and ("projectId" in data or "name" in data)
    )


def _json_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _normalize_generation_input(inp: Any) -> dict[str, Any] | None:
    """Convert common Langfuse generation input shapes to OpenAI messages."""
    if isinstance(inp, dict):
        messages = inp.get("messages")
        if isinstance(messages, list) and messages:
            return dict(inp)

        role = inp.get("role")
        content = inp.get("content")
        if isinstance(role, str) and role and content is not None:
            normalized = {
                "messages": [
                    {
                        "role": role,
                        "content": _json_text(content),
                    }
                ]
            }
            if isinstance(inp.get("tools"), list):
                normalized["tools"] = inp["tools"]
            if isinstance(inp.get("model"), str):
                normalized["model"] = inp["model"]
            return normalized

    if isinstance(inp, str) and inp:
        return {"messages": [{"role": "user", "content": inp}]}

    return None


def parse_langfuse_trace(data: dict[str, Any]) -> Session:
    """Extract all GENERATION observations and build a Session.

    Returns a Session with one APIRequest per GENERATION, sorted by startTime.
    """
    observations = data.get("observations", [])

    generation_observations = [
        obs for obs in observations
        if obs.get("type") == "GENERATION"
    ]
    generation_observations.sort(key=lambda x: x.get("startTime", ""))

    requests: list[APIRequest] = []
    unsupported_generation_count = 0
    for gen in generation_observations:
        normalized_input = _normalize_generation_input(gen.get("input"))
        if normalized_input is None:
            unsupported_generation_count += 1
            continue

        req = _openai.parse(normalized_input)
        req.request_index = len(requests)
        req.source_format = "langfuse"
        req.model = gen.get("model") or req.model
        requests.append(req)

    warnings: list[str] = []
    if unsupported_generation_count:
        warnings.append(
            f"Skipped {unsupported_generation_count} Langfuse GENERATION observations without analyzable input messages."
        )
    if generation_observations and not requests:
        warnings.append("Langfuse trace contains GENERATION observations, but none include analyzable input messages.")
    if not generation_observations:
        warnings.append("Langfuse trace contains no GENERATION observations.")

    metadata = {
        "trace_id": data.get("id"),
        "trace_name": data.get("name"),
        "project_id": data.get("projectId"),
        "session_id": data.get("sessionId"),
        "timestamp": data.get("timestamp"),
        "total_generations": len(requests),
        "total_generation_observations": len(generation_observations),
        "unsupported_generation_observations": unsupported_generation_count,
        "total_observations": len(observations),
        "source_format": "langfuse",
        "warnings": warnings,
    }

    return Session(requests=requests, metadata=metadata)
