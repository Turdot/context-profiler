"""Adapter for Langfuse trace export format.

Parses a Langfuse trace JSON (exported from UI or API) and extracts all
GENERATION observations as a Session of APIRequests.

Each GENERATION observation contains an 'input' dict with {tools, messages}
in OpenAI format, which we delegate to the OpenAI adapter for parsing.
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


def parse_langfuse_trace(data: dict[str, Any]) -> Session:
    """Extract all GENERATION observations and build a Session.

    Returns a Session with one APIRequest per GENERATION, sorted by startTime.
    """
    observations = data.get("observations", [])

    generations = [
        obs for obs in observations
        if obs.get("type") == "GENERATION"
        and isinstance(obs.get("input"), dict)
        and "messages" in obs["input"]
    ]

    generations.sort(key=lambda x: x.get("startTime", ""))

    requests: list[APIRequest] = []
    for idx, gen in enumerate(generations):
        inp = gen["input"]
        req = _openai.parse(inp)
        req.request_index = idx
        req.model = gen.get("model", req.model)
        requests.append(req)

    metadata = {
        "trace_id": data.get("id"),
        "trace_name": data.get("name"),
        "project_id": data.get("projectId"),
        "session_id": data.get("sessionId"),
        "timestamp": data.get("timestamp"),
        "total_generations": len(generations),
        "total_observations": len(observations),
        "source_format": "langfuse",
    }

    return Session(requests=requests, metadata=metadata)
