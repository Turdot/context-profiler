"""Validation and canonical normalization helpers for CLI commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from context_profiler.adapters.auto_detect import detect_adapter
from context_profiler.adapters.agent_trace_adapter import is_agent_trace
from context_profiler.adapters.langfuse_adapter import is_langfuse_trace, parse_langfuse_trace
from context_profiler.adapters.transcript_adapter import TRANSCRIPT_FORMATS
from context_profiler.models import APIRequest, Session
from context_profiler.profiler import load_session


NEXT_STEPS = [
    "Run: context-profiler formats list --json",
    "Run: context-profiler schema trace --json",
    "If this is a custom trace, have the agent normalize it to ContextTrace and pipe it to context-profiler diagnose - --format context-trace --json",
]


def _normalize_format_hint(format_hint: str | None) -> str | None:
    return None if format_hint in (None, "auto") else format_hint


def _adapter_format_name(adapter: object) -> str:
    name = adapter.__class__.__name__
    if name == "OpenAIAdapter":
        return "openai"
    if name == "AnthropicAdapter":
        return "anthropic"
    return name.removesuffix("Adapter").lower()


def load_json_input(path: str) -> Any:
    if path == "-":
        import sys

        return json.loads(sys.stdin.read())
    with open(Path(path), encoding="utf-8") as f:
        return json.load(f)


def validate_input(path: str, format_hint: str | None = None) -> dict[str, Any]:
    format_hint = _normalize_format_hint(format_hint)
    try:
        if path != "-" and Path(path).suffix == ".jsonl":
            session = load_session(Path(path), format_hint=format_hint)
            detected_format = format_hint or session.metadata.get("source_format") or "jsonl"
            return _valid_result(detected_format)

        data = load_json_input(path)
        if isinstance(data, dict) and data.get("schema_version") == "0.1" and "runs" in data:
            return _valid_result("context-trace")
        if isinstance(data, dict) and is_langfuse_trace(data):
            session = parse_langfuse_trace(data)
            if not session.requests:
                return _invalid_langfuse_result(session)
            return _valid_result("langfuse")
        if isinstance(data, dict) and is_agent_trace(data):
            return _valid_result("agent-trace")
        if isinstance(data, dict):
            adapter = detect_adapter(data) if format_hint is None else None
            return _valid_result(format_hint or _adapter_format_name(adapter))
        return _invalid_result(
            code="UNSUPPORTED_SHAPE",
            message="Input must be a JSON object or supported JSONL transcript.",
        )
    except Exception as exc:
        return _invalid_result(
            code="UNSUPPORTED_SHAPE",
            message=str(exc),
        )


def _valid_result(detected_format: str) -> dict[str, Any]:
    return {
        "valid": True,
        "detected_format": detected_format,
        "errors": [],
        "next_steps": [],
    }


def _invalid_result(code: str, message: str) -> dict[str, Any]:
    return {
        "valid": False,
        "detected_format": "unknown",
        "errors": [
            {
                "code": code,
                "message": message,
                "expected": "OpenAI/Anthropic request, Langfuse trace, supported agent transcript JSONL, or ContextTrace.",
                "agent_action": "Inspect formats/schema and convert the input into ContextTrace if no supported format matches.",
            }
        ],
        "next_steps": NEXT_STEPS,
    }


def _invalid_langfuse_result(session: Session) -> dict[str, Any]:
    warnings = session.metadata.get("warnings") or [
        "Langfuse trace did not contain analyzable generation input messages."
    ]
    return {
        "valid": False,
        "detected_format": "langfuse",
        "errors": [
            {
                "code": "NO_ANALYZABLE_GENERATIONS",
                "message": "Langfuse trace was recognized, but no generation input messages could be analyzed.",
                "expected": "Langfuse GENERATION observations with input.messages, input {role, content}, or string input.",
                "agent_action": "Fetch full Langfuse trace observations or normalize generation inputs into ContextTrace before running diagnose.",
                "warnings": warnings,
            }
        ],
        "next_steps": NEXT_STEPS,
    }


def request_to_event(request: APIRequest) -> dict[str, Any]:
    return {
        "event_type": "llm_request",
        "request_index": request.request_index,
        "trace_index": request.trace_index,
        "source_format": request.source_format,
        "model": request.model,
        "total_input_tokens": request.total_input_tokens,
        "message_count": len(request.messages),
        "tool_count": len(request.tools),
    }


def normalize_input(path: str, format_hint: str | None = None) -> dict[str, Any]:
    format_hint = _normalize_format_hint(format_hint)
    if path != "-" and (Path(path).suffix == ".jsonl" or format_hint in TRANSCRIPT_FORMATS):
        session = load_session(Path(path), format_hint=format_hint)
        return _session_to_context_trace(session)

    data = load_json_input(path)
    if isinstance(data, dict) and data.get("schema_version") == "0.1" and "runs" in data:
        return data

    if isinstance(data, dict) and is_langfuse_trace(data):
        session = parse_langfuse_trace(data)
    elif isinstance(data, dict) and is_agent_trace(data):
        session = load_session(Path(path), format_hint="agent-trace")
    elif path != "-":
        session = load_session(Path(path), format_hint=format_hint)
    elif isinstance(data, dict):
        adapter = detect_adapter(data)
        session = Session(requests=[adapter.parse(data)])
    else:
        raise ValueError("Input must be a JSON object")

    return _session_to_context_trace(session)


def _session_to_context_trace(session: Session) -> dict[str, Any]:
    run_id = session.metadata.get("trace_id") or "run-0"
    source_format = session.metadata.get("source_format")
    if source_format is None and session.requests:
        source_format = session.requests[0].source_format

    return {
        "schema_version": "0.1",
        "runs": [
            {
                "run_id": run_id,
                "source_format": source_format or "unknown",
                "metadata": session.metadata,
                "events": [request_to_event(req) for req in session.requests],
            }
        ],
    }
