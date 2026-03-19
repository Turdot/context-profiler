"""Profiler orchestrator — runs all analyzers and aggregates results."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from context_profiler.adapters.auto_detect import detect_adapter
from context_profiler.adapters.langfuse_adapter import is_langfuse_trace, parse_langfuse_trace
from context_profiler.analyzers.base import AnalyzerResult, BaseAnalyzer
from context_profiler.analyzers.token_counter import TokenCounterAnalyzer
from context_profiler.models import APIRequest, Session

ALL_ANALYZERS: list[BaseAnalyzer] = [
    TokenCounterAnalyzer(),
]


@dataclass
class ProfileResult:
    """Aggregated result from profiling a request or session."""

    source: str
    mode: str  # "snapshot" or "session"
    analyzer_results: dict[str, AnalyzerResult] = field(default_factory=dict)
    session_timeline: list[dict[str, Any]] | None = None
    all_warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "source": self.source,
            "mode": self.mode,
            "analyzers": {},
            "warnings": self.all_warnings,
        }
        for name, ar in self.analyzer_results.items():
            result["analyzers"][name] = {
                "summary": ar.summary,
                "details": ar.details,
                "warnings": ar.warnings,
            }
        if self.session_timeline:
            result["session_timeline"] = self.session_timeline
        return result


def load_request(path: Path, format_hint: str | None = None) -> APIRequest:
    """Load and parse a single API request JSON file."""
    with open(path) as f:
        data = json.load(f)

    if format_hint:
        from context_profiler.adapters.openai_adapter import OpenAIAdapter
        from context_profiler.adapters.anthropic_adapter import AnthropicAdapter

        adapters = {"openai": OpenAIAdapter(), "anthropic": AnthropicAdapter()}
        adapter = adapters.get(format_hint)
        if adapter is None:
            raise ValueError(f"Unknown format: {format_hint}")
    else:
        adapter = detect_adapter(data)

    return adapter.parse(data)


def load_langfuse_trace(path: Path) -> Session:
    """Load a Langfuse trace JSON and extract all generations as a Session."""
    with open(path) as f:
        data = json.load(f)
    return parse_langfuse_trace(data)


def try_load_langfuse(path: Path) -> Session | None:
    """Try to load as Langfuse trace. Returns None if not a Langfuse format."""
    if not path.is_file() or path.suffix not in (".json",):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        if is_langfuse_trace(data):
            return parse_langfuse_trace(data)
    except (json.JSONDecodeError, KeyError):
        pass
    return None


def load_multi_trace_session(paths: list[Path], format_hint: str | None = None) -> Session:
    """Load multiple Langfuse trace files and merge into a single Session.

    Traces are sorted by timestamp, requests are re-indexed sequentially,
    and each request is tagged with a trace_index indicating its origin turn.
    """
    trace_sessions: list[tuple[str, Session]] = []
    for p in paths:
        session = load_session(p, format_hint=format_hint)
        timestamp = session.metadata.get("timestamp", "")
        trace_sessions.append((timestamp, session))

    # Sort traces by timestamp
    trace_sessions.sort(key=lambda x: x[0])

    merged_requests = []
    turn_boundaries: list[int] = []  # request_index where each new turn starts
    global_idx = 0

    for trace_idx, (_ts, session) in enumerate(trace_sessions):
        if session.requests:
            turn_boundaries.append(global_idx)
        for req in session.requests:
            req.request_index = global_idx
            req.trace_index = trace_idx
            merged_requests.append(req)
            global_idx += 1

    metadata = {
        "source_format": "langfuse",
        "num_traces": len(trace_sessions),
        "turn_boundaries": turn_boundaries,
    }
    # Inherit session_id from first trace if available
    if trace_sessions:
        first_meta = trace_sessions[0][1].metadata
        metadata["session_id"] = first_meta.get("session_id")

    return Session(requests=merged_requests, metadata=metadata)


def load_session(path: Path, format_hint: str | None = None) -> Session:
    """Load a session from a JSONL file, directory, or Langfuse trace."""
    if format_hint == "langfuse":
        return load_langfuse_trace(path)

    langfuse_session = try_load_langfuse(path)
    if langfuse_session is not None:
        return langfuse_session

    requests: list[APIRequest] = []

    if path.is_dir():
        files = sorted(path.glob("*.json"))
        for idx, file in enumerate(files):
            req = load_request(file, format_hint)
            req.request_index = idx
            requests.append(req)
    elif path.suffix == ".jsonl":
        with open(path) as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                adapter = detect_adapter(data)
                req = adapter.parse(data)
                req.request_index = idx
                requests.append(req)
    else:
        req = load_request(path, format_hint)
        requests.append(req)

    return Session(requests=requests)


def profile_request(
    request: APIRequest,
    source: str = "",
    analyzers: list[BaseAnalyzer] | None = None,
) -> ProfileResult:
    """Profile a single API request (snapshot mode)."""
    if analyzers is None:
        analyzers = ALL_ANALYZERS

    result = ProfileResult(source=source, mode="snapshot")

    for analyzer in analyzers:
        ar = analyzer.analyze(request)
        result.analyzer_results[analyzer.name] = ar
        result.all_warnings.extend(ar.warnings)

    return result


def profile_session(
    session: Session,
    source: str = "",
    analyzers: list[BaseAnalyzer] | None = None,
) -> ProfileResult:
    """Profile a session — analyze the last request + build timeline."""
    if analyzers is None:
        analyzers = ALL_ANALYZERS

    if not session.requests:
        return ProfileResult(source=source, mode="session")

    # Analyze the last (most bloated) request
    last_request = session.requests[-1]
    result = profile_request(last_request, source=source, analyzers=analyzers)
    result.mode = "session"

    # Build timeline across all requests
    timeline: list[dict[str, Any]] = []
    for req in session.requests:
        token_result = TokenCounterAnalyzer().analyze(req)

        timeline.append({
            "request_index": req.request_index,
            "trace_index": req.trace_index,
            "total_tokens": token_result.summary.get("total_input_tokens", 0),
            "system_tokens": token_result.summary.get("system_prompt_tokens", 0),
            "tool_def_tokens": token_result.summary.get("tool_definition_tokens", 0),
            "message_tokens": token_result.summary.get("message_tokens", 0),
            "by_role": token_result.summary.get("by_role", {}),
        })

    result.session_timeline = timeline
    return result
