"""Convert profile results into stable agent-readable diagnosis reports."""

from __future__ import annotations

from typing import Any

from context_profiler.context_diff import analyze_context_diff
from context_profiler.formats import describe_format
from context_profiler.models import Session
from context_profiler.pricing import estimate_cost
from context_profiler.profiler import ProfileResult
from context_profiler.session_insights import analyze_session_insights

_TOOL_USE_DOMINATES_RATIO = 0.5
_TOOL_USE_DOMINATES_MIN_TOKENS = 100
_TOOL_INPUT_BLOAT_RATIO = 0.3
_TOOL_INPUT_BLOAT_MIN_TOKENS = 100
_TOOL_RESULT_DOMINATES_RATIO = 0.4
_TOOL_RESULT_DOMINATES_MIN_TOKENS = 100
_TOP_TOOL_HOTSPOT_RATIO = 0.3
_TOP_TOOL_HOTSPOT_MIN_TOKENS = 100
_REPEATED_FIELD_MIN_WASTE_TOKENS = 500


def _severity_for_ratio(ratio: float) -> str:
    if ratio >= 0.3:
        return "critical"
    if ratio >= 0.1:
        return "warning"
    return "info"


def diagnose_result(result: ProfileResult, session: Session | None = None) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    scope = _analysis_scope(result, session=session)
    diff = analyze_context_diff(session)
    session_insights = analyze_session_insights(session)

    token = result.analyzer_results.get("token_counter")
    if token:
        summary = token.summary
        total = summary.get("total_input_tokens", 0) or 0
        tool_defs = summary.get("tool_definition_tokens", 0) or 0
        by_content_type = summary.get("by_content_type", {})
        tool_use_tokens = by_content_type.get("tool_use", 0) or 0
        if total and tool_defs / total > 0.3:
            issues.append({
                "code": "STATIC_CONTEXT_BLOAT",
                "severity": _severity_for_ratio(tool_defs / total),
                "message": "Tool definitions consume a large share of input context.",
                "evidence": {"tool_definition_tokens": tool_defs, "total_input_tokens": total},
                "recommendation": "Audit unused or oversized tool schemas and consider smaller tool descriptions.",
            })
        if total and tool_use_tokens >= _TOOL_USE_DOMINATES_MIN_TOKENS:
            tool_use_ratio = tool_use_tokens / total
            if tool_use_ratio >= _TOOL_USE_DOMINATES_RATIO:
                issues.append({
                    "code": "TOOL_USE_DOMINATES_CONTEXT",
                    "severity": _severity_for_ratio(tool_use_ratio),
                    "message": "Tool inputs dominate the visible context.",
                    "evidence": {
                        "tool_use_tokens": tool_use_tokens,
                        "total_input_tokens": total,
                        "ratio": tool_use_ratio,
                    },
                    "recommendation": "Consider externalizing large tool inputs or replacing bulky payloads with artifact references.",
                })

        # Separate tool input and result analysis
        tool_use_tokens = summary.get("tool_use_tokens", 0) or 0
        tool_result_tokens = summary.get("tool_result_tokens", 0) or 0

        if total and tool_use_tokens >= _TOOL_INPUT_BLOAT_MIN_TOKENS:
            tool_input_ratio = tool_use_tokens / total
            if tool_input_ratio >= _TOOL_INPUT_BLOAT_RATIO:
                issues.append({
                    "code": "TOOL_INPUT_BLOAT",
                    "severity": _severity_for_ratio(tool_input_ratio),
                    "message": "Tool inputs (not results) consume a large share of context.",
                    "evidence": {
                        "tool_input_tokens": tool_use_tokens,
                        "total_input_tokens": total,
                        "ratio": tool_input_ratio,
                    },
                    "recommendation": "Tool inputs are often compressible. Consider using artifact references, shorter identifiers, or externalizing large payloads.",
                })

        if total and tool_result_tokens >= _TOOL_RESULT_DOMINATES_MIN_TOKENS:
            tool_result_ratio = tool_result_tokens / total
            if tool_result_ratio >= _TOOL_RESULT_DOMINATES_RATIO:
                issues.append({
                    "code": "TOOL_RESULT_DOMINATES",
                    "severity": _severity_for_ratio(tool_result_ratio),
                    "message": "Tool results (not inputs) dominate the context budget.",
                    "evidence": {
                        "tool_result_tokens": tool_result_tokens,
                        "total_input_tokens": total,
                        "ratio": tool_result_ratio,
                    },
                    "recommendation": "Tool results often contain real data that cannot be compressed. Consider summarizing large outputs, paginating results, or using streaming.",
                })

        top_tools = summary.get("top_tools_by_tokens", [])
        if total and top_tools:
            top_tool_name, top_tool_tokens = top_tools[0]
            top_tool_ratio = top_tool_tokens / total
            if (
                top_tool_tokens >= _TOP_TOOL_HOTSPOT_MIN_TOKENS
                and top_tool_ratio >= _TOP_TOOL_HOTSPOT_RATIO
            ):
                # Enhanced message with input/result breakdown
                top_tools_input = summary.get("top_tools_by_input_tokens", [])
                top_tools_result = summary.get("top_tools_by_result_tokens", [])
                input_tokens = dict(top_tools_input).get(top_tool_name, 0)
                result_tokens = dict(top_tools_result).get(top_tool_name, 0)

                issues.append({
                    "code": "TOP_TOOL_CONTEXT_HOTSPOT",
                    "severity": _severity_for_ratio(top_tool_ratio),
                    "message": f"{top_tool_name} is the largest visible tool context hotspot.",
                    "evidence": {
                        "tool_name": top_tool_name,
                        "tool_tokens": top_tool_tokens,
                        "tool_input_tokens": input_tokens,
                        "tool_result_tokens": result_tokens,
                        "total_input_tokens": total,
                        "ratio": top_tool_ratio,
                    },
                    "recommendation": "Inspect this tool's inputs/results for repeated large payloads, patch bodies, or values that should be stored as references.",
                })

    content = result.analyzer_results.get("content_repeat")
    if content:
        wasted = content.summary.get("total_wasted_tokens", 0)
        ratio = content.summary.get("waste_ratio", 0.0)
        if wasted > 0:
            issues.append({
                "code": "REPEATED_CONTENT_BLOCK",
                "severity": _severity_for_ratio(ratio),
                "message": "Repeated content blocks waste context tokens.",
                "evidence": {"total_wasted_tokens": wasted, "waste_ratio": ratio},
                "recommendation": "Replace repeated long content with references or summaries.",
            })

    field = result.analyzer_results.get("field_repeat")
    if field:
        wasted = field.summary.get("total_wasted_tokens", 0)
        ratio = field.summary.get("waste_ratio", 0.0)
        if wasted >= _REPEATED_FIELD_MIN_WASTE_TOKENS:
            issues.append({
                "code": "REPEATED_TOOL_INPUT",
                "severity": _severity_for_ratio(ratio),
                "message": "Tool input fields repeat large similar values across calls.",
                "evidence": {
                    "total_wasted_tokens": wasted,
                    "waste_ratio": ratio,
                    "top_offenders": field.summary.get("top_offenders", []),
                },
                "recommendation": "Move stable repeated arguments into references or shorter identifiers.",
            })

    # Context overflow risk from budget forecast
    forecast = session_insights.get("budget_forecast")
    if forecast and forecast.get("estimated_overflow_turn") is not None:
        current_turn_count = len(session.requests) if session else 0
        overflow_turn = forecast["estimated_overflow_turn"]
        utilization = forecast["current_utilization"]
        # Trigger if overflow is within 2x the current turn count
        if current_turn_count > 0 and overflow_turn <= current_turn_count * 2:
            if utilization > 0.8:
                severity = "critical"
            elif utilization > 0.5:
                severity = "warning"
            else:
                severity = "info"
            issues.append({
                "code": "CONTEXT_OVERFLOW_RISK",
                "severity": severity,
                "message": "Context is projected to overflow the model window at the current growth rate.",
                "evidence": {
                    "growth_rate_per_turn": forecast["growth_rate_per_turn"],
                    "current_utilization": forecast["current_utilization"],
                    "estimated_overflow_turn": forecast["estimated_overflow_turn"],
                    "context_window_tokens": forecast["context_window_tokens"],
                    "model": forecast["model"],
                },
                "recommendation": "Consider compacting earlier turns, summarizing tool results, or removing stale context before the window fills.",
            })

    # Cost estimation
    cost_info = _compute_cost(result, session)

    return {
        "schema_version": "0.1",
        "source": result.source,
        "mode": result.mode,
        "analysis_scope": scope,
        "summary": {
            "issue_count": len(issues),
            "warnings": result.all_warnings,
        },
        "issues": issues,
        "cost": cost_info,
        "diff_summary": diff["diff_summary"],
        "diff_hints": diff["diff_hints"] + session_insights["hints"],
        "session_insights": session_insights,
    }


def _compute_cost(result: ProfileResult, session: Session | None = None) -> dict[str, Any] | None:
    """Compute cost estimation for the profiled request or session."""
    token = result.analyzer_results.get("token_counter")
    if not token:
        return None

    summary = token.summary
    model = summary.get("model", "unknown")

    if session and session.requests:
        # Session mode: sum input tokens across all requests
        total_input = sum(req.total_input_tokens for req in session.requests)
        cost = estimate_cost(input_tokens=total_input, model=model)
        if cost:
            cost["mode"] = "session"
            cost["num_requests"] = len(session.requests)
        return cost
    else:
        # Snapshot mode: single request
        total_input = summary.get("total_input_tokens", 0)
        cost = estimate_cost(input_tokens=total_input, model=model)
        if cost:
            cost["mode"] = "snapshot"
        return cost


def _analysis_scope(result: ProfileResult, session: Session | None = None) -> dict[str, Any]:
    source_format = _source_format(result) or (
        session.metadata.get("source_format") if session is not None else None
    )
    if source_format:
        try:
            spec = describe_format(source_format)
            return {
                "format": source_format,
                "input_kind": spec["input_kind"],
                "confidence": spec["confidence"],
                "analysis_scope": spec["analysis_scope"],
                "limitations": spec["limitations"],
            }
        except ValueError:
            pass

    return {
        "format": source_format or "unknown",
        "input_kind": "unknown",
        "confidence": "unknown",
        "analysis_scope": ["Visible parsed content only."],
        "limitations": ["Input format was not recognized by the format registry."],
    }


def _source_format(result: ProfileResult) -> str | None:
    if result.session_timeline:
        for point in result.session_timeline:
            source_format = point.get("source_format")
            if source_format:
                return source_format
    return _source_format_from_profile(result)


def _source_format_from_profile(result: ProfileResult) -> str | None:
    token = result.analyzer_results.get("token_counter")
    if not token:
        return None
    return token.summary.get("source_format")
