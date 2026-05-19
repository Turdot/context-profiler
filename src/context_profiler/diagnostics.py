"""Convert profile results into stable agent-readable diagnosis reports."""

from __future__ import annotations

from typing import Any

from context_profiler.context_diff import analyze_context_diff
from context_profiler.formats import describe_format
from context_profiler.models import Session
from context_profiler.profiler import ProfileResult
from context_profiler.session_insights import analyze_session_insights

_TOOL_USE_DOMINATES_RATIO = 0.5
_TOOL_USE_DOMINATES_MIN_TOKENS = 100
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

        top_tools = summary.get("top_tools_by_tokens", [])
        if total and top_tools:
            top_tool_name, top_tool_tokens = top_tools[0]
            top_tool_ratio = top_tool_tokens / total
            if (
                top_tool_tokens >= _TOP_TOOL_HOTSPOT_MIN_TOKENS
                and top_tool_ratio >= _TOP_TOOL_HOTSPOT_RATIO
            ):
                issues.append({
                    "code": "TOP_TOOL_CONTEXT_HOTSPOT",
                    "severity": _severity_for_ratio(top_tool_ratio),
                    "message": f"{top_tool_name} is the largest visible tool context hotspot.",
                    "evidence": {
                        "tool_name": top_tool_name,
                        "tool_tokens": top_tool_tokens,
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
        "diff_summary": diff["diff_summary"],
        "diff_hints": diff["diff_hints"] + session_insights["hints"],
        "session_insights": session_insights,
    }


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
