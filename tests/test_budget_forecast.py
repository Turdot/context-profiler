"""Tests for context budget forecasting."""

from context_profiler.diagnostics import diagnose_result
from context_profiler.models import APIRequest, BlockType, ContentBlock, Message, Role, Session
from context_profiler.profiler import profile_session
from context_profiler.session_insights import (
    _resolve_context_window,
    analyze_session_insights,
    budget_forecast,
)


def _request(index: int, token_count: int, model: str = "gpt-4o") -> APIRequest:
    """Create a minimal request with a given total token count."""
    messages = [
        Message(
            role=Role.USER,
            blocks=[ContentBlock(BlockType.TEXT, "content", token_count=token_count)],
            index=0,
        )
    ]
    return APIRequest(messages=messages, request_index=index, model=model, source_format="openai")


# ---------------------------------------------------------------------------
# _resolve_context_window
# ---------------------------------------------------------------------------


def test_resolve_context_window_claude():
    model, size = _resolve_context_window("claude-3-opus-20240229")
    assert model == "claude"
    assert size == 200_000


def test_resolve_context_window_gpt4o():
    model, size = _resolve_context_window("gpt-4o-2024-05-13")
    assert model == "gpt-4o"
    assert size == 128_000


def test_resolve_context_window_gpt4_turbo():
    model, size = _resolve_context_window("gpt-4-turbo-preview")
    assert model == "gpt-4-turbo"
    assert size == 128_000


def test_resolve_context_window_gpt4o_mini():
    model, size = _resolve_context_window("gpt-4o-mini")
    assert model == "gpt-4o-mini"
    assert size == 128_000


def test_resolve_context_window_unknown_falls_back():
    model, size = _resolve_context_window("some-custom-model")
    assert model == "default"
    assert size == 128_000


# ---------------------------------------------------------------------------
# budget_forecast — basic behavior
# ---------------------------------------------------------------------------


def test_budget_forecast_returns_none_for_single_request():
    session = Session(requests=[_request(0, 10_000)])
    assert budget_forecast(session) is None


def test_budget_forecast_returns_none_for_empty_session():
    session = Session(requests=[])
    assert budget_forecast(session) is None


def test_budget_forecast_returns_none_for_none_session():
    assert budget_forecast(None) is None


def test_budget_forecast_linear_growth():
    """With steady 10K growth per turn, should predict overflow correctly."""
    session = Session(requests=[
        _request(0, 10_000),
        _request(1, 20_000),
        _request(2, 30_000),
        _request(3, 40_000),
        _request(4, 50_000),
    ])

    forecast = budget_forecast(session)

    assert forecast is not None
    assert forecast["growth_rate_per_turn"] == 10_000.0
    assert forecast["model"] == "gpt-4o"
    assert forecast["context_window_tokens"] == 128_000
    # Current utilization: 50000 / 128000
    assert abs(forecast["current_utilization"] - 50_000 / 128_000) < 0.001
    # Remaining: 78000 tokens, at 10K/turn = 7.8 turns from turn 5
    # So overflow at turn 5 + 7 = 12
    assert forecast["estimated_overflow_turn"] == 12


def test_budget_forecast_no_growth():
    """Flat token usage should not predict overflow."""
    session = Session(requests=[
        _request(0, 50_000),
        _request(1, 50_000),
        _request(2, 50_000),
    ])

    forecast = budget_forecast(session)

    assert forecast is not None
    assert forecast["growth_rate_per_turn"] == 0.0
    assert forecast["estimated_overflow_turn"] is None


def test_budget_forecast_shrinking_context():
    """Decreasing token usage should not predict overflow."""
    session = Session(requests=[
        _request(0, 80_000),
        _request(1, 60_000),
        _request(2, 40_000),
    ])

    forecast = budget_forecast(session)

    assert forecast is not None
    assert forecast["growth_rate_per_turn"] < 0
    assert forecast["estimated_overflow_turn"] is None


def test_budget_forecast_claude_model_uses_200k_window():
    session = Session(requests=[
        _request(0, 50_000, model="claude-3-sonnet-20240229"),
        _request(1, 100_000, model="claude-3-sonnet-20240229"),
    ])

    forecast = budget_forecast(session)

    assert forecast is not None
    assert forecast["context_window_tokens"] == 200_000
    assert forecast["model"] == "claude"
    # Growth: 50K/turn, remaining: 100K, overflow at turn 2 + 2 = 4
    assert forecast["estimated_overflow_turn"] == 4


# ---------------------------------------------------------------------------
# Integration with analyze_session_insights
# ---------------------------------------------------------------------------


def test_session_insights_includes_budget_forecast():
    session = Session(requests=[
        _request(0, 10_000),
        _request(1, 20_000),
        _request(2, 30_000),
    ])

    insights = analyze_session_insights(session)

    assert "budget_forecast" in insights
    forecast = insights["budget_forecast"]
    assert forecast is not None
    assert "growth_rate_per_turn" in forecast
    assert "current_utilization" in forecast
    assert "estimated_overflow_turn" in forecast
    assert "context_window_tokens" in forecast
    assert "model" in forecast


def test_session_insights_budget_forecast_none_for_single_request():
    session = Session(requests=[_request(0, 10_000)])
    insights = analyze_session_insights(session)
    assert insights["budget_forecast"] is None


# ---------------------------------------------------------------------------
# CONTEXT_OVERFLOW_RISK diagnostic
# ---------------------------------------------------------------------------


def test_diagnostic_context_overflow_risk_critical():
    """High utilization + imminent overflow triggers critical issue."""
    # 5 turns, 22K growth/turn, at 110K now on 128K window
    # utilization = 110000/128000 = 0.859 > 0.8 => critical
    # overflow at turn 5 + int(18000/22000) = 5 + 0 = 5
    # 2x current = 10, overflow 5 <= 10 => triggers
    session = Session(requests=[
        _request(0, 22_000),
        _request(1, 44_000),
        _request(2, 66_000),
        _request(3, 88_000),
        _request(4, 110_000),
    ])

    result = profile_session(session)
    diagnosis = diagnose_result(result, session=session)

    overflow_issues = [i for i in diagnosis["issues"] if i["code"] == "CONTEXT_OVERFLOW_RISK"]
    assert len(overflow_issues) == 1
    issue = overflow_issues[0]
    assert issue["severity"] == "critical"
    assert issue["evidence"]["growth_rate_per_turn"] == 22_000.0
    assert issue["evidence"]["estimated_overflow_turn"] is not None


def test_diagnostic_context_overflow_risk_warning():
    """Moderate utilization + near overflow triggers warning."""
    # 4 turns, 20K growth/turn, at 80K on 128K window
    # utilization = 80000/128000 = 0.625 > 0.5 => warning
    # overflow at turn 4 + int(48000/20000) = 4 + 2 = 6
    # 2x current = 8, overflow 6 <= 8 => triggers
    session = Session(requests=[
        _request(0, 20_000),
        _request(1, 40_000),
        _request(2, 60_000),
        _request(3, 80_000),
    ])

    result = profile_session(session)
    diagnosis = diagnose_result(result, session=session)

    overflow_issues = [i for i in diagnosis["issues"] if i["code"] == "CONTEXT_OVERFLOW_RISK"]
    assert len(overflow_issues) == 1
    assert overflow_issues[0]["severity"] == "warning"


def test_diagnostic_no_overflow_risk_when_far_away():
    """Low utilization with overflow far in the future should not trigger."""
    # 5 turns, 1K growth/turn, at 5K on 128K window
    # overflow at turn 5 + int(123000/1000) = 5 + 123 = 128
    # 2x current = 10, overflow 128 > 10 => no trigger
    session = Session(requests=[
        _request(0, 1_000),
        _request(1, 2_000),
        _request(2, 3_000),
        _request(3, 4_000),
        _request(4, 5_000),
    ])

    result = profile_session(session)
    diagnosis = diagnose_result(result, session=session)

    overflow_issues = [i for i in diagnosis["issues"] if i["code"] == "CONTEXT_OVERFLOW_RISK"]
    assert len(overflow_issues) == 0


def test_diagnostic_no_overflow_risk_when_shrinking():
    """Shrinking context should never trigger overflow risk."""
    session = Session(requests=[
        _request(0, 80_000),
        _request(1, 60_000),
        _request(2, 40_000),
    ])

    result = profile_session(session)
    diagnosis = diagnose_result(result, session=session)

    overflow_issues = [i for i in diagnosis["issues"] if i["code"] == "CONTEXT_OVERFLOW_RISK"]
    assert len(overflow_issues) == 0


def test_diagnostic_budget_forecast_in_session_insights():
    """Budget forecast should appear in the diagnosis JSON under session_insights."""
    session = Session(requests=[
        _request(0, 10_000),
        _request(1, 20_000),
        _request(2, 30_000),
    ])

    result = profile_session(session)
    diagnosis = diagnose_result(result, session=session)

    assert "budget_forecast" in diagnosis["session_insights"]
    forecast = diagnosis["session_insights"]["budget_forecast"]
    assert forecast is not None
    assert forecast["growth_rate_per_turn"] == 10_000.0
