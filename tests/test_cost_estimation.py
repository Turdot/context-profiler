"""Tests for cost estimation feature."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from context_profiler.pricing import estimate_cost, lookup_pricing, ModelPricing
from context_profiler.models import APIRequest, Message, ContentBlock, BlockType, Role, Session
from context_profiler.analyzers.token_counter import TokenCounterAnalyzer
from context_profiler.diagnostics import diagnose_result
from context_profiler.profiler import profile_request, profile_session

FIXTURES = Path(__file__).parent / "fixtures"


class TestLookupPricing:
    """Test model name pattern matching."""

    def test_claude_sonnet_35(self):
        pricing = lookup_pricing("claude-3-5-sonnet-20241022")
        assert pricing is not None
        assert pricing.display_name == "Claude 3.5 Sonnet"
        assert pricing.input_per_1m == 3.0

    def test_claude_opus_4(self):
        pricing = lookup_pricing("claude-opus-4-20250514")
        assert pricing is not None
        assert pricing.display_name == "Claude Opus 4"
        assert pricing.input_per_1m == 15.0

    def test_claude_sonnet_4(self):
        pricing = lookup_pricing("claude-sonnet-4-20250514")
        assert pricing is not None
        assert pricing.display_name == "Claude Sonnet 4"

    def test_claude_haiku_35(self):
        pricing = lookup_pricing("claude-3-5-haiku-20241022")
        assert pricing is not None
        assert pricing.display_name == "Claude 3.5 Haiku"

    def test_gpt4o(self):
        pricing = lookup_pricing("gpt-4o-2024-08-06")
        assert pricing is not None
        assert pricing.display_name == "GPT-4o"
        assert pricing.input_per_1m == 2.50

    def test_gpt4o_mini(self):
        pricing = lookup_pricing("gpt-4o-mini-2024-07-18")
        assert pricing is not None
        assert pricing.display_name == "GPT-4o mini"
        assert pricing.input_per_1m == 0.15

    def test_gpt4_turbo(self):
        pricing = lookup_pricing("gpt-4-turbo-2024-04-09")
        assert pricing is not None
        assert pricing.display_name == "GPT-4 Turbo"

    def test_unknown_model_returns_none(self):
        assert lookup_pricing("unknown") is None
        assert lookup_pricing("") is None
        assert lookup_pricing("some-custom-model") is None

    def test_case_insensitive(self):
        pricing = lookup_pricing("Claude-3-5-Sonnet-20241022")
        assert pricing is not None
        assert pricing.display_name == "Claude 3.5 Sonnet"


class TestEstimateCost:
    """Test cost calculation."""

    def test_basic_cost_calculation(self):
        cost = estimate_cost(input_tokens=1_000_000, model="gpt-4o")
        assert cost is not None
        assert cost["estimated_input_cost_usd"] == 2.50
        assert cost["estimated_output_cost_usd"] == 0.0
        assert cost["estimated_total_cost_usd"] == 2.50
        assert cost["estimated_model"] == "GPT-4o"

    def test_fractional_tokens(self):
        cost = estimate_cost(input_tokens=50_000, model="claude-3-5-sonnet-20241022")
        assert cost is not None
        # 50K tokens at $3/1M = $0.15
        assert cost["estimated_input_cost_usd"] == 0.15

    def test_with_output_tokens(self):
        cost = estimate_cost(input_tokens=100_000, output_tokens=50_000, model="gpt-4o")
        assert cost is not None
        # Input: 100K at $2.50/1M = $0.25
        # Output: 50K at $10/1M = $0.50
        assert cost["estimated_input_cost_usd"] == 0.25
        assert cost["estimated_output_cost_usd"] == 0.5
        assert cost["estimated_total_cost_usd"] == 0.75

    def test_unknown_model_returns_none(self):
        cost = estimate_cost(input_tokens=1000, model="unknown")
        assert cost is None

    def test_zero_tokens(self):
        cost = estimate_cost(input_tokens=0, model="gpt-4o")
        assert cost is not None
        assert cost["estimated_input_cost_usd"] == 0.0


class TestTokenCounterCostIntegration:
    """Test that cost appears in token_counter analyzer output."""

    def _make_request(self, model: str, num_blocks: int = 3) -> APIRequest:
        blocks = [
            ContentBlock(block_type=BlockType.TEXT, text="hello " * 100, token_count=100)
            for _ in range(num_blocks)
        ]
        messages = [Message(role=Role.USER, blocks=blocks, index=0)]
        return APIRequest(messages=messages, model=model)

    def test_cost_in_summary_known_model(self):
        req = self._make_request("claude-3-5-sonnet-20241022")
        analyzer = TokenCounterAnalyzer()
        result = analyzer.analyze(req)
        assert "cost" in result.summary
        cost = result.summary["cost"]
        assert cost["estimated_model"] == "Claude 3.5 Sonnet"
        assert cost["estimated_input_cost_usd"] > 0

    def test_no_cost_for_unknown_model(self):
        req = self._make_request("unknown")
        analyzer = TokenCounterAnalyzer()
        result = analyzer.analyze(req)
        assert "cost" not in result.summary


class TestDiagnosticsCost:
    """Test that cost appears in diagnosis JSON output."""

    def _make_session(self, model: str, num_requests: int = 3) -> Session:
        requests = []
        for i in range(num_requests):
            blocks = [
                ContentBlock(block_type=BlockType.TEXT, text="x " * 500, token_count=500)
            ]
            messages = [Message(role=Role.USER, blocks=blocks, index=0)]
            req = APIRequest(messages=messages, model=model, request_index=i)
            requests.append(req)
        return Session(requests=requests)

    def test_cost_in_diagnosis_snapshot(self):
        blocks = [
            ContentBlock(block_type=BlockType.TEXT, text="hello " * 100, token_count=1000)
        ]
        messages = [Message(role=Role.USER, blocks=blocks, index=0)]
        req = APIRequest(messages=messages, model="gpt-4o")
        result = profile_request(req, source="test")
        diagnosis = diagnose_result(result)
        assert "cost" in diagnosis
        cost = diagnosis["cost"]
        assert cost["estimated_model"] == "GPT-4o"
        assert cost["mode"] == "snapshot"

    def test_cost_in_diagnosis_session(self):
        session = self._make_session("claude-3-5-sonnet-20241022", num_requests=3)
        result = profile_session(session, source="test")
        diagnosis = diagnose_result(result, session=session)
        assert "cost" in diagnosis
        cost = diagnosis["cost"]
        assert cost["estimated_model"] == "Claude 3.5 Sonnet"
        assert cost["mode"] == "session"
        assert cost["num_requests"] == 3
        # 3 requests x 500 tokens = 1500 tokens at $3/1M
        assert cost["input_tokens"] == 1500
        assert cost["estimated_input_cost_usd"] > 0

    def test_no_cost_for_unknown_model_in_diagnosis(self):
        blocks = [
            ContentBlock(block_type=BlockType.TEXT, text="hello", token_count=100)
        ]
        messages = [Message(role=Role.USER, blocks=blocks, index=0)]
        req = APIRequest(messages=messages, model="unknown")
        result = profile_request(req, source="test")
        diagnosis = diagnose_result(result)
        assert diagnosis["cost"] is None


class TestCLIReporterCost:
    """Test that cost shows up in CLI output."""

    def test_cost_in_cli_output(self):
        from click.testing import CliRunner
        from context_profiler.cli import main

        fixture = FIXTURES / "repeated_tool_calls.json"
        # The fixture uses "gpt-4" which isn't in our pricing table,
        # so let's create a temp fixture with a known model
        runner = CliRunner()
        with runner.isolated_filesystem():
            data = json.loads(fixture.read_text())
            data["model"] = "gpt-4o"
            Path("test_request.json").write_text(json.dumps(data))
            result = runner.invoke(main, ["analyze", "test_request.json", "--format", "openai"])
            assert result.exit_code == 0
            assert "Estimated Cost" in result.output
            assert "GPT-4o" in result.output

    def test_no_cost_line_for_unknown_model(self):
        from click.testing import CliRunner
        from context_profiler.cli import main

        fixture = FIXTURES / "repeated_tool_calls.json"
        runner = CliRunner()
        # The fixture uses "gpt-4" which is not in our pricing table
        result = runner.invoke(main, ["analyze", str(fixture), "--format", "openai"])
        assert result.exit_code == 0
        assert "Estimated Cost" not in result.output
