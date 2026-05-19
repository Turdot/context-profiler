"""Model pricing table for cost estimation.

Maps model name patterns to input/output pricing per 1M tokens (USD).
Prices are approximate and should be updated as providers change rates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ModelPricing:
    """Pricing for a single model tier."""

    input_per_1m: float  # USD per 1M input tokens
    output_per_1m: float  # USD per 1M output tokens
    display_name: str


# Patterns are matched in order; first match wins.
# Use lowercase substrings for matching against model identifiers.
PRICING_TABLE: list[tuple[list[str], ModelPricing]] = [
    # Claude models
    (
        ["claude-opus-4", "claude-4-opus"],
        ModelPricing(input_per_1m=15.0, output_per_1m=75.0, display_name="Claude Opus 4"),
    ),
    (
        ["claude-sonnet-4", "claude-4-sonnet"],
        ModelPricing(input_per_1m=3.0, output_per_1m=15.0, display_name="Claude Sonnet 4"),
    ),
    (
        ["claude-3-5-sonnet", "claude-3.5-sonnet"],
        ModelPricing(input_per_1m=3.0, output_per_1m=15.0, display_name="Claude 3.5 Sonnet"),
    ),
    (
        ["claude-3-5-haiku", "claude-3.5-haiku"],
        ModelPricing(input_per_1m=0.80, output_per_1m=4.0, display_name="Claude 3.5 Haiku"),
    ),
    (
        ["claude-3-opus"],
        ModelPricing(input_per_1m=15.0, output_per_1m=75.0, display_name="Claude 3 Opus"),
    ),
    (
        ["claude-3-sonnet"],
        ModelPricing(input_per_1m=3.0, output_per_1m=15.0, display_name="Claude 3 Sonnet"),
    ),
    (
        ["claude-3-haiku"],
        ModelPricing(input_per_1m=0.25, output_per_1m=1.25, display_name="Claude 3 Haiku"),
    ),
    # GPT models
    (
        ["gpt-4o-mini"],
        ModelPricing(input_per_1m=0.15, output_per_1m=0.60, display_name="GPT-4o mini"),
    ),
    (
        ["gpt-4o"],
        ModelPricing(input_per_1m=2.50, output_per_1m=10.0, display_name="GPT-4o"),
    ),
    (
        ["gpt-4-turbo"],
        ModelPricing(input_per_1m=10.0, output_per_1m=30.0, display_name="GPT-4 Turbo"),
    ),
]


def lookup_pricing(model: str) -> ModelPricing | None:
    """Find pricing for a model by matching name patterns.

    Returns None if no match is found.
    """
    if not model or model == "unknown":
        return None

    model_lower = model.lower()
    for patterns, pricing in PRICING_TABLE:
        for pattern in patterns:
            if pattern in model_lower:
                return pricing
    return None


def estimate_cost(
    input_tokens: int,
    output_tokens: int = 0,
    model: str = "unknown",
) -> dict[str, Any] | None:
    """Estimate cost for a request given token counts and model.

    Returns a dict with cost breakdown, or None if model is unknown.
    """
    pricing = lookup_pricing(model)
    if pricing is None:
        return None

    input_cost = (input_tokens / 1_000_000) * pricing.input_per_1m
    output_cost = (output_tokens / 1_000_000) * pricing.output_per_1m

    return {
        "estimated_input_cost_usd": round(input_cost, 6),
        "estimated_output_cost_usd": round(output_cost, 6),
        "estimated_total_cost_usd": round(input_cost + output_cost, 6),
        "estimated_model": pricing.display_name,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
