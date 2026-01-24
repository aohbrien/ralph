"""Pricing calculations for Claude models.

This module provides cost calculation based on token usage and model pricing.
Supports Claude 4.5, 4, 3.5, and 3 variants with fallback pricing by model family.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ralph.usage import UsageRecord


@dataclass(frozen=True)
class ModelPricing:
    """Pricing rates for a model (per million tokens)."""

    input: float
    output: float
    cache_creation: float
    cache_read: float


# Model pricing in USD per million tokens
# Based on official Anthropic pricing
MODEL_PRICING: dict[str, ModelPricing] = {
    # Claude Opus 4.5 (2025-11-01)
    "claude-opus-4-5-20251101": ModelPricing(
        input=5.0,
        output=25.0,
        cache_creation=6.25,
        cache_read=0.5,
    ),
    # Claude Opus 4 (2025-05-14)
    "claude-opus-4-20250514": ModelPricing(
        input=15.0,
        output=75.0,
        cache_creation=18.75,
        cache_read=1.5,
    ),
    # Claude Sonnet 4 (2025-05-14)
    "claude-sonnet-4-20250514": ModelPricing(
        input=3.0,
        output=15.0,
        cache_creation=3.75,
        cache_read=0.3,
    ),
    # Claude 3.5 Sonnet variants
    "claude-3-5-sonnet-20241022": ModelPricing(
        input=3.0,
        output=15.0,
        cache_creation=3.75,
        cache_read=0.3,
    ),
    "claude-3-5-sonnet-20240620": ModelPricing(
        input=3.0,
        output=15.0,
        cache_creation=3.75,
        cache_read=0.3,
    ),
    # Claude 3.5 Haiku
    "claude-3-5-haiku-20241022": ModelPricing(
        input=0.80,
        output=4.0,
        cache_creation=1.0,
        cache_read=0.08,
    ),
    # Claude 3 Opus
    "claude-3-opus-20240229": ModelPricing(
        input=15.0,
        output=75.0,
        cache_creation=18.75,
        cache_read=1.5,
    ),
    # Claude 3 Sonnet
    "claude-3-sonnet-20240229": ModelPricing(
        input=3.0,
        output=15.0,
        cache_creation=3.75,
        cache_read=0.3,
    ),
    # Claude 3 Haiku
    "claude-3-haiku-20240307": ModelPricing(
        input=0.25,
        output=1.25,
        cache_creation=0.3,
        cache_read=0.03,
    ),
}

# Fallback pricing by model family (used when exact model not found)
FALLBACK_PRICING: dict[str, ModelPricing] = {
    "opus-4.5": ModelPricing(
        input=5.0,
        output=25.0,
        cache_creation=6.25,
        cache_read=0.5,
    ),
    "opus": ModelPricing(
        input=15.0,
        output=75.0,
        cache_creation=18.75,
        cache_read=1.5,
    ),
    "sonnet": ModelPricing(
        input=3.0,
        output=15.0,
        cache_creation=3.75,
        cache_read=0.3,
    ),
    "haiku": ModelPricing(
        input=0.25,
        output=1.25,
        cache_creation=0.3,
        cache_read=0.03,
    ),
}


def normalize_model_name(model: str) -> str:
    """
    Normalize model name for pricing lookup.

    Handles common variations in model naming.

    Args:
        model: Raw model name from API

    Returns:
        Normalized model name for lookup
    """
    if not model:
        return ""

    # Already a known model ID
    if model in MODEL_PRICING:
        return model

    # Try lowercase
    model_lower = model.lower()
    for known_model in MODEL_PRICING:
        if known_model.lower() == model_lower:
            return known_model

    return model


def get_model_family(model: str) -> str:
    """
    Determine the model family for fallback pricing.

    Args:
        model: Model name

    Returns:
        Family name: 'opus-4.5', 'opus', 'sonnet', or 'haiku'
    """
    if not model:
        return "sonnet"  # Default fallback

    model_lower = model.lower()

    if "opus" in model_lower:
        # Check for Opus 4.5 variants
        if "4-5" in model_lower or "4.5" in model_lower or "45" in model_lower:
            return "opus-4.5"
        return "opus"
    elif "haiku" in model_lower:
        return "haiku"
    else:
        return "sonnet"


def get_pricing_for_model(model: str | None) -> ModelPricing:
    """
    Get pricing for a model with fallback logic.

    Args:
        model: Model name (can be None)

    Returns:
        ModelPricing for the model
    """
    if not model:
        return FALLBACK_PRICING["sonnet"]

    # Try exact match
    normalized = normalize_model_name(model)
    if normalized in MODEL_PRICING:
        return MODEL_PRICING[normalized]

    # Fallback to family pricing
    family = get_model_family(model)
    return FALLBACK_PRICING[family]


class PricingCalculator:
    """Calculates costs based on model pricing with caching support."""

    def __init__(self) -> None:
        """Initialize the calculator with an empty cache."""
        self._cache: dict[str, float] = {}

    def calculate_cost(
        self,
        model: str | None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_creation_tokens: int = 0,
        cache_read_tokens: int = 0,
    ) -> float:
        """
        Calculate cost for given token counts.

        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cache_creation_tokens: Number of cache creation tokens
            cache_read_tokens: Number of cache read tokens

        Returns:
            Total cost in USD
        """
        # Create cache key
        cache_key = (
            f"{model}:{input_tokens}:{output_tokens}:"
            f"{cache_creation_tokens}:{cache_read_tokens}"
        )

        # Check cache
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Get pricing
        pricing = get_pricing_for_model(model)

        # Calculate cost (pricing is per million tokens)
        cost = (
            (input_tokens / 1_000_000) * pricing.input
            + (output_tokens / 1_000_000) * pricing.output
            + (cache_creation_tokens / 1_000_000) * pricing.cache_creation
            + (cache_read_tokens / 1_000_000) * pricing.cache_read
        )

        # Round to 6 decimal places
        cost = round(cost, 6)

        # Cache result
        self._cache[cache_key] = cost
        return cost

    def calculate_cost_for_record(self, record: "UsageRecord") -> float:
        """
        Calculate cost for a UsageRecord.

        Args:
            record: UsageRecord with token counts and model

        Returns:
            Cost in USD
        """
        return self.calculate_cost(
            model=record.model,
            input_tokens=record.input_tokens,
            output_tokens=record.output_tokens,
            cache_creation_tokens=record.cache_creation_input_tokens,
            cache_read_tokens=record.cache_read_input_tokens,
        )

    def clear_cache(self) -> None:
        """Clear the pricing cache."""
        self._cache.clear()


# Global calculator instance for convenience
_calculator: PricingCalculator | None = None


def get_calculator() -> PricingCalculator:
    """Get or create the global pricing calculator."""
    global _calculator
    if _calculator is None:
        _calculator = PricingCalculator()
    return _calculator


def calculate_cost(
    model: str | None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> float:
    """
    Calculate cost for given token counts using the global calculator.

    Args:
        model: Model name
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        cache_creation_tokens: Number of cache creation tokens
        cache_read_tokens: Number of cache read tokens

    Returns:
        Total cost in USD
    """
    return get_calculator().calculate_cost(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_tokens=cache_creation_tokens,
        cache_read_tokens=cache_read_tokens,
    )


def calculate_cost_for_record(record: "UsageRecord") -> float:
    """
    Calculate cost for a UsageRecord using the global calculator.

    Args:
        record: UsageRecord with token counts and model

    Returns:
        Cost in USD
    """
    return get_calculator().calculate_cost_for_record(record)
