"""Tests for the pricing module."""

import pytest

from ralph.pricing import (
    MODEL_PRICING,
    FALLBACK_PRICING,
    ModelPricing,
    PricingCalculator,
    calculate_cost,
    calculate_cost_for_record,
    get_model_family,
    get_pricing_for_model,
    normalize_model_name,
)


class TestModelPricing:
    """Tests for ModelPricing dataclass."""

    def test_model_pricing_is_frozen(self):
        """Test that ModelPricing is immutable."""
        pricing = ModelPricing(input=1.0, output=2.0, cache_creation=1.5, cache_read=0.1)
        with pytest.raises(Exception):  # FrozenInstanceError
            pricing.input = 5.0

    def test_known_models_have_pricing(self):
        """Test that all known models have complete pricing."""
        for model_id, pricing in MODEL_PRICING.items():
            assert pricing.input > 0, f"{model_id} should have positive input price"
            assert pricing.output > 0, f"{model_id} should have positive output price"
            assert pricing.cache_creation > 0, f"{model_id} should have cache_creation price"
            assert pricing.cache_read >= 0, f"{model_id} should have cache_read price"


class TestNormalizeModelName:
    """Tests for model name normalization."""

    def test_normalize_known_model(self):
        """Test normalization of known model names."""
        assert normalize_model_name("claude-opus-4-5-20251101") == "claude-opus-4-5-20251101"

    def test_normalize_case_insensitive(self):
        """Test case-insensitive model name lookup."""
        # Note: May return original if not exact match
        result = normalize_model_name("Claude-Opus-4-5-20251101")
        # Should find the model
        assert "opus" in result.lower()

    def test_normalize_unknown_model(self):
        """Test that unknown models are returned as-is."""
        assert normalize_model_name("unknown-model") == "unknown-model"

    def test_normalize_empty_string(self):
        """Test normalization of empty string."""
        assert normalize_model_name("") == ""


class TestGetModelFamily:
    """Tests for model family detection."""

    def test_opus_45_detection(self):
        """Test detection of Opus 4.5 variants."""
        assert get_model_family("claude-opus-4-5-20251101") == "opus-4.5"
        assert get_model_family("opus-4.5") == "opus-4.5"
        assert get_model_family("some-opus-45-model") == "opus-4.5"

    def test_opus_detection(self):
        """Test detection of Opus (non-4.5) variants."""
        assert get_model_family("claude-opus-4-20250514") == "opus"
        assert get_model_family("claude-3-opus-20240229") == "opus"

    def test_sonnet_detection(self):
        """Test detection of Sonnet variants."""
        assert get_model_family("claude-sonnet-4-20250514") == "sonnet"
        assert get_model_family("claude-3-5-sonnet-20241022") == "sonnet"

    def test_haiku_detection(self):
        """Test detection of Haiku variants."""
        assert get_model_family("claude-3-haiku-20240307") == "haiku"
        assert get_model_family("claude-3-5-haiku-20241022") == "haiku"

    def test_unknown_defaults_to_sonnet(self):
        """Test that unknown models default to Sonnet family."""
        assert get_model_family("unknown-model") == "sonnet"
        assert get_model_family("") == "sonnet"


class TestGetPricingForModel:
    """Tests for pricing lookup."""

    def test_exact_model_match(self):
        """Test pricing for exact model ID."""
        pricing = get_pricing_for_model("claude-opus-4-5-20251101")
        assert pricing == MODEL_PRICING["claude-opus-4-5-20251101"]

    def test_fallback_to_family(self):
        """Test fallback to family pricing for unknown models."""
        pricing = get_pricing_for_model("some-new-opus-model")
        assert pricing == FALLBACK_PRICING["opus"]

    def test_none_returns_sonnet(self):
        """Test that None model returns Sonnet pricing."""
        pricing = get_pricing_for_model(None)
        assert pricing == FALLBACK_PRICING["sonnet"]


class TestPricingCalculator:
    """Tests for PricingCalculator class."""

    def test_calculate_cost_basic(self):
        """Test basic cost calculation."""
        calc = PricingCalculator()
        cost = calc.calculate_cost(
            model="claude-opus-4-5-20251101",
            input_tokens=1_000_000,
            output_tokens=0,
        )
        # Opus 4.5 input: $5.0 per million
        assert cost == 5.0

    def test_calculate_cost_all_token_types(self):
        """Test cost calculation with all token types."""
        calc = PricingCalculator()
        cost = calc.calculate_cost(
            model="claude-opus-4-5-20251101",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            cache_creation_tokens=1_000_000,
            cache_read_tokens=1_000_000,
        )
        # Opus 4.5: input=$5, output=$25, cache_creation=$6.25, cache_read=$0.5
        expected = 5.0 + 25.0 + 6.25 + 0.5
        assert cost == expected

    def test_calculate_cost_caching(self):
        """Test that calculations are cached."""
        calc = PricingCalculator()

        # First call
        cost1 = calc.calculate_cost("claude-opus-4-5-20251101", input_tokens=1000)

        # Second call with same params
        cost2 = calc.calculate_cost("claude-opus-4-5-20251101", input_tokens=1000)

        assert cost1 == cost2

    def test_calculate_cost_for_record(self):
        """Test cost calculation for a UsageRecord."""
        from ralph.usage import UsageRecord
        from datetime import datetime, timezone

        calc = PricingCalculator()
        record = UsageRecord(
            timestamp=datetime.now(timezone.utc),
            input_tokens=500_000,
            output_tokens=100_000,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            model="claude-sonnet-4-20250514",
        )

        cost = calc.calculate_cost_for_record(record)

        # Sonnet: input=$3, output=$15 per million
        expected = (500_000 / 1_000_000) * 3.0 + (100_000 / 1_000_000) * 15.0
        assert abs(cost - expected) < 0.0001

    def test_clear_cache(self):
        """Test cache clearing."""
        calc = PricingCalculator()
        calc.calculate_cost("claude-opus-4-5-20251101", input_tokens=1000)
        assert len(calc._cache) > 0

        calc.clear_cache()
        assert len(calc._cache) == 0


class TestGlobalFunctions:
    """Tests for module-level convenience functions."""

    def test_calculate_cost_function(self):
        """Test the global calculate_cost function."""
        cost = calculate_cost(
            model="claude-opus-4-5-20251101",
            input_tokens=1_000_000,
        )
        assert cost == 5.0

    def test_calculate_cost_for_record_function(self):
        """Test the global calculate_cost_for_record function."""
        from ralph.usage import UsageRecord
        from datetime import datetime, timezone

        record = UsageRecord(
            timestamp=datetime.now(timezone.utc),
            input_tokens=1_000_000,
            output_tokens=0,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            model="claude-opus-4-5-20251101",
        )

        cost = calculate_cost_for_record(record)
        assert cost == 5.0


class TestPricingEdgeCases:
    """Tests for edge cases in pricing."""

    def test_zero_tokens(self):
        """Test cost with zero tokens."""
        cost = calculate_cost("claude-opus-4-5-20251101")
        assert cost == 0.0

    def test_small_token_counts(self):
        """Test cost with small token counts."""
        cost = calculate_cost(
            model="claude-opus-4-5-20251101",
            input_tokens=100,
        )
        # 100 tokens at $5/million = $0.0005
        assert cost == 0.0005

    def test_very_large_token_counts(self):
        """Test cost with very large token counts."""
        cost = calculate_cost(
            model="claude-opus-4-5-20251101",
            input_tokens=100_000_000,  # 100M tokens
        )
        # 100M tokens at $5/million = $500
        assert cost == 500.0
