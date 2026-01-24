"""Tests for the P90 limit calculator module."""

from datetime import datetime, timedelta, timezone

import pytest

from ralph.p90 import (
    COMMON_TOKEN_LIMITS,
    DEFAULT_MIN_LIMIT,
    LIMIT_DETECTION_THRESHOLD,
    P90Calculator,
    P90Config,
    _calculate_p90_from_windows,
    _did_hit_limit,
    _is_complete_window,
    get_calculator,
    get_p90_limit_with_fallback,
)
from ralph.usage import UsageAggregate


def create_window(
    tokens: int,
    hours: float = 5.0,
    start_time: datetime | None = None,
) -> UsageAggregate:
    """Helper to create a UsageAggregate for testing."""
    if start_time is None:
        start_time = datetime.now(timezone.utc) - timedelta(hours=hours)

    return UsageAggregate(
        window_start=start_time,
        window_end=start_time + timedelta(hours=hours),
        input_tokens=tokens,
        output_tokens=0,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        message_count=1,
        request_count=1,
        cost_usd=0.0,
    )


class TestDidHitLimit:
    """Tests for limit hit detection."""

    def test_hits_free_limit(self):
        """Test detection of free tier limit hit."""
        # 95% of 30,000 = 28,500
        assert _did_hit_limit(29_000, COMMON_TOKEN_LIMITS, LIMIT_DETECTION_THRESHOLD)

    def test_hits_pro_limit(self):
        """Test detection of pro tier limit hit."""
        # 95% of 300,000 = 285,000
        assert _did_hit_limit(290_000, COMMON_TOKEN_LIMITS, LIMIT_DETECTION_THRESHOLD)

    def test_hits_max5x_limit(self):
        """Test detection of max5x limit hit."""
        # 95% of 1,500,000 = 1,425,000
        assert _did_hit_limit(1_450_000, COMMON_TOKEN_LIMITS, LIMIT_DETECTION_THRESHOLD)

    def test_below_all_limits(self):
        """Test that usage below all limits returns False."""
        # 20,000 is below 95% of the lowest limit (30,000 * 0.95 = 28,500)
        assert not _did_hit_limit(20_000, COMMON_TOKEN_LIMITS, LIMIT_DETECTION_THRESHOLD)

    def test_custom_threshold(self):
        """Test with custom threshold."""
        # 80% of 30,000 = 24,000
        assert _did_hit_limit(25_000, (30_000,), 0.80)
        assert not _did_hit_limit(23_000, (30_000,), 0.80)


class TestIsCompleteWindow:
    """Tests for complete window detection."""

    def test_complete_window(self):
        """Test that a 5-hour window is marked complete."""
        window = create_window(100_000, hours=5.0)
        assert _is_complete_window(window)

    def test_incomplete_window(self):
        """Test that a partial window is marked incomplete."""
        window = create_window(100_000, hours=3.0)
        assert not _is_complete_window(window)

    def test_nearly_complete_window(self):
        """Test that a nearly complete window (4.9h) is marked complete."""
        window = create_window(100_000, hours=4.9)
        assert _is_complete_window(window)


class TestCalculateP90FromWindows:
    """Tests for P90 calculation."""

    @pytest.fixture
    def default_config(self) -> P90Config:
        """Create default P90 config."""
        return P90Config(
            common_limits=COMMON_TOKEN_LIMITS,
            limit_threshold=LIMIT_DETECTION_THRESHOLD,
            default_min_limit=DEFAULT_MIN_LIMIT,
            cache_ttl_seconds=3600,
        )

    def test_empty_windows_returns_default(self, default_config):
        """Test that empty windows return default limit."""
        result = _calculate_p90_from_windows([], default_config)
        assert result == DEFAULT_MIN_LIMIT

    def test_single_window_returns_max(self, default_config):
        """Test P90 with single window."""
        windows = [create_window(500_000)]
        result = _calculate_p90_from_windows(windows, default_config)
        # Single value should return max of value and default
        assert result >= DEFAULT_MIN_LIMIT

    def test_multiple_windows_p90(self, default_config):
        """Test P90 calculation with multiple windows."""
        # Create windows with varying usage
        windows = [
            create_window(100_000),
            create_window(200_000),
            create_window(300_000),
            create_window(400_000),
            create_window(500_000),
            create_window(600_000),
            create_window(700_000),
            create_window(800_000),
            create_window(900_000),
            create_window(1_000_000),  # P90 should be around this
        ]
        result = _calculate_p90_from_windows(windows, default_config)
        # P90 should be around 900,000 (90th percentile of 100k-1M range)
        assert result >= 800_000

    def test_filters_incomplete_windows(self, default_config):
        """Test that incomplete windows are filtered out."""
        windows = [
            create_window(100_000, hours=5.0),  # Complete
            create_window(1_000_000, hours=2.0),  # Incomplete - should be ignored
        ]
        result = _calculate_p90_from_windows(windows, default_config)
        # Only the 100K window should count
        assert result == max(100_000, DEFAULT_MIN_LIMIT)

    def test_filters_zero_token_windows(self, default_config):
        """Test that zero-token windows are filtered out."""
        windows = [
            create_window(0),
            create_window(500_000),
        ]
        result = _calculate_p90_from_windows(windows, default_config)
        assert result >= 500_000

    def test_respects_minimum_limit(self, default_config):
        """Test that result is never below minimum limit."""
        windows = [create_window(10_000), create_window(20_000)]
        result = _calculate_p90_from_windows(windows, default_config)
        assert result >= DEFAULT_MIN_LIMIT


class TestP90Calculator:
    """Tests for P90Calculator class."""

    def test_calculate_with_data(self):
        """Test P90 calculation with data."""
        calc = P90Calculator()
        windows = [
            create_window(300_000),
            create_window(350_000),
            create_window(400_000),
        ]
        result = calc.calculate_p90_limit(windows)
        assert result is not None
        assert result >= DEFAULT_MIN_LIMIT

    def test_calculate_with_none_returns_none(self):
        """Test that None input returns None."""
        calc = P90Calculator()
        assert calc.calculate_p90_limit(None) is None

    def test_calculate_with_empty_returns_none(self):
        """Test that empty list returns None."""
        calc = P90Calculator()
        assert calc.calculate_p90_limit([]) is None

    def test_caching_enabled(self):
        """Test that caching works."""
        calc = P90Calculator()
        windows = [create_window(500_000), create_window(600_000)]

        # First call
        result1 = calc.calculate_p90_limit(windows, use_cache=True)

        # Second call with same windows should use cache
        result2 = calc.calculate_p90_limit(windows, use_cache=True)

        assert result1 == result2

    def test_caching_disabled(self):
        """Test calculation without caching."""
        calc = P90Calculator()
        windows = [create_window(500_000)]
        result = calc.calculate_p90_limit(windows, use_cache=False)
        assert result is not None

    def test_clear_cache(self):
        """Test cache clearing."""
        calc = P90Calculator()
        windows = [create_window(500_000)]
        calc.calculate_p90_limit(windows, use_cache=True)
        calc.clear_cache()
        # Should still work after clearing cache
        result = calc.calculate_p90_limit(windows, use_cache=True)
        assert result is not None

    def test_custom_config(self):
        """Test calculator with custom config."""
        config = P90Config(
            common_limits=(100_000,),
            limit_threshold=0.90,
            default_min_limit=50_000,
            cache_ttl_seconds=60,
        )
        calc = P90Calculator(config)
        windows = [create_window(30_000)]
        result = calc.calculate_p90_limit(windows)
        # Should return at least the custom minimum
        assert result is not None
        assert result >= 50_000


class TestGlobalFunctions:
    """Tests for module-level convenience functions."""

    def test_get_calculator_singleton(self):
        """Test that get_calculator returns consistent instance."""
        calc1 = get_calculator()
        calc2 = get_calculator()
        assert calc1 is calc2

    def test_get_p90_limit_with_fallback(self):
        """Test P90 with fallback when no data."""
        # This will likely return fallback since we don't have real data
        result = get_p90_limit_with_fallback(
            days=1,
            fallback_limit=500_000,
        )
        # Should return either calculated P90 or fallback
        assert result >= 0


class TestP90EdgeCases:
    """Tests for edge cases in P90 calculation."""

    def test_all_identical_values(self):
        """Test P90 when all windows have identical usage."""
        windows = [create_window(300_000) for _ in range(10)]
        calc = P90Calculator()
        result = calc.calculate_p90_limit(windows)
        assert result == 300_000

    def test_very_high_usage(self):
        """Test P90 with very high usage values."""
        windows = [create_window(10_000_000) for _ in range(5)]
        calc = P90Calculator()
        result = calc.calculate_p90_limit(windows)
        assert result >= 10_000_000

    def test_mixed_complete_incomplete(self):
        """Test with mix of complete and incomplete windows."""
        now = datetime.now(timezone.utc)
        windows = [
            # Complete windows with moderate usage
            create_window(300_000, hours=5.0, start_time=now - timedelta(hours=15)),
            create_window(350_000, hours=5.0, start_time=now - timedelta(hours=10)),
            # Incomplete current window with high usage (should be ignored)
            create_window(1_000_000, hours=2.0, start_time=now - timedelta(hours=2)),
        ]
        calc = P90Calculator()
        result = calc.calculate_p90_limit(windows)
        # Should be based on complete windows only
        assert result < 1_000_000
