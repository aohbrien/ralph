"""P90-based token limit auto-detection.

This module calculates the 90th percentile of historical usage to auto-detect
the user's actual token limits based on observed usage patterns.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import timedelta
from functools import lru_cache
from pathlib import Path
from statistics import quantiles
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ralph.usage import UsageAggregate

# Common token limits across Claude plans
# Used to detect when a user "hit" their limit
COMMON_TOKEN_LIMITS: tuple[int, ...] = (
    30_000,      # Free tier estimate
    300_000,     # Pro tier
    1_500_000,   # Max 5x
    6_000_000,   # Max 20x
)

# Threshold for detecting a "limit hit" (95% of limit)
LIMIT_DETECTION_THRESHOLD = 0.95

# Default minimum limit (Pro tier)
DEFAULT_MIN_LIMIT = 300_000

# Cache TTL in seconds (1 hour)
DEFAULT_CACHE_TTL = 60 * 60


@dataclass(frozen=True)
class P90Config:
    """Configuration for P90 limit calculation."""

    common_limits: tuple[int, ...]
    limit_threshold: float
    default_min_limit: int
    cache_ttl_seconds: int


def _did_hit_limit(
    tokens: int,
    common_limits: tuple[int, ...],
    threshold: float,
) -> bool:
    """
    Check if a token count suggests the user hit a limit.

    Args:
        tokens: Token count for a window
        common_limits: Known common limits
        threshold: Percentage threshold (0-1) for "hitting" a limit

    Returns:
        True if usage suggests the user hit any common limit
    """
    return any(tokens >= limit * threshold for limit in common_limits)


def _is_complete_window(window: "UsageAggregate") -> bool:
    """
    Check if a window is complete (full 5 hours).

    A window is incomplete if its duration is less than 5 hours,
    which happens for the current/active window.

    Args:
        window: UsageAggregate to check

    Returns:
        True if window is a complete 5-hour window
    """
    duration = window.window_end - window.window_start
    # Allow some tolerance (4.9 hours minimum)
    return duration >= timedelta(hours=4, minutes=54)


def _get_window_tokens(window: "UsageAggregate") -> int:
    """Get the token count for rate limiting purposes.

    Uses rate_limited_tokens which excludes cache reads.
    """
    return window.rate_limited_tokens


def _calculate_p90_from_windows(
    windows: list["UsageAggregate"],
    config: P90Config,
) -> int:
    """
    Calculate P90 limit from historical windows.

    Args:
        windows: List of UsageAggregate objects
        config: P90 calculation configuration

    Returns:
        Calculated P90 limit in tokens
    """
    # Extract token counts from windows that "hit" a limit
    # Only consider complete windows (not the current/active one)
    # Use rate_limited_tokens (excludes cache reads)
    hit_windows = [
        _get_window_tokens(w)
        for w in windows
        if _is_complete_window(w)
        and _get_window_tokens(w) > 0
        and _did_hit_limit(_get_window_tokens(w), config.common_limits, config.limit_threshold)
    ]

    # If no limit-hitting windows, fall back to all complete windows
    if not hit_windows:
        hit_windows = [
            _get_window_tokens(w)
            for w in windows
            if _is_complete_window(w) and _get_window_tokens(w) > 0
        ]

    # If still no data, return default
    if not hit_windows:
        return config.default_min_limit

    # Need at least 2 data points for quantiles
    if len(hit_windows) < 2:
        return max(hit_windows[0], config.default_min_limit)

    # Calculate 90th percentile (index 8 of 10 quantiles)
    try:
        q = quantiles(hit_windows, n=10)[8]
        return max(int(q), config.default_min_limit)
    except Exception:
        # Fallback if quantiles fails
        return max(max(hit_windows), config.default_min_limit)


class P90Calculator:
    """Calculates P90-based token limits with caching."""

    def __init__(self, config: P90Config | None = None) -> None:
        """
        Initialize the calculator.

        Args:
            config: Configuration for P90 calculation. Uses defaults if None.
        """
        if config is None:
            config = P90Config(
                common_limits=COMMON_TOKEN_LIMITS,
                limit_threshold=LIMIT_DETECTION_THRESHOLD,
                default_min_limit=DEFAULT_MIN_LIMIT,
                cache_ttl_seconds=DEFAULT_CACHE_TTL,
            )
        self._config = config

    @lru_cache(maxsize=1)
    def _cached_calculate(
        self,
        cache_key: int,
        windows_hash: tuple[tuple[int, int], ...],
    ) -> int:
        """
        Internal cached calculation.

        Args:
            cache_key: Time-based cache key for TTL
            windows_hash: Hashable representation of windows

        Returns:
            Calculated P90 limit
        """
        # This is a bit of a hack - we can't pass the actual windows
        # to an lru_cache function, so we pass a hash and store the
        # windows in an instance variable for access
        return _calculate_p90_from_windows(self._windows_for_cache, self._config)

    def calculate_p90_limit(
        self,
        windows: list["UsageAggregate"] | None = None,
        use_cache: bool = True,
    ) -> int | None:
        """
        Calculate the P90 limit from historical windows.

        Args:
            windows: List of UsageAggregate objects. If None, returns None.
            use_cache: Whether to use caching (default True)

        Returns:
            Calculated P90 limit in tokens, or None if no data
        """
        if not windows:
            return None

        if not use_cache:
            return _calculate_p90_from_windows(windows, self._config)

        # Store windows for cache access
        self._windows_for_cache = windows

        # Create time-based cache key (expires every cache_ttl_seconds)
        ttl = self._config.cache_ttl_seconds
        cache_key = int(time.time() // ttl)

        # Create hashable representation of windows
        windows_hash = tuple(
            (int(w.window_start.timestamp()), w.total_tokens)
            for w in windows
        )

        return self._cached_calculate(cache_key, windows_hash)

    def clear_cache(self) -> None:
        """Clear the calculation cache."""
        self._cached_calculate.cache_clear()


# Global calculator instance
_calculator: P90Calculator | None = None


def get_calculator() -> P90Calculator:
    """Get or create the global P90 calculator."""
    global _calculator
    if _calculator is None:
        _calculator = P90Calculator()
    return _calculator


def get_p90_limit(
    days: int = 14,
    claude_dir: Path | None = None,
    use_cache: bool = True,
) -> int | None:
    """
    Get the P90-calculated token limit.

    This function integrates with the usage module to get historical
    5-hour windows and calculate the P90 limit.

    Args:
        days: Number of days to look back for history
        claude_dir: Claude projects directory (for usage data)
        use_cache: Whether to use caching

    Returns:
        Calculated P90 limit in tokens, or None if insufficient data
    """
    from ralph.usage import get_historical_5hour_windows

    # Get historical windows
    windows = get_historical_5hour_windows(days=days, claude_dir=claude_dir)

    if not windows:
        return None

    # Calculate P90
    calculator = get_calculator()
    return calculator.calculate_p90_limit(windows, use_cache=use_cache)


def get_p90_limit_with_fallback(
    days: int = 14,
    fallback_limit: int = DEFAULT_MIN_LIMIT,
    claude_dir: Path | None = None,
) -> int:
    """
    Get the P90-calculated token limit with a fallback.

    Args:
        days: Number of days to look back for history
        fallback_limit: Limit to use if P90 calculation fails
        claude_dir: Claude projects directory (for usage data)

    Returns:
        Calculated P90 limit or fallback if insufficient data
    """
    p90 = get_p90_limit(days=days, claude_dir=claude_dir)
    return p90 if p90 is not None else fallback_limit
