"""Tests for ralph.limit_detector — per-account 5h-limit auto-detection."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from ralph.limit_detector import (
    DETECTION_WINDOW_DAYS,
    DetectedLimit,
    _cache_fresh,
    clear_detection_cache,
    detect_limit,
)
from ralph.usage import UsageAggregate


def _window(start: datetime, duration_hours: float, tokens: int) -> UsageAggregate:
    end = start + timedelta(hours=duration_hours)
    # rate_limited_tokens = input + output + cache_creation; stuff everything
    # into input_tokens to keep the math obvious.
    return UsageAggregate(
        window_start=start,
        window_end=end,
        input_tokens=tokens,
        output_tokens=0,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        message_count=1,
        request_count=1,
        cost_usd=0.0,
        oldest_record_timestamp=start,
    )


class TestDetectedLimitDataclass:
    def test_is_empirical_only_for_observed_sources(self):
        assert DetectedLimit(value=1, source="detected").is_empirical is True
        assert DetectedLimit(value=1, source="cached").is_empirical is True
        assert DetectedLimit(value=1, source="default").is_empirical is False
        assert DetectedLimit(value=1, source="override").is_empirical is False

    def test_describe_default(self):
        d = DetectedLimit(value=300_000, source="default")
        assert "default" in d.describe()

    def test_describe_detected_shows_hits_and_window(self):
        d = DetectedLimit(
            value=1_500_000, source="detected", hit_count=12, window_days=14
        )
        text = d.describe()
        assert "12 hits" in text
        assert "14d" in text


class TestDetectLimit:
    def test_none_root_returns_default(self, tmp_path: Path):
        result = detect_limit(None, default=42_000, cache_path=tmp_path / "c.json")
        assert result.value == 42_000
        assert result.source == "default"

    def test_root_with_no_history_returns_default(self, tmp_path: Path):
        root = tmp_path / "projects"
        root.mkdir()
        result = detect_limit(root, default=99_000, cache_path=tmp_path / "c.json")
        assert result.value == 99_000
        assert result.source == "default"
        assert result.hit_count == 0

    def test_detected_value_is_p90_of_hit_windows(self, tmp_path: Path):
        """If the account has a mix of windows near Max-5x's 1.5M ceiling,
        detection should land on that scale, not 300k."""
        root = tmp_path / "projects"
        root.mkdir()
        base = datetime.now(timezone.utc) - timedelta(days=7)
        synthetic = [
            _window(base + timedelta(hours=i * 6), 5.0, t)
            for i, t in enumerate([
                1_400_000, 1_450_000, 1_500_000, 1_480_000, 1_510_000,
                1_420_000, 1_490_000, 1_475_000, 1_495_000, 1_500_000,
            ])
        ]

        with patch(
            "ralph.usage.get_historical_5hour_windows",
            return_value=synthetic,
        ):
            result = detect_limit(
                root, default=300_000, cache_path=tmp_path / "cache.json"
            )

        assert result.source == "detected"
        assert result.hit_count == 10
        # All hits above 1.425M (Max-5x * 0.95) — P90 sits in the 1.4-1.5M range.
        assert 1_400_000 <= result.value <= 1_550_000
        # max_observed captures the single heaviest window.
        assert result.max_observed == 1_510_000

    def test_max_observed_exceeds_p90_for_bursty_history(self, tmp_path: Path):
        """When the bulk of hit-windows cluster well below a few peaks, P90
        reflects the common case and ``max_observed`` reveals the spike —
        this is what makes the 'P90 X — max Y' range useful. Models the
        personal2-account pattern we observed in the real data (38 hits
        from ~270k up to 14.2M, P90 ≈ 4.6M, max ≈ 14.2M)."""
        root = tmp_path / "projects"
        root.mkdir()
        base = datetime.now(timezone.utc) - timedelta(days=14)
        # 35 typical small windows (around 1-2M) + 3 big-ish + 1 spike.
        # With ≥40 samples, P90 stays around the 90th-percentile sample instead
        # of being pulled to the top by linear interpolation.
        token_counts = (
            [1_000_000] * 10 + [1_500_000] * 15 + [2_000_000] * 10
            + [4_500_000] * 3 + [14_000_000]
        )
        synthetic = [
            _window(base + timedelta(hours=i * 6), 5.0, t)
            for i, t in enumerate(token_counts)
        ]

        with patch(
            "ralph.usage.get_historical_5hour_windows",
            return_value=synthetic,
        ):
            result = detect_limit(
                root, default=300_000, cache_path=tmp_path / "cache.json"
            )

        assert result.source == "detected"
        assert result.hit_count == len(token_counts)
        assert result.max_observed == 14_000_000
        # P90 reflects the bulk of activity, not the single spike.
        assert result.value < result.max_observed
        assert result.value <= 5_000_000, (
            f"P90 ({result.value}) should be well under max spike "
            f"({result.max_observed}) for bursty history"
        )

    def test_cached_entry_is_reused(self, tmp_path: Path):
        root = tmp_path / "projects"
        root.mkdir()
        cache_path = tmp_path / "cache.json"
        now = datetime.now(timezone.utc).isoformat()
        cache_path.write_text(json.dumps({
            str(root.resolve()): {
                "value": 2_000_000,
                "source": "detected",
                "hit_count": 7,
                "window_days": 14,
                "detected_at": now,
                "max_observed": 5_500_000,
            }
        }))

        with patch(
            "ralph.usage.get_historical_5hour_windows"
        ) as mock_hist:
            result = detect_limit(root, default=300_000, cache_path=cache_path)
            # Fresh cache hit → no JSONL scan performed.
            mock_hist.assert_not_called()

        assert result.source == "cached"
        assert result.value == 2_000_000
        assert result.hit_count == 7
        assert result.max_observed == 5_500_000

    def test_force_bypasses_cache(self, tmp_path: Path):
        root = tmp_path / "projects"
        root.mkdir()
        cache_path = tmp_path / "cache.json"
        now = datetime.now(timezone.utc).isoformat()
        cache_path.write_text(json.dumps({
            str(root.resolve()): {
                "value": 2_000_000,
                "source": "detected",
                "hit_count": 7,
                "window_days": 14,
                "detected_at": now,
            }
        }))

        with patch(
            "ralph.usage.get_historical_5hour_windows",
            return_value=[],
        ):
            result = detect_limit(
                root, default=300_000, cache_path=cache_path, force=True
            )
        # No history → default returned despite the pre-existing cache entry.
        assert result.source == "default"
        assert result.value == 300_000

    def test_expired_cache_entry_triggers_recompute(self, tmp_path: Path):
        root = tmp_path / "projects"
        root.mkdir()
        cache_path = tmp_path / "cache.json"
        old = (
            datetime.now(timezone.utc) - timedelta(days=2)
        ).isoformat()
        cache_path.write_text(json.dumps({
            str(root.resolve()): {
                "value": 2_000_000,
                "source": "detected",
                "hit_count": 7,
                "window_days": 14,
                "detected_at": old,
            }
        }))

        with patch(
            "ralph.usage.get_historical_5hour_windows",
            return_value=[],
        ):
            result = detect_limit(root, default=300_000, cache_path=cache_path)
        # Expired entry → fresh compute → default (no history).
        assert result.source == "default"


class TestClearDetectionCache:
    def test_removes_file_and_reports_count(self, tmp_path: Path):
        cache_path = tmp_path / "cache.json"
        cache_path.write_text(json.dumps({"a": {}, "b": {}, "c": {}}))
        n = clear_detection_cache(cache_path)
        assert n == 3
        assert not cache_path.exists()

    def test_returns_zero_if_cache_missing(self, tmp_path: Path):
        n = clear_detection_cache(tmp_path / "nonexistent.json")
        assert n == 0


class TestCacheFreshnessHelper:
    def test_fresh_timestamp_passes(self):
        now_iso = datetime.now(timezone.utc).isoformat()
        import time as _time
        assert _cache_fresh({"detected_at": now_iso}, _time.time()) is True

    def test_missing_timestamp_fails(self):
        import time as _time
        assert _cache_fresh({}, _time.time()) is False

    def test_malformed_timestamp_fails(self):
        import time as _time
        assert _cache_fresh({"detected_at": "not-a-date"}, _time.time()) is False
