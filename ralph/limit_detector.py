"""Per-bucket 5-hour token limit auto-detection.

Replaces the global ``DEFAULT_5H_LIMIT = 300_000`` fallback with an empirical,
per-(provider, account) estimate derived from the account's own JSONL history.

Why this exists: Anthropic tunes rate-limit caps over time and they differ by
plan. With CCS, each account can be on a different plan entirely. A single
hardcoded number is always wrong for *somebody*. We let the data answer the
question: "what ceiling has this account actually been pushing against?"

Algorithm: reuse ``ralph.p90.get_p90_limit`` — for each account, fetch the last
N days of 5-hour windows, filter to those that hit ≥95% of any common plan
ceiling, take the 9th decile. Falls back to the caller's default when there
aren't enough samples yet.

Persistence: results are cached to ``~/.ralph/detected-limits.json`` keyed on
the account's projects-dir path, so fresh Ralph invocations don't re-scan
JSONL on every startup. Cache entries carry a detected_at timestamp and bust
after ``DETECTION_TTL_SECONDS`` (default 24h).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from ralph.p90 import DEFAULT_MIN_LIMIT, get_p90_limit
from ralph.session import DEFAULT_RALPH_DIR

logger = logging.getLogger(__name__)

DETECTED_LIMITS_PATH = DEFAULT_RALPH_DIR / "detected-limits.json"
DETECTION_WINDOW_DAYS = 14
DETECTION_TTL_SECONDS = 24 * 3600


@dataclass
class DetectedLimit:
    """A 5-hour token limit paired with provenance.

    Callers care about more than just the number — the UI should distinguish
    "this was observed from N hits in 14 days" from "we fell back to the Pro
    default because we haven't seen you yet". ``source`` carries that signal.

    We keep two observed values:
      - ``value`` is the P90 of hit-windows — conservative, used for pacing.
      - ``max_observed`` is the single highest hit-window seen in the history
        period — a ceiling hint, useful for the UI to say "here's the most
        you've ever pushed against" and for users with bursty patterns where
        P90 under-estimates their real cap.
    """

    value: int
    source: str  # "detected" | "cached" | "default" | "override"
    hit_count: int = 0
    window_days: int = DETECTION_WINDOW_DAYS
    detected_at: str | None = None  # ISO timestamp
    max_observed: int = 0  # single highest hit-window in the period

    @property
    def is_empirical(self) -> bool:
        return self.source in ("detected", "cached")

    def describe(self) -> str:
        """Short human-readable suffix, e.g. 'detected from 12 hits over 14d'."""
        if self.source == "detected" or self.source == "cached":
            age = ""
            if self.source == "cached" and self.detected_at:
                age = f", {_format_age(self.detected_at)} ago"
            return f"detected from {self.hit_count} hits over {self.window_days}d{age}"
        if self.source == "override":
            return "override"
        return "default, no hits yet"


# ---------------------------------------------------------------------------
# Persistence layer
# ---------------------------------------------------------------------------


def _load_cache(path: Path = DETECTED_LIMITS_PATH) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        return cast(
            dict[str, dict[str, Any]],
            json.loads(path.read_text(encoding="utf-8")),
        )
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug(f"Could not read {path}: {exc}")
        return {}


def _save_cache(
    cache: dict[str, dict[str, Any]], path: Path = DETECTED_LIMITS_PATH
) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Simple write — no atomic rename dance; the file is advisory cache,
        # a partial write just means we re-detect next time.
        path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")
    except OSError as exc:
        logger.debug(f"Could not write {path}: {exc}")


def _cache_fresh(entry: dict[str, Any], now: float) -> bool:
    ts = entry.get("detected_at")
    if not isinstance(ts, str):
        return False
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return False
    age = now - dt.timestamp()
    return 0 <= age < DETECTION_TTL_SECONDS


def _format_age(iso_ts: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return "?"
    age_s = (datetime.now(timezone.utc) - dt).total_seconds()
    if age_s < 60:
        return f"{int(age_s)}s"
    if age_s < 3600:
        return f"{int(age_s // 60)}m"
    if age_s < 86400:
        return f"{int(age_s // 3600)}h"
    return f"{int(age_s // 86400)}d"


# ---------------------------------------------------------------------------
# Detection entry points
# ---------------------------------------------------------------------------


def detect_limit(
    root: Path | None,
    default: int = DEFAULT_MIN_LIMIT,
    days: int = DETECTION_WINDOW_DAYS,
    cache_path: Path = DETECTED_LIMITS_PATH,
    force: bool = False,
) -> DetectedLimit:
    """Detect the 5-hour limit for one account by its JSONL root.

    Args:
        root: Claude projects dir (e.g. ``~/.ccs/instances/personal3/projects``).
            ``None`` skips detection entirely → returns the default.
        default: Value to return when the account has no history yet.
        days: Window of history to consider.
        cache_path: Where persistent cache lives.
        force: Skip cache lookup and recompute.

    Returns:
        DetectedLimit with provenance. Never raises — falls back to default on
        any error.
    """
    if root is None:
        return DetectedLimit(value=default, source="default")

    key = str(root.resolve()) if root.exists() else str(root)
    cache = _load_cache(cache_path)
    now_ts = time.time()

    if not force:
        entry = cache.get(key)
        if entry and _cache_fresh(entry, now_ts):
            return DetectedLimit(
                value=int(entry["value"]),
                source="cached",
                hit_count=int(entry.get("hit_count", 0)),
                window_days=int(entry.get("window_days", days)),
                detected_at=entry.get("detected_at"),
                max_observed=int(entry.get("max_observed", 0)),
            )

    # Compute fresh
    try:
        p90, hit_count, max_observed = _compute_p90_with_count(root, days)
    except Exception as exc:
        logger.debug(f"P90 detection failed for {root}: {exc}")
        return DetectedLimit(value=default, source="default")

    # p90 returns its own default floor (300k) even when there are zero hits;
    # treat zero-hit results as "default" so the UI doesn't claim we detected
    # something when we really just fell back.
    if p90 is None or hit_count == 0:
        return DetectedLimit(
            value=default,
            source="default",
            hit_count=0,
            window_days=days,
        )

    now_iso = datetime.now(timezone.utc).isoformat()
    result = DetectedLimit(
        value=p90,
        source="detected",
        hit_count=hit_count,
        window_days=days,
        detected_at=now_iso,
        max_observed=max_observed,
    )
    cache[key] = asdict(result)
    _save_cache(cache, cache_path)
    return result


def _compute_p90_with_count(root: Path, days: int) -> tuple[int | None, int, int]:
    """Run P90 detection and also report hit count and the single max hit.

    Reimplements the filter half of ``ralph.p90._calculate_p90_from_windows`` so
    we can surface the hit count and max in the UI; for the actual P90 value we
    still defer to the shared algorithm.

    Returns:
        ``(p90, hit_count, max_observed)``. ``p90`` is None when there's no
        history at all. ``max_observed`` is 0 when there are no hits.
    """
    from ralph.p90 import (
        COMMON_TOKEN_LIMITS,
        LIMIT_DETECTION_THRESHOLD,
        _did_hit_limit,
        _is_complete_window,
    )
    from ralph.usage import get_historical_5hour_windows

    windows = get_historical_5hour_windows(days=days, claude_dir=root)
    if not windows:
        return None, 0, 0

    hits = [
        w.rate_limited_tokens
        for w in windows
        if _is_complete_window(w)
        and w.rate_limited_tokens > 0
        and _did_hit_limit(
            w.rate_limited_tokens, COMMON_TOKEN_LIMITS, LIMIT_DETECTION_THRESHOLD
        )
    ]
    hit_count = len(hits)
    max_observed = max(hits) if hits else 0

    value = get_p90_limit(days=days, claude_dir=root, use_cache=False)
    return value, hit_count, max_observed


def clear_detection_cache(cache_path: Path = DETECTED_LIMITS_PATH) -> int:
    """Wipe the persistent cache. Returns the number of entries removed."""
    cache = _load_cache(cache_path)
    n = len(cache)
    if cache_path.exists():
        try:
            cache_path.unlink()
        except OSError as exc:
            logger.debug(f"Could not delete {cache_path}: {exc}")
    return n
