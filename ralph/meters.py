"""Multi-bucket usage accounting for Ralph.

A "bucket" is a (provider, account) view of the 5-hour rate-limit window. The
old code collapsed all tools into a single bucket rooted at ``~/.claude/projects``,
which silently reported 0% for every CCS account (each account is an isolated
Claude home under ``~/.ccs/instances/<name>/``).

This module replaces that with one bucket per identity, plus a per-bucket EMA
forecaster that predicts how many more iterations fit before saturating.

Design notes:
- CCS accounts are discovered by enumerating ``~/.ccs/instances/<name>/projects``.
  Filesystem is authoritative; the config file can lag.
- ``compute_bucket()`` merges live-stream records into the JSONL aggregate, so
  fresh tokens from the current iteration show up before the tool flushes.
- Forecasters live in ``BucketRegistry`` and are updated at iteration close with
  the iteration's total rate-limited-token cost.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from ralph import usage as _usage
from ralph.process import Tool
from ralph.usage import UsageAggregate, UsageRecord

logger = logging.getLogger(__name__)


DEFAULT_5H_LIMIT = 300_000  # Pro tier default — mirrors usage.check_usage_before_run
DEFAULT_WINDOW = timedelta(hours=5)

HOST_CLAUDE_DIR = Path.home() / ".claude" / "projects"
CCS_INSTANCES_DIR = Path.home() / ".ccs" / "instances"


# ---------------------------------------------------------------------------
# Identity and roots
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MeterIdentity:
    """Stable identity for a usage bucket."""

    provider: str
    account: str | None = None

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.account}" if self.account else self.provider


def ccs_projects_dir(account: str) -> Path:
    """Path to the Claude projects directory for a CCS account."""
    return CCS_INSTANCES_DIR / account / "projects"


def discover_ccs_accounts(instances_dir: Path = CCS_INSTANCES_DIR) -> list[str]:
    """List CCS accounts that have a ``projects/`` subdirectory.

    We enumerate the filesystem rather than parsing ``~/.ccs/config.yaml`` —
    no YAML dependency and the filesystem is the authoritative record of which
    accounts actually have session data.
    """
    if not instances_dir.exists() or not instances_dir.is_dir():
        return []

    accounts: list[str] = []
    try:
        entries = sorted(instances_dir.iterdir())
    except OSError as e:
        logger.debug(f"Could not list {instances_dir}: {e}")
        return []

    for entry in entries:
        if not entry.is_dir():
            continue
        if entry.name.startswith("."):  # skip .locks/, .cache/, etc.
            continue
        if (entry / "projects").exists():
            accounts.append(entry.name)
    return accounts


def usage_roots(
    tool: Tool,
    ccs_profile: str | None = None,
    ccs_pool: list[str] | None = None,
) -> list[tuple[MeterIdentity, Path]]:
    """Select the (identity, Claude-projects dir) pairs for a given run.

    Pool wins over single-profile. Tools without JSONL storage return ``[]`` —
    their buckets are stream-fed only (see BucketRegistry.ensure_stream_bucket).
    """
    if tool == Tool.CLAUDE:
        return [(MeterIdentity(provider="anthropic"), HOST_CLAUDE_DIR)]

    if tool == Tool.CCS:
        if ccs_pool:
            return [
                (MeterIdentity(provider="anthropic", account=a), ccs_projects_dir(a))
                for a in ccs_pool
            ]
        if ccs_profile:
            return [
                (
                    MeterIdentity(provider="anthropic", account=ccs_profile),
                    ccs_projects_dir(ccs_profile),
                )
            ]
        discovered = discover_ccs_accounts()
        return [
            (MeterIdentity(provider="anthropic", account=a), ccs_projects_dir(a))
            for a in discovered
        ]

    # amp, opencode: stream is the only source.
    return []


# ---------------------------------------------------------------------------
# Bucket data model
# ---------------------------------------------------------------------------


@dataclass
class Bucket:
    """A snapshot of one (provider, account) 5-hour window."""

    identity: MeterIdentity
    tokens_used: int
    limit: int
    percentage: float
    resets_at: datetime | None
    cost_usd: float
    source: str  # "jsonl" | "stream" | "merged" | "empty"
    forecast_iters_to_90: int | None = None
    forecast_iters_to_100: int | None = None
    is_active: bool = False  # this bucket is being drawn from *right now*
    # Provenance for the `limit` value: "detected" | "cached" | "default" | "override"
    limit_source: str = "default"
    limit_hit_count: int = 0
    # Highest single hit-window observed in detection period. ``limit`` (P90)
    # is what we pace against; ``limit_max_observed`` is shown for context so
    # the UI can disclose when the real cap is likely higher than P90.
    limit_max_observed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity.label,
            "provider": self.identity.provider,
            "account": self.identity.account,
            "tokens_used": self.tokens_used,
            "limit": self.limit,
            "percentage": self.percentage,
            "resets_at": self.resets_at.isoformat() if self.resets_at else None,
            "cost_usd": self.cost_usd,
            "source": self.source,
            "forecast_iters_to_90": self.forecast_iters_to_90,
            "forecast_iters_to_100": self.forecast_iters_to_100,
            "is_active": self.is_active,
            "limit_source": self.limit_source,
            "limit_hit_count": self.limit_hit_count,
            "limit_max_observed": self.limit_max_observed,
        }


# ---------------------------------------------------------------------------
# Forecaster
# ---------------------------------------------------------------------------


@dataclass
class EMAForecaster:
    """Exponential moving average of per-iteration rate-limited-token cost.

    ``alpha=0.3`` weights recent iterations ~3x the iteration before that; fast
    enough to track regime changes (e.g. switching from cheap re-evaluation
    iterations to heavy coding iterations) without being thrashy.
    """

    alpha: float = 0.3
    value: float | None = None
    samples: int = 0

    def update(self, tokens: int) -> None:
        if tokens <= 0:
            return
        self.samples += 1
        if self.value is None:
            self.value = float(tokens)
        else:
            self.value = self.alpha * tokens + (1 - self.alpha) * self.value

    def seed(self, samples: list[int]) -> None:
        self.value = None
        self.samples = 0
        for s in samples:
            self.update(s)

    def iterations_until(self, tokens_remaining: int) -> int | None:
        if self.value is None or self.value <= 0:
            return None
        if tokens_remaining <= 0:
            return 0
        return int(tokens_remaining / self.value)


# ---------------------------------------------------------------------------
# Bucket computation
# ---------------------------------------------------------------------------


def _merge_record(agg: UsageAggregate, record: UsageRecord) -> UsageAggregate:
    new_oldest = agg.oldest_record_timestamp
    if new_oldest is None or record.timestamp < new_oldest:
        new_oldest = record.timestamp
    return replace(
        agg,
        input_tokens=agg.input_tokens + record.input_tokens,
        output_tokens=agg.output_tokens + record.output_tokens,
        cache_creation_input_tokens=agg.cache_creation_input_tokens
        + record.cache_creation_input_tokens,
        cache_read_input_tokens=agg.cache_read_input_tokens
        + record.cache_read_input_tokens,
        message_count=agg.message_count + 1,
        request_count=agg.request_count + 1,
        cost_usd=agg.cost_usd + record.cost_usd,
        oldest_record_timestamp=new_oldest,
    )


def _empty_aggregate(now: datetime, window: timedelta) -> UsageAggregate:
    return UsageAggregate(
        window_start=now - window,
        window_end=now,
        input_tokens=0,
        output_tokens=0,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        message_count=0,
        request_count=0,
        cost_usd=0.0,
        oldest_record_timestamp=None,
    )


def compute_bucket(
    identity: MeterIdentity,
    root: Path | None,
    limit: int,
    window: timedelta = DEFAULT_WINDOW,
    now: datetime | None = None,
    extra_records: list[UsageRecord] | None = None,
    forecaster: EMAForecaster | None = None,
    limit_source: str = "default",
    limit_hit_count: int = 0,
    limit_max_observed: int = 0,
) -> Bucket:
    """Build a Bucket for one identity by unioning JSONL history + live stream.

    ``limit_source`` / ``limit_hit_count`` carry the provenance of ``limit`` so
    the UI can show "detected from 12 hits over 14d" vs "default, no hits yet".
    ``limit_max_observed`` is the single highest hit window seen in the
    detection period — passed through so the UI can render a range
    ("P90 X — max Y") when the two differ.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    if root is not None:
        # Indirect access so tests that patch("ralph.usage.get_5hour_window_usage")
        # still intercept calls from this module.
        agg = _usage.get_5hour_window_usage(claude_dir=root, now=now)
        source = "jsonl" if agg.rate_limited_tokens > 0 else "empty"
    else:
        agg = _empty_aggregate(now, window)
        source = "empty"

    window_start = now - window
    if extra_records:
        for r in extra_records:
            if r.timestamp < window_start or r.timestamp > now:
                continue
            agg = _merge_record(agg, r)
        if agg.rate_limited_tokens > 0 and source == "empty":
            source = "stream"
        elif agg.rate_limited_tokens > 0 and source == "jsonl":
            source = "merged"

    tokens_used = agg.rate_limited_tokens
    percentage = min(100.0, (tokens_used / limit) * 100) if limit > 0 else 0.0

    remaining_to_100 = max(0, limit - tokens_used)
    remaining_to_90 = max(0, int(limit * 0.9) - tokens_used)

    fc_90: int | None = None
    fc_100: int | None = None
    if forecaster is not None:
        fc_90 = forecaster.iterations_until(remaining_to_90)
        fc_100 = forecaster.iterations_until(remaining_to_100)

    resets_at = (
        agg.oldest_record_timestamp + window
        if agg.oldest_record_timestamp is not None
        else None
    )

    return Bucket(
        identity=identity,
        tokens_used=tokens_used,
        limit=limit,
        percentage=percentage,
        resets_at=resets_at,
        cost_usd=agg.cost_usd,
        source=source,
        forecast_iters_to_90=fc_90,
        forecast_iters_to_100=fc_100,
        limit_source=limit_source,
        limit_hit_count=limit_hit_count,
        limit_max_observed=limit_max_observed,
    )


# ---------------------------------------------------------------------------
# BucketRegistry: runtime state for one Ralph run
# ---------------------------------------------------------------------------


class BucketRegistry:
    """Owns the buckets for a single Ralph run.

    Responsibilities:
      - Know which (identity, root) pairs exist for this run.
      - Buffer live-stream UsageRecord events per-identity.
      - Produce fresh bucket snapshots on demand.
      - Roll up each iteration's total cost into the forecaster.
    """

    def __init__(
        self,
        roots: list[tuple[MeterIdentity, Path]],
        limit: int = DEFAULT_5H_LIMIT,
        window: timedelta = DEFAULT_WINDOW,
        limit_for: Callable[
            [MeterIdentity, Path | None], tuple[int, str, int, int]
        ]
        | None = None,
    ):
        """
        Args:
            roots: Known (identity, claude-projects-dir) pairs for this run.
            limit: Fallback limit used when ``limit_for`` returns ``"default"``
                and for stream-only buckets that have no root to detect from.
            window: Window duration (default 5h).
            limit_for: Optional per-identity limit resolver returning
                ``(value, source, hit_count, max_observed)``. Wired by the
                runner to the P90 detector in :mod:`ralph.limit_detector`.
                Called once per identity at construction time and cached; use
                ``refresh_limits()`` to re-query.
        """
        self._roots: dict[MeterIdentity, Path] = dict(roots)
        self._limit = limit
        self._window = window
        self._limit_for = limit_for
        self._forecasters: dict[MeterIdentity, EMAForecaster] = {
            ident: EMAForecaster() for ident in self._roots
        }
        self._iter_records: dict[MeterIdentity, list[UsageRecord]] = {
            ident: [] for ident in self._roots
        }
        # Cache per-identity (value, source, hit_count, max_observed).
        self._per_identity_limits: dict[
            MeterIdentity, tuple[int, str, int, int]
        ] = {}
        self.refresh_limits()

    def refresh_limits(self) -> None:
        """(Re)query the per-identity limit resolver.

        Called once at construction, plus whenever the caller wants to bust
        the cache (e.g. after `--redetect`). Safe to call with no resolver —
        then every identity inherits the registry's global fallback.
        """
        self._per_identity_limits = {}
        if self._limit_for is None:
            return
        for ident in self._roots:
            try:
                value, source, hits, max_obs = self._limit_for(
                    ident, self._roots.get(ident)
                )
            except Exception:
                logger.debug("limit_for failed for %s", ident.label, exc_info=True)
                continue
            self._per_identity_limits[ident] = (
                int(value), source, int(hits), int(max_obs),
            )

    def limit_for_identity(
        self, identity: MeterIdentity
    ) -> tuple[int, str, int, int]:
        """Resolved (limit, source, hit_count, max_observed) for one identity.

        Falls back to ``(self._limit, "default", 0, 0)`` when nothing is cached
        for this identity — happens for stream-only buckets (amp/opencode)
        registered lazily via ``ensure_stream_bucket``.
        """
        cached = self._per_identity_limits.get(identity)
        if cached is not None:
            return cached
        return (self._limit, "default", 0, 0)

    @property
    def identities(self) -> list[MeterIdentity]:
        return list(self._roots.keys())

    @property
    def limit(self) -> int:
        return self._limit

    def ensure_stream_bucket(self, identity: MeterIdentity) -> None:
        """Register an identity that has no JSONL root (amp/opencode)."""
        if identity not in self._roots:
            # Use None sentinel path via separate dict — but keep API uniform
            # by storing the identity in _iter_records/_forecasters only.
            self._iter_records.setdefault(identity, [])
            self._forecasters.setdefault(identity, EMAForecaster())

    def add_stream_record(self, identity: MeterIdentity, record: UsageRecord) -> None:
        """Record a live-stream UsageRecord for the current iteration."""
        self.ensure_stream_bucket(identity)
        self._iter_records[identity].append(record)

    def snapshot(
        self,
        active: MeterIdentity | None = None,
        now: datetime | None = None,
    ) -> list[Bucket]:
        """Compute a fresh list of buckets for all known identities."""
        all_identities = set(self._iter_records.keys()) | set(self._roots.keys())
        buckets: list[Bucket] = []
        for ident in sorted(all_identities, key=lambda i: i.label):
            root = self._roots.get(ident)
            fc = self._forecasters.get(ident)
            extras = self._iter_records.get(ident, [])
            per_limit, limit_source, hit_count, max_obs = (
                self.limit_for_identity(ident)
            )
            b = compute_bucket(
                identity=ident,
                root=root,
                limit=per_limit,
                window=self._window,
                now=now,
                extra_records=extras,
                forecaster=fc,
                limit_source=limit_source,
                limit_hit_count=hit_count,
                limit_max_observed=max_obs,
            )
            if active is not None and ident == active:
                b.is_active = True
            buckets.append(b)
        return buckets

    def close_iteration(self) -> dict[MeterIdentity, int]:
        """Finalize the iteration: update forecasters and clear the live buffer.

        Returns per-identity rate-limited-token totals for this iteration.
        """
        totals: dict[MeterIdentity, int] = {}
        for ident, records in self._iter_records.items():
            total = sum(r.rate_limited_tokens for r in records)
            totals[ident] = total
            if total > 0:
                fc = self._forecasters.setdefault(ident, EMAForecaster())
                fc.update(total)
        for ident in list(self._iter_records.keys()):
            self._iter_records[ident] = []
        return totals


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def hottest_bucket(buckets: list[Bucket]) -> Bucket | None:
    """Return the bucket with the highest percentage, or None if empty."""
    return max(buckets, key=lambda b: b.percentage) if buckets else None
