"""Tests for ralph.meters — bucket model, discovery, and EMA forecaster."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from ralph.meters import (
    BucketRegistry,
    EMAForecaster,
    MeterIdentity,
    ccs_projects_dir,
    compute_bucket,
    discover_ccs_accounts,
    hottest_bucket,
    usage_roots,
)
from ralph.process import Tool


class TestDiscoverCCSAccounts:
    def test_returns_empty_when_instances_dir_missing(self, tmp_path: Path):
        assert discover_ccs_accounts(tmp_path / "nonexistent") == []

    def test_lists_accounts_with_projects_subdir(self, tmp_path: Path):
        for acct in ("personal", "personal2", "personal3"):
            (tmp_path / acct / "projects").mkdir(parents=True)
        # A dir without projects/ is ignored (account not yet initialized)
        (tmp_path / "unused").mkdir()
        # Dot-prefixed entries (like .locks) are ignored
        (tmp_path / ".locks" / "projects").mkdir(parents=True)

        assert discover_ccs_accounts(tmp_path) == ["personal", "personal2", "personal3"]


class TestUsageRoots:
    def test_claude_returns_host_dir(self):
        roots = usage_roots(Tool.CLAUDE)
        assert len(roots) == 1
        ident, path = roots[0]
        assert ident == MeterIdentity(provider="anthropic")

    def test_ccs_with_profile_returns_account_dir(self):
        roots = usage_roots(Tool.CCS, ccs_profile="personal2")
        assert len(roots) == 1
        ident, path = roots[0]
        assert ident == MeterIdentity(provider="anthropic", account="personal2")
        assert str(path).endswith("/.ccs/instances/personal2/projects")

    def test_ccs_with_pool_returns_one_per_account(self):
        roots = usage_roots(Tool.CCS, ccs_pool=["a", "b", "c"])
        assert len(roots) == 3
        accounts = [r[0].account for r in roots]
        assert accounts == ["a", "b", "c"]

    def test_pool_takes_precedence_over_profile(self):
        roots = usage_roots(Tool.CCS, ccs_profile="personal", ccs_pool=["a", "b"])
        accounts = [r[0].account for r in roots]
        assert accounts == ["a", "b"]

    def test_amp_and_opencode_return_empty(self):
        assert usage_roots(Tool.AMP) == []
        assert usage_roots(Tool.OPENCODE) == []


class TestMeterIdentityLabel:
    def test_label_with_account(self):
        ident = MeterIdentity(provider="anthropic", account="personal3")
        assert ident.label == "anthropic:personal3"

    def test_label_without_account(self):
        ident = MeterIdentity(provider="anthropic")
        assert ident.label == "anthropic"


class TestEMAForecaster:
    def test_initial_value_is_first_sample(self):
        fc = EMAForecaster(alpha=0.3)
        fc.update(1000)
        assert fc.value == 1000.0
        assert fc.samples == 1

    def test_subsequent_updates_weight_alpha(self):
        fc = EMAForecaster(alpha=0.5)
        fc.update(1000)
        fc.update(2000)
        # 0.5 * 2000 + 0.5 * 1000 = 1500
        assert fc.value == 1500.0

    def test_zero_and_negative_samples_ignored(self):
        fc = EMAForecaster()
        fc.update(0)
        fc.update(-100)
        assert fc.value is None
        assert fc.samples == 0

    def test_iterations_until_is_none_with_no_samples(self):
        fc = EMAForecaster()
        assert fc.iterations_until(10_000) is None

    def test_iterations_until_divides_remaining_by_ema(self):
        fc = EMAForecaster()
        fc.update(50_000)
        # 150k remaining / 50k per iter = 3
        assert fc.iterations_until(150_000) == 3

    def test_iterations_until_zero_when_no_headroom(self):
        fc = EMAForecaster()
        fc.update(1000)
        assert fc.iterations_until(0) == 0

    def test_seed_resets_and_feeds_history(self):
        fc = EMAForecaster(alpha=0.3)
        fc.update(999_999)  # should be wiped
        fc.seed([1000, 2000, 3000])
        # Final EMA after seeding: 0.3*3000 + 0.7*(0.3*2000 + 0.7*1000)
        expected = 0.3 * 3000 + 0.7 * (0.3 * 2000 + 0.7 * 1000)
        assert abs(fc.value - expected) < 1e-6


class TestBucketRegistry:
    def test_snapshot_returns_one_bucket_per_root(self, tmp_path: Path):
        identities = [
            (MeterIdentity(provider="anthropic", account="a"), tmp_path / "a"),
            (MeterIdentity(provider="anthropic", account="b"), tmp_path / "b"),
        ]
        for _, p in identities:
            p.mkdir()
        registry = BucketRegistry(roots=identities, limit=1000)
        buckets = registry.snapshot()
        assert len(buckets) == 2
        assert [b.identity.account for b in buckets] == ["a", "b"]

    def test_close_iteration_updates_forecaster(self, tmp_path: Path):
        ident = MeterIdentity(provider="anthropic", account="x")
        registry = BucketRegistry(
            roots=[(ident, tmp_path / "x")], limit=300_000,
        )
        # Simulate a stream-fed iteration
        from ralph.usage import UsageRecord
        now = datetime.now(timezone.utc)
        registry.add_stream_record(
            ident,
            UsageRecord(
                timestamp=now,
                input_tokens=10_000,
                output_tokens=5_000,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
            ),
        )
        registry.add_stream_record(
            ident,
            UsageRecord(
                timestamp=now,
                input_tokens=8_000,
                output_tokens=4_000,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
            ),
        )

        totals = registry.close_iteration()
        assert totals[ident] == 27_000  # 10k + 5k + 8k + 4k
        # Buffer cleared after close_iteration
        assert registry._iter_records[ident] == []
        # Forecaster was updated
        assert registry._forecasters[ident].value == 27_000.0

    def test_active_identity_marked_in_snapshot(self, tmp_path: Path):
        ident_a = MeterIdentity(provider="anthropic", account="a")
        ident_b = MeterIdentity(provider="anthropic", account="b")
        registry = BucketRegistry(
            roots=[(ident_a, tmp_path / "a"), (ident_b, tmp_path / "b")],
            limit=1000,
        )
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        buckets = registry.snapshot(active=ident_b)
        by_account = {b.identity.account: b for b in buckets}
        assert by_account["b"].is_active is True
        assert by_account["a"].is_active is False


class TestHottestBucket:
    def test_returns_bucket_with_max_percentage(self):
        from ralph.meters import Bucket
        b1 = Bucket(
            identity=MeterIdentity(provider="anthropic", account="a"),
            tokens_used=100, limit=1000, percentage=10.0,
            resets_at=None, cost_usd=0.0, source="jsonl",
        )
        b2 = Bucket(
            identity=MeterIdentity(provider="anthropic", account="b"),
            tokens_used=800, limit=1000, percentage=80.0,
            resets_at=None, cost_usd=0.0, source="jsonl",
        )
        assert hottest_bucket([b1, b2]).identity.account == "b"

    def test_returns_none_for_empty_list(self):
        assert hottest_bucket([]) is None


class TestBucketRegistryLimitFor:
    """The limit_for callable gives each identity its own detected limit."""

    def test_limit_for_is_invoked_per_identity(self, tmp_path: Path):
        ident_a = MeterIdentity(provider="anthropic", account="a")
        ident_b = MeterIdentity(provider="anthropic", account="b")
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()

        seen: list[MeterIdentity] = []

        def limit_for(ident: MeterIdentity, root):
            seen.append(ident)
            return (
                {"a": 1_000_000, "b": 2_000_000}[ident.account],
                "detected",
                5,
                {"a": 1_200_000, "b": 2_800_000}[ident.account],
            )

        registry = BucketRegistry(
            roots=[(ident_a, tmp_path / "a"), (ident_b, tmp_path / "b")],
            limit=300_000,
            limit_for=limit_for,
        )
        assert set(seen) == {ident_a, ident_b}

        buckets = registry.snapshot()
        by_account = {b.identity.account: b for b in buckets}
        assert by_account["a"].limit == 1_000_000
        assert by_account["b"].limit == 2_000_000
        assert by_account["a"].limit_source == "detected"
        assert by_account["a"].limit_hit_count == 5
        assert by_account["a"].limit_max_observed == 1_200_000
        assert by_account["b"].limit_max_observed == 2_800_000

    def test_missing_identity_falls_back_to_global_limit(self, tmp_path: Path):
        ident = MeterIdentity(provider="anthropic", account="x")
        (tmp_path / "x").mkdir()

        def limit_for(_id, _root):
            raise RuntimeError("detector transient failure")

        registry = BucketRegistry(
            roots=[(ident, tmp_path / "x")],
            limit=99_000,
            limit_for=limit_for,
        )
        value, source, hits, max_obs = registry.limit_for_identity(ident)
        # Detector threw → nothing cached for this identity → global fallback.
        assert value == 99_000
        assert source == "default"
        assert hits == 0
        assert max_obs == 0

    def test_stream_only_identity_uses_global_limit(self, tmp_path: Path):
        """amp/opencode buckets register via ensure_stream_bucket and have no
        root — limit_for isn't called for them. They inherit the registry's
        global limit."""
        registry = BucketRegistry(roots=[], limit=555_000)
        stream_id = MeterIdentity(provider="amp")
        registry.ensure_stream_bucket(stream_id)
        value, source, _hits, _max = registry.limit_for_identity(stream_id)
        assert value == 555_000
        assert source == "default"

    def test_refresh_limits_can_update_values(self, tmp_path: Path):
        ident = MeterIdentity(provider="anthropic", account="only")
        (tmp_path / "only").mkdir()
        state = {"value": 1_000_000}

        def limit_for(_id, _root):
            return (state["value"], "detected", 1, state["value"] + 500_000)

        registry = BucketRegistry(
            roots=[(ident, tmp_path / "only")],
            limit=300_000,
            limit_for=limit_for,
        )
        assert registry.limit_for_identity(ident)[0] == 1_000_000

        # Simulate a fresh detection that found a higher cap.
        state["value"] = 5_000_000
        registry.refresh_limits()
        assert registry.limit_for_identity(ident)[0] == 5_000_000


class TestComputeBucketWithStreamMerge:
    """compute_bucket unions JSONL history with live-stream records."""

    def test_stream_records_contribute_to_percentage(self, tmp_path: Path):
        from ralph.usage import UsageRecord
        ident = MeterIdentity(provider="anthropic", account="x")
        now = datetime.now(timezone.utc)
        stream_records = [
            UsageRecord(
                timestamp=now - timedelta(minutes=1),
                input_tokens=50_000,
                output_tokens=25_000,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
            ),
        ]
        # Root doesn't exist → no JSONL data, stream-only.
        b = compute_bucket(
            identity=ident,
            root=tmp_path / "nonexistent",
            limit=300_000,
            extra_records=stream_records,
            now=now,
        )
        assert b.tokens_used == 75_000
        assert b.source == "stream"
        assert b.percentage == pytest.approx(25.0)

    def test_empty_bucket_has_zero_percentage(self, tmp_path: Path):
        b = compute_bucket(
            identity=MeterIdentity(provider="anthropic"),
            root=tmp_path,
            limit=1000,
        )
        assert b.tokens_used == 0
        assert b.percentage == 0.0
        assert b.source == "empty"
