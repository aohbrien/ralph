"""Tests for the account-claim extensions in ralph.session.

Ensures that peer Ralph instances can publish in-flight account claims via the
file-locked registry and that ``pick_least_loaded_account`` honors them so two
concurrent instances don't hotspot the same CCS account.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from ralph.session import (
    SessionInfo,
    clear_session_claim,
    get_concurrent_claims,
    pick_least_loaded_account,
    register_session,
    set_session_claim,
    unregister_session,
)


@pytest.fixture
def temp_lock_file(tmp_path: Path) -> Path:
    return tmp_path / "usage.lock"


class TestSessionInfoClaimFields:
    def test_defaults_are_none_and_zero(self):
        s = SessionInfo(pid=1, started_at="", last_heartbeat="")
        assert s.active_account is None
        assert s.iter_estimate_tokens == 0
        assert s.claim_expires_at is None
        assert s.claim_is_active is False

    def test_roundtrip_preserves_claim_fields(self):
        s = SessionInfo(
            pid=1,
            started_at="2026-04-21T12:00:00+00:00",
            last_heartbeat="2026-04-21T12:00:00+00:00",
            active_account="personal2",
            iter_estimate_tokens=55_000,
            claim_expires_at="2026-04-21T12:30:00+00:00",
        )
        data = s.to_dict()
        restored = SessionInfo.from_dict(data)
        assert restored.active_account == "personal2"
        assert restored.iter_estimate_tokens == 55_000
        assert restored.claim_expires_at == "2026-04-21T12:30:00+00:00"

    def test_backward_compat_legacy_dict_without_claim_fields(self):
        """Older on-disk registries don't have the new fields — defaults kick in."""
        legacy = {
            "pid": 123,
            "started_at": "2026-04-21T12:00:00+00:00",
            "last_heartbeat": "2026-04-21T12:00:00+00:00",
        }
        s = SessionInfo.from_dict(legacy)
        assert s.active_account is None
        assert s.iter_estimate_tokens == 0

    def test_claim_is_active_true_while_valid(self):
        future = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
        s = SessionInfo(
            pid=1, started_at="", last_heartbeat="",
            active_account="acct", iter_estimate_tokens=100,
            claim_expires_at=future,
        )
        assert s.claim_is_active is True

    def test_claim_is_active_false_when_expired(self):
        past = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        s = SessionInfo(
            pid=1, started_at="", last_heartbeat="",
            active_account="acct", iter_estimate_tokens=100,
            claim_expires_at=past,
        )
        assert s.claim_is_active is False


class TestSetAndClearClaim:
    def test_set_claim_publishes_to_registry(self, temp_lock_file: Path):
        register_session(lock_file=temp_lock_file)
        try:
            ok = set_session_claim(
                active_account="personal2",
                iter_estimate_tokens=42_000,
                claim_duration_seconds=120,
                lock_file=temp_lock_file,
            )
            assert ok is True

            claims = get_concurrent_claims(
                exclude_pid=-1,  # include our own in the sum
                lock_file=temp_lock_file,
            )
            assert claims.get("personal2") == 42_000
        finally:
            unregister_session(lock_file=temp_lock_file)

    def test_clear_claim_drops_from_registry(self, temp_lock_file: Path):
        register_session(lock_file=temp_lock_file)
        try:
            set_session_claim("acct", 10_000, 120, lock_file=temp_lock_file)
            clear_session_claim(lock_file=temp_lock_file)
            claims = get_concurrent_claims(
                exclude_pid=-1, lock_file=temp_lock_file,
            )
            assert "acct" not in claims
        finally:
            unregister_session(lock_file=temp_lock_file)

    def test_set_claim_returns_false_if_not_registered(self, temp_lock_file: Path):
        ok = set_session_claim("acct", 1000, 60, lock_file=temp_lock_file)
        assert ok is False


class TestPickLeastLoadedAccount:
    def test_picks_highest_headroom_when_no_peers(self, temp_lock_file: Path):
        picked = pick_least_loaded_account(
            pool=["a", "b", "c"],
            headroom_by_account={"a": 100_000, "b": 250_000, "c": 50_000},
            lock_file=temp_lock_file,
        )
        assert picked == "b"

    def test_peer_claim_reduces_effective_headroom(self, temp_lock_file: Path):
        """A peer drawing 200k from account b brings b's effective headroom
        below a's, so we pick a instead."""
        # Peer 999 has a 200k claim on account b
        with patch("os.getpid", return_value=999):
            register_session(lock_file=temp_lock_file)
            set_session_claim(
                active_account="b",
                iter_estimate_tokens=200_000,
                claim_duration_seconds=120,
                lock_file=temp_lock_file,
            )

        try:
            # From our perspective (a different PID), b's effective headroom is
            # 250k - 200k = 50k, so a (100k) wins.
            picked = pick_least_loaded_account(
                pool=["a", "b"],
                headroom_by_account={"a": 100_000, "b": 250_000},
                lock_file=temp_lock_file,
            )
            assert picked == "a"
        finally:
            with patch("os.getpid", return_value=999):
                unregister_session(lock_file=temp_lock_file)

    def test_expired_claims_are_ignored(self, temp_lock_file: Path):
        # Peer has an EXPIRED claim on b (claim_duration_seconds < 0 via mocking)
        with patch("os.getpid", return_value=1001):
            register_session(lock_file=temp_lock_file)
            # Publish, then rewrite expiry into the past.
            set_session_claim("b", 500_000, 60, lock_file=temp_lock_file)
            # Directly expire the claim timestamp.
            from ralph.session import (
                _read_registry_locked,
                _write_registry_and_unlock,
            )
            reg, f = _read_registry_locked(temp_lock_file)
            peer = reg.get_session(1001)
            assert peer is not None
            peer.claim_expires_at = (
                datetime.now(timezone.utc) - timedelta(seconds=1)
            ).isoformat()
            reg.add_session(peer)
            _write_registry_and_unlock(reg, temp_lock_file, f)

        try:
            picked = pick_least_loaded_account(
                pool=["a", "b"],
                headroom_by_account={"a": 100_000, "b": 250_000},
                lock_file=temp_lock_file,
            )
            # Expired claim ignored — b still wins on raw headroom.
            assert picked == "b"
        finally:
            with patch("os.getpid", return_value=1001):
                unregister_session(lock_file=temp_lock_file)

    def test_empty_pool_returns_none(self, temp_lock_file: Path):
        assert pick_least_loaded_account([], {}, lock_file=temp_lock_file) is None

    def test_stable_tiebreak_favors_earlier_pool_entry(self, temp_lock_file: Path):
        """Equal effective headroom → the account listed first in the pool wins.
        This keeps rotation deterministic and spreads consecutive iterations in
        a predictable pattern."""
        picked = pick_least_loaded_account(
            pool=["first", "second"],
            headroom_by_account={"first": 100_000, "second": 100_000},
            lock_file=temp_lock_file,
        )
        assert picked == "first"
