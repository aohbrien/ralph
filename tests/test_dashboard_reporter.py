"""Tests for the DashboardReporter writer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from ralph.dashboard.reporter import (
    DEFAULT_MAX_TAIL_BYTES,
    DashboardReporter,
    InstanceMetadata,
)


@dataclass
class _FakeStory:
    id: str
    title: str


def _make_reporter(tmp_path: Path, pid: int = 4242, **kwargs) -> DashboardReporter:
    return DashboardReporter(instances_dir=tmp_path, pid=pid, **kwargs)


def _make_metadata(tmp_path: Path) -> InstanceMetadata:
    return InstanceMetadata(
        prd_path=tmp_path / "prd.json",
        prd_project="Test Project",
        tool="claude",
        two_phase=False,
        max_iterations=20,
        cwd=tmp_path,
    )


def test_start_writes_snapshot(tmp_path: Path) -> None:
    reporter = _make_reporter(tmp_path)
    reporter.start(_make_metadata(tmp_path))

    state_file = tmp_path / "4242.json"
    assert state_file.exists()

    data = json.loads(state_file.read_text())
    assert data["pid"] == 4242
    assert data["prd_project"] == "Test Project"
    assert data["tool"] == "claude"
    assert data["iteration"] == 0
    assert data["max_iterations"] == 20

    reporter.close()
    assert not state_file.exists()


def test_iteration_lifecycle_updates_file(tmp_path: Path) -> None:
    reporter = _make_reporter(tmp_path)
    reporter.start(_make_metadata(tmp_path))

    reporter.on_iteration_start(3, _FakeStory(id="US-007", title="Add feature"), phase="single")

    data = json.loads((tmp_path / "4242.json").read_text())
    assert data["iteration"] == 3
    assert data["phase"] == "single"
    assert data["current_story"] == {"id": "US-007", "title": "Add feature"}

    reporter.on_iteration_end(
        return_code=0,
        completed=True,
        timed_out=False,
        interrupted=False,
        prd_passing=4,
        prd_total=10,
        usage={"percentage": 33.0, "tokens_used": 123},
    )

    data = json.loads((tmp_path / "4242.json").read_text())
    assert data["last_iteration_result"] == {
        "return_code": 0,
        "completed": True,
        "timed_out": False,
        "interrupted": False,
    }
    assert data["prd_progress"] == {"passing": 4, "total": 10}
    assert data["usage"]["percentage"] == 33.0
    assert data["phase"] == "idle"

    reporter.close()


def test_tail_writes_and_caps(tmp_path: Path) -> None:
    # Use tiny cap so we can verify ring-buffer behavior without writing MB.
    reporter = _make_reporter(
        tmp_path,
        max_tail_bytes=128,
        tail_flush_interval_ms=0,
        tail_flush_bytes=1,
    )
    reporter.start(_make_metadata(tmp_path))

    for i in range(50):
        reporter.on_output_line(f"line-{i:03d}-filler-content\n")

    tail_path = tmp_path / "4242.tail"
    assert tail_path.exists()
    size = tail_path.stat().st_size
    assert size <= 128, f"tail exceeded cap: {size}"

    content = tail_path.read_bytes().decode()
    # The most recent line should survive the ring-buffer trim.
    assert "line-049" in content

    reporter.close()


def test_atomic_write_round_trip(tmp_path: Path) -> None:
    """Ensure we don't leave temp files lying around in the spool dir."""
    reporter = _make_reporter(tmp_path)
    reporter.start(_make_metadata(tmp_path))
    reporter.on_iteration_start(1, None, phase="single")
    reporter.heartbeat(usage={"percentage": 12.5})

    entries = {p.name for p in tmp_path.iterdir()}
    # Only the state + tail files (plus any test tmp dirs the fixture makes).
    suspicious = {n for n in entries if n.endswith(".tmp") or n.startswith(".4242.json.")}
    assert not suspicious, f"stray temp files: {suspicious}"

    reporter.close()


def test_close_is_idempotent(tmp_path: Path) -> None:
    reporter = _make_reporter(tmp_path)
    reporter.start(_make_metadata(tmp_path))
    reporter.close()
    reporter.close()  # must not raise


def test_two_reporters_do_not_clobber(tmp_path: Path) -> None:
    r1 = _make_reporter(tmp_path, pid=1001)
    r2 = _make_reporter(tmp_path, pid=1002)
    r1.start(_make_metadata(tmp_path))
    r2.start(_make_metadata(tmp_path))
    r1.on_iteration_start(1, None, phase="single")
    r2.on_iteration_start(2, None, phase="single")

    d1 = json.loads((tmp_path / "1001.json").read_text())
    d2 = json.loads((tmp_path / "1002.json").read_text())
    assert d1["pid"] == 1001 and d1["iteration"] == 1
    assert d2["pid"] == 1002 and d2["iteration"] == 2

    r1.close()
    r2.close()


def test_reporter_no_ops_before_start(tmp_path: Path) -> None:
    """Events before start() should be silently ignored, not crash."""
    reporter = _make_reporter(tmp_path)
    reporter.on_iteration_start(1, None, phase="single")
    reporter.heartbeat()
    reporter.on_output_line("nothing to see here")
    reporter.close()
    # No files written because start() was never called.
    assert not (tmp_path / "4242.json").exists()


def test_start_writes_initial_progress(tmp_path: Path) -> None:
    """On resume, start() must ship current PRD progress, not 0/0."""
    reporter = _make_reporter(tmp_path)
    reporter.start(_make_metadata(tmp_path), prd_passing=154, prd_total=266)

    data = json.loads((tmp_path / "4242.json").read_text())
    assert data["prd_progress"] == {"passing": 154, "total": 266}
    reporter.close()


def test_iteration_start_updates_progress(tmp_path: Path) -> None:
    """PRD progress is refreshed on each iteration start so the dashboard doesn't
    lag until the iteration completes — important for long iterations."""
    reporter = _make_reporter(tmp_path)
    reporter.start(_make_metadata(tmp_path), prd_passing=10, prd_total=100)
    reporter.on_iteration_start(
        5, _FakeStory(id="US-050", title="x"), phase="single",
        prd_passing=11, prd_total=100,
    )
    data = json.loads((tmp_path / "4242.json").read_text())
    assert data["prd_progress"] == {"passing": 11, "total": 100}
    reporter.close()


def test_output_refreshes_heartbeat_when_past_interval(tmp_path: Path) -> None:
    """A long iteration that streams output should keep the state file fresh."""
    reporter = _make_reporter(tmp_path, state_heartbeat_interval_ms=0)
    reporter.start(_make_metadata(tmp_path))
    state_file = tmp_path / "4242.json"
    first_mtime = state_file.stat().st_mtime_ns

    # Need a small delay so the filesystem mtime can advance.
    import time as _t
    _t.sleep(0.01)

    reporter.on_output_line("a line of output\n")
    second_mtime = state_file.stat().st_mtime_ns
    assert second_mtime > first_mtime, "output should have rewritten the state file"
    reporter.close()


def test_output_does_not_refresh_heartbeat_within_interval(tmp_path: Path) -> None:
    """The heartbeat-on-output throttle must not trigger a write per line."""
    reporter = _make_reporter(tmp_path, state_heartbeat_interval_ms=10_000_000)
    reporter.start(_make_metadata(tmp_path))
    state_file = tmp_path / "4242.json"
    first_mtime = state_file.stat().st_mtime_ns

    for _ in range(20):
        reporter.on_output_line("noise\n")

    second_mtime = state_file.stat().st_mtime_ns
    assert first_mtime == second_mtime, "state file should not rewrite on every output line"
    reporter.close()


def test_default_max_tail_bytes_is_reasonable() -> None:
    # Sanity bound — keeps us from blowing up disk if nobody tunes it.
    assert DEFAULT_MAX_TAIL_BYTES <= 1024 * 1024
