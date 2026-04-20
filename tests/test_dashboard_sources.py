"""Tests for LocalFilesSource — aggregation, stale filtering, tail reading."""

from __future__ import annotations

import os
from pathlib import Path

from ralph.dashboard.sources import LocalFilesSource


def test_empty_directory_returns_empty_list(tmp_path: Path) -> None:
    source = LocalFilesSource(instances_dir=tmp_path)
    assert source.list_instances() == []


def test_live_instance_included(tmp_path: Path, write_instance) -> None:
    write_instance(tmp_path, pid=os.getpid())
    source = LocalFilesSource(instances_dir=tmp_path)
    snapshots = source.list_instances()
    assert len(snapshots) == 1
    assert snapshots[0].pid == os.getpid()
    assert snapshots[0].status == "running"


def test_stale_instance_keeps_file_but_marks_status(tmp_path: Path, write_instance) -> None:
    # Live PID (our own) but heartbeat older than threshold.
    write_instance(tmp_path, pid=os.getpid(), heartbeat_age_seconds=7200)
    source = LocalFilesSource(instances_dir=tmp_path, stale_threshold_seconds=3600)
    snapshots = source.list_instances()
    assert len(snapshots) == 1
    assert snapshots[0].status == "stale"


def test_dead_pid_within_grace_is_reported_stopped(tmp_path: Path, write_instance) -> None:
    dead_pid = 99999999  # almost certainly not running
    write_instance(tmp_path, pid=dead_pid, heartbeat_age_seconds=5)
    source = LocalFilesSource(instances_dir=tmp_path, dead_grace_seconds=60)
    snapshots = source.list_instances()
    assert len(snapshots) == 1
    assert snapshots[0].status == "stopped"


def test_dead_pid_past_grace_is_removed(tmp_path: Path, write_instance) -> None:
    dead_pid = 99999999
    write_instance(tmp_path, pid=dead_pid, heartbeat_age_seconds=120)
    (tmp_path / f"{dead_pid}.tail").write_bytes(b"some tail data")

    source = LocalFilesSource(instances_dir=tmp_path, dead_grace_seconds=60)
    snapshots = source.list_instances()
    assert snapshots == []
    assert not (tmp_path / f"{dead_pid}.json").exists()
    assert not (tmp_path / f"{dead_pid}.tail").exists()


def test_corrupt_json_is_skipped(tmp_path: Path, write_instance) -> None:
    (tmp_path / "123.json").write_text("{not valid json")
    write_instance(tmp_path, pid=os.getpid())
    source = LocalFilesSource(instances_dir=tmp_path)
    snapshots = source.list_instances()
    assert len(snapshots) == 1
    assert snapshots[0].pid == os.getpid()


def test_get_tail_returns_trailing_bytes(tmp_path: Path, write_instance) -> None:
    pid = os.getpid()
    write_instance(tmp_path, pid=pid)
    tail_path = tmp_path / f"{pid}.tail"
    tail_path.write_bytes(b"x" * 1000 + b"MARKER")

    source = LocalFilesSource(instances_dir=tmp_path)
    data = source.get_tail(pid, max_bytes=50)
    assert len(data) == 50
    assert data.endswith(b"MARKER")


def test_get_tail_missing_file(tmp_path: Path) -> None:
    source = LocalFilesSource(instances_dir=tmp_path)
    assert source.get_tail(12345, max_bytes=100) == b""
