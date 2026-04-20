"""Sources of dashboard instance data.

`InstanceSource` is the abstraction the dashboard HTTP server reads from. The
local-files implementation scans `~/.ralph/instances/*.json` and filters out
dead or stale processes. A future remote-HTTP ingest mode can plug in as an
alternative implementation of the same Protocol without touching the server.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from ralph.session import (
    DEFAULT_RALPH_DIR,
    STALE_SESSION_THRESHOLD_SECONDS,
)

logger = logging.getLogger(__name__)

DEFAULT_INSTANCES_DIR = DEFAULT_RALPH_DIR / "instances"

# Instances whose heartbeat is older than this and whose PID is dead are
# removed from the dashboard entirely. Live-but-quiet instances are kept.
DEAD_INSTANCE_GRACE_SECONDS = 60


@dataclass
class InstanceSnapshot:
    """Point-in-time view of one ralph instance for dashboard rendering."""

    pid: int
    host: str
    cwd: str
    prd_path: str | None
    prd_project: str | None
    tool: str
    two_phase: bool
    phase: str | None
    started_at: str
    updated_at: str
    last_heartbeat: str
    iteration: int
    max_iterations: int
    current_story: dict[str, Any] | None
    prd_progress: dict[str, int]
    usage: dict[str, Any]
    last_iteration_result: dict[str, Any] | None
    tail_file: str | None
    tail_bytes: int
    # Derived fields — not written to disk
    status: str = "running"  # running | stopped | stale
    heartbeat_age_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "host": self.host,
            "cwd": self.cwd,
            "prd_path": self.prd_path,
            "prd_project": self.prd_project,
            "tool": self.tool,
            "two_phase": self.two_phase,
            "phase": self.phase,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "last_heartbeat": self.last_heartbeat,
            "iteration": self.iteration,
            "max_iterations": self.max_iterations,
            "current_story": self.current_story,
            "prd_progress": self.prd_progress,
            "usage": self.usage,
            "last_iteration_result": self.last_iteration_result,
            "tail_file": self.tail_file,
            "tail_bytes": self.tail_bytes,
            "status": self.status,
            "heartbeat_age_seconds": self.heartbeat_age_seconds,
        }

    @classmethod
    def from_file_dict(cls, data: dict[str, Any]) -> InstanceSnapshot:
        return cls(
            pid=int(data["pid"]),
            host=data.get("host", ""),
            cwd=data.get("cwd", ""),
            prd_path=data.get("prd_path"),
            prd_project=data.get("prd_project"),
            tool=data.get("tool", ""),
            two_phase=bool(data.get("two_phase", False)),
            phase=data.get("phase"),
            started_at=data.get("started_at", ""),
            updated_at=data.get("updated_at", ""),
            last_heartbeat=data.get("last_heartbeat", ""),
            iteration=int(data.get("iteration", 0)),
            max_iterations=int(data.get("max_iterations", 0)),
            current_story=data.get("current_story"),
            prd_progress=data.get("prd_progress") or {"passing": 0, "total": 0},
            usage=data.get("usage") or {},
            last_iteration_result=data.get("last_iteration_result"),
            tail_file=data.get("tail_file"),
            tail_bytes=int(data.get("tail_bytes", 0)),
        )


class InstanceSource(Protocol):
    """How the dashboard discovers and reads ralph instances."""

    def list_instances(self) -> list[InstanceSnapshot]: ...

    def get_tail(self, pid: int, max_bytes: int) -> bytes: ...


def _is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but owned by another user.
        return True
    except OSError:
        return False


def _heartbeat_age_seconds(heartbeat_iso: str) -> float:
    if not heartbeat_iso:
        return float("inf")
    try:
        hb = datetime.fromisoformat(heartbeat_iso)
        if hb.tzinfo is None:
            hb = hb.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - hb).total_seconds()
    except ValueError:
        return float("inf")


@dataclass
class LocalFilesSource:
    """Read instance snapshots from ``~/.ralph/instances/*.json``."""

    instances_dir: Path = field(default_factory=lambda: DEFAULT_INSTANCES_DIR)
    stale_threshold_seconds: int = STALE_SESSION_THRESHOLD_SECONDS
    dead_grace_seconds: int = DEAD_INSTANCE_GRACE_SECONDS

    def list_instances(self) -> list[InstanceSnapshot]:
        if not self.instances_dir.exists():
            return []

        snapshots: list[InstanceSnapshot] = []
        for path in sorted(self.instances_dir.glob("*.json")):
            snap = self._load_one(path)
            if snap is None:
                continue
            snapshots.append(snap)

        self._sweep_dead_files(snapshots)

        # Drop instances whose files have already been deleted by the sweep.
        return [s for s in snapshots if s.status != "_removed"]

    def get_tail(self, pid: int, max_bytes: int) -> bytes:
        tail_path = self.instances_dir / f"{pid}.tail"
        if not tail_path.exists():
            return b""
        try:
            size = tail_path.stat().st_size
        except OSError:
            return b""
        read_bytes = min(max_bytes, size) if max_bytes > 0 else size
        try:
            with tail_path.open("rb") as f:
                if size > read_bytes:
                    f.seek(size - read_bytes)
                return f.read()
        except OSError:
            return b""

    def _load_one(self, path: Path) -> InstanceSnapshot | None:
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            return None
        if not raw.strip():
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Partial write from the instance — skip this poll cycle.
            logger.debug("Skipping %s: JSON decode error", path)
            return None
        try:
            snap = InstanceSnapshot.from_file_dict(data)
        except (KeyError, TypeError, ValueError) as exc:
            logger.debug("Skipping %s: %s", path, exc)
            return None

        snap.heartbeat_age_seconds = _heartbeat_age_seconds(snap.last_heartbeat)
        snap.status = self._classify(snap)
        return snap

    def _classify(self, snap: InstanceSnapshot) -> str:
        if not _is_pid_alive(snap.pid):
            return "stopped"
        if snap.heartbeat_age_seconds > self.stale_threshold_seconds:
            return "stale"
        return "running"

    def _sweep_dead_files(self, snapshots: list[InstanceSnapshot]) -> None:
        """Delete state files for instances that have been stopped past the grace window."""
        for snap in snapshots:
            if snap.status != "stopped":
                continue
            if snap.heartbeat_age_seconds < self.dead_grace_seconds:
                continue
            self._remove_instance_files(snap.pid)
            snap.status = "_removed"

    def _remove_instance_files(self, pid: int) -> None:
        for suffix in (".json", ".tail"):
            path = self.instances_dir / f"{pid}{suffix}"
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                logger.debug("Could not remove %s: %s", path, exc)


def now_iso() -> str:
    """UTC ISO-8601 timestamp — shared helper so reader/writer agree on format."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def monotonic_ms() -> int:
    """Monotonic clock in milliseconds, for write throttles."""
    return int(time.monotonic() * 1000)
