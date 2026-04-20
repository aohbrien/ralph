"""Writer-side reporter that publishes a ralph instance's state for the dashboard.

Each running `ralph run` owns one reporter, which writes
``~/.ralph/instances/<pid>.json`` (snapshot, rewritten on every event) and
``~/.ralph/instances/<pid>.tail`` (streaming output, throttled ring buffer).
The dashboard reads both via :mod:`ralph.dashboard.sources`.

Design notes:
- JSON writes are atomic (temp file in the same dir + ``os.replace``) so the
  dashboard never sees a truncated file.
- The tail is a disk-backed ring buffer capped at ``max_tail_bytes``. Writes
  are batched on a throttle to avoid rewriting a 64 KB file per line.
- ``close()`` removes both files and is safe to call multiple times. It's
  wired via ``atexit`` plus signal handlers so crashes leave at worst a stale
  file that the dashboard's sweep will clean up.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import socket
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ralph.dashboard.sources import DEFAULT_INSTANCES_DIR, now_iso

logger = logging.getLogger(__name__)

# State file schema version — bump if we rename/remove fields the dashboard relies on.
SCHEMA_VERSION = 1

DEFAULT_MAX_TAIL_BYTES = 64 * 1024
DEFAULT_TAIL_FLUSH_INTERVAL_MS = 500
DEFAULT_TAIL_FLUSH_BYTES = 4 * 1024
# While output is streaming, refresh the state file at most this often so
# `last_heartbeat` stays fresh during long iterations without triggering
# excessive writes.
DEFAULT_STATE_HEARTBEAT_INTERVAL_MS = 30_000


@dataclass
class InstanceMetadata:
    """Static info known at start time — doesn't change across iterations."""

    prd_path: Path | None
    prd_project: str | None
    tool: str
    two_phase: bool
    max_iterations: int
    cwd: Path


@dataclass
class _IterationState:
    """Mutable fields tracked across iteration events."""

    iteration: int = 0
    phase: str | None = None
    current_story: dict[str, Any] | None = None
    prd_progress: dict[str, int] = field(default_factory=lambda: {"passing": 0, "total": 0})
    usage: dict[str, Any] = field(default_factory=dict)
    last_iteration_result: dict[str, Any] | None = None


class DashboardReporter:
    """Publish this process's state to the dashboard spool directory.

    Thread-safety: Output lines may arrive from a subprocess-stream thread
    while iteration events come from the main loop — all writes hold
    ``self._lock``.
    """

    def __init__(
        self,
        instances_dir: Path = DEFAULT_INSTANCES_DIR,
        pid: int | None = None,
        max_tail_bytes: int = DEFAULT_MAX_TAIL_BYTES,
        tail_flush_interval_ms: int = DEFAULT_TAIL_FLUSH_INTERVAL_MS,
        tail_flush_bytes: int = DEFAULT_TAIL_FLUSH_BYTES,
        state_heartbeat_interval_ms: int = DEFAULT_STATE_HEARTBEAT_INTERVAL_MS,
    ):
        self.instances_dir = instances_dir
        self.pid = pid if pid is not None else os.getpid()
        self.max_tail_bytes = max_tail_bytes
        self.tail_flush_interval_ms = tail_flush_interval_ms
        self.tail_flush_bytes = tail_flush_bytes
        self.state_heartbeat_interval_ms = state_heartbeat_interval_ms

        self._lock = threading.Lock()
        self._state_path = self.instances_dir / f"{self.pid}.json"
        self._tail_path = self.instances_dir / f"{self.pid}.tail"

        self._metadata: InstanceMetadata | None = None
        self._started_at: str | None = None
        self._iter = _IterationState()

        # Line-buffered tail input; flushed on throttle.
        self._tail_pending: bytearray = bytearray()
        self._tail_last_flush_ms: int = 0
        # Tracks the last state-file rewrite so the streaming-output path can
        # refresh last_heartbeat without hammering the disk.
        self._state_last_write_ms: int = 0

        self._closed = False
        self._atexit_registered = False

    # ---- Public API ---------------------------------------------------

    def start(
        self,
        metadata: InstanceMetadata,
        prd_passing: int = 0,
        prd_total: int = 0,
    ) -> None:
        """Register this instance and write its initial snapshot.

        ``prd_passing``/``prd_total`` let the reporter ship a meaningful
        progress number in the first write — important when resuming a run
        that's already partway through the PRD.
        """
        with self._lock:
            if self._started_at is not None:
                return  # idempotent
            self._metadata = metadata
            self._started_at = now_iso()
            self._iter.prd_progress = {"passing": prd_passing, "total": prd_total}
            self.instances_dir.mkdir(parents=True, exist_ok=True)
            # Truncate any stale tail file from a previous process reusing this PID.
            try:
                self._tail_path.unlink(missing_ok=True)
            except OSError:
                pass
            self._write_state_locked()

        if not self._atexit_registered:
            atexit.register(self.close)
            self._atexit_registered = True

    def on_iteration_start(
        self,
        iteration: int,
        story: Any | None,
        phase: str,
        prd_passing: int | None = None,
        prd_total: int | None = None,
    ) -> None:
        with self._lock:
            self._iter.iteration = iteration
            self._iter.phase = phase
            self._iter.current_story = _story_to_dict(story)
            self._iter.last_iteration_result = None
            if prd_passing is not None and prd_total is not None:
                self._iter.prd_progress = {"passing": prd_passing, "total": prd_total}
            self._write_state_locked()

    def on_iteration_end(
        self,
        *,
        return_code: int,
        completed: bool,
        timed_out: bool,
        interrupted: bool,
        prd_passing: int,
        prd_total: int,
        usage: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self._iter.last_iteration_result = {
                "return_code": return_code,
                "completed": completed,
                "timed_out": timed_out,
                "interrupted": interrupted,
            }
            self._iter.prd_progress = {"passing": prd_passing, "total": prd_total}
            if usage is not None:
                self._iter.usage = usage
            self._iter.phase = "idle"
            self._write_state_locked()

    def on_output_line(self, text: str) -> None:
        """Append output text to the tail buffer. Safe to call from any thread."""
        if not text:
            return
        data = text.encode("utf-8", errors="replace")
        with self._lock:
            if self._closed:
                return
            self._tail_pending.extend(data)
            if self._should_flush_tail_locked():
                self._flush_tail_locked()
            # Refresh the state file's last_heartbeat on a throttle so the
            # dashboard doesn't flag healthy long-running iterations as stale.
            if self._should_refresh_state_locked():
                self._write_state_locked()

    def heartbeat(self, usage: dict[str, Any] | None = None) -> None:
        """Update the last_heartbeat timestamp (and optionally usage) and flush state."""
        with self._lock:
            if self._started_at is None or self._closed:
                return
            if usage is not None:
                self._iter.usage = usage
            self._write_state_locked()
            # Also drain any pending tail bytes on each heartbeat.
            if self._tail_pending:
                self._flush_tail_locked()

    def close(self) -> None:
        """Remove this instance's state files. Safe to call multiple times."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            # Flush any remaining tail bytes before deletion so that, if the
            # file is kept around for debugging, it's at least complete. But
            # the default is to remove — see below.
            if self._tail_pending:
                try:
                    self._flush_tail_locked()
                except Exception:
                    pass
            for path in (self._state_path, self._tail_path):
                try:
                    path.unlink(missing_ok=True)
                except OSError as exc:
                    logger.debug("Could not remove %s: %s", path, exc)

    # ---- Internals ----------------------------------------------------

    def _should_flush_tail_locked(self) -> bool:
        if len(self._tail_pending) >= self.tail_flush_bytes:
            return True
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        return (now_ms - self._tail_last_flush_ms) >= self.tail_flush_interval_ms

    def _should_refresh_state_locked(self) -> bool:
        if self._metadata is None or self._started_at is None:
            return False
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        return (now_ms - self._state_last_write_ms) >= self.state_heartbeat_interval_ms

    def _flush_tail_locked(self) -> None:
        if not self._tail_pending:
            return
        self.instances_dir.mkdir(parents=True, exist_ok=True)
        try:
            # Append first (cheap), then truncate back to cap if we've exceeded it.
            with self._tail_path.open("ab") as f:
                f.write(self._tail_pending)
            self._tail_pending.clear()
            size = self._tail_path.stat().st_size
            if size > self.max_tail_bytes:
                with self._tail_path.open("rb") as f:
                    f.seek(size - self.max_tail_bytes)
                    trimmed = f.read()
                # Atomic replace of the trimmed tail.
                self._atomic_write_bytes(self._tail_path, trimmed)
        except OSError as exc:
            logger.debug("Tail flush failed: %s", exc)
        finally:
            self._tail_last_flush_ms = int(
                datetime.now(timezone.utc).timestamp() * 1000
            )

    def _write_state_locked(self) -> None:
        if self._metadata is None or self._started_at is None:
            return
        payload = self._build_payload_locked()
        data = (json.dumps(payload, indent=2) + "\n").encode("utf-8")
        try:
            self._atomic_write_bytes(self._state_path, data)
            self._state_last_write_ms = int(
                datetime.now(timezone.utc).timestamp() * 1000
            )
        except OSError as exc:
            logger.debug("State write failed: %s", exc)

    def _build_payload_locked(self) -> dict[str, Any]:
        assert self._metadata is not None
        assert self._started_at is not None
        now = now_iso()
        m = self._metadata
        try:
            tail_bytes = self._tail_path.stat().st_size
        except OSError:
            tail_bytes = 0
        return {
            "schema_version": SCHEMA_VERSION,
            "pid": self.pid,
            "host": socket.gethostname(),
            "cwd": str(m.cwd),
            "prd_path": str(m.prd_path) if m.prd_path is not None else None,
            "prd_project": m.prd_project,
            "tool": m.tool,
            "two_phase": m.two_phase,
            "phase": self._iter.phase,
            "started_at": self._started_at,
            "updated_at": now,
            "last_heartbeat": now,
            "iteration": self._iter.iteration,
            "max_iterations": m.max_iterations,
            "current_story": self._iter.current_story,
            "prd_progress": dict(self._iter.prd_progress),
            "usage": dict(self._iter.usage),
            "last_iteration_result": self._iter.last_iteration_result,
            "tail_file": str(self._tail_path),
            "tail_bytes": tail_bytes,
        }

    @staticmethod
    def _atomic_write_bytes(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # NamedTemporaryFile in the same directory → os.replace for atomicity.
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
        )
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass
            os.replace(tmp_name, path)
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise


def _story_to_dict(story: Any | None) -> dict[str, Any] | None:
    if story is None:
        return None
    sid = getattr(story, "id", None)
    title = getattr(story, "title", None)
    if sid is None and title is None:
        return None
    return {"id": sid, "title": title}


class _NullReporter:
    """No-op fallback used when a Runner runs without a reporter attached."""

    def start(self, metadata: InstanceMetadata, prd_passing: int = 0, prd_total: int = 0) -> None: ...
    def on_iteration_start(
        self,
        iteration: int,
        story: Any | None,
        phase: str,
        prd_passing: int | None = None,
        prd_total: int | None = None,
    ) -> None: ...
    def on_iteration_end(self, **kwargs: Any) -> None: ...
    def on_output_line(self, text: str) -> None: ...
    def heartbeat(self, usage: dict[str, Any] | None = None) -> None: ...
    def close(self) -> None: ...


NULL_REPORTER = _NullReporter()
