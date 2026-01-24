"""Cross-session coordination for Ralph.

Coordinates rate limiting across multiple Ralph instances running on the same
machine using a shared lock file (~/.ralph/usage.lock).
"""

from __future__ import annotations

import atexit
import fcntl
import json
import logging
import os
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_RALPH_DIR = Path.home() / ".ralph"
DEFAULT_LOCK_FILE = DEFAULT_RALPH_DIR / "usage.lock"

# Stale session threshold in seconds (1 hour)
STALE_SESSION_THRESHOLD_SECONDS = 3600


@dataclass
class SessionInfo:
    """Information about an active Ralph session."""

    pid: int
    started_at: str  # ISO format timestamp
    last_heartbeat: str  # ISO format timestamp
    prd_path: str | None = None
    allocated_budget: int = 0  # Tokens allocated to this session

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "pid": self.pid,
            "started_at": self.started_at,
            "last_heartbeat": self.last_heartbeat,
            "prd_path": self.prd_path,
            "allocated_budget": self.allocated_budget,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionInfo":
        """Create from dictionary."""
        return cls(
            pid=data["pid"],
            started_at=data["started_at"],
            last_heartbeat=data["last_heartbeat"],
            prd_path=data.get("prd_path"),
            allocated_budget=data.get("allocated_budget", 0),
        )

    def is_stale(self, threshold_seconds: int = STALE_SESSION_THRESHOLD_SECONDS) -> bool:
        """Check if this session is stale (last heartbeat > threshold ago)."""
        try:
            last_hb = datetime.fromisoformat(self.last_heartbeat)
            now = datetime.now(timezone.utc)
            # Handle naive datetimes by assuming UTC
            if last_hb.tzinfo is None:
                last_hb = last_hb.replace(tzinfo=timezone.utc)
            age_seconds = (now - last_hb).total_seconds()
            return age_seconds > threshold_seconds
        except (ValueError, TypeError):
            # If we can't parse the timestamp, consider it stale
            return True

    def is_process_running(self) -> bool:
        """Check if the process with this PID is still running."""
        try:
            # os.kill with signal 0 doesn't actually send a signal,
            # but raises an error if the process doesn't exist
            os.kill(self.pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            # Process exists but we don't have permission to signal it
            return True


@dataclass
class SessionRegistry:
    """Registry of active Ralph sessions."""

    sessions: dict[str, SessionInfo] = field(default_factory=dict)
    last_updated: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "sessions": {k: v.to_dict() for k, v in self.sessions.items()},
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionRegistry":
        """Create from dictionary."""
        sessions_data = data.get("sessions", {})
        sessions = {k: SessionInfo.from_dict(v) for k, v in sessions_data.items()}
        return cls(
            sessions=sessions,
            last_updated=data.get("last_updated"),
        )

    def add_session(self, session: SessionInfo) -> None:
        """Add or update a session."""
        key = str(session.pid)
        self.sessions[key] = session
        self.last_updated = datetime.now(timezone.utc).isoformat()

    def remove_session(self, pid: int) -> bool:
        """Remove a session by PID. Returns True if removed."""
        key = str(pid)
        if key in self.sessions:
            del self.sessions[key]
            self.last_updated = datetime.now(timezone.utc).isoformat()
            return True
        return False

    def get_session(self, pid: int) -> SessionInfo | None:
        """Get a session by PID."""
        return self.sessions.get(str(pid))

    def cleanup_stale_sessions(
        self, threshold_seconds: int = STALE_SESSION_THRESHOLD_SECONDS
    ) -> list[int]:
        """
        Remove stale sessions (>threshold old or process not running).

        Returns:
            List of PIDs that were removed.
        """
        removed_pids: list[int] = []
        keys_to_remove: list[str] = []

        for key, session in self.sessions.items():
            # Check if process is still running
            if not session.is_process_running():
                logger.debug(f"Session {session.pid} process not running, removing")
                keys_to_remove.append(key)
                removed_pids.append(session.pid)
            # Check if session is stale (>1 hour without heartbeat)
            elif session.is_stale(threshold_seconds):
                logger.debug(f"Session {session.pid} is stale, removing")
                keys_to_remove.append(key)
                removed_pids.append(session.pid)

        for key in keys_to_remove:
            del self.sessions[key]

        if keys_to_remove:
            self.last_updated = datetime.now(timezone.utc).isoformat()

        return removed_pids

    def active_session_count(self) -> int:
        """Return the number of active sessions."""
        return len(self.sessions)


def _ensure_ralph_dir(ralph_dir: Path = DEFAULT_RALPH_DIR) -> None:
    """Ensure the Ralph config directory exists."""
    ralph_dir.mkdir(parents=True, exist_ok=True)


def _read_registry_locked(lock_file: Path) -> tuple[SessionRegistry, Any]:
    """
    Read the session registry with an exclusive file lock.

    Returns:
        Tuple of (SessionRegistry, file handle) - caller must close the file
        to release the lock.
    """
    _ensure_ralph_dir(lock_file.parent)

    # Open file for read+write (create if doesn't exist)
    # Use 'a+' to create if not exists, then seek to beginning
    try:
        f = open(lock_file, "r+", encoding="utf-8")
    except FileNotFoundError:
        # Create the file if it doesn't exist
        f = open(lock_file, "w+", encoding="utf-8")

    # Acquire exclusive lock (blocking)
    fcntl.flock(f.fileno(), fcntl.LOCK_EX)

    # Read current content
    f.seek(0)
    content = f.read()

    if content.strip():
        try:
            data = json.loads(content)
            registry = SessionRegistry.from_dict(data)
        except json.JSONDecodeError:
            logger.warning("Corrupted lock file, starting fresh")
            registry = SessionRegistry()
    else:
        registry = SessionRegistry()

    return registry, f


def _write_registry_and_unlock(
    registry: SessionRegistry, lock_file: Path, f: Any
) -> None:
    """Write the registry and release the lock."""
    try:
        # Truncate and write new content
        f.seek(0)
        f.truncate()
        json.dump(registry.to_dict(), f, indent=2)
        f.write("\n")
        f.flush()
    finally:
        # Release lock and close file
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        f.close()


def register_session(
    prd_path: Path | None = None,
    lock_file: Path = DEFAULT_LOCK_FILE,
) -> SessionInfo:
    """
    Register the current process as an active Ralph session.

    Args:
        prd_path: Path to the PRD file this session is working on
        lock_file: Path to the lock file

    Returns:
        SessionInfo for this session
    """
    registry, f = _read_registry_locked(lock_file)

    # Clean up stale sessions first
    removed = registry.cleanup_stale_sessions()
    if removed:
        logger.debug(f"Cleaned up {len(removed)} stale sessions: {removed}")

    # Create session info
    now = datetime.now(timezone.utc).isoformat()
    session = SessionInfo(
        pid=os.getpid(),
        started_at=now,
        last_heartbeat=now,
        prd_path=str(prd_path) if prd_path else None,
    )

    # Add to registry
    registry.add_session(session)

    _write_registry_and_unlock(registry, lock_file, f)

    logger.debug(f"Registered session: PID={session.pid}")
    return session


def unregister_session(
    pid: int | None = None,
    lock_file: Path = DEFAULT_LOCK_FILE,
) -> bool:
    """
    Unregister a Ralph session.

    Args:
        pid: Process ID to unregister (defaults to current process)
        lock_file: Path to the lock file

    Returns:
        True if session was found and removed
    """
    if pid is None:
        pid = os.getpid()

    try:
        registry, f = _read_registry_locked(lock_file)
        removed = registry.remove_session(pid)
        _write_registry_and_unlock(registry, lock_file, f)

        if removed:
            logger.debug(f"Unregistered session: PID={pid}")
        return removed
    except (OSError, IOError) as e:
        logger.warning(f"Failed to unregister session: {e}")
        return False


def update_heartbeat(
    pid: int | None = None,
    lock_file: Path = DEFAULT_LOCK_FILE,
) -> bool:
    """
    Update the heartbeat timestamp for a session.

    Args:
        pid: Process ID to update (defaults to current process)
        lock_file: Path to the lock file

    Returns:
        True if session was found and updated
    """
    if pid is None:
        pid = os.getpid()

    try:
        registry, f = _read_registry_locked(lock_file)

        session = registry.get_session(pid)
        if session:
            session.last_heartbeat = datetime.now(timezone.utc).isoformat()
            registry.add_session(session)
            _write_registry_and_unlock(registry, lock_file, f)
            return True
        else:
            _write_registry_and_unlock(registry, lock_file, f)
            return False
    except (OSError, IOError) as e:
        logger.warning(f"Failed to update heartbeat: {e}")
        return False


def get_active_sessions(
    lock_file: Path = DEFAULT_LOCK_FILE,
    cleanup_stale: bool = True,
) -> list[SessionInfo]:
    """
    Get all active Ralph sessions.

    Args:
        lock_file: Path to the lock file
        cleanup_stale: If True, clean up stale sessions before returning

    Returns:
        List of active SessionInfo objects
    """
    if not lock_file.exists():
        return []

    try:
        registry, f = _read_registry_locked(lock_file)

        if cleanup_stale:
            registry.cleanup_stale_sessions()

        sessions = list(registry.sessions.values())
        _write_registry_and_unlock(registry, lock_file, f)
        return sessions
    except (OSError, IOError) as e:
        logger.warning(f"Failed to get active sessions: {e}")
        return []


def get_active_session_count(
    lock_file: Path = DEFAULT_LOCK_FILE,
    cleanup_stale: bool = True,
) -> int:
    """
    Get the count of active Ralph sessions.

    Args:
        lock_file: Path to the lock file
        cleanup_stale: If True, clean up stale sessions before counting

    Returns:
        Number of active sessions
    """
    sessions = get_active_sessions(lock_file, cleanup_stale)
    return len(sessions)


def calculate_session_budget(
    total_budget: int,
    lock_file: Path = DEFAULT_LOCK_FILE,
) -> int:
    """
    Calculate the budget allocation for the current session.

    Divides the remaining budget evenly across all active sessions.

    Args:
        total_budget: Total available budget (tokens)
        lock_file: Path to the lock file

    Returns:
        Budget allocated to this session
    """
    sessions = get_active_sessions(lock_file, cleanup_stale=True)
    num_sessions = len(sessions) if sessions else 1

    # Divide budget evenly
    return total_budget // num_sessions


def update_session_budget(
    budget: int,
    pid: int | None = None,
    lock_file: Path = DEFAULT_LOCK_FILE,
) -> bool:
    """
    Update the allocated budget for a session.

    Args:
        budget: New budget allocation
        pid: Process ID to update (defaults to current process)
        lock_file: Path to the lock file

    Returns:
        True if session was found and updated
    """
    if pid is None:
        pid = os.getpid()

    try:
        registry, f = _read_registry_locked(lock_file)

        session = registry.get_session(pid)
        if session:
            session.allocated_budget = budget
            registry.add_session(session)
            _write_registry_and_unlock(registry, lock_file, f)
            return True
        else:
            _write_registry_and_unlock(registry, lock_file, f)
            return False
    except (OSError, IOError) as e:
        logger.warning(f"Failed to update session budget: {e}")
        return False


def cleanup_stale_sessions(
    lock_file: Path = DEFAULT_LOCK_FILE,
    threshold_seconds: int = STALE_SESSION_THRESHOLD_SECONDS,
) -> list[int]:
    """
    Clean up stale sessions from the registry.

    Sessions are considered stale if:
    - Their last heartbeat is older than threshold_seconds (default: 1 hour)
    - Their process is no longer running

    Args:
        lock_file: Path to the lock file
        threshold_seconds: Seconds after which a session is considered stale

    Returns:
        List of PIDs that were removed
    """
    if not lock_file.exists():
        return []

    try:
        registry, f = _read_registry_locked(lock_file)
        removed = registry.cleanup_stale_sessions(threshold_seconds)
        _write_registry_and_unlock(registry, lock_file, f)
        return removed
    except (OSError, IOError) as e:
        logger.warning(f"Failed to cleanup stale sessions: {e}")
        return []


class SessionManager:
    """
    Context manager for Ralph session lifecycle.

    Automatically registers the session on entry and unregisters on exit,
    including crash scenarios.

    Usage:
        with SessionManager(prd_path) as session:
            # Run Ralph loop
            pass  # Session is automatically unregistered on exit
    """

    def __init__(
        self,
        prd_path: Path | None = None,
        lock_file: Path = DEFAULT_LOCK_FILE,
        heartbeat_interval: float = 60.0,  # seconds
    ):
        self.prd_path = prd_path
        self.lock_file = lock_file
        self.heartbeat_interval = heartbeat_interval
        self.session: SessionInfo | None = None
        self._original_sigint: Any = None
        self._original_sigterm: Any = None
        self._cleanup_registered = False

    def __enter__(self) -> SessionInfo:
        """Register the session and set up cleanup handlers."""
        self.session = register_session(self.prd_path, self.lock_file)

        # Register atexit handler for normal exit
        atexit.register(self._cleanup)
        self._cleanup_registered = True

        # Save original signal handlers and install our own
        self._original_sigint = signal.signal(signal.SIGINT, self._signal_handler)
        self._original_sigterm = signal.signal(signal.SIGTERM, self._signal_handler)

        return self.session

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Unregister the session."""
        self._cleanup()

        # Restore original signal handlers
        if self._original_sigint is not None:
            signal.signal(signal.SIGINT, self._original_sigint)
        if self._original_sigterm is not None:
            signal.signal(signal.SIGTERM, self._original_sigterm)

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle signals by cleaning up and re-raising."""
        self._cleanup()

        # Restore original handler and re-raise the signal
        if signum == signal.SIGINT and self._original_sigint is not None:
            signal.signal(signal.SIGINT, self._original_sigint)
            os.kill(os.getpid(), signal.SIGINT)
        elif signum == signal.SIGTERM and self._original_sigterm is not None:
            signal.signal(signal.SIGTERM, self._original_sigterm)
            os.kill(os.getpid(), signal.SIGTERM)

    def _cleanup(self) -> None:
        """Unregister the session."""
        if self.session is not None:
            unregister_session(self.session.pid, self.lock_file)
            self.session = None

        # Unregister atexit handler if registered
        if self._cleanup_registered:
            try:
                atexit.unregister(self._cleanup)
            except Exception:
                pass
            self._cleanup_registered = False

    def heartbeat(self) -> bool:
        """Update the session heartbeat."""
        if self.session is not None:
            return update_heartbeat(self.session.pid, self.lock_file)
        return False


def get_budget_for_session(
    remaining_budget: int,
    lock_file: Path = DEFAULT_LOCK_FILE,
) -> tuple[int, int]:
    """
    Get the budget allocation for the current session.

    This function should be called periodically during a Ralph run to get
    the current session's share of the remaining budget.

    Args:
        remaining_budget: Total remaining budget across all sessions
        lock_file: Path to the lock file

    Returns:
        Tuple of (session_budget, active_session_count)
    """
    sessions = get_active_sessions(lock_file, cleanup_stale=True)
    num_sessions = len(sessions) if sessions else 1

    session_budget = remaining_budget // num_sessions
    return session_budget, num_sessions
