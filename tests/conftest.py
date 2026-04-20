"""Shared pytest fixtures."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import pytest


@pytest.fixture
def write_instance() -> Callable[..., Path]:
    """Factory fixture that writes a dashboard instance state file under a given dir."""

    def _write(
        dir: Path,
        *,
        pid: int,
        heartbeat_age_seconds: float = 0.0,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        hb = (
            datetime.now(timezone.utc) - timedelta(seconds=heartbeat_age_seconds)
        ).isoformat()
        data: dict[str, Any] = {
            "schema_version": 1,
            "pid": pid,
            "host": "testhost",
            "cwd": str(dir),
            "prd_path": str(dir / "prd.json"),
            "prd_project": f"project-{pid}",
            "tool": "claude",
            "two_phase": False,
            "phase": "single",
            "started_at": hb,
            "updated_at": hb,
            "last_heartbeat": hb,
            "iteration": 3,
            "max_iterations": 10,
            "current_story": {"id": "US-001", "title": "test"},
            "prd_progress": {"passing": 1, "total": 5},
            "usage": {},
            "last_iteration_result": None,
            "tail_file": str(dir / f"{pid}.tail"),
            "tail_bytes": 0,
        }
        if extra:
            data.update(extra)
        path = dir / f"{pid}.json"
        path.write_text(json.dumps(data))
        return path

    return _write
