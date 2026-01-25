"""Run state persistence for Ralph resume functionality."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


STATE_FILENAME = ".ralph-state.json"


@dataclass
class RunState:
    """State of a Ralph run for resume functionality."""

    prd_path: str
    tool: str
    last_iteration: int
    last_story_id: str | None
    started_at: str
    updated_at: str
    story_count: int | None = None
    last_reeval_iteration: int | None = None

    @classmethod
    def create(
        cls,
        prd_path: Path,
        tool: str,
        iteration: int,
        story_id: str | None,
        story_count: int | None = None,
    ) -> RunState:
        """Create a new run state."""
        now = datetime.now().isoformat()
        return cls(
            prd_path=str(prd_path.resolve()),
            tool=tool,
            last_iteration=iteration,
            last_story_id=story_id,
            started_at=now,
            updated_at=now,
            story_count=story_count,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunState:
        """Create a RunState from a dictionary."""
        return cls(
            prd_path=data["prd_path"],
            tool=data["tool"],
            last_iteration=data["last_iteration"],
            last_story_id=data.get("last_story_id"),
            started_at=data["started_at"],
            updated_at=data["updated_at"],
            story_count=data.get("story_count"),
            last_reeval_iteration=data.get("last_reeval_iteration"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "prd_path": self.prd_path,
            "tool": self.tool,
            "last_iteration": self.last_iteration,
            "last_story_id": self.last_story_id,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "story_count": self.story_count,
            "last_reeval_iteration": self.last_reeval_iteration,
        }

    def update(self, iteration: int, story_id: str | None) -> None:
        """Update the state with new iteration info."""
        self.last_iteration = iteration
        self.last_story_id = story_id
        self.updated_at = datetime.now().isoformat()

    def update_reeval(self, iteration: int) -> None:
        """Update the state with the last re-evaluation iteration."""
        self.last_reeval_iteration = iteration
        self.updated_at = datetime.now().isoformat()


def get_state_path(base_dir: Path) -> Path:
    """Get the path to the state file."""
    return base_dir / STATE_FILENAME


def load_state(base_dir: Path) -> RunState | None:
    """
    Load run state from file.

    Returns None if no state file exists or if it's invalid.
    """
    state_path = get_state_path(base_dir)
    if not state_path.exists():
        return None

    try:
        data = json.loads(state_path.read_text())
        return RunState.from_dict(data)
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def save_state(base_dir: Path, state: RunState) -> None:
    """Save run state to file."""
    state_path = get_state_path(base_dir)
    state_path.write_text(json.dumps(state.to_dict(), indent=2) + "\n")


def clear_state(base_dir: Path) -> bool:
    """
    Clear run state file.

    Returns True if state file was deleted, False if it didn't exist.
    """
    state_path = get_state_path(base_dir)
    if state_path.exists():
        state_path.unlink()
        return True
    return False
