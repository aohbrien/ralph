"""Test fixtures for Claude session data.

This module provides sample JSONL files that mimic Claude's session format
for reliable, reproducible testing without requiring real Claude data.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

# Directory containing static fixture files
FIXTURES_DIR = Path(__file__).parent


@dataclass
class SessionRecord:
    """A record to be written to a session JSONL file."""

    record_type: str
    timestamp: datetime
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    session_id: str | None = None

    def to_jsonl_line(self) -> str:
        """Convert to a JSONL line matching Claude's format."""
        ts = self.timestamp.isoformat()
        if self.timestamp.tzinfo is not None:
            # Convert to Z suffix for UTC
            if ts.endswith("+00:00"):
                ts = ts[:-6] + "Z"

        data: dict = {
            "type": self.record_type,
            "timestamp": ts,
        }

        if self.session_id:
            data["sessionId"] = self.session_id

        if self.record_type == "assistant":
            message: dict = {}
            if self.model:
                message["model"] = self.model

            message["usage"] = {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "cache_creation_input_tokens": self.cache_creation_input_tokens,
                "cache_read_input_tokens": self.cache_read_input_tokens,
            }
            data["message"] = message
        elif self.record_type == "user":
            data["message"] = {"content": "User message content"}

        return json.dumps(data)


def generate_session_records(
    start_time: datetime,
    num_records: int = 5,
    model: str = "claude-sonnet-4-20250514",
    input_tokens_range: tuple[int, int] = (1000, 5000),
    output_tokens_range: tuple[int, int] = (500, 2000),
    cache_creation_range: tuple[int, int] = (0, 1000),
    cache_read_range: tuple[int, int] = (0, 500),
    interval_minutes: int = 10,
    include_user_messages: bool = True,
    session_id: str | None = None,
) -> Iterator[SessionRecord]:
    """
    Generate a sequence of session records.

    Args:
        start_time: Starting timestamp for the first record
        num_records: Number of assistant message records to generate
        model: Model name to use
        input_tokens_range: Min/max for input_tokens
        output_tokens_range: Min/max for output_tokens
        cache_creation_range: Min/max for cache_creation_input_tokens
        cache_read_range: Min/max for cache_read_input_tokens
        interval_minutes: Time between messages
        include_user_messages: Whether to include user messages between assistant messages
        session_id: Session ID to use (generated if None)

    Yields:
        SessionRecord objects
    """
    import random

    if session_id is None:
        session_id = str(uuid.uuid4())

    current_time = start_time

    for i in range(num_records):
        # Optionally generate a user message first
        if include_user_messages:
            yield SessionRecord(
                record_type="user",
                timestamp=current_time,
                session_id=session_id,
            )
            current_time += timedelta(seconds=30)

        # Generate assistant message with usage
        yield SessionRecord(
            record_type="assistant",
            timestamp=current_time,
            model=model,
            input_tokens=random.randint(*input_tokens_range),
            output_tokens=random.randint(*output_tokens_range),
            cache_creation_input_tokens=random.randint(*cache_creation_range),
            cache_read_input_tokens=random.randint(*cache_read_range),
            session_id=session_id,
        )

        current_time += timedelta(minutes=interval_minutes)


def write_session_file(
    records: list[SessionRecord] | Iterator[SessionRecord],
    file_path: Path,
) -> None:
    """Write session records to a JSONL file."""
    with open(file_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(record.to_jsonl_line() + "\n")


class FixtureDirectory:
    """
    A temporary directory structure mimicking Claude's session storage.

    Structure:
        temp_dir/
            project-name-hash/
                session-uuid-1.jsonl
                session-uuid-2.jsonl
            other-project-hash/
                session-uuid-3.jsonl

    Usage:
        with FixtureDirectory() as fixture:
            # Create a session
            fixture.create_session(
                project="my-project",
                records=[...],
            )
            # Use fixture.path as the claude_dir argument
            usage = parse_all_sessions(fixture.path)
    """

    def __init__(self, base_dir: Path | None = None):
        """
        Initialize the fixture directory.

        Args:
            base_dir: Optional base directory. If None, creates a temp directory.
        """
        self._temp_dir: tempfile.TemporaryDirectory | None = None
        self._base_dir = base_dir
        self._path: Path | None = None

    def __enter__(self) -> "FixtureDirectory":
        if self._base_dir is None:
            self._temp_dir = tempfile.TemporaryDirectory()
            self._path = Path(self._temp_dir.name)
        else:
            self._path = self._base_dir
            self._path.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._temp_dir is not None:
            self._temp_dir.cleanup()

    @property
    def path(self) -> Path:
        """Get the path to the fixture directory."""
        if self._path is None:
            raise RuntimeError("FixtureDirectory not initialized. Use as context manager.")
        return self._path

    def create_project(self, project_name: str) -> Path:
        """Create a project directory and return its path."""
        # Claude uses a hash-based directory name, but we can use a simple name for tests
        project_dir = self.path / project_name
        project_dir.mkdir(parents=True, exist_ok=True)
        return project_dir

    def create_session(
        self,
        project: str = "test-project",
        session_id: str | None = None,
        records: list[SessionRecord] | None = None,
    ) -> Path:
        """
        Create a session file in a project directory.

        Args:
            project: Project name/identifier
            session_id: Session UUID (generated if None)
            records: List of SessionRecord objects to write

        Returns:
            Path to the created session file
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        project_dir = self.create_project(project)
        session_file = project_dir / f"{session_id}.jsonl"

        if records:
            write_session_file(records, session_file)
        else:
            # Create empty file
            session_file.touch()

        return session_file

    def create_session_with_usage(
        self,
        project: str = "test-project",
        session_id: str | None = None,
        start_time: datetime | None = None,
        num_records: int = 5,
        model: str = "claude-sonnet-4-20250514",
        **kwargs,
    ) -> Path:
        """
        Create a session file with generated usage data.

        Args:
            project: Project name/identifier
            session_id: Session UUID (generated if None)
            start_time: Starting timestamp (defaults to 1 hour ago)
            num_records: Number of assistant messages to generate
            model: Model name to use
            **kwargs: Additional arguments passed to generate_session_records

        Returns:
            Path to the created session file
        """
        if start_time is None:
            start_time = datetime.now(timezone.utc) - timedelta(hours=1)

        if session_id is None:
            session_id = str(uuid.uuid4())

        records = list(
            generate_session_records(
                start_time=start_time,
                num_records=num_records,
                model=model,
                session_id=session_id,
                **kwargs,
            )
        )

        return self.create_session(
            project=project,
            session_id=session_id,
            records=records,
        )


def create_temp_fixture_directory() -> FixtureDirectory:
    """
    Create a temporary fixture directory for tests.

    This is a helper function that creates a FixtureDirectory instance.
    Use it as a context manager:

        with create_temp_fixture_directory() as fixture:
            session_file = fixture.create_session_with_usage(...)
            # Run tests using fixture.path
    """
    return FixtureDirectory()


# Pre-built fixture scenarios for common test cases


def create_single_session_fixture(
    fixture_dir: FixtureDirectory,
    start_time: datetime | None = None,
) -> Path:
    """
    Create a fixture with a single session containing normal usage data.

    Returns the path to the session file.
    """
    if start_time is None:
        start_time = datetime.now(timezone.utc) - timedelta(hours=2)

    return fixture_dir.create_session_with_usage(
        project="single-session-project",
        start_time=start_time,
        num_records=10,
        model="claude-sonnet-4-20250514",
        input_tokens_range=(2000, 4000),
        output_tokens_range=(1000, 2000),
    )


def create_multiple_sessions_fixture(
    fixture_dir: FixtureDirectory,
    num_sessions: int = 3,
    start_time: datetime | None = None,
) -> list[Path]:
    """
    Create a fixture with multiple sessions across different projects.

    Returns list of paths to session files.
    """
    if start_time is None:
        start_time = datetime.now(timezone.utc) - timedelta(hours=4)

    paths = []

    # Session 1: Sonnet in project A
    paths.append(
        fixture_dir.create_session_with_usage(
            project="project-a-hash123",
            start_time=start_time,
            num_records=5,
            model="claude-sonnet-4-20250514",
        )
    )

    # Session 2: Opus in project A
    paths.append(
        fixture_dir.create_session_with_usage(
            project="project-a-hash123",
            start_time=start_time + timedelta(hours=1),
            num_records=3,
            model="claude-opus-4-20250514",
            input_tokens_range=(5000, 10000),
            output_tokens_range=(2000, 5000),
        )
    )

    # Session 3: Sonnet in project B
    if num_sessions >= 3:
        paths.append(
            fixture_dir.create_session_with_usage(
                project="project-b-hash456",
                start_time=start_time + timedelta(hours=2),
                num_records=7,
                model="claude-sonnet-4-20250514",
            )
        )

    return paths


def create_malformed_data_fixture(fixture_dir: FixtureDirectory) -> Path:
    """
    Create a fixture with malformed/edge case data.

    Contains:
    - Valid records
    - Invalid JSON lines
    - Lines with missing usage data
    - Lines with non-integer token values
    """
    project_dir = fixture_dir.create_project("malformed-project")
    session_file = project_dir / f"{uuid.uuid4()}.jsonl"

    now = datetime.now(timezone.utc)
    lines = [
        # Valid assistant message
        SessionRecord(
            record_type="assistant",
            timestamp=now - timedelta(hours=1),
            model="claude-sonnet-4-20250514",
            input_tokens=1000,
            output_tokens=500,
        ).to_jsonl_line(),
        # Invalid JSON
        "this is not valid json{",
        # Empty line
        "",
        # Valid but non-assistant type
        json.dumps(
            {
                "type": "user",
                "timestamp": (now - timedelta(minutes=50)).isoformat(),
                "message": {"content": "Hello"},
            }
        ),
        # Assistant without usage
        json.dumps(
            {
                "type": "assistant",
                "timestamp": (now - timedelta(minutes=45)).isoformat(),
                "message": {"content": "Response without usage"},
            }
        ),
        # Assistant with null usage values
        json.dumps(
            {
                "type": "assistant",
                "timestamp": (now - timedelta(minutes=40)).isoformat(),
                "message": {
                    "model": "claude-sonnet-4-20250514",
                    "usage": {
                        "input_tokens": None,
                        "output_tokens": 500,
                    },
                },
            }
        ),
        # Valid assistant message at the end
        SessionRecord(
            record_type="assistant",
            timestamp=now - timedelta(minutes=30),
            model="claude-sonnet-4-20250514",
            input_tokens=2000,
            output_tokens=1000,
        ).to_jsonl_line(),
        # Line with trailing whitespace (should be handled)
        SessionRecord(
            record_type="assistant",
            timestamp=now - timedelta(minutes=20),
            model="claude-sonnet-4-20250514",
            input_tokens=1500,
            output_tokens=750,
        ).to_jsonl_line()
        + "   ",
    ]

    with open(session_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return session_file


def create_empty_file_fixture(fixture_dir: FixtureDirectory) -> Path:
    """Create a fixture with an empty session file."""
    return fixture_dir.create_session(
        project="empty-project",
        records=[],
    )


def create_realistic_usage_fixture(
    fixture_dir: FixtureDirectory,
    hours_of_history: int = 12,
    messages_per_hour: int = 3,
) -> list[Path]:
    """
    Create a fixture with realistic usage patterns over multiple hours.

    Simulates a typical coding session with varying activity levels.
    """
    start_time = datetime.now(timezone.utc) - timedelta(hours=hours_of_history)
    paths = []

    # Create multiple sessions representing different coding sessions
    session_start = start_time

    while session_start < datetime.now(timezone.utc) - timedelta(hours=1):
        # Vary the number of records per session
        import random

        num_records = random.randint(2, 8)

        # Alternate between models occasionally
        model = (
            "claude-opus-4-20250514"
            if random.random() < 0.2
            else "claude-sonnet-4-20250514"
        )

        paths.append(
            fixture_dir.create_session_with_usage(
                project=f"coding-project-{random.randint(1, 3)}",
                start_time=session_start,
                num_records=num_records,
                model=model,
                interval_minutes=random.randint(5, 15),
                input_tokens_range=(1500, 8000),
                output_tokens_range=(500, 3000),
                cache_creation_range=(100, 2000),
                cache_read_range=(500, 5000),
            )
        )

        # Gap between sessions
        session_start += timedelta(
            hours=random.randint(1, 3), minutes=random.randint(0, 30)
        )

    return paths
