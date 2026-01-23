"""Parse Claude session data for usage tracking."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

# Default Claude projects directory
DEFAULT_CLAUDE_DIR = Path.home() / ".claude" / "projects"


@dataclass
class UsageRecord:
    """A single usage record from a Claude session."""

    timestamp: datetime
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    model: str | None = None
    session_id: str | None = None

    @property
    def total_input_tokens(self) -> int:
        """Total input tokens including cache tokens."""
        return (
            self.input_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
        )

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output)."""
        return self.total_input_tokens + self.output_tokens

    @property
    def is_opus(self) -> bool:
        """Check if this record is from an Opus model."""
        return self.model is not None and "opus" in self.model.lower()

    @property
    def is_sonnet(self) -> bool:
        """Check if this record is from a Sonnet model."""
        return self.model is not None and "sonnet" in self.model.lower()


def discover_session_files(
    claude_dir: Path | None = None,
) -> list[Path]:
    """
    Discover all Claude session JSONL files.

    Args:
        claude_dir: Path to Claude projects directory.
                   Defaults to ~/.claude/projects/

    Returns:
        List of paths to JSONL session files, sorted by modification time (newest first).
    """
    if claude_dir is None:
        claude_dir = DEFAULT_CLAUDE_DIR

    if not claude_dir.exists():
        logger.debug(f"Claude directory not found: {claude_dir}")
        return []

    # Find all .jsonl files in project subdirectories
    jsonl_files = list(claude_dir.glob("*/*.jsonl"))

    # Sort by modification time, newest first
    jsonl_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    return jsonl_files


def _parse_timestamp(timestamp_str: str) -> datetime | None:
    """Parse an ISO timestamp string to datetime."""
    try:
        # Handle ISO format with Z suffix (UTC)
        if timestamp_str.endswith("Z"):
            timestamp_str = timestamp_str[:-1] + "+00:00"
        return datetime.fromisoformat(timestamp_str)
    except (ValueError, TypeError):
        return None


def _parse_usage_from_line(line: str, session_id: str | None = None) -> UsageRecord | None:
    """
    Parse a single JSONL line and extract usage data if present.

    Args:
        line: A single line from a JSONL file
        session_id: Optional session ID to include in the record

    Returns:
        UsageRecord if the line contains assistant usage data, None otherwise.
    """
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        logger.debug("Skipping malformed JSON line")
        return None

    # Only process assistant messages
    if data.get("type") != "assistant":
        return None

    # Get the message object
    message = data.get("message")
    if not isinstance(message, dict):
        return None

    # Extract usage data
    usage = message.get("usage")
    if not isinstance(usage, dict):
        return None

    # Parse timestamp
    timestamp_str = data.get("timestamp")
    if not timestamp_str:
        return None

    timestamp = _parse_timestamp(timestamp_str)
    if timestamp is None:
        logger.debug(f"Could not parse timestamp: {timestamp_str}")
        return None

    # Extract token counts with defaults
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    cache_creation = usage.get("cache_creation_input_tokens", 0)
    cache_read = usage.get("cache_read_input_tokens", 0)

    # Ensure all values are integers
    try:
        input_tokens = int(input_tokens) if input_tokens else 0
        output_tokens = int(output_tokens) if output_tokens else 0
        cache_creation = int(cache_creation) if cache_creation else 0
        cache_read = int(cache_read) if cache_read else 0
    except (ValueError, TypeError):
        logger.debug("Invalid token count values")
        return None

    # Extract model if available
    model = message.get("model")

    # Get session ID from data if not provided
    if session_id is None:
        session_id = data.get("sessionId")

    return UsageRecord(
        timestamp=timestamp,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=cache_creation,
        cache_read_input_tokens=cache_read,
        model=model,
        session_id=session_id,
    )


def parse_session_file(
    file_path: Path,
) -> Iterator[UsageRecord]:
    """
    Parse a Claude session JSONL file and yield usage records.

    Args:
        file_path: Path to the JSONL file

    Yields:
        UsageRecord for each assistant message with usage data.

    Note:
        Malformed lines are silently skipped with debug logging.
    """
    # Extract session ID from filename (UUID.jsonl)
    session_id = file_path.stem

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                record = _parse_usage_from_line(line, session_id)
                if record is not None:
                    yield record
    except OSError as e:
        logger.warning(f"Could not read session file {file_path}: {e}")
    except UnicodeDecodeError as e:
        logger.warning(f"Encoding error in session file {file_path}: {e}")


def parse_all_sessions(
    claude_dir: Path | None = None,
    since: datetime | None = None,
) -> list[UsageRecord]:
    """
    Parse all Claude session files and return usage records.

    Args:
        claude_dir: Path to Claude projects directory.
                   Defaults to ~/.claude/projects/
        since: If provided, only return records after this timestamp.

    Returns:
        List of UsageRecord objects sorted by timestamp (oldest first).
    """
    session_files = discover_session_files(claude_dir)

    records: list[UsageRecord] = []
    for file_path in session_files:
        for record in parse_session_file(file_path):
            if since is None or record.timestamp >= since:
                records.append(record)

    # Sort by timestamp, oldest first
    records.sort(key=lambda r: r.timestamp)

    return records


@dataclass
class UsageAggregate:
    """Aggregated usage data for a time window."""

    window_start: datetime
    window_end: datetime
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    message_count: int
    request_count: int

    @property
    def total_input_tokens(self) -> int:
        """Total input tokens including cache tokens."""
        return (
            self.input_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
        )

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output)."""
        return self.total_input_tokens + self.output_tokens


def _aggregate_records(
    records: list[UsageRecord],
    window_start: datetime,
    window_end: datetime,
) -> UsageAggregate:
    """
    Aggregate a list of usage records into a single aggregate.

    Args:
        records: List of UsageRecord objects to aggregate
        window_start: Start of the time window
        window_end: End of the time window

    Returns:
        UsageAggregate with totals for all records.
    """
    input_tokens = 0
    output_tokens = 0
    cache_creation = 0
    cache_read = 0

    for record in records:
        input_tokens += record.input_tokens
        output_tokens += record.output_tokens
        cache_creation += record.cache_creation_input_tokens
        cache_read += record.cache_read_input_tokens

    # Each record is one message/request
    message_count = len(records)
    request_count = len(records)

    return UsageAggregate(
        window_start=window_start,
        window_end=window_end,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=cache_creation,
        cache_read_input_tokens=cache_read,
        message_count=message_count,
        request_count=request_count,
    )


def _filter_by_model(
    records: list[UsageRecord],
    model_filter: str | None = None,
) -> list[UsageRecord]:
    """
    Filter records by model type.

    Args:
        records: List of UsageRecord objects
        model_filter: Filter string - "opus", "sonnet", or None for all

    Returns:
        Filtered list of records.
    """
    if model_filter is None:
        return records

    model_filter = model_filter.lower()

    if model_filter == "opus":
        return [r for r in records if r.is_opus]
    elif model_filter == "sonnet":
        return [r for r in records if r.is_sonnet]
    else:
        # Unknown filter, return all
        logger.warning(f"Unknown model filter: {model_filter}")
        return records


def _filter_by_time_window(
    records: list[UsageRecord],
    window_start: datetime,
    window_end: datetime,
) -> list[UsageRecord]:
    """
    Filter records to those within a time window.

    Args:
        records: List of UsageRecord objects
        window_start: Start of time window (inclusive)
        window_end: End of time window (exclusive)

    Returns:
        Filtered list of records within the window.
    """
    # Make timestamps timezone-aware for comparison if needed
    filtered = []
    for record in records:
        ts = record.timestamp
        # Ensure comparison works with timezone-aware datetimes
        if ts >= window_start and ts < window_end:
            filtered.append(record)
    return filtered


def get_5hour_window_usage(
    claude_dir: Path | None = None,
    model_filter: str | None = None,
    now: datetime | None = None,
) -> UsageAggregate:
    """
    Calculate usage for the current 5-hour rolling window.

    Args:
        claude_dir: Path to Claude projects directory.
                   Defaults to ~/.claude/projects/
        model_filter: Filter by model - "opus", "sonnet", or None for all
        now: Current time (defaults to datetime.now(timezone.utc))

    Returns:
        UsageAggregate for the 5-hour window ending at 'now'.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    window_start = now - timedelta(hours=5)
    window_end = now

    # Parse all records since window start
    records = parse_all_sessions(claude_dir, since=window_start)

    # Filter by model if specified
    records = _filter_by_model(records, model_filter)

    # Filter to window (parse_all_sessions uses 'since' but we want exact bounds)
    records = _filter_by_time_window(records, window_start, window_end)

    return _aggregate_records(records, window_start, window_end)


def get_weekly_window_usage(
    claude_dir: Path | None = None,
    model_filter: str | None = None,
    now: datetime | None = None,
) -> UsageAggregate:
    """
    Calculate usage for the current weekly (7-day) rolling window.

    Args:
        claude_dir: Path to Claude projects directory.
                   Defaults to ~/.claude/projects/
        model_filter: Filter by model - "opus", "sonnet", or None for all
        now: Current time (defaults to datetime.now(timezone.utc))

    Returns:
        UsageAggregate for the 7-day window ending at 'now'.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    window_start = now - timedelta(days=7)
    window_end = now

    # Parse all records since window start
    records = parse_all_sessions(claude_dir, since=window_start)

    # Filter by model if specified
    records = _filter_by_model(records, model_filter)

    # Filter to window (parse_all_sessions uses 'since' but we want exact bounds)
    records = _filter_by_time_window(records, window_start, window_end)

    return _aggregate_records(records, window_start, window_end)
