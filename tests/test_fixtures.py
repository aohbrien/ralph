"""Tests for the fixture module itself.

These tests verify that the fixture generation and helper functions
work correctly and produce valid Claude session data format.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from ralph.usage import (
    UsageRecord,
    discover_session_files,
    parse_all_sessions,
    parse_session_file,
)
from tests.fixtures import (
    FIXTURES_DIR,
    FixtureDirectory,
    SessionRecord,
    create_empty_file_fixture,
    create_malformed_data_fixture,
    create_multiple_sessions_fixture,
    create_single_session_fixture,
    create_temp_fixture_directory,
    generate_session_records,
    write_session_file,
)


class TestSessionRecord:
    """Tests for SessionRecord dataclass."""

    def test_assistant_record_to_jsonl(self):
        """Test converting an assistant record to JSONL format."""
        record = SessionRecord(
            record_type="assistant",
            timestamp=datetime(2025, 1, 20, 10, 0, 0, tzinfo=timezone.utc),
            model="claude-sonnet-4-20250514",
            input_tokens=1000,
            output_tokens=500,
            cache_creation_input_tokens=100,
            cache_read_input_tokens=50,
            session_id="test-session-123",
        )

        line = record.to_jsonl_line()

        # Verify it's valid JSON by parsing
        import json

        data = json.loads(line)

        assert data["type"] == "assistant"
        assert "2025-01-20T10:00:00" in data["timestamp"]
        assert data["sessionId"] == "test-session-123"
        assert data["message"]["model"] == "claude-sonnet-4-20250514"
        assert data["message"]["usage"]["input_tokens"] == 1000
        assert data["message"]["usage"]["output_tokens"] == 500
        assert data["message"]["usage"]["cache_creation_input_tokens"] == 100
        assert data["message"]["usage"]["cache_read_input_tokens"] == 50

    def test_user_record_to_jsonl(self):
        """Test converting a user record to JSONL format."""
        record = SessionRecord(
            record_type="user",
            timestamp=datetime(2025, 1, 20, 10, 0, 0, tzinfo=timezone.utc),
            session_id="test-session-123",
        )

        line = record.to_jsonl_line()

        import json

        data = json.loads(line)

        assert data["type"] == "user"
        assert data["sessionId"] == "test-session-123"
        assert "message" in data


class TestGenerateSessionRecords:
    """Tests for generate_session_records function."""

    def test_generates_correct_number_of_records(self):
        """Test that the function generates the requested number of records."""
        start = datetime(2025, 1, 20, 10, 0, 0, tzinfo=timezone.utc)
        records = list(
            generate_session_records(
                start_time=start,
                num_records=5,
                include_user_messages=False,
            )
        )

        assert len(records) == 5
        assert all(r.record_type == "assistant" for r in records)

    def test_includes_user_messages(self):
        """Test that user messages are included when requested."""
        start = datetime(2025, 1, 20, 10, 0, 0, tzinfo=timezone.utc)
        records = list(
            generate_session_records(
                start_time=start,
                num_records=3,
                include_user_messages=True,
            )
        )

        # Should have 3 user + 3 assistant = 6 records
        assert len(records) == 6
        user_records = [r for r in records if r.record_type == "user"]
        assistant_records = [r for r in records if r.record_type == "assistant"]
        assert len(user_records) == 3
        assert len(assistant_records) == 3

    def test_timestamps_are_sequential(self):
        """Test that timestamps increase over time."""
        start = datetime(2025, 1, 20, 10, 0, 0, tzinfo=timezone.utc)
        records = list(
            generate_session_records(
                start_time=start,
                num_records=5,
                include_user_messages=False,
            )
        )

        timestamps = [r.timestamp for r in records]
        for i in range(1, len(timestamps)):
            assert timestamps[i] > timestamps[i - 1]

    def test_token_counts_in_range(self):
        """Test that token counts are within specified ranges."""
        start = datetime(2025, 1, 20, 10, 0, 0, tzinfo=timezone.utc)
        records = list(
            generate_session_records(
                start_time=start,
                num_records=10,
                include_user_messages=False,
                input_tokens_range=(100, 200),
                output_tokens_range=(50, 100),
            )
        )

        for r in records:
            assert 100 <= r.input_tokens <= 200
            assert 50 <= r.output_tokens <= 100


class TestFixtureDirectory:
    """Tests for FixtureDirectory context manager."""

    def test_creates_temp_directory(self):
        """Test that a temp directory is created."""
        with FixtureDirectory() as fixture:
            assert fixture.path.exists()
            assert fixture.path.is_dir()

    def test_cleanup_on_exit(self):
        """Test that temp directory is cleaned up on exit."""
        with FixtureDirectory() as fixture:
            path = fixture.path
        assert not path.exists()

    def test_create_project(self):
        """Test creating a project directory."""
        with FixtureDirectory() as fixture:
            project_path = fixture.create_project("my-project")
            assert project_path.exists()
            assert project_path.is_dir()
            assert project_path.name == "my-project"

    def test_create_empty_session(self):
        """Test creating an empty session file."""
        with FixtureDirectory() as fixture:
            session_path = fixture.create_session(
                project="test-project",
                session_id="test-session",
                records=[],
            )
            assert session_path.exists()
            assert session_path.suffix == ".jsonl"
            assert session_path.stat().st_size == 0

    def test_create_session_with_records(self):
        """Test creating a session file with records."""
        with FixtureDirectory() as fixture:
            records = [
                SessionRecord(
                    record_type="assistant",
                    timestamp=datetime.now(timezone.utc),
                    model="claude-sonnet-4-20250514",
                    input_tokens=1000,
                    output_tokens=500,
                )
            ]
            session_path = fixture.create_session(
                project="test-project",
                records=records,
            )
            assert session_path.exists()
            assert session_path.stat().st_size > 0

    def test_create_session_with_usage(self):
        """Test the convenience method for creating sessions with generated usage."""
        with FixtureDirectory() as fixture:
            session_path = fixture.create_session_with_usage(
                project="test-project",
                num_records=5,
            )
            assert session_path.exists()

            # Verify the file has valid records
            records = list(parse_session_file(session_path))
            assert len(records) == 5


class TestCreateTempFixtureDirectory:
    """Tests for the create_temp_fixture_directory helper function."""

    def test_returns_fixture_directory(self):
        """Test that helper returns a FixtureDirectory instance."""
        fixture = create_temp_fixture_directory()
        assert isinstance(fixture, FixtureDirectory)

    def test_works_as_context_manager(self):
        """Test using the helper as a context manager."""
        with create_temp_fixture_directory() as fixture:
            session_path = fixture.create_session_with_usage(num_records=3)
            assert session_path.exists()


class TestFixtureScenarios:
    """Tests for pre-built fixture scenarios."""

    def test_single_session_fixture(self):
        """Test creating a single session fixture."""
        with create_temp_fixture_directory() as fixture:
            session_path = create_single_session_fixture(fixture)
            assert session_path.exists()

            records = list(parse_session_file(session_path))
            assert len(records) == 10

    def test_multiple_sessions_fixture(self):
        """Test creating multiple sessions fixture."""
        with create_temp_fixture_directory() as fixture:
            paths = create_multiple_sessions_fixture(fixture, num_sessions=3)
            assert len(paths) == 3

            # Verify we have both sonnet and opus models
            all_records = []
            for path in paths:
                all_records.extend(parse_session_file(path))

            models = {r.model for r in all_records if r.model}
            assert any("sonnet" in m.lower() for m in models)
            assert any("opus" in m.lower() for m in models)

    def test_malformed_data_fixture(self):
        """Test creating malformed data fixture."""
        with create_temp_fixture_directory() as fixture:
            session_path = create_malformed_data_fixture(fixture)
            assert session_path.exists()

            # Should parse some valid records despite malformed lines
            records = list(parse_session_file(session_path))
            # The fixture has 3 valid assistant messages with usage
            assert len(records) >= 2

    def test_empty_file_fixture(self):
        """Test creating empty file fixture."""
        with create_temp_fixture_directory() as fixture:
            session_path = create_empty_file_fixture(fixture)
            assert session_path.exists()
            assert session_path.stat().st_size == 0

            records = list(parse_session_file(session_path))
            assert len(records) == 0


class TestStaticFixtureFiles:
    """Tests for static fixture files."""

    def test_single_session_file_exists(self):
        """Test that single_session.jsonl exists and is valid."""
        fixture_file = FIXTURES_DIR / "single_session.jsonl"
        assert fixture_file.exists()

        records = list(parse_session_file(fixture_file))
        assert len(records) == 5  # 5 assistant messages

    def test_opus_session_file(self):
        """Test that session_opus.jsonl exists and contains opus records."""
        fixture_file = FIXTURES_DIR / "session_opus.jsonl"
        assert fixture_file.exists()

        records = list(parse_session_file(fixture_file))
        assert len(records) >= 1
        assert all(r.is_opus for r in records)

    def test_mixed_models_file(self):
        """Test that session_mixed_models.jsonl contains both models."""
        fixture_file = FIXTURES_DIR / "session_mixed_models.jsonl"
        assert fixture_file.exists()

        records = list(parse_session_file(fixture_file))
        assert len(records) >= 2

        has_sonnet = any(r.is_sonnet for r in records)
        has_opus = any(r.is_opus for r in records)
        assert has_sonnet and has_opus

    def test_malformed_file_handles_errors(self):
        """Test that malformed.jsonl can be parsed without crashing."""
        fixture_file = FIXTURES_DIR / "malformed.jsonl"
        assert fixture_file.exists()

        # Should not raise an exception
        records = list(parse_session_file(fixture_file))
        # Should still get some valid records
        assert len(records) >= 1

    def test_empty_file(self):
        """Test that empty.jsonl returns no records."""
        fixture_file = FIXTURES_DIR / "empty.jsonl"
        assert fixture_file.exists()

        records = list(parse_session_file(fixture_file))
        assert len(records) == 0

    def test_edge_cases_file(self):
        """Test that edge_cases.jsonl is handled correctly."""
        fixture_file = FIXTURES_DIR / "edge_cases.jsonl"
        assert fixture_file.exists()

        records = list(parse_session_file(fixture_file))
        # Should handle various edge cases like zero tokens, different timestamp formats
        assert len(records) >= 1

    def test_high_usage_file(self):
        """Test that high_usage.jsonl has realistic high token counts."""
        fixture_file = FIXTURES_DIR / "high_usage.jsonl"
        assert fixture_file.exists()

        records = list(parse_session_file(fixture_file))
        assert len(records) >= 1

        # Verify high token counts
        total_input = sum(r.total_input_tokens for r in records)
        assert total_input > 100000  # Should have high usage


class TestIntegrationWithUsageModule:
    """Integration tests verifying fixtures work with ralph/usage.py."""

    def test_discover_session_files_finds_fixtures(self):
        """Test that discover_session_files finds files in fixture directories."""
        with create_temp_fixture_directory() as fixture:
            # Create multiple sessions
            fixture.create_session_with_usage(project="proj1", num_records=3)
            fixture.create_session_with_usage(project="proj2", num_records=3)

            files = discover_session_files(fixture.path)
            assert len(files) == 2

    def test_parse_all_sessions_aggregates_fixtures(self):
        """Test that parse_all_sessions correctly aggregates fixture data."""
        with create_temp_fixture_directory() as fixture:
            start = datetime.now(timezone.utc) - timedelta(hours=2)

            fixture.create_session_with_usage(
                project="proj1",
                start_time=start,
                num_records=5,
            )
            fixture.create_session_with_usage(
                project="proj2",
                start_time=start + timedelta(hours=1),
                num_records=3,
            )

            records = parse_all_sessions(fixture.path)
            assert len(records) == 8  # 5 + 3

            # Verify records are sorted by timestamp
            timestamps = [r.timestamp for r in records]
            assert timestamps == sorted(timestamps)

    def test_parse_all_sessions_with_since_filter(self):
        """Test filtering records by timestamp."""
        with create_temp_fixture_directory() as fixture:
            old_time = datetime.now(timezone.utc) - timedelta(hours=10)
            recent_time = datetime.now(timezone.utc) - timedelta(hours=1)

            fixture.create_session_with_usage(
                project="old-proj",
                start_time=old_time,
                num_records=5,
            )
            fixture.create_session_with_usage(
                project="recent-proj",
                start_time=recent_time,
                num_records=3,
            )

            # Filter to only recent records
            since = datetime.now(timezone.utc) - timedelta(hours=2)
            records = parse_all_sessions(fixture.path, since=since)

            # Should only get the 3 recent records
            assert len(records) == 3
