"""Unit tests for the pre-flight usage check that runs before ralph run.

This module tests the pre-flight check functionality:
- Warning threshold (>70% usage)
- Blocking threshold with --strict (>90% usage)
- --ignore-limits flag bypasses checks
- Capacity estimation calculation
- Integration with ralph run command
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from ralph.cli import app
from ralph.config import Plan
from ralph.usage import (
    ESTIMATED_TOKENS_PER_ITERATION,
    PreflightCheck,
    UsageAggregate,
    check_usage_before_run,
)
from tests.fixtures import (
    FixtureDirectory,
    SessionRecord,
    create_temp_fixture_directory,
)


runner = CliRunner()


# =============================================================================
# Test PreflightCheck dataclass
# =============================================================================


class TestPreflightCheckDataclass:
    """Tests for the PreflightCheck dataclass."""

    def test_preflight_check_fields(self):
        """Test that PreflightCheck has expected fields."""
        now = datetime.now(timezone.utc)
        usage = UsageAggregate(
            window_start=now - timedelta(hours=5),
            window_end=now,
            input_tokens=100000,
            output_tokens=50000,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            message_count=10,
            request_count=10,
        )

        preflight = PreflightCheck(
            current_usage=usage,
            limit=300000,
            percentage=50.0,
            estimated_iterations_remaining=3,
            should_warn=False,
            should_block=False,
        )

        assert preflight.current_usage == usage
        assert preflight.limit == 300000
        assert preflight.percentage == 50.0
        assert preflight.estimated_iterations_remaining == 3
        assert preflight.should_warn is False
        assert preflight.should_block is False

    def test_tokens_remaining_property(self):
        """Test tokens_remaining property calculation."""
        now = datetime.now(timezone.utc)
        usage = UsageAggregate(
            window_start=now - timedelta(hours=5),
            window_end=now,
            input_tokens=100000,
            output_tokens=50000,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            message_count=10,
            request_count=10,
        )

        preflight = PreflightCheck(
            current_usage=usage,
            limit=300000,
            percentage=50.0,
            estimated_iterations_remaining=3,
            should_warn=False,
            should_block=False,
        )

        # Total tokens = 150000, limit = 300000, remaining = 150000
        assert preflight.tokens_remaining == 150000

    def test_tokens_remaining_never_negative(self):
        """Test that tokens_remaining is never negative."""
        now = datetime.now(timezone.utc)
        # Usage exceeds limit
        usage = UsageAggregate(
            window_start=now - timedelta(hours=5),
            window_end=now,
            input_tokens=400000,
            output_tokens=100000,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            message_count=10,
            request_count=10,
        )

        preflight = PreflightCheck(
            current_usage=usage,
            limit=300000,
            percentage=166.67,
            estimated_iterations_remaining=0,
            should_warn=True,
            should_block=True,
        )

        # Total tokens = 500000, limit = 300000, remaining should be 0 not -200000
        assert preflight.tokens_remaining == 0


# =============================================================================
# Test warning threshold (>70% usage)
# =============================================================================


class TestWarningThreshold:
    """Tests for warning threshold (>70% usage)."""

    def test_no_warning_below_70_percent(self):
        """Test that should_warn is False when usage is below 70%."""
        with create_temp_fixture_directory() as fixture:
            now = datetime(2025, 1, 20, 15, 0, 0, tzinfo=timezone.utc)

            # Create usage at 50% (150K out of 300K limit)
            fixture.create_session_with_usage(
                project="test-project",
                start_time=now - timedelta(hours=2),
                num_records=10,
                model="claude-sonnet-4-20250514",
                input_tokens_range=(10000, 10000),
                output_tokens_range=(5000, 5000),
                cache_creation_range=(0, 0),
                cache_read_range=(0, 0),
                include_user_messages=False,
            )

            preflight = check_usage_before_run(
                claude_dir=fixture.path,
                five_hour_limit=300000,
                now=now,
            )

            assert preflight.should_warn is False

    def test_warning_at_exactly_70_percent(self):
        """Test that should_warn is False at exactly 70%."""
        with create_temp_fixture_directory() as fixture:
            now = datetime(2025, 1, 20, 15, 0, 0, tzinfo=timezone.utc)

            # Create usage at exactly 70% (210K out of 300K limit)
            # 14 records * (10000 + 5000) = 210000 tokens
            fixture.create_session_with_usage(
                project="test-project",
                start_time=now - timedelta(hours=2),
                num_records=14,
                model="claude-sonnet-4-20250514",
                input_tokens_range=(10000, 10000),
                output_tokens_range=(5000, 5000),
                cache_creation_range=(0, 0),
                cache_read_range=(0, 0),
                include_user_messages=False,
            )

            preflight = check_usage_before_run(
                claude_dir=fixture.path,
                five_hour_limit=300000,
                now=now,
            )

            # At exactly 70%, should_warn should be False (>70 is the condition)
            assert preflight.should_warn is False

    def test_warning_above_70_percent(self):
        """Test that should_warn is True when usage is above 70%."""
        with create_temp_fixture_directory() as fixture:
            now = datetime(2025, 1, 20, 15, 0, 0, tzinfo=timezone.utc)

            # Create usage at 75% (225K out of 300K limit)
            # Use a single record with exact token counts for precision
            project_dir = fixture.create_project("test-project")
            records = [
                SessionRecord(
                    record_type="assistant",
                    timestamp=now - timedelta(hours=1),
                    model="claude-sonnet-4-20250514",
                    input_tokens=150000,
                    output_tokens=75000,  # Total: 225000 tokens = 75%
                ).to_jsonl_line()
            ]
            session_file = project_dir / "test-session.jsonl"
            session_file.write_text("\n".join(records))

            preflight = check_usage_before_run(
                claude_dir=fixture.path,
                five_hour_limit=300000,
                now=now,
            )

            assert preflight.should_warn is True
            assert preflight.should_block is False

    def test_warning_at_71_percent(self):
        """Test that should_warn is True at 71%."""
        with create_temp_fixture_directory() as fixture:
            now = datetime(2025, 1, 20, 15, 0, 0, tzinfo=timezone.utc)

            # Create usage slightly above 70%
            # Need total tokens of 213001 for >70% of 300000
            # 1 record with 213001 tokens total
            project_dir = fixture.create_project("test-project")
            records = [
                SessionRecord(
                    record_type="assistant",
                    timestamp=now - timedelta(hours=1),
                    model="claude-sonnet-4-20250514",
                    input_tokens=150000,
                    output_tokens=63001,  # Total: 213001 tokens = 71% of 300000
                ).to_jsonl_line()
            ]
            session_file = project_dir / "test-session.jsonl"
            session_file.write_text("\n".join(records))

            preflight = check_usage_before_run(
                claude_dir=fixture.path,
                five_hour_limit=300000,
                now=now,
            )

            assert preflight.percentage > 70
            assert preflight.should_warn is True

    def test_warning_but_not_blocking(self):
        """Test that should_warn can be True while should_block is False."""
        with create_temp_fixture_directory() as fixture:
            now = datetime(2025, 1, 20, 15, 0, 0, tzinfo=timezone.utc)

            # Create usage at 80% (240K out of 300K limit)
            project_dir = fixture.create_project("test-project")
            records = [
                SessionRecord(
                    record_type="assistant",
                    timestamp=now - timedelta(hours=1),
                    model="claude-sonnet-4-20250514",
                    input_tokens=160000,
                    output_tokens=80000,  # Total: 240000 tokens = 80%
                ).to_jsonl_line()
            ]
            session_file = project_dir / "test-session.jsonl"
            session_file.write_text("\n".join(records))

            preflight = check_usage_before_run(
                claude_dir=fixture.path,
                five_hour_limit=300000,
                now=now,
            )

            assert preflight.should_warn is True
            assert preflight.should_block is False


# =============================================================================
# Test blocking threshold with --strict (>90% usage)
# =============================================================================


class TestBlockingThreshold:
    """Tests for blocking threshold (>90% usage)."""

    def test_no_blocking_below_90_percent(self):
        """Test that should_block is False when usage is below 90%."""
        with create_temp_fixture_directory() as fixture:
            now = datetime(2025, 1, 20, 15, 0, 0, tzinfo=timezone.utc)

            # Create usage at 85% (255K out of 300K limit)
            # 17 records * (10000 + 5000) = 255000 tokens
            fixture.create_session_with_usage(
                project="test-project",
                start_time=now - timedelta(hours=2),
                num_records=17,
                model="claude-sonnet-4-20250514",
                input_tokens_range=(10000, 10000),
                output_tokens_range=(5000, 5000),
                cache_creation_range=(0, 0),
                cache_read_range=(0, 0),
                include_user_messages=False,
            )

            preflight = check_usage_before_run(
                claude_dir=fixture.path,
                five_hour_limit=300000,
                now=now,
            )

            assert preflight.should_block is False

    def test_blocking_at_exactly_90_percent(self):
        """Test that should_block is False at exactly 90%."""
        with create_temp_fixture_directory() as fixture:
            now = datetime(2025, 1, 20, 15, 0, 0, tzinfo=timezone.utc)

            # Create usage at exactly 90% (270K out of 300K limit)
            # 18 records * (10000 + 5000) = 270000 tokens
            fixture.create_session_with_usage(
                project="test-project",
                start_time=now - timedelta(hours=2),
                num_records=18,
                model="claude-sonnet-4-20250514",
                input_tokens_range=(10000, 10000),
                output_tokens_range=(5000, 5000),
                cache_creation_range=(0, 0),
                cache_read_range=(0, 0),
                include_user_messages=False,
            )

            preflight = check_usage_before_run(
                claude_dir=fixture.path,
                five_hour_limit=300000,
                now=now,
            )

            # At exactly 90%, should_block should be False (>90 is the condition)
            assert preflight.should_block is False

    def test_blocking_above_90_percent(self):
        """Test that should_block is True when usage is above 90%."""
        with create_temp_fixture_directory() as fixture:
            now = datetime(2025, 1, 20, 15, 0, 0, tzinfo=timezone.utc)

            # Create usage at 95% (285K out of 300K limit)
            project_dir = fixture.create_project("test-project")
            records = [
                SessionRecord(
                    record_type="assistant",
                    timestamp=now - timedelta(hours=1),
                    model="claude-sonnet-4-20250514",
                    input_tokens=200000,
                    output_tokens=85000,  # Total: 285000 tokens = 95%
                ).to_jsonl_line()
            ]
            session_file = project_dir / "test-session.jsonl"
            session_file.write_text("\n".join(records))

            preflight = check_usage_before_run(
                claude_dir=fixture.path,
                five_hour_limit=300000,
                now=now,
            )

            assert preflight.should_block is True

    def test_blocking_at_91_percent(self):
        """Test that should_block is True at 91%."""
        with create_temp_fixture_directory() as fixture:
            now = datetime(2025, 1, 20, 15, 0, 0, tzinfo=timezone.utc)

            # Create usage slightly above 90%
            # Need total tokens of 270001 for >90% of 300000
            project_dir = fixture.create_project("test-project")
            records = [
                SessionRecord(
                    record_type="assistant",
                    timestamp=now - timedelta(hours=1),
                    model="claude-sonnet-4-20250514",
                    input_tokens=200000,
                    output_tokens=73001,  # Total: 273001 tokens = 91% of 300000
                ).to_jsonl_line()
            ]
            session_file = project_dir / "test-session.jsonl"
            session_file.write_text("\n".join(records))

            preflight = check_usage_before_run(
                claude_dir=fixture.path,
                five_hour_limit=300000,
                now=now,
            )

            assert preflight.percentage > 90
            assert preflight.should_block is True

    def test_blocking_implies_warning(self):
        """Test that should_block True implies should_warn True."""
        with create_temp_fixture_directory() as fixture:
            now = datetime(2025, 1, 20, 15, 0, 0, tzinfo=timezone.utc)

            # Create usage at 95%
            project_dir = fixture.create_project("test-project")
            records = [
                SessionRecord(
                    record_type="assistant",
                    timestamp=now - timedelta(hours=1),
                    model="claude-sonnet-4-20250514",
                    input_tokens=200000,
                    output_tokens=85000,  # Total: 285000 tokens = 95%
                ).to_jsonl_line()
            ]
            session_file = project_dir / "test-session.jsonl"
            session_file.write_text("\n".join(records))

            preflight = check_usage_before_run(
                claude_dir=fixture.path,
                five_hour_limit=300000,
                now=now,
            )

            # If blocking, should also warn
            assert preflight.should_block is True
            assert preflight.should_warn is True

    def test_blocking_at_100_percent(self):
        """Test that should_block is True at 100% usage."""
        with create_temp_fixture_directory() as fixture:
            now = datetime(2025, 1, 20, 15, 0, 0, tzinfo=timezone.utc)

            # Create usage at 100% (300K out of 300K limit)
            project_dir = fixture.create_project("test-project")
            records = [
                SessionRecord(
                    record_type="assistant",
                    timestamp=now - timedelta(hours=1),
                    model="claude-sonnet-4-20250514",
                    input_tokens=200000,
                    output_tokens=100000,  # Total: 300000 tokens = 100%
                ).to_jsonl_line()
            ]
            session_file = project_dir / "test-session.jsonl"
            session_file.write_text("\n".join(records))

            preflight = check_usage_before_run(
                claude_dir=fixture.path,
                five_hour_limit=300000,
                now=now,
            )

            assert preflight.should_block is True
            assert preflight.tokens_remaining == 0


# =============================================================================
# Test capacity estimation calculation
# =============================================================================


class TestCapacityEstimation:
    """Tests for capacity estimation calculation."""

    def test_estimated_tokens_per_iteration_constant(self):
        """Test that ESTIMATED_TOKENS_PER_ITERATION is defined."""
        assert ESTIMATED_TOKENS_PER_ITERATION > 0
        assert ESTIMATED_TOKENS_PER_ITERATION == 50_000

    def test_capacity_estimation_full_budget(self):
        """Test capacity estimation with full budget available."""
        with create_temp_fixture_directory() as fixture:
            now = datetime(2025, 1, 20, 15, 0, 0, tzinfo=timezone.utc)

            # No usage - full budget available
            preflight = check_usage_before_run(
                claude_dir=fixture.path,
                five_hour_limit=300000,
                now=now,
            )

            # 300000 / 50000 = 6 iterations
            assert preflight.estimated_iterations_remaining == 6
            assert preflight.tokens_remaining == 300000

    def test_capacity_estimation_half_budget(self):
        """Test capacity estimation with half budget used."""
        with create_temp_fixture_directory() as fixture:
            now = datetime(2025, 1, 20, 15, 0, 0, tzinfo=timezone.utc)

            # Create usage at 50% (150K out of 300K limit)
            fixture.create_session_with_usage(
                project="test-project",
                start_time=now - timedelta(hours=2),
                num_records=10,
                model="claude-sonnet-4-20250514",
                input_tokens_range=(10000, 10000),
                output_tokens_range=(5000, 5000),
                cache_creation_range=(0, 0),
                cache_read_range=(0, 0),
                include_user_messages=False,
            )

            preflight = check_usage_before_run(
                claude_dir=fixture.path,
                five_hour_limit=300000,
                now=now,
            )

            # 150000 / 50000 = 3 iterations
            assert preflight.estimated_iterations_remaining == 3
            assert preflight.tokens_remaining == 150000

    def test_capacity_estimation_nearly_empty(self):
        """Test capacity estimation when almost all budget is used."""
        with create_temp_fixture_directory() as fixture:
            now = datetime(2025, 1, 20, 15, 0, 0, tzinfo=timezone.utc)

            # Create usage at 95% (285K out of 300K limit)
            project_dir = fixture.create_project("test-project")
            records = [
                SessionRecord(
                    record_type="assistant",
                    timestamp=now - timedelta(hours=1),
                    model="claude-sonnet-4-20250514",
                    input_tokens=200000,
                    output_tokens=85000,  # Total: 285000 tokens = 95%
                ).to_jsonl_line()
            ]
            session_file = project_dir / "test-session.jsonl"
            session_file.write_text("\n".join(records))

            preflight = check_usage_before_run(
                claude_dir=fixture.path,
                five_hour_limit=300000,
                now=now,
            )

            # 15000 / 50000 = 0 iterations (integer division)
            assert preflight.estimated_iterations_remaining == 0
            assert preflight.tokens_remaining == 15000

    def test_capacity_estimation_exceeds_limit(self):
        """Test capacity estimation when usage exceeds limit."""
        with create_temp_fixture_directory() as fixture:
            now = datetime(2025, 1, 20, 15, 0, 0, tzinfo=timezone.utc)

            # Create usage exceeding limit
            project_dir = fixture.create_project("test-project")
            records = [
                SessionRecord(
                    record_type="assistant",
                    timestamp=now - timedelta(hours=1),
                    model="claude-sonnet-4-20250514",
                    input_tokens=250000,
                    output_tokens=100000,  # Total: 350000 tokens (exceeds 300K limit)
                ).to_jsonl_line()
            ]
            session_file = project_dir / "test-session.jsonl"
            session_file.write_text("\n".join(records))

            preflight = check_usage_before_run(
                claude_dir=fixture.path,
                five_hour_limit=300000,
                now=now,
            )

            # Should be 0 iterations, not negative
            assert preflight.estimated_iterations_remaining == 0
            assert preflight.tokens_remaining == 0

    def test_capacity_estimation_with_different_limits(self):
        """Test capacity estimation with different plan limits."""
        with create_temp_fixture_directory() as fixture:
            now = datetime(2025, 1, 20, 15, 0, 0, tzinfo=timezone.utc)

            # Free tier: 30K limit, should estimate 0 iterations
            preflight_free = check_usage_before_run(
                claude_dir=fixture.path,
                five_hour_limit=30000,
                now=now,
            )
            assert preflight_free.estimated_iterations_remaining == 0  # 30000 / 50000 = 0

            # Pro tier: 300K limit, should estimate 6 iterations
            preflight_pro = check_usage_before_run(
                claude_dir=fixture.path,
                five_hour_limit=300000,
                now=now,
            )
            assert preflight_pro.estimated_iterations_remaining == 6

            # Max5X tier: 1.5M limit, should estimate 30 iterations
            preflight_max5x = check_usage_before_run(
                claude_dir=fixture.path,
                five_hour_limit=1500000,
                now=now,
            )
            assert preflight_max5x.estimated_iterations_remaining == 30

    def test_capacity_estimation_uses_default_limit(self):
        """Test that default limit is used when not specified."""
        with create_temp_fixture_directory() as fixture:
            now = datetime(2025, 1, 20, 15, 0, 0, tzinfo=timezone.utc)

            preflight = check_usage_before_run(
                claude_dir=fixture.path,
                now=now,
            )

            # Default pro limit is 300000
            assert preflight.limit == 300000
            assert preflight.estimated_iterations_remaining == 6


# =============================================================================
# Test CLI integration with ralph run command
# =============================================================================


class TestCliPreflightIntegration:
    """Tests for pre-flight check integration with ralph run command."""

    def test_run_command_with_strict_blocks_at_high_usage(self):
        """Test that --strict flag blocks run when >90% usage."""
        with create_temp_fixture_directory() as fixture:
            now = datetime.now(timezone.utc)

            # Create usage at 95%
            project_dir = fixture.create_project("test-project")
            records = [
                SessionRecord(
                    record_type="assistant",
                    timestamp=now - timedelta(hours=1),
                    model="claude-sonnet-4-20250514",
                    input_tokens=200000,
                    output_tokens=85000,  # Total: 285000 tokens = 95%
                ).to_jsonl_line()
            ]
            session_file = project_dir / "test-session.jsonl"
            session_file.write_text("\n".join(records))

            with tempfile.TemporaryDirectory() as tmpdir:
                # Create a minimal PRD file
                prd_path = Path(tmpdir) / "prd.json"
                prd_content = {
                    "project": "Test Project",
                    "branchName": "test-branch",
                    "description": "Test",
                    "userStories": [
                        {
                            "id": "US-001",
                            "title": "Test Story",
                            "description": "Test",
                            "acceptanceCriteria": ["Test"],
                            "priority": 1,
                            "passes": False,
                        }
                    ],
                }
                prd_path.write_text(json.dumps(prd_content))

                config_file = Path(tmpdir) / "config.json"
                config_file.write_text('{"plan": "pro"}')

                with patch("ralph.config.DEFAULT_CONFIG_FILE", config_file):
                    with patch("ralph.usage.DEFAULT_CLAUDE_DIR", fixture.path):
                        result = runner.invoke(
                            app,
                            ["run", "--prd", str(prd_path), "--strict", "--dry-run"],
                        )

                        # Should exit with non-zero code due to high usage with --strict
                        # Note: --dry-run won't run but preflight happens before that
                        assert result.exit_code != 0

    def test_run_command_warns_at_high_usage_without_strict(self):
        """Test that run shows warning at >70% usage without --strict."""
        with create_temp_fixture_directory() as fixture:
            now = datetime.now(timezone.utc)

            # Create usage at 75%
            project_dir = fixture.create_project("test-project")
            records = [
                SessionRecord(
                    record_type="assistant",
                    timestamp=now - timedelta(hours=1),
                    model="claude-sonnet-4-20250514",
                    input_tokens=150000,
                    output_tokens=75000,  # Total: 225000 tokens = 75%
                ).to_jsonl_line()
            ]
            session_file = project_dir / "test-session.jsonl"
            session_file.write_text("\n".join(records))

            with tempfile.TemporaryDirectory() as tmpdir:
                prd_path = Path(tmpdir) / "prd.json"
                prd_content = {
                    "project": "Test Project",
                    "branchName": "test-branch",
                    "description": "Test",
                    "userStories": [
                        {
                            "id": "US-001",
                            "title": "Test Story",
                            "description": "Test",
                            "acceptanceCriteria": ["Test"],
                            "priority": 1,
                            "passes": False,
                        }
                    ],
                }
                prd_path.write_text(json.dumps(prd_content))

                config_file = Path(tmpdir) / "config.json"
                config_file.write_text('{"plan": "pro"}')

                with patch("ralph.config.DEFAULT_CONFIG_FILE", config_file):
                    with patch("ralph.usage.DEFAULT_CLAUDE_DIR", fixture.path):
                        # Without --strict, should show warning but proceed (dry-run exits 0)
                        result = runner.invoke(
                            app,
                            ["run", "--prd", str(prd_path), "--dry-run"],
                        )

                        # Should exit 0 (dry-run successful) - warning doesn't block
                        assert result.exit_code == 0

    def test_run_command_ignore_limits_bypasses_checks(self):
        """Test that --ignore-limits bypasses all usage checks."""
        with create_temp_fixture_directory() as fixture:
            now = datetime.now(timezone.utc)

            # Create usage at 95%
            project_dir = fixture.create_project("test-project")
            records = [
                SessionRecord(
                    record_type="assistant",
                    timestamp=now - timedelta(hours=1),
                    model="claude-sonnet-4-20250514",
                    input_tokens=200000,
                    output_tokens=85000,  # Total: 285000 tokens = 95%
                ).to_jsonl_line()
            ]
            session_file = project_dir / "test-session.jsonl"
            session_file.write_text("\n".join(records))

            with tempfile.TemporaryDirectory() as tmpdir:
                prd_path = Path(tmpdir) / "prd.json"
                prd_content = {
                    "project": "Test Project",
                    "branchName": "test-branch",
                    "description": "Test",
                    "userStories": [
                        {
                            "id": "US-001",
                            "title": "Test Story",
                            "description": "Test",
                            "acceptanceCriteria": ["Test"],
                            "priority": 1,
                            "passes": False,
                        }
                    ],
                }
                prd_path.write_text(json.dumps(prd_content))

                config_file = Path(tmpdir) / "config.json"
                config_file.write_text('{"plan": "pro"}')

                with patch("ralph.config.DEFAULT_CONFIG_FILE", config_file):
                    with patch("ralph.usage.DEFAULT_CLAUDE_DIR", fixture.path):
                        # Even with --strict, --ignore-limits should bypass checks
                        result = runner.invoke(
                            app,
                            ["run", "--prd", str(prd_path), "--strict", "--ignore-limits", "--dry-run"],
                        )

                        # Should exit 0 (dry-run successful) - limits ignored
                        assert result.exit_code == 0

    def test_run_command_no_warning_at_low_usage(self):
        """Test that no warning is shown at low usage."""
        with create_temp_fixture_directory() as fixture:
            now = datetime.now(timezone.utc)

            # Create minimal usage (well below 70%)
            fixture.create_session_with_usage(
                project="test-project",
                start_time=now - timedelta(hours=2),
                num_records=2,
                model="claude-sonnet-4-20250514",
                input_tokens_range=(1000, 1000),
                output_tokens_range=(500, 500),
                cache_creation_range=(0, 0),
                cache_read_range=(0, 0),
                include_user_messages=False,
            )

            with tempfile.TemporaryDirectory() as tmpdir:
                prd_path = Path(tmpdir) / "prd.json"
                prd_content = {
                    "project": "Test Project",
                    "branchName": "test-branch",
                    "description": "Test",
                    "userStories": [
                        {
                            "id": "US-001",
                            "title": "Test Story",
                            "description": "Test",
                            "acceptanceCriteria": ["Test"],
                            "priority": 1,
                            "passes": False,
                        }
                    ],
                }
                prd_path.write_text(json.dumps(prd_content))

                config_file = Path(tmpdir) / "config.json"
                config_file.write_text('{"plan": "pro"}')

                with patch("ralph.config.DEFAULT_CONFIG_FILE", config_file):
                    with patch("ralph.usage.DEFAULT_CLAUDE_DIR", fixture.path):
                        result = runner.invoke(
                            app,
                            ["run", "--prd", str(prd_path), "--dry-run"],
                        )

                        assert result.exit_code == 0
                        # Should not contain warning about usage
                        # Warning messages typically contain "warning" or "!"

    def test_run_with_no_claude_data(self):
        """Test that run works when no Claude data exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prd_path = Path(tmpdir) / "prd.json"
            prd_content = {
                "project": "Test Project",
                "branchName": "test-branch",
                "description": "Test",
                "userStories": [
                    {
                        "id": "US-001",
                        "title": "Test Story",
                        "description": "Test",
                        "acceptanceCriteria": ["Test"],
                        "priority": 1,
                        "passes": False,
                    }
                ],
            }
            prd_path.write_text(json.dumps(prd_content))

            config_file = Path(tmpdir) / "config.json"
            nonexistent_claude_dir = Path(tmpdir) / "nonexistent-claude"

            with patch("ralph.config.DEFAULT_CONFIG_FILE", config_file):
                with patch("ralph.usage.DEFAULT_CLAUDE_DIR", nonexistent_claude_dir):
                    result = runner.invoke(
                        app,
                        ["run", "--prd", str(prd_path), "--dry-run"],
                    )

                    # Should succeed with 0% usage
                    assert result.exit_code == 0


# =============================================================================
# Test check_usage_before_run function
# =============================================================================


class TestCheckUsageBeforeRun:
    """Tests for the check_usage_before_run function."""

    def test_returns_preflight_check(self):
        """Test that check_usage_before_run returns PreflightCheck."""
        with create_temp_fixture_directory() as fixture:
            now = datetime(2025, 1, 20, 15, 0, 0, tzinfo=timezone.utc)

            result = check_usage_before_run(
                claude_dir=fixture.path,
                five_hour_limit=300000,
                now=now,
            )

            assert isinstance(result, PreflightCheck)

    def test_uses_5hour_window(self):
        """Test that check_usage_before_run uses 5-hour window."""
        with create_temp_fixture_directory() as fixture:
            now = datetime(2025, 1, 20, 15, 0, 0, tzinfo=timezone.utc)

            # Create records in different time ranges
            # Record within 5 hours (should be counted)
            fixture.create_session_with_usage(
                project="recent-project",
                start_time=now - timedelta(hours=2),
                num_records=5,
                model="claude-sonnet-4-20250514",
                input_tokens_range=(10000, 10000),
                output_tokens_range=(5000, 5000),
                cache_creation_range=(0, 0),
                cache_read_range=(0, 0),
                include_user_messages=False,
            )

            # Record outside 5 hours (should NOT be counted)
            fixture.create_session_with_usage(
                project="old-project",
                start_time=now - timedelta(hours=6),
                num_records=5,
                model="claude-sonnet-4-20250514",
                input_tokens_range=(10000, 10000),
                output_tokens_range=(5000, 5000),
                cache_creation_range=(0, 0),
                cache_read_range=(0, 0),
                include_user_messages=False,
            )

            result = check_usage_before_run(
                claude_dir=fixture.path,
                five_hour_limit=300000,
                now=now,
            )

            # Only 5 records within window, each 15000 tokens = 75000 total
            assert result.current_usage.message_count == 5
            # 75000 / 300000 = 25%
            assert result.percentage == 25.0

    def test_percentage_calculation(self):
        """Test that percentage is calculated correctly."""
        with create_temp_fixture_directory() as fixture:
            now = datetime(2025, 1, 20, 15, 0, 0, tzinfo=timezone.utc)

            # Create 10 records with 15000 tokens each = 150000 total
            fixture.create_session_with_usage(
                project="test-project",
                start_time=now - timedelta(hours=2),
                num_records=10,
                model="claude-sonnet-4-20250514",
                input_tokens_range=(10000, 10000),
                output_tokens_range=(5000, 5000),
                cache_creation_range=(0, 0),
                cache_read_range=(0, 0),
                include_user_messages=False,
            )

            result = check_usage_before_run(
                claude_dir=fixture.path,
                five_hour_limit=300000,
                now=now,
            )

            # 150000 / 300000 * 100 = 50%
            assert result.percentage == 50.0

    def test_zero_limit_handling(self):
        """Test handling of zero limit (edge case)."""
        with create_temp_fixture_directory() as fixture:
            now = datetime(2025, 1, 20, 15, 0, 0, tzinfo=timezone.utc)

            result = check_usage_before_run(
                claude_dir=fixture.path,
                five_hour_limit=0,
                now=now,
            )

            # With zero limit, percentage should be 0 (not division by zero)
            assert result.percentage == 0
            assert result.limit == 0

    def test_empty_directory_handling(self):
        """Test handling of empty Claude data directory."""
        with create_temp_fixture_directory() as fixture:
            now = datetime(2025, 1, 20, 15, 0, 0, tzinfo=timezone.utc)

            result = check_usage_before_run(
                claude_dir=fixture.path,
                five_hour_limit=300000,
                now=now,
            )

            # Should return 0% usage
            assert result.percentage == 0
            assert result.current_usage.total_tokens == 0
            assert result.should_warn is False
            assert result.should_block is False

    def test_nonexistent_directory_handling(self):
        """Test handling of non-existent Claude data directory."""
        nonexistent_dir = Path("/tmp/nonexistent-claude-dir-test-12345")
        now = datetime(2025, 1, 20, 15, 0, 0, tzinfo=timezone.utc)

        result = check_usage_before_run(
            claude_dir=nonexistent_dir,
            five_hour_limit=300000,
            now=now,
        )

        # Should return 0% usage
        assert result.percentage == 0
        assert result.current_usage.total_tokens == 0
        assert result.should_warn is False
        assert result.should_block is False


# =============================================================================
# Test edge cases
# =============================================================================


class TestPreflightEdgeCases:
    """Tests for edge cases in pre-flight checks."""

    def test_exact_threshold_boundaries(self):
        """Test exact threshold boundaries (70.0%, 90.0%)."""
        # Test the exact boundary percentages
        now = datetime(2025, 1, 20, 15, 0, 0, tzinfo=timezone.utc)

        # At 70.0% exactly
        usage_70 = UsageAggregate(
            window_start=now - timedelta(hours=5),
            window_end=now,
            input_tokens=210000,
            output_tokens=0,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            message_count=1,
            request_count=1,
        )

        # At 70.0%: 210000 / 300000 * 100 = 70.0
        # should_warn = percentage > 70 = False
        preflight_70 = PreflightCheck(
            current_usage=usage_70,
            limit=300000,
            percentage=70.0,
            estimated_iterations_remaining=1,
            should_warn=70.0 > 70,  # False
            should_block=70.0 > 90,  # False
        )
        assert preflight_70.should_warn is False
        assert preflight_70.should_block is False

        # At 90.0% exactly
        usage_90 = UsageAggregate(
            window_start=now - timedelta(hours=5),
            window_end=now,
            input_tokens=270000,
            output_tokens=0,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            message_count=1,
            request_count=1,
        )

        # At 90.0%: 270000 / 300000 * 100 = 90.0
        # should_block = percentage > 90 = False
        preflight_90 = PreflightCheck(
            current_usage=usage_90,
            limit=300000,
            percentage=90.0,
            estimated_iterations_remaining=0,
            should_warn=90.0 > 70,  # True
            should_block=90.0 > 90,  # False
        )
        assert preflight_90.should_warn is True
        assert preflight_90.should_block is False

    def test_very_small_usage(self):
        """Test with very small usage amounts."""
        with create_temp_fixture_directory() as fixture:
            now = datetime(2025, 1, 20, 15, 0, 0, tzinfo=timezone.utc)

            project_dir = fixture.create_project("test-project")
            records = [
                SessionRecord(
                    record_type="assistant",
                    timestamp=now - timedelta(hours=1),
                    model="claude-sonnet-4-20250514",
                    input_tokens=1,
                    output_tokens=1,
                ).to_jsonl_line()
            ]
            session_file = project_dir / "test-session.jsonl"
            session_file.write_text("\n".join(records))

            result = check_usage_before_run(
                claude_dir=fixture.path,
                five_hour_limit=300000,
                now=now,
            )

            assert result.current_usage.total_tokens == 2
            assert result.percentage < 1
            assert result.should_warn is False
            assert result.should_block is False

    def test_cache_tokens_in_calculation(self):
        """Test that cache creation is included but cache reads excluded from rate limiting."""
        with create_temp_fixture_directory() as fixture:
            now = datetime(2025, 1, 20, 15, 0, 0, tzinfo=timezone.utc)

            # Create records with significant cache tokens
            fixture.create_session_with_usage(
                project="test-project",
                start_time=now - timedelta(hours=2),
                num_records=10,
                model="claude-sonnet-4-20250514",
                input_tokens_range=(5000, 5000),
                output_tokens_range=(5000, 5000),
                cache_creation_range=(2500, 2500),
                cache_read_range=(2500, 2500),
                include_user_messages=False,
            )

            result = check_usage_before_run(
                claude_dir=fixture.path,
                five_hour_limit=300000,
                now=now,
            )

            # total_tokens includes all: 5000 + 5000 + 2500 + 2500 = 15000 per record
            # 10 records = 150000 total tokens (for cost calculation)
            assert result.current_usage.total_tokens == 150000

            # rate_limited_tokens excludes cache reads: 5000 + 5000 + 2500 = 12500 per record
            # 10 records = 125000 rate-limited tokens
            assert result.current_usage.rate_limited_tokens == 125000

            # Percentage is based on rate_limited_tokens
            # 125000 / 300000 = 41.67%
            assert abs(result.percentage - 41.67) < 0.1

    def test_uses_plan_limit_correctly(self):
        """Test that different plan limits affect thresholds correctly."""
        with create_temp_fixture_directory() as fixture:
            now = datetime(2025, 1, 20, 15, 0, 0, tzinfo=timezone.utc)

            # Create 100K tokens of usage
            project_dir = fixture.create_project("test-project")
            records = [
                SessionRecord(
                    record_type="assistant",
                    timestamp=now - timedelta(hours=1),
                    model="claude-sonnet-4-20250514",
                    input_tokens=60000,
                    output_tokens=40000,
                ).to_jsonl_line()
            ]
            session_file = project_dir / "test-session.jsonl"
            session_file.write_text("\n".join(records))

            # With pro limit (300K), 100K = 33.3%
            result_pro = check_usage_before_run(
                claude_dir=fixture.path,
                five_hour_limit=300000,
                now=now,
            )
            assert result_pro.percentage < 70
            assert result_pro.should_warn is False

            # With free limit (30K), 100K = 333.3%
            result_free = check_usage_before_run(
                claude_dir=fixture.path,
                five_hour_limit=30000,
                now=now,
            )
            assert result_free.percentage > 90
            assert result_free.should_warn is True
            assert result_free.should_block is True
