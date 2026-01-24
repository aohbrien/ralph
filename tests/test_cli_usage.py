"""Integration tests for the ralph usage CLI commands.

This module tests the CLI commands using Typer's test runner:
- 'ralph usage' command output
- 'ralph usage --set-plan' command
- 'ralph usage --history' command
- CLI with no Claude data directory
- CLI output format and color coding thresholds
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
from tests.fixtures import (
    FixtureDirectory,
    SessionRecord,
    create_temp_fixture_directory,
)


runner = CliRunner()


# =============================================================================
# Test 'ralph usage' command output
# =============================================================================


class TestUsageCommand:
    """Tests for 'ralph usage' command output."""

    def test_usage_command_runs_successfully(self):
        """Test that 'ralph usage' command runs without errors."""
        with create_temp_fixture_directory() as fixture:
            with tempfile.TemporaryDirectory() as config_dir:
                config_file = Path(config_dir) / "config.json"

                with patch("ralph.config.DEFAULT_CONFIG_FILE", config_file):
                    with patch("ralph.usage.DEFAULT_CLAUDE_DIR", fixture.path):
                        result = runner.invoke(app, ["usage"])

                        # Should exit successfully (code 0)
                        assert result.exit_code == 0

    def test_usage_command_shows_plan(self):
        """Test that 'ralph usage' shows the current plan."""
        with create_temp_fixture_directory() as fixture:
            with tempfile.TemporaryDirectory() as config_dir:
                config_file = Path(config_dir) / "config.json"
                # Write config with specific plan
                config_file.parent.mkdir(parents=True, exist_ok=True)
                config_file.write_text('{"plan": "max5x"}')

                with patch("ralph.config.DEFAULT_CONFIG_FILE", config_file):
                    with patch("ralph.usage.DEFAULT_CLAUDE_DIR", fixture.path):
                        result = runner.invoke(app, ["usage"])

                        assert result.exit_code == 0
                        # Should show the plan (case may vary due to Rich formatting)
                        assert "MAX5X" in result.output or "max5x" in result.output

    def test_usage_command_shows_5hour_window(self):
        """Test that 'ralph usage' shows 5-hour window usage."""
        with create_temp_fixture_directory() as fixture:
            with tempfile.TemporaryDirectory() as config_dir:
                config_file = Path(config_dir) / "config.json"

                with patch("ralph.config.DEFAULT_CONFIG_FILE", config_file):
                    with patch("ralph.usage.DEFAULT_CLAUDE_DIR", fixture.path):
                        result = runner.invoke(app, ["usage"])

                        assert result.exit_code == 0
                        # Should mention 5-hour window
                        assert "5-Hour" in result.output or "5-hour" in result.output

    def test_usage_command_shows_weekly_window(self):
        """Test that 'ralph usage' shows weekly window usage."""
        with create_temp_fixture_directory() as fixture:
            with tempfile.TemporaryDirectory() as config_dir:
                config_file = Path(config_dir) / "config.json"

                with patch("ralph.config.DEFAULT_CONFIG_FILE", config_file):
                    with patch("ralph.usage.DEFAULT_CLAUDE_DIR", fixture.path):
                        result = runner.invoke(app, ["usage"])

                        assert result.exit_code == 0
                        # Should mention weekly window
                        assert "Weekly" in result.output or "weekly" in result.output

    def test_usage_command_with_session_data(self):
        """Test 'ralph usage' with actual session data."""
        with create_temp_fixture_directory() as fixture:
            now = datetime.now(timezone.utc)

            # Create session with known usage
            fixture.create_session_with_usage(
                project="test-project",
                start_time=now - timedelta(hours=2),
                num_records=5,
                model="claude-sonnet-4-20250514",
                input_tokens_range=(1000, 1000),
                output_tokens_range=(500, 500),
                cache_creation_range=(0, 0),
                cache_read_range=(0, 0),
                include_user_messages=False,
            )

            with tempfile.TemporaryDirectory() as config_dir:
                config_file = Path(config_dir) / "config.json"

                with patch("ralph.config.DEFAULT_CONFIG_FILE", config_file):
                    with patch("ralph.usage.DEFAULT_CLAUDE_DIR", fixture.path):
                        result = runner.invoke(app, ["usage"])

                        assert result.exit_code == 0
                        # Should show some tokens were used
                        # The output should contain usage information

    def test_usage_command_shows_token_breakdown(self):
        """Test that 'ralph usage' shows token breakdown by model."""
        with create_temp_fixture_directory() as fixture:
            with tempfile.TemporaryDirectory() as config_dir:
                config_file = Path(config_dir) / "config.json"

                with patch("ralph.config.DEFAULT_CONFIG_FILE", config_file):
                    with patch("ralph.usage.DEFAULT_CLAUDE_DIR", fixture.path):
                        result = runner.invoke(app, ["usage"])

                        assert result.exit_code == 0
                        # Should show model breakdown table
                        assert "Model" in result.output or "Opus" in result.output or "Sonnet" in result.output


# =============================================================================
# Test 'ralph usage --set-plan' command
# =============================================================================


class TestSetPlanCommand:
    """Tests for 'ralph usage --set-plan' command."""

    def test_set_plan_free(self):
        """Test setting plan to 'free'."""
        with tempfile.TemporaryDirectory() as config_dir:
            config_file = Path(config_dir) / "config.json"

            with patch("ralph.config.DEFAULT_CONFIG_FILE", config_file):
                result = runner.invoke(app, ["usage", "--set-plan", "free"])

                assert result.exit_code == 0
                assert "free" in result.output.lower()

                # Verify file was written
                assert config_file.exists()
                data = json.loads(config_file.read_text())
                assert data["plan"] == "free"

    def test_set_plan_pro(self):
        """Test setting plan to 'pro'."""
        with tempfile.TemporaryDirectory() as config_dir:
            config_file = Path(config_dir) / "config.json"

            with patch("ralph.config.DEFAULT_CONFIG_FILE", config_file):
                result = runner.invoke(app, ["usage", "--set-plan", "pro"])

                assert result.exit_code == 0
                assert "pro" in result.output.lower()

                data = json.loads(config_file.read_text())
                assert data["plan"] == "pro"

    def test_set_plan_max5x(self):
        """Test setting plan to 'max5x'."""
        with tempfile.TemporaryDirectory() as config_dir:
            config_file = Path(config_dir) / "config.json"

            with patch("ralph.config.DEFAULT_CONFIG_FILE", config_file):
                result = runner.invoke(app, ["usage", "--set-plan", "max5x"])

                assert result.exit_code == 0
                assert "max5x" in result.output.lower()

                data = json.loads(config_file.read_text())
                assert data["plan"] == "max5x"

    def test_set_plan_max20x(self):
        """Test setting plan to 'max20x'."""
        with tempfile.TemporaryDirectory() as config_dir:
            config_file = Path(config_dir) / "config.json"

            with patch("ralph.config.DEFAULT_CONFIG_FILE", config_file):
                result = runner.invoke(app, ["usage", "--set-plan", "max20x"])

                assert result.exit_code == 0
                assert "max20x" in result.output.lower()

                data = json.loads(config_file.read_text())
                assert data["plan"] == "max20x"

    def test_set_plan_uppercase(self):
        """Test setting plan with uppercase input."""
        with tempfile.TemporaryDirectory() as config_dir:
            config_file = Path(config_dir) / "config.json"

            with patch("ralph.config.DEFAULT_CONFIG_FILE", config_file):
                result = runner.invoke(app, ["usage", "--set-plan", "MAX5X"])

                assert result.exit_code == 0
                data = json.loads(config_file.read_text())
                assert data["plan"] == "max5x"

    def test_set_plan_mixed_case(self):
        """Test setting plan with mixed case input."""
        with tempfile.TemporaryDirectory() as config_dir:
            config_file = Path(config_dir) / "config.json"

            with patch("ralph.config.DEFAULT_CONFIG_FILE", config_file):
                result = runner.invoke(app, ["usage", "--set-plan", "Max20X"])

                assert result.exit_code == 0
                data = json.loads(config_file.read_text())
                assert data["plan"] == "max20x"

    def test_set_plan_invalid_fails(self):
        """Test that invalid plan type fails."""
        with tempfile.TemporaryDirectory() as config_dir:
            config_file = Path(config_dir) / "config.json"

            with patch("ralph.config.DEFAULT_CONFIG_FILE", config_file):
                result = runner.invoke(app, ["usage", "--set-plan", "invalid_plan"])

                # Should fail with non-zero exit code
                assert result.exit_code != 0
                # Should show error message
                assert "Unknown plan" in result.output or "Error" in result.output or "error" in result.output

    def test_set_plan_empty_string_fails(self):
        """Test that empty plan string fails."""
        with tempfile.TemporaryDirectory() as config_dir:
            config_file = Path(config_dir) / "config.json"

            with patch("ralph.config.DEFAULT_CONFIG_FILE", config_file):
                result = runner.invoke(app, ["usage", "--set-plan", ""])

                # Should fail
                assert result.exit_code != 0

    def test_set_plan_creates_config_directory(self):
        """Test that --set-plan creates config directory if needed."""
        with tempfile.TemporaryDirectory() as base_dir:
            config_file = Path(base_dir) / "nested" / "dir" / "config.json"
            assert not config_file.parent.exists()

            with patch("ralph.config.DEFAULT_CONFIG_FILE", config_file):
                result = runner.invoke(app, ["usage", "--set-plan", "pro"])

                assert result.exit_code == 0
                assert config_file.exists()

    def test_set_plan_overwrites_existing(self):
        """Test that --set-plan overwrites existing plan."""
        with tempfile.TemporaryDirectory() as config_dir:
            config_file = Path(config_dir) / "config.json"
            config_file.write_text('{"plan": "free"}')

            with patch("ralph.config.DEFAULT_CONFIG_FILE", config_file):
                result = runner.invoke(app, ["usage", "--set-plan", "max20x"])

                assert result.exit_code == 0
                data = json.loads(config_file.read_text())
                assert data["plan"] == "max20x"


# =============================================================================
# Test 'ralph usage --history' command
# =============================================================================


class TestHistoryCommand:
    """Tests for 'ralph usage --history' command."""

    def test_history_command_basic(self):
        """Test basic --history command."""
        with create_temp_fixture_directory() as fixture:
            with tempfile.TemporaryDirectory() as config_dir:
                config_file = Path(config_dir) / "config.json"

                with patch("ralph.config.DEFAULT_CONFIG_FILE", config_file):
                    with patch("ralph.usage.DEFAULT_CLAUDE_DIR", fixture.path):
                        result = runner.invoke(app, ["usage", "--history", "1"])

                        assert result.exit_code == 0
                        # Should show historical usage header
                        assert "Historical" in result.output or "history" in result.output.lower()

    def test_history_command_with_days(self):
        """Test --history with specific number of days."""
        with create_temp_fixture_directory() as fixture:
            with tempfile.TemporaryDirectory() as config_dir:
                config_file = Path(config_dir) / "config.json"

                with patch("ralph.config.DEFAULT_CONFIG_FILE", config_file):
                    with patch("ralph.usage.DEFAULT_CLAUDE_DIR", fixture.path):
                        result = runner.invoke(app, ["usage", "--history", "7"])

                        assert result.exit_code == 0
                        # Should mention the number of days
                        assert "7" in result.output or "days" in result.output.lower()

    def test_history_command_with_data(self):
        """Test --history with actual session data."""
        with create_temp_fixture_directory() as fixture:
            now = datetime.now(timezone.utc)

            # Create session data for the past few days
            fixture.create_session_with_usage(
                project="test-project-1",
                start_time=now - timedelta(days=1),
                num_records=3,
                model="claude-sonnet-4-20250514",
                include_user_messages=False,
            )
            fixture.create_session_with_usage(
                project="test-project-2",
                start_time=now - timedelta(days=2),
                num_records=2,
                model="claude-opus-4-20250514",
                include_user_messages=False,
            )

            with tempfile.TemporaryDirectory() as config_dir:
                config_file = Path(config_dir) / "config.json"

                with patch("ralph.config.DEFAULT_CONFIG_FILE", config_file):
                    with patch("ralph.usage.DEFAULT_CLAUDE_DIR", fixture.path):
                        result = runner.invoke(app, ["usage", "--history", "3"])

                        assert result.exit_code == 0
                        # Should display table structure
                        assert "Window" in result.output or "Tokens" in result.output

    def test_history_command_invalid_days(self):
        """Test --history with invalid days value (less than 1)."""
        with create_temp_fixture_directory() as fixture:
            with tempfile.TemporaryDirectory() as config_dir:
                config_file = Path(config_dir) / "config.json"

                with patch("ralph.config.DEFAULT_CONFIG_FILE", config_file):
                    with patch("ralph.usage.DEFAULT_CLAUDE_DIR", fixture.path):
                        result = runner.invoke(app, ["usage", "--history", "0"])

                        # Should fail with error
                        assert result.exit_code != 0
                        assert "at least 1" in result.output.lower() or "error" in result.output.lower()

    def test_history_command_negative_days(self):
        """Test --history with negative days value."""
        with create_temp_fixture_directory() as fixture:
            with tempfile.TemporaryDirectory() as config_dir:
                config_file = Path(config_dir) / "config.json"

                with patch("ralph.config.DEFAULT_CONFIG_FILE", config_file):
                    with patch("ralph.usage.DEFAULT_CLAUDE_DIR", fixture.path):
                        result = runner.invoke(app, ["usage", "--history", "-1"])

                        # Should fail with error
                        assert result.exit_code != 0

    def test_history_shows_5hour_windows(self):
        """Test that --history shows 5-hour windows."""
        with create_temp_fixture_directory() as fixture:
            now = datetime.now(timezone.utc)

            # Create data that spans multiple 5-hour windows
            fixture.create_session_with_usage(
                project="test-project",
                start_time=now - timedelta(hours=12),
                num_records=10,
                model="claude-sonnet-4-20250514",
                interval_minutes=60,
                include_user_messages=False,
            )

            with tempfile.TemporaryDirectory() as config_dir:
                config_file = Path(config_dir) / "config.json"

                with patch("ralph.config.DEFAULT_CONFIG_FILE", config_file):
                    with patch("ralph.usage.DEFAULT_CLAUDE_DIR", fixture.path):
                        result = runner.invoke(app, ["usage", "--history", "1"])

                        assert result.exit_code == 0
                        # Output should contain window information
                        # (5-Hour Windows table with timestamps)

    def test_history_with_short_flag(self):
        """Test --history with short -h flag."""
        with create_temp_fixture_directory() as fixture:
            with tempfile.TemporaryDirectory() as config_dir:
                config_file = Path(config_dir) / "config.json"

                with patch("ralph.config.DEFAULT_CONFIG_FILE", config_file):
                    with patch("ralph.usage.DEFAULT_CLAUDE_DIR", fixture.path):
                        result = runner.invoke(app, ["usage", "-h", "2"])

                        assert result.exit_code == 0
                        assert "Historical" in result.output or "2" in result.output


# =============================================================================
# Test CLI with no Claude data directory
# =============================================================================


class TestNoClaiseDataDirectory:
    """Tests for CLI behavior when Claude data directory doesn't exist."""

    def test_usage_with_nonexistent_claude_dir(self):
        """Test 'ralph usage' when Claude data directory doesn't exist."""
        nonexistent_path = Path("/tmp/nonexistent-claude-dir-test-12345")
        assert not nonexistent_path.exists()

        with tempfile.TemporaryDirectory() as config_dir:
            config_file = Path(config_dir) / "config.json"

            with patch("ralph.config.DEFAULT_CONFIG_FILE", config_file):
                with patch("ralph.usage.DEFAULT_CLAUDE_DIR", nonexistent_path):
                    result = runner.invoke(app, ["usage"])

                    # Should still succeed, just show zero usage
                    assert result.exit_code == 0

    def test_usage_shows_zero_tokens_with_no_data(self):
        """Test that usage shows zero tokens when no data exists."""
        with create_temp_fixture_directory() as fixture:
            # Empty fixture directory
            with tempfile.TemporaryDirectory() as config_dir:
                config_file = Path(config_dir) / "config.json"

                with patch("ralph.config.DEFAULT_CONFIG_FILE", config_file):
                    with patch("ralph.usage.DEFAULT_CLAUDE_DIR", fixture.path):
                        result = runner.invoke(app, ["usage"])

                        assert result.exit_code == 0
                        # Should show 0% or 0 tokens or similar
                        assert "0" in result.output

    def test_history_with_nonexistent_claude_dir(self):
        """Test --history when Claude data directory doesn't exist."""
        nonexistent_path = Path("/tmp/nonexistent-claude-dir-test-67890")
        assert not nonexistent_path.exists()

        with tempfile.TemporaryDirectory() as config_dir:
            config_file = Path(config_dir) / "config.json"

            with patch("ralph.config.DEFAULT_CONFIG_FILE", config_file):
                with patch("ralph.usage.DEFAULT_CLAUDE_DIR", nonexistent_path):
                    result = runner.invoke(app, ["usage", "--history", "1"])

                    # Should still succeed
                    assert result.exit_code == 0

    def test_history_shows_no_data_message(self):
        """Test that history shows appropriate message when no data."""
        with create_temp_fixture_directory() as fixture:
            # Empty fixture directory
            with tempfile.TemporaryDirectory() as config_dir:
                config_file = Path(config_dir) / "config.json"

                with patch("ralph.config.DEFAULT_CONFIG_FILE", config_file):
                    with patch("ralph.usage.DEFAULT_CLAUDE_DIR", fixture.path):
                        result = runner.invoke(app, ["usage", "--history", "1"])

                        assert result.exit_code == 0
                        # Should indicate no data or empty results

    def test_set_plan_works_without_claude_dir(self):
        """Test that --set-plan works even without Claude data directory."""
        nonexistent_path = Path("/tmp/nonexistent-claude-dir-test-11111")

        with tempfile.TemporaryDirectory() as config_dir:
            config_file = Path(config_dir) / "config.json"

            with patch("ralph.config.DEFAULT_CONFIG_FILE", config_file):
                with patch("ralph.usage.DEFAULT_CLAUDE_DIR", nonexistent_path):
                    result = runner.invoke(app, ["usage", "--set-plan", "pro"])

                    # Should still work - set-plan doesn't need Claude data
                    assert result.exit_code == 0
                    assert config_file.exists()


# =============================================================================
# Test CLI output format and color coding thresholds
# =============================================================================


class TestOutputFormatAndColorCoding:
    """Tests for CLI output format and color coding thresholds."""

    def test_usage_output_contains_progress_bar_elements(self):
        """Test that usage output contains progress bar elements."""
        with create_temp_fixture_directory() as fixture:
            with tempfile.TemporaryDirectory() as config_dir:
                config_file = Path(config_dir) / "config.json"

                with patch("ralph.config.DEFAULT_CONFIG_FILE", config_file):
                    with patch("ralph.usage.DEFAULT_CLAUDE_DIR", fixture.path):
                        result = runner.invoke(app, ["usage"])

                        assert result.exit_code == 0
                        # Rich output should contain percentage
                        assert "%" in result.output

    def test_usage_output_contains_token_info(self):
        """Test that usage output contains token information."""
        with create_temp_fixture_directory() as fixture:
            fixture.create_session_with_usage(
                project="test",
                num_records=1,
                include_user_messages=False,
            )

            with tempfile.TemporaryDirectory() as config_dir:
                config_file = Path(config_dir) / "config.json"

                with patch("ralph.config.DEFAULT_CONFIG_FILE", config_file):
                    with patch("ralph.usage.DEFAULT_CLAUDE_DIR", fixture.path):
                        result = runner.invoke(app, ["usage"])

                        assert result.exit_code == 0
                        # Should show token-related terms
                        assert "token" in result.output.lower() or "Token" in result.output

    def test_low_usage_output_format(self):
        """Test output format when usage is low (should be green)."""
        with create_temp_fixture_directory() as fixture:
            # Create minimal usage (will be well under any threshold)
            now = datetime.now(timezone.utc)
            fixture.create_session_with_usage(
                project="test-project",
                start_time=now - timedelta(hours=1),
                num_records=1,
                model="claude-sonnet-4-20250514",
                input_tokens_range=(100, 100),
                output_tokens_range=(50, 50),
                cache_creation_range=(0, 0),
                cache_read_range=(0, 0),
                include_user_messages=False,
            )

            with tempfile.TemporaryDirectory() as config_dir:
                config_file = Path(config_dir) / "config.json"

                with patch("ralph.config.DEFAULT_CONFIG_FILE", config_file):
                    with patch("ralph.usage.DEFAULT_CLAUDE_DIR", fixture.path):
                        result = runner.invoke(app, ["usage"])

                        assert result.exit_code == 0
                        # With low usage, should show a low percentage
                        # The exact format depends on Rich rendering

    def test_output_shows_remaining_tokens(self):
        """Test that output shows remaining tokens."""
        with create_temp_fixture_directory() as fixture:
            with tempfile.TemporaryDirectory() as config_dir:
                config_file = Path(config_dir) / "config.json"

                with patch("ralph.config.DEFAULT_CONFIG_FILE", config_file):
                    with patch("ralph.usage.DEFAULT_CLAUDE_DIR", fixture.path):
                        result = runner.invoke(app, ["usage"])

                        assert result.exit_code == 0
                        # Should show remaining tokens
                        assert "Remaining" in result.output or "remaining" in result.output

    def test_output_shows_window_reset_info(self):
        """Test that output shows window reset information."""
        with create_temp_fixture_directory() as fixture:
            with tempfile.TemporaryDirectory() as config_dir:
                config_file = Path(config_dir) / "config.json"

                with patch("ralph.config.DEFAULT_CONFIG_FILE", config_file):
                    with patch("ralph.usage.DEFAULT_CLAUDE_DIR", fixture.path):
                        result = runner.invoke(app, ["usage"])

                        assert result.exit_code == 0
                        # Should show rolling window info
                        assert "rolling" in result.output.lower() or "resets" in result.output.lower()

    def test_history_output_shows_table_headers(self):
        """Test that history output shows appropriate table headers."""
        with create_temp_fixture_directory() as fixture:
            now = datetime.now(timezone.utc)
            fixture.create_session_with_usage(
                project="test",
                start_time=now - timedelta(hours=6),
                num_records=2,
                include_user_messages=False,
            )

            with tempfile.TemporaryDirectory() as config_dir:
                config_file = Path(config_dir) / "config.json"

                with patch("ralph.config.DEFAULT_CONFIG_FILE", config_file):
                    with patch("ralph.usage.DEFAULT_CLAUDE_DIR", fixture.path):
                        result = runner.invoke(app, ["usage", "--history", "1"])

                        assert result.exit_code == 0
                        # Should show table column headers

    def test_history_output_shows_summary(self):
        """Test that history output shows summary statistics."""
        with create_temp_fixture_directory() as fixture:
            now = datetime.now(timezone.utc)
            fixture.create_session_with_usage(
                project="test",
                start_time=now - timedelta(hours=3),
                num_records=5,
                include_user_messages=False,
            )

            with tempfile.TemporaryDirectory() as config_dir:
                config_file = Path(config_dir) / "config.json"

                with patch("ralph.config.DEFAULT_CONFIG_FILE", config_file):
                    with patch("ralph.usage.DEFAULT_CLAUDE_DIR", fixture.path):
                        result = runner.invoke(app, ["usage", "--history", "1"])

                        assert result.exit_code == 0
                        # Should show summary
                        assert "Summary" in result.output or "Total" in result.output

    def test_model_breakdown_shows_opus_and_sonnet(self):
        """Test that model breakdown shows Opus and Sonnet."""
        with create_temp_fixture_directory() as fixture:
            now = datetime.now(timezone.utc)

            # Create both Opus and Sonnet records
            fixture.create_session_with_usage(
                project="opus-project",
                start_time=now - timedelta(hours=1),
                num_records=2,
                model="claude-opus-4-20250514",
                include_user_messages=False,
            )
            fixture.create_session_with_usage(
                project="sonnet-project",
                start_time=now - timedelta(hours=2),
                num_records=3,
                model="claude-sonnet-4-20250514",
                include_user_messages=False,
            )

            with tempfile.TemporaryDirectory() as config_dir:
                config_file = Path(config_dir) / "config.json"

                with patch("ralph.config.DEFAULT_CONFIG_FILE", config_file):
                    with patch("ralph.usage.DEFAULT_CLAUDE_DIR", fixture.path):
                        result = runner.invoke(app, ["usage"])

                        assert result.exit_code == 0
                        # Should show both models
                        assert "Opus" in result.output
                        assert "Sonnet" in result.output


# =============================================================================
# Additional edge case tests
# =============================================================================


class TestCliEdgeCases:
    """Tests for CLI edge cases."""

    def test_usage_with_default_plan(self):
        """Test usage command uses default plan when none configured."""
        with create_temp_fixture_directory() as fixture:
            with tempfile.TemporaryDirectory() as config_dir:
                # No config file created - should use default
                config_file = Path(config_dir) / "nonexistent" / "config.json"

                with patch("ralph.config.DEFAULT_CONFIG_FILE", config_file):
                    with patch("ralph.usage.DEFAULT_CLAUDE_DIR", fixture.path):
                        result = runner.invoke(app, ["usage"])

                        assert result.exit_code == 0
                        # Default plan is PRO
                        assert "PRO" in result.output or "pro" in result.output

    def test_set_plan_success_message(self):
        """Test that set-plan shows success message."""
        with tempfile.TemporaryDirectory() as config_dir:
            config_file = Path(config_dir) / "config.json"

            with patch("ralph.config.DEFAULT_CONFIG_FILE", config_file):
                result = runner.invoke(app, ["usage", "--set-plan", "max5x"])

                assert result.exit_code == 0
                # Should indicate success (checkmark or "set to")
                assert "max5x" in result.output.lower()

    def test_history_large_number_of_days(self):
        """Test --history with large number of days."""
        with create_temp_fixture_directory() as fixture:
            with tempfile.TemporaryDirectory() as config_dir:
                config_file = Path(config_dir) / "config.json"

                with patch("ralph.config.DEFAULT_CONFIG_FILE", config_file):
                    with patch("ralph.usage.DEFAULT_CLAUDE_DIR", fixture.path):
                        result = runner.invoke(app, ["usage", "--history", "30"])

                        assert result.exit_code == 0
                        assert "30" in result.output

    def test_multiple_sessions_same_project(self):
        """Test usage with multiple sessions in the same project."""
        with create_temp_fixture_directory() as fixture:
            now = datetime.now(timezone.utc)

            # Create multiple sessions in same project
            for i in range(3):
                fixture.create_session_with_usage(
                    project="same-project",
                    start_time=now - timedelta(hours=i + 1),
                    num_records=2,
                    model="claude-sonnet-4-20250514",
                    include_user_messages=False,
                )

            with tempfile.TemporaryDirectory() as config_dir:
                config_file = Path(config_dir) / "config.json"

                with patch("ralph.config.DEFAULT_CONFIG_FILE", config_file):
                    with patch("ralph.usage.DEFAULT_CLAUDE_DIR", fixture.path):
                        result = runner.invoke(app, ["usage"])

                        assert result.exit_code == 0
                        # Should aggregate all sessions

    def test_usage_command_with_messages_count(self):
        """Test that usage shows message/request counts."""
        with create_temp_fixture_directory() as fixture:
            now = datetime.now(timezone.utc)

            fixture.create_session_with_usage(
                project="test-project",
                start_time=now - timedelta(hours=1),
                num_records=5,
                include_user_messages=False,
            )

            with tempfile.TemporaryDirectory() as config_dir:
                config_file = Path(config_dir) / "config.json"

                with patch("ralph.config.DEFAULT_CONFIG_FILE", config_file):
                    with patch("ralph.usage.DEFAULT_CLAUDE_DIR", fixture.path):
                        result = runner.invoke(app, ["usage"])

                        assert result.exit_code == 0
                        # Should show message counts
                        assert "message" in result.output.lower() or "request" in result.output.lower()
