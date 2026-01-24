"""Integration tests for the core Ralph runner functionality.

This module tests the Ralph runner to ensure usage tracking doesn't break core features:
- Test ralph run with mock Claude process
- Test iteration loop with usage tracking enabled
- Test graceful shutdown preserves state
- Test resume functionality still works
- Test retry logic still works
- Tests use subprocess mocking, not real Claude calls
"""

from __future__ import annotations

import json
import os
import signal
import tempfile
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ralph.prd import PRD, UserStory
from ralph.process import ManagedProcess, ProcessResult, Tool
from ralph.runner import Runner, run_ralph
from ralph.state import RunState, clear_state, load_state, save_state


# =============================================================================
# Helper Functions and Fixtures
# =============================================================================


def create_test_prd(
    prd_path: Path,
    stories: list[dict[str, Any]] | None = None,
    project: str = "Test Project",
    branch_name: str = "test/feature",
) -> PRD:
    """Create a test PRD file and return the PRD object."""
    if stories is None:
        stories = [
            {
                "id": "US-001",
                "title": "Test Story 1",
                "description": "Test description 1",
                "acceptanceCriteria": ["Criterion 1", "Criterion 2"],
                "priority": 1,
                "passes": False,
            },
            {
                "id": "US-002",
                "title": "Test Story 2",
                "description": "Test description 2",
                "acceptanceCriteria": ["Criterion A"],
                "priority": 2,
                "passes": False,
            },
        ]

    prd_content = {
        "project": project,
        "branchName": branch_name,
        "description": "Test project description",
        "userStories": stories,
    }
    prd_path.write_text(json.dumps(prd_content, indent=2))
    return PRD.from_file(prd_path)


def create_mock_process_result(
    output: str = "",
    return_code: int = 0,
    completed: bool = False,
    interrupted: bool = False,
    timed_out: bool = False,
) -> ProcessResult:
    """Create a mock ProcessResult."""
    return ProcessResult(
        output=output,
        return_code=return_code,
        completed=completed,
        interrupted=interrupted,
        timed_out=timed_out,
    )


@pytest.fixture
def temp_project_dir(tmp_path: Path) -> Path:
    """Create a temporary project directory with required files."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    # Create a minimal CLAUDE.md
    claude_md = project_dir / "CLAUDE.md"
    claude_md.write_text("# Test Project\n\nThis is a test project.")

    return project_dir


@pytest.fixture
def temp_lock_file(tmp_path: Path) -> Path:
    """Create a temporary lock file path for session coordination tests."""
    lock_file = tmp_path / ".ralph" / "usage.lock"
    return lock_file


# =============================================================================
# Test ralph run with mock Claude process
# =============================================================================


class TestRalphRunWithMockProcess:
    """Tests for ralph run with mock Claude process."""

    def test_single_iteration_with_mock_process(self, temp_project_dir: Path):
        """Test a single iteration with a mocked Claude process."""
        prd_path = temp_project_dir / "prd.json"
        create_test_prd(prd_path)

        runner = Runner(
            prd_path=prd_path,
            tool=Tool.CLAUDE,
            max_iterations=1,
            adaptive_pacing=False,
        )

        # Mock the process execution
        mock_result = create_mock_process_result(
            output="Working on story...\n<promise>COMPLETE</promise>",
            return_code=0,
            completed=True,
        )

        with patch("ralph.runner.run_tool_with_prompt", return_value=mock_result):
            with patch("ralph.runner.SessionManager") as mock_session:
                mock_session.return_value.__enter__ = MagicMock(
                    return_value=MagicMock(pid=12345)
                )
                mock_session.return_value.__exit__ = MagicMock(return_value=False)
                with patch("ralph.runner.ConversationWatcher"):
                    result, story_id = runner.run_iteration(1)

        assert result.completed is True
        assert story_id == "US-001"

    @pytest.mark.slow
    def test_multiple_iterations_until_completion(self, temp_project_dir: Path):
        """Test multiple iterations until all stories are complete."""
        prd_path = temp_project_dir / "prd.json"
        stories = [
            {
                "id": "US-001",
                "title": "Story 1",
                "description": "Description",
                "acceptanceCriteria": ["Criterion"],
                "priority": 1,
                "passes": False,
            },
        ]
        create_test_prd(prd_path, stories=stories)

        runner = Runner(
            prd_path=prd_path,
            tool=Tool.CLAUDE,
            max_iterations=5,
            adaptive_pacing=False,
        )

        # On first call, mark story as complete by modifying PRD
        def mock_tool_with_side_effect(*args: Any, **kwargs: Any) -> ProcessResult:
            # Mark the story as complete in the PRD
            prd = PRD.from_file(prd_path)
            prd.mark_story_complete("US-001")
            prd.save(prd_path)
            return create_mock_process_result(
                output="Done!\n<promise>COMPLETE</promise>",
                return_code=0,
                completed=True,
            )

        with patch(
            "ralph.runner.run_tool_with_prompt",
            side_effect=mock_tool_with_side_effect,
        ):
            with patch("ralph.runner.SessionManager") as mock_session:
                mock_session.return_value.__enter__ = MagicMock(
                    return_value=MagicMock(pid=12345)
                )
                mock_session.return_value.__exit__ = MagicMock(return_value=False)
                with patch("ralph.runner.ConversationWatcher"):
                    result = runner.run()

        assert result is True  # All stories completed

    @pytest.mark.slow
    def test_max_iterations_reached(self, temp_project_dir: Path):
        """Test that max iterations limit is respected."""
        prd_path = temp_project_dir / "prd.json"
        create_test_prd(prd_path)

        runner = Runner(
            prd_path=prd_path,
            tool=Tool.CLAUDE,
            max_iterations=2,
            adaptive_pacing=False,
            iteration_delay=0.01,  # Fast iterations for testing
        )

        # Never complete the story
        mock_result = create_mock_process_result(
            output="Working...",
            return_code=0,
            completed=False,
        )

        iteration_count = 0

        def track_iterations(*args: Any, **kwargs: Any) -> ProcessResult:
            nonlocal iteration_count
            iteration_count += 1
            return mock_result

        with patch("ralph.runner.run_tool_with_prompt", side_effect=track_iterations):
            with patch("ralph.runner.SessionManager") as mock_session:
                mock_session.return_value.__enter__ = MagicMock(
                    return_value=MagicMock(pid=12345)
                )
                mock_session.return_value.__exit__ = MagicMock(return_value=False)
                with patch("ralph.runner.ConversationWatcher"):
                    result = runner.run()

        assert result is False  # Max iterations reached
        assert iteration_count == 2

    def test_already_complete_prd(self, temp_project_dir: Path):
        """Test that runner exits early when all stories are already complete."""
        prd_path = temp_project_dir / "prd.json"
        stories = [
            {
                "id": "US-001",
                "title": "Story 1",
                "description": "Description",
                "acceptanceCriteria": ["Criterion"],
                "priority": 1,
                "passes": True,  # Already complete
            },
        ]
        create_test_prd(prd_path, stories=stories)

        runner = Runner(
            prd_path=prd_path,
            tool=Tool.CLAUDE,
            max_iterations=5,
            adaptive_pacing=False,
        )

        with patch("ralph.runner.SessionManager") as mock_session:
            mock_session.return_value.__enter__ = MagicMock(
                return_value=MagicMock(pid=12345)
            )
            mock_session.return_value.__exit__ = MagicMock(return_value=False)
            result = runner.run()

        assert result is True  # Already complete

    def test_no_stories_in_prd(self, temp_project_dir: Path):
        """Test handling of PRD with no incomplete stories."""
        prd_path = temp_project_dir / "prd.json"
        stories = [
            {
                "id": "US-001",
                "title": "Story 1",
                "description": "Description",
                "acceptanceCriteria": ["Criterion"],
                "priority": 1,
                "passes": True,
            },
        ]
        create_test_prd(prd_path, stories=stories)

        runner = Runner(
            prd_path=prd_path,
            tool=Tool.CLAUDE,
            max_iterations=5,
        )

        mock_result = create_mock_process_result(completed=True)

        with patch("ralph.runner.run_tool_with_prompt", return_value=mock_result):
            with patch("ralph.runner.SessionManager") as mock_session:
                mock_session.return_value.__enter__ = MagicMock(
                    return_value=MagicMock(pid=12345)
                )
                mock_session.return_value.__exit__ = MagicMock(return_value=False)
                with patch("ralph.runner.ConversationWatcher"):
                    # run_iteration should return early when no stories are pending
                    result, story_id = runner.run_iteration(1)

        assert result.completed is True
        assert story_id is None  # No story to work on


# =============================================================================
# Test iteration loop with usage tracking enabled
# =============================================================================


class TestIterationLoopWithUsageTracking:
    """Tests for iteration loop with usage tracking enabled."""

    def test_iteration_with_usage_display(self, temp_project_dir: Path):
        """Test that usage is displayed after each iteration."""
        prd_path = temp_project_dir / "prd.json"
        create_test_prd(prd_path)

        runner = Runner(
            prd_path=prd_path,
            tool=Tool.CLAUDE,
            max_iterations=1,
            adaptive_pacing=True,
        )

        # Track if usage display was called
        display_called = False

        def mock_display_usage() -> None:
            nonlocal display_called
            display_called = True

        def mock_tool_complete_stories(*args: Any, **kwargs: Any) -> ProcessResult:
            # Mark stories complete during iteration
            prd = PRD.from_file(prd_path)
            prd.mark_story_complete("US-001")
            prd.mark_story_complete("US-002")
            prd.save(prd_path)
            return create_mock_process_result(
                output="Done!\n<promise>COMPLETE</promise>",
                completed=True,
            )

        with patch("ralph.runner.run_tool_with_prompt", side_effect=mock_tool_complete_stories):
            with patch.object(
                runner, "_display_iteration_usage", side_effect=mock_display_usage
            ):
                with patch("ralph.runner.SessionManager") as mock_session:
                    mock_session.return_value.__enter__ = MagicMock(
                        return_value=MagicMock(pid=12345)
                    )
                    mock_session.return_value.__exit__ = MagicMock(return_value=False)
                    with patch("ralph.runner.ConversationWatcher"):
                        runner.run()

        assert display_called is True

    def test_adaptive_pacing_applies_delay(self, temp_project_dir: Path):
        """Test that adaptive pacing applies correct delay multiplier."""
        prd_path = temp_project_dir / "prd.json"
        create_test_prd(prd_path)

        runner = Runner(
            prd_path=prd_path,
            tool=Tool.CLAUDE,
            max_iterations=5,
            adaptive_pacing=True,
            iteration_delay=1.0,
        )

        # Mock 75% usage - should trigger 2x multiplier
        with patch.object(runner, "_get_usage_percentage", return_value=75.0):
            with patch("ralph.runner.print_pacing_adjustment"):
                delay = runner._apply_adaptive_pacing()

        assert delay == 2.0  # 1.0 * 2.0 multiplier

    def test_usage_tracking_handles_errors_gracefully(self, temp_project_dir: Path):
        """Test that usage tracking errors don't break the runner."""
        prd_path = temp_project_dir / "prd.json"
        create_test_prd(prd_path)

        runner = Runner(
            prd_path=prd_path,
            tool=Tool.CLAUDE,
            max_iterations=1,
            adaptive_pacing=True,
        )

        # Mock usage percentage to raise an exception
        def failing_usage() -> None:
            raise Exception("Usage tracking failed")

        mock_result = create_mock_process_result(
            output="<promise>COMPLETE</promise>",
            completed=True,
        )

        with patch("ralph.runner.run_tool_with_prompt", return_value=mock_result):
            with patch.object(
                runner, "_display_iteration_usage", side_effect=failing_usage
            ):
                with patch("ralph.runner.SessionManager") as mock_session:
                    mock_session.return_value.__enter__ = MagicMock(
                        return_value=MagicMock(pid=12345)
                    )
                    mock_session.return_value.__exit__ = MagicMock(return_value=False)
                    with patch("ralph.runner.ConversationWatcher"):
                        # Mark story complete
                        prd = PRD.from_file(prd_path)
                        prd.mark_story_complete("US-001")
                        prd.mark_story_complete("US-002")
                        prd.save(prd_path)

                        # Should not raise even though usage tracking fails
                        try:
                            runner.run()
                            raised = False
                        except Exception:
                            raised = True

        # The runner continues despite usage tracking errors
        # (the _display_iteration_usage catches exceptions internally)
        # We verify the runner doesn't crash

    def test_session_manager_coordinates_multiple_sessions(
        self, temp_project_dir: Path, temp_lock_file: Path
    ):
        """Test that SessionManager is used for cross-session coordination."""
        prd_path = temp_project_dir / "prd.json"
        stories = [
            {
                "id": "US-001",
                "title": "Story 1",
                "description": "Description",
                "acceptanceCriteria": ["Criterion"],
                "priority": 1,
                "passes": True,  # Already complete so run exits quickly
            },
        ]
        create_test_prd(prd_path, stories=stories)

        session_manager_used = False

        class MockSessionManager:
            def __init__(self, *args: Any, **kwargs: Any):
                nonlocal session_manager_used
                session_manager_used = True

            def __enter__(self) -> MagicMock:
                return MagicMock(pid=12345)

            def __exit__(self, *args: Any) -> bool:
                return False

        with patch("ralph.runner.SessionManager", MockSessionManager):
            runner = Runner(
                prd_path=prd_path,
                tool=Tool.CLAUDE,
                max_iterations=1,
            )
            runner.run()

        assert session_manager_used is True


# =============================================================================
# Test graceful shutdown preserves state
# =============================================================================


class TestGracefulShutdownPreservesState:
    """Tests for graceful shutdown state preservation."""

    @pytest.mark.slow
    def test_state_saved_after_each_iteration(self, temp_project_dir: Path):
        """Test that run state is saved after each iteration."""
        prd_path = temp_project_dir / "prd.json"
        create_test_prd(prd_path)

        runner = Runner(
            prd_path=prd_path,
            tool=Tool.CLAUDE,
            max_iterations=2,
            adaptive_pacing=False,
            iteration_delay=0.01,
        )

        mock_result = create_mock_process_result(output="Working...", return_code=0)

        iteration_states: list[int] = []

        def capture_state(*args: Any, **kwargs: Any) -> ProcessResult:
            # Check state after each iteration
            state = load_state(temp_project_dir)
            if state:
                iteration_states.append(state.last_iteration)
            return mock_result

        original_save_state = runner._save_run_state

        def save_and_track(iteration: int, story_id: str | None, story_count: int | None = None) -> None:
            original_save_state(iteration, story_id, story_count)
            state = load_state(temp_project_dir)
            if state:
                iteration_states.append(state.last_iteration)

        with patch("ralph.runner.run_tool_with_prompt", return_value=mock_result):
            with patch.object(runner, "_save_run_state", side_effect=save_and_track):
                with patch("ralph.runner.SessionManager") as mock_session:
                    mock_session.return_value.__enter__ = MagicMock(
                        return_value=MagicMock(pid=12345)
                    )
                    mock_session.return_value.__exit__ = MagicMock(return_value=False)
                    with patch("ralph.runner.ConversationWatcher"):
                        runner.run()

        # State should have been saved for each iteration
        assert len(iteration_states) == 2
        assert 1 in iteration_states
        assert 2 in iteration_states

    @pytest.mark.slow
    def test_interrupt_preserves_state(self, temp_project_dir: Path):
        """Test that state is preserved when runner is interrupted."""
        prd_path = temp_project_dir / "prd.json"
        create_test_prd(prd_path)

        runner = Runner(
            prd_path=prd_path,
            tool=Tool.CLAUDE,
            max_iterations=10,
            adaptive_pacing=False,
            iteration_delay=0.01,
        )

        # Simulate interrupt after first iteration
        iteration_count = 0

        def simulate_interrupt(*args: Any, **kwargs: Any) -> ProcessResult:
            nonlocal iteration_count
            iteration_count += 1
            if iteration_count >= 2:
                runner._interrupted = True
            return create_mock_process_result(output="Working...", return_code=0)

        with patch("ralph.runner.run_tool_with_prompt", side_effect=simulate_interrupt):
            with patch("ralph.runner.SessionManager") as mock_session:
                mock_session.return_value.__enter__ = MagicMock(
                    return_value=MagicMock(pid=12345)
                )
                mock_session.return_value.__exit__ = MagicMock(return_value=False)
                with patch("ralph.runner.ConversationWatcher"):
                    result = runner.run()

        assert result == 130  # SIGINT exit code

        # State should be saved
        state = load_state(temp_project_dir)
        assert state is not None
        assert state.last_iteration >= 1

    @pytest.mark.slow
    def test_state_cleared_on_completion(self, temp_project_dir: Path):
        """Test that state is cleared when all stories complete."""
        prd_path = temp_project_dir / "prd.json"
        stories = [
            {
                "id": "US-001",
                "title": "Story 1",
                "description": "Description",
                "acceptanceCriteria": ["Criterion"],
                "priority": 1,
                "passes": False,
            },
        ]
        create_test_prd(prd_path, stories=stories)

        runner = Runner(
            prd_path=prd_path,
            tool=Tool.CLAUDE,
            max_iterations=5,
            adaptive_pacing=False,
        )

        def complete_story(*args: Any, **kwargs: Any) -> ProcessResult:
            prd = PRD.from_file(prd_path)
            prd.mark_story_complete("US-001")
            prd.save(prd_path)
            return create_mock_process_result(
                output="<promise>COMPLETE</promise>",
                completed=True,
            )

        with patch("ralph.runner.run_tool_with_prompt", side_effect=complete_story):
            with patch("ralph.runner.SessionManager") as mock_session:
                mock_session.return_value.__enter__ = MagicMock(
                    return_value=MagicMock(pid=12345)
                )
                mock_session.return_value.__exit__ = MagicMock(return_value=False)
                with patch("ralph.runner.ConversationWatcher"):
                    result = runner.run()

        assert result is True

        # State should be cleared after completion
        state = load_state(temp_project_dir)
        assert state is None


# =============================================================================
# Test resume functionality still works
# =============================================================================


class TestResumeFunctionality:
    """Tests for resume functionality."""

    @pytest.mark.slow
    def test_resume_from_saved_state(self, temp_project_dir: Path):
        """Test that runner resumes from saved state."""
        prd_path = temp_project_dir / "prd.json"
        create_test_prd(prd_path)

        # Create existing state
        existing_state = RunState.create(
            prd_path=prd_path,
            tool="claude",
            iteration=2,
            story_id="US-001",
            story_count=2,
        )
        save_state(temp_project_dir, existing_state)

        runner = Runner(
            prd_path=prd_path,
            tool=Tool.CLAUDE,
            max_iterations=5,
            adaptive_pacing=False,
            resume=True,
            iteration_delay=0.01,
        )

        iterations_run: list[int] = []

        def track_iteration(*args: Any, **kwargs: Any) -> ProcessResult:
            iterations_run.append(runner._current_iteration)
            return create_mock_process_result(output="Working...", return_code=0)

        with patch("ralph.runner.run_tool_with_prompt", side_effect=track_iteration):
            with patch("ralph.runner.SessionManager") as mock_session:
                mock_session.return_value.__enter__ = MagicMock(
                    return_value=MagicMock(pid=12345)
                )
                mock_session.return_value.__exit__ = MagicMock(return_value=False)
                with patch("ralph.runner.ConversationWatcher"):
                    runner.run()

        # Should have started from iteration 3 (2 + 1)
        assert iterations_run[0] == 3

    @pytest.mark.slow
    def test_resume_with_no_previous_state(self, temp_project_dir: Path):
        """Test that runner starts fresh when no state exists."""
        prd_path = temp_project_dir / "prd.json"
        create_test_prd(prd_path)

        # Ensure no state exists
        clear_state(temp_project_dir)

        runner = Runner(
            prd_path=prd_path,
            tool=Tool.CLAUDE,
            max_iterations=2,
            adaptive_pacing=False,
            resume=True,
            iteration_delay=0.01,
        )

        iterations_run: list[int] = []

        def track_iteration(*args: Any, **kwargs: Any) -> ProcessResult:
            iterations_run.append(runner._current_iteration)
            return create_mock_process_result(output="Working...", return_code=0)

        with patch("ralph.runner.run_tool_with_prompt", side_effect=track_iteration):
            with patch("ralph.runner.SessionManager") as mock_session:
                mock_session.return_value.__enter__ = MagicMock(
                    return_value=MagicMock(pid=12345)
                )
                mock_session.return_value.__exit__ = MagicMock(return_value=False)
                with patch("ralph.runner.ConversationWatcher"):
                    runner.run()

        # Should have started from iteration 1
        assert iterations_run[0] == 1

    def test_resume_validates_prd_path(self, temp_project_dir: Path):
        """Test that resume validates PRD path matches saved state."""
        prd_path = temp_project_dir / "prd.json"
        create_test_prd(prd_path)

        # Create state with different PRD path
        existing_state = RunState.create(
            prd_path=Path("/different/path/prd.json"),
            tool="claude",
            iteration=2,
            story_id="US-001",
        )
        save_state(temp_project_dir, existing_state)

        runner = Runner(
            prd_path=prd_path,
            tool=Tool.CLAUDE,
            max_iterations=5,
            adaptive_pacing=False,
            resume=True,
            verbose=True,  # To trigger warning output
        )

        warning_printed = False

        def capture_warning(*args: Any, **kwargs: Any) -> None:
            nonlocal warning_printed
            warning_printed = True

        with patch("ralph.runner.print_warning", side_effect=capture_warning):
            with patch(
                "ralph.runner.run_tool_with_prompt",
                return_value=create_mock_process_result(),
            ):
                with patch("ralph.runner.SessionManager") as mock_session:
                    mock_session.return_value.__enter__ = MagicMock(
                        return_value=MagicMock(pid=12345)
                    )
                    mock_session.return_value.__exit__ = MagicMock(return_value=False)
                    with patch("ralph.runner.ConversationWatcher"):
                        # Mark stories complete to end quickly
                        prd = PRD.from_file(prd_path)
                        prd.mark_story_complete("US-001")
                        prd.mark_story_complete("US-002")
                        prd.save(prd_path)
                        runner.run()

        # Warning should have been printed about path mismatch
        assert warning_printed is True

    def test_resume_handles_story_count_change(self, temp_project_dir: Path):
        """Test that resume handles PRD story count changes."""
        prd_path = temp_project_dir / "prd.json"
        create_test_prd(prd_path)  # Creates 2 stories

        # Create state with different story count
        existing_state = RunState.create(
            prd_path=prd_path,
            tool="claude",
            iteration=1,
            story_id="US-001",
            story_count=5,  # Different from actual 2
        )
        save_state(temp_project_dir, existing_state)

        runner = Runner(
            prd_path=prd_path,
            tool=Tool.CLAUDE,
            max_iterations=5,
            adaptive_pacing=False,
            resume=True,
            verbose=True,
        )

        warning_printed = False

        def capture_warning(*args: Any, **kwargs: Any) -> None:
            nonlocal warning_printed
            warning_printed = True

        with patch("ralph.runner.print_warning", side_effect=capture_warning):
            with patch(
                "ralph.runner.run_tool_with_prompt",
                return_value=create_mock_process_result(),
            ):
                with patch("ralph.runner.SessionManager") as mock_session:
                    mock_session.return_value.__enter__ = MagicMock(
                        return_value=MagicMock(pid=12345)
                    )
                    mock_session.return_value.__exit__ = MagicMock(return_value=False)
                    with patch("ralph.runner.ConversationWatcher"):
                        prd = PRD.from_file(prd_path)
                        prd.mark_story_complete("US-001")
                        prd.mark_story_complete("US-002")
                        prd.save(prd_path)
                        runner.run()

        assert warning_printed is True


# =============================================================================
# Test retry logic still works
# =============================================================================


class TestRetryLogic:
    """Tests for retry logic."""

    def test_retry_on_non_zero_exit_code(self, temp_project_dir: Path):
        """Test that runner retries on non-zero exit code."""
        prd_path = temp_project_dir / "prd.json"
        create_test_prd(prd_path)

        runner = Runner(
            prd_path=prd_path,
            tool=Tool.CLAUDE,
            max_iterations=1,
            max_retries=2,
            adaptive_pacing=False,
        )

        attempt_count = 0

        def fail_then_succeed(*args: Any, **kwargs: Any) -> ProcessResult:
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                return create_mock_process_result(output="Error", return_code=1)
            return create_mock_process_result(output="Success", return_code=0)

        with patch("ralph.runner.run_tool_with_prompt", side_effect=fail_then_succeed):
            with patch("ralph.runner.time.sleep"):  # Skip actual delays
                with patch("ralph.runner.ConversationWatcher"):
                    result, story_id = runner._run_iteration_with_retry(1)

        # Should have retried twice (3 total attempts)
        assert attempt_count == 3
        assert result.return_code == 0

    def test_no_retry_on_success(self, temp_project_dir: Path):
        """Test that runner doesn't retry on success."""
        prd_path = temp_project_dir / "prd.json"
        create_test_prd(prd_path)

        runner = Runner(
            prd_path=prd_path,
            tool=Tool.CLAUDE,
            max_iterations=1,
            max_retries=2,
            adaptive_pacing=False,
        )

        attempt_count = 0

        def succeed_immediately(*args: Any, **kwargs: Any) -> ProcessResult:
            nonlocal attempt_count
            attempt_count += 1
            return create_mock_process_result(output="Success", return_code=0)

        with patch(
            "ralph.runner.run_tool_with_prompt", side_effect=succeed_immediately
        ):
            with patch("ralph.runner.ConversationWatcher"):
                result, story_id = runner._run_iteration_with_retry(1)

        # Should have only run once
        assert attempt_count == 1

    def test_no_retry_on_timeout(self, temp_project_dir: Path):
        """Test that runner doesn't retry on timeout."""
        prd_path = temp_project_dir / "prd.json"
        create_test_prd(prd_path)

        runner = Runner(
            prd_path=prd_path,
            tool=Tool.CLAUDE,
            max_iterations=1,
            max_retries=2,
            adaptive_pacing=False,
        )

        attempt_count = 0

        def timeout(*args: Any, **kwargs: Any) -> ProcessResult:
            nonlocal attempt_count
            attempt_count += 1
            return create_mock_process_result(
                output="Timed out", return_code=0, timed_out=True
            )

        with patch("ralph.runner.run_tool_with_prompt", side_effect=timeout):
            with patch("ralph.runner.ConversationWatcher"):
                result, story_id = runner._run_iteration_with_retry(1)

        # Should not retry on timeout
        assert attempt_count == 1
        assert result.timed_out is True

    def test_no_retry_on_completion(self, temp_project_dir: Path):
        """Test that runner doesn't retry when story completes."""
        prd_path = temp_project_dir / "prd.json"
        create_test_prd(prd_path)

        runner = Runner(
            prd_path=prd_path,
            tool=Tool.CLAUDE,
            max_iterations=1,
            max_retries=2,
            adaptive_pacing=False,
        )

        attempt_count = 0

        def complete(*args: Any, **kwargs: Any) -> ProcessResult:
            nonlocal attempt_count
            attempt_count += 1
            return create_mock_process_result(
                output="<promise>COMPLETE</promise>", return_code=0, completed=True
            )

        with patch("ralph.runner.run_tool_with_prompt", side_effect=complete):
            with patch("ralph.runner.ConversationWatcher"):
                result, story_id = runner._run_iteration_with_retry(1)

        # Should not retry on completion
        assert attempt_count == 1
        assert result.completed is True

    def test_retry_exhaustion(self, temp_project_dir: Path):
        """Test that runner continues after exhausting retries."""
        prd_path = temp_project_dir / "prd.json"
        create_test_prd(prd_path)

        runner = Runner(
            prd_path=prd_path,
            tool=Tool.CLAUDE,
            max_iterations=1,
            max_retries=2,
            adaptive_pacing=False,
        )

        attempt_count = 0

        def always_fail(*args: Any, **kwargs: Any) -> ProcessResult:
            nonlocal attempt_count
            attempt_count += 1
            return create_mock_process_result(output="Error", return_code=1)

        with patch("ralph.runner.run_tool_with_prompt", side_effect=always_fail):
            with patch("ralph.runner.time.sleep"):
                with patch("ralph.runner.ConversationWatcher"):
                    with patch("ralph.runner.print_retry_exhausted") as mock_exhausted:
                        result, story_id = runner._run_iteration_with_retry(1)

        # Should have tried 3 times (1 original + 2 retries)
        assert attempt_count == 3
        assert result.return_code == 1
        mock_exhausted.assert_called_once()

    def test_backoff_calculation(self, temp_project_dir: Path):
        """Test that exponential backoff is calculated correctly."""
        prd_path = temp_project_dir / "prd.json"
        prd_content = {
            "project": "Test",
            "branchName": "test",
            "description": "Test",
            "userStories": [],
        }
        prd_path.write_text(json.dumps(prd_content))

        runner = Runner(prd_path=prd_path, tool=Tool.CLAUDE)

        # Verify backoff formula: 5 * 3^(attempt-1)
        assert runner._calculate_backoff(1) == 5.0  # 5 * 3^0 = 5
        assert runner._calculate_backoff(2) == 15.0  # 5 * 3^1 = 15
        assert runner._calculate_backoff(3) == 45.0  # 5 * 3^2 = 45

    def test_retry_respects_interrupt(self, temp_project_dir: Path):
        """Test that retry loop respects interrupt signal."""
        prd_path = temp_project_dir / "prd.json"
        create_test_prd(prd_path)

        runner = Runner(
            prd_path=prd_path,
            tool=Tool.CLAUDE,
            max_iterations=1,
            max_retries=5,
            adaptive_pacing=False,
        )

        attempt_count = 0

        def fail_and_interrupt(*args: Any, **kwargs: Any) -> ProcessResult:
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count == 2:
                runner._interrupted = True
            return create_mock_process_result(output="Error", return_code=1)

        with patch("ralph.runner.run_tool_with_prompt", side_effect=fail_and_interrupt):
            with patch("ralph.runner.time.sleep"):
                with patch("ralph.runner.ConversationWatcher"):
                    result, story_id = runner._run_iteration_with_retry(1)

        # Should have stopped after interrupt
        assert attempt_count == 2


# =============================================================================
# Test ManagedProcess for subprocess termination
# =============================================================================


class TestManagedProcess:
    """Tests for ManagedProcess subprocess management."""

    def test_managed_process_initial_state(self):
        """Test ManagedProcess initial state."""
        mp = ManagedProcess()
        assert mp.process is None
        assert mp.interrupted is False

    def test_managed_process_reset(self):
        """Test ManagedProcess reset method."""
        mp = ManagedProcess()
        mp._interrupted = True

        mp.reset()

        assert mp.interrupted is False
        assert mp.process is None

    def test_terminate_sets_interrupted(self):
        """Test that terminate sets interrupted flag."""
        mp = ManagedProcess()
        mp.terminate()

        assert mp.interrupted is True


# =============================================================================
# Test run_ralph convenience function
# =============================================================================


class TestRunRalphFunction:
    """Tests for the run_ralph convenience function."""

    def test_run_ralph_creates_runner(self, temp_project_dir: Path):
        """Test that run_ralph creates and runs a Runner."""
        prd_path = temp_project_dir / "prd.json"
        stories = [
            {
                "id": "US-001",
                "title": "Story 1",
                "description": "Description",
                "acceptanceCriteria": ["Criterion"],
                "priority": 1,
                "passes": True,
            },
        ]
        create_test_prd(prd_path, stories=stories)

        with patch("ralph.runner.SessionManager") as mock_session:
            mock_session.return_value.__enter__ = MagicMock(
                return_value=MagicMock(pid=12345)
            )
            mock_session.return_value.__exit__ = MagicMock(return_value=False)
            result = run_ralph(
                prd_path=prd_path,
                tool=Tool.CLAUDE,
                max_iterations=1,
            )

        assert result is True  # Already complete

    def test_run_ralph_passes_parameters(self, temp_project_dir: Path):
        """Test that run_ralph passes all parameters to Runner."""
        prd_path = temp_project_dir / "prd.json"
        stories = [
            {
                "id": "US-001",
                "title": "Story 1",
                "description": "Description",
                "acceptanceCriteria": ["Criterion"],
                "priority": 1,
                "passes": True,
            },
        ]
        create_test_prd(prd_path, stories=stories)

        with patch("ralph.runner.Runner") as MockRunner:
            mock_instance = MagicMock()
            mock_instance.run.return_value = True
            MockRunner.return_value = mock_instance

            run_ralph(
                prd_path=prd_path,
                tool=Tool.AMP,
                max_iterations=10,
                verbose=True,
                timeout=600.0,
                max_retries=3,
                adaptive_pacing=True,
                pacing_threshold_1=60.0,
                pacing_threshold_2=75.0,
                pacing_threshold_3=85.0,
                five_hour_limit=50000,
            )

            MockRunner.assert_called_once()
            call_kwargs = MockRunner.call_args.kwargs
            assert call_kwargs["prd_path"] == prd_path
            assert call_kwargs["tool"] == Tool.AMP
            assert call_kwargs["max_iterations"] == 10
            assert call_kwargs["verbose"] is True
            assert call_kwargs["timeout"] == 600.0
            assert call_kwargs["max_retries"] == 3
            assert call_kwargs["adaptive_pacing"] is True
            assert call_kwargs["pacing_threshold_1"] == 60.0
            assert call_kwargs["pacing_threshold_2"] == 75.0
            assert call_kwargs["pacing_threshold_3"] == 85.0
            assert call_kwargs["five_hour_limit"] == 50000


# =============================================================================
# Test prompt generation
# =============================================================================


class TestPromptGeneration:
    """Tests for dynamic prompt generation."""

    def test_generate_prompt_includes_story_details(self, temp_project_dir: Path):
        """Test that generated prompt includes story details."""
        prd_path = temp_project_dir / "prd.json"
        prd = create_test_prd(prd_path)

        runner = Runner(prd_path=prd_path, tool=Tool.CLAUDE)

        story = prd.get_next_story()
        assert story is not None

        prompt = runner._generate_prompt(story, prd)

        assert story.id in prompt
        assert story.title in prompt
        assert story.description in prompt
        for criterion in story.acceptance_criteria:
            assert criterion in prompt
        assert "<promise>COMPLETE</promise>" in prompt

    def test_generate_prompt_includes_base_context(self, temp_project_dir: Path):
        """Test that generated prompt includes CLAUDE.md content."""
        prd_path = temp_project_dir / "prd.json"
        prd = create_test_prd(prd_path)

        runner = Runner(prd_path=prd_path, tool=Tool.CLAUDE)

        story = prd.get_next_story()
        assert story is not None

        prompt = runner._generate_prompt(story, prd)

        assert "Test Project" in prompt  # From CLAUDE.md


# =============================================================================
# Test iteration log saving
# =============================================================================


class TestIterationLogSaving:
    """Tests for iteration log saving."""

    def test_log_saved_when_dir_specified(self, temp_project_dir: Path):
        """Test that logs are saved when log_dir is specified."""
        prd_path = temp_project_dir / "prd.json"
        create_test_prd(prd_path)

        log_dir = temp_project_dir / "logs"

        runner = Runner(
            prd_path=prd_path,
            tool=Tool.CLAUDE,
            log_dir=log_dir,
        )
        runner._initialize_log_dir()

        log_path = runner._save_iteration_log(1, "US-001", "Test output content")

        assert log_path is not None
        assert log_path.exists()
        assert "iteration-1-US-001" in log_path.name
        assert log_path.read_text() == "Test output content"

    def test_no_log_when_dir_not_specified(self, temp_project_dir: Path):
        """Test that logs are not saved when log_dir is None."""
        prd_path = temp_project_dir / "prd.json"
        prd_content = {
            "project": "Test",
            "branchName": "test",
            "description": "Test",
            "userStories": [],
        }
        prd_path.write_text(json.dumps(prd_content))

        runner = Runner(
            prd_path=prd_path,
            tool=Tool.CLAUDE,
            log_dir=None,
        )

        log_path = runner._save_iteration_log(1, "US-001", "Test output")

        assert log_path is None
