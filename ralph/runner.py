"""Main orchestration loop for Ralph."""

from __future__ import annotations

import logging
import signal
import time
from datetime import datetime
from pathlib import Path
from types import FrameType
from typing import Any, Callable, cast

logger = logging.getLogger("ralph.runner")

from ralph.archive import handle_branch_change
from ralph.console import (
    console,
    print_completion,
    print_error,
    print_info,
    print_interrupt,
    print_iteration_header,
    print_iteration_usage,
    print_max_iterations_reached,
    print_pacing_adjustment,
    print_phase_header,
    print_plan_extraction_result,
    print_prd_status,
    print_reeval_cancelled,
    print_reeval_changes,
    print_reeval_error,
    print_reeval_header,
    print_reeval_proposed_changes,
    print_reeval_skipped,
    print_resume_info,
    print_retry,
    print_retry_exhausted,
    print_task_list,
    print_timeout,
    print_two_phase_summary,
    print_usage_critical,
    print_usage_warning,
    print_warning,
    prompt_reeval_confirmation,
)
from ralph.conversation import ConversationWatcher
from ralph.dashboard.reporter import (
    DashboardReporter,
    InstanceMetadata,
    NULL_REPORTER,
)
from ralph.prd import PRD, UserStory
from ralph.process import ManagedProcess, ProcessResult, Tool, run_tool_with_prompt
from ralph.session import SessionManager, get_budget_for_session, update_heartbeat
from ralph.state import RunState, clear_state, load_state, save_state
from ralph.tasks import get_all_tasks_for_project, ClaudeTask


DEFAULT_MAX_ITERATIONS = 10
DEFAULT_ITERATION_DELAY = 2.0  # seconds
DEFAULT_TIMEOUT = 30 * 60  # 30 minutes in seconds
DEFAULT_MAX_RETRIES = 2  # per-iteration retry attempts
RETRY_BACKOFF_BASE = 5.0  # seconds for first retry (5s, 15s, 45s with multiplier 3)
DEFAULT_REEVAL_INTERVAL = 10  # Run re-evaluation every N iterations
REEVAL_TIMEOUT = 5 * 60  # 5 minutes for re-evaluation
DEFAULT_PLANNING_TIMEOUT = 10 * 60  # 10 minutes for planning phase
DEFAULT_CODING_TIMEOUT = 30 * 60  # 30 minutes for coding phase

# Adaptive pacing thresholds and multipliers
DEFAULT_PACING_THRESHOLD_1 = 70.0  # 70% usage - 2x delay
DEFAULT_PACING_THRESHOLD_2 = 80.0  # 80% usage - 4x delay
DEFAULT_PACING_THRESHOLD_3 = 90.0  # 90% usage - 8x delay
DEFAULT_PACING_MULTIPLIER_1 = 2.0
DEFAULT_PACING_MULTIPLIER_2 = 4.0
DEFAULT_PACING_MULTIPLIER_3 = 8.0


class Runner:
    """Main orchestration loop for Ralph."""

    def __init__(
        self,
        prd_path: Path,
        tool: Tool = Tool.CLAUDE,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        verbose: bool = False,
        watch_tasks: bool = False,
        iteration_delay: float = DEFAULT_ITERATION_DELAY,
        timeout: float | None = DEFAULT_TIMEOUT,
        log_dir: Path | None = None,
        resume: bool = False,
        max_retries: int = DEFAULT_MAX_RETRIES,
        adaptive_pacing: bool = True,
        pacing_threshold_1: float = DEFAULT_PACING_THRESHOLD_1,
        pacing_threshold_2: float = DEFAULT_PACING_THRESHOLD_2,
        pacing_threshold_3: float = DEFAULT_PACING_THRESHOLD_3,
        five_hour_limit: int | None = None,
        reeval_interval: int = DEFAULT_REEVAL_INTERVAL,
        no_reeval: bool = False,
        reeval_confirm: bool = False,
        reeval_dry_run: bool = False,
        two_phase: bool = False,
        planning_tool: Tool = Tool.CLAUDE,
        coding_tool: Tool = Tool.OPENCODE,
        planning_timeout: float | None = DEFAULT_PLANNING_TIMEOUT,
        coding_timeout: float | None = DEFAULT_CODING_TIMEOUT,
        ccs_profile: str | None = None,
        ccs_args: str | None = None,
    ):
        self.prd_path = prd_path
        self.tool = tool
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.watch_tasks = watch_tasks
        self.iteration_delay = iteration_delay
        self.timeout = timeout
        self.log_dir = log_dir
        self.resume = resume
        self.max_retries = max_retries

        # Adaptive pacing configuration
        self.adaptive_pacing = adaptive_pacing
        self.pacing_threshold_1 = pacing_threshold_1
        self.pacing_threshold_2 = pacing_threshold_2
        self.pacing_threshold_3 = pacing_threshold_3
        self.five_hour_limit = five_hour_limit

        # Re-evaluation configuration
        self.reeval_interval = reeval_interval
        self.no_reeval = no_reeval
        self.reeval_confirm = reeval_confirm
        self.reeval_dry_run = reeval_dry_run

        # Two-phase orchestration configuration
        self.two_phase = two_phase
        self.planning_tool = planning_tool
        self.coding_tool = coding_tool
        self.planning_timeout = planning_timeout
        self.coding_timeout = coding_timeout

        # ccs-specific configuration (profile + passthrough args)
        self.ccs_profile = ccs_profile
        self.ccs_args = ccs_args

        # Track last pacing multiplier to only log changes
        self._last_pacing_multiplier: float = 1.0

        # Derive paths
        self.base_dir = prd_path.parent
        self.prompt_path = self._get_prompt_path()

        # Signal handling state
        self._interrupted = False
        self._current_iteration = 0
        self._managed_process = ManagedProcess()
        self._original_sigint: signal.Handlers = signal.SIG_DFL
        self._original_sigterm: signal.Handlers = signal.SIG_DFL

        # Run state for resume functionality
        self._run_state: RunState | None = None
        self._start_iteration = 1

        # Dashboard reporter — publishes instance state to ~/.ralph/instances/
        # Lifecycle managed inside run() so tests that construct Runner directly
        # without calling run() don't leave stale state files.
        self._reporter: Any = NULL_REPORTER

    def _get_prompt_path(self) -> Path:
        """Get the prompt file path for the current tool."""
        if self.tool == Tool.CLAUDE:
            return self.base_dir / "CLAUDE.md"
        else:
            return self.base_dir / "prompt.md"

    def _generate_prompt(
        self, story: UserStory, prd: PRD, iteration: int
    ) -> str:
        """Generate a dynamic prompt for the current iteration."""
        # Format acceptance criteria as a checklist
        criteria_list = "\n".join(f"- [ ] {c}" for c in story.acceptance_criteria)

        # Build notes section only if present
        notes_section = f"\n**Notes:** {story.notes}" if story.notes else ""

        prompt = f"""# Task: Implement User Story {story.id}

## Current Context
- **Iteration:** {iteration} of {self.max_iterations}
- **PRD Path:** {self.prd_path}
- **Project:** {prd.project}

## Story Details
**ID:** {story.id}
**Title:** {story.title}
**Description:** {story.description}

**Acceptance Criteria:**
{criteria_list}
{notes_section}

## Available CLI Tools

Ralph provides these commands to manage PRD state:

- `ralph story {story.id} --prd {self.prd_path}` - View this story's details and current status
- `ralph mark-complete {story.id} --prd {self.prd_path}` - Mark this story as complete
- `ralph status --prd {self.prd_path}` - View overall PRD progress

## Instructions

1. Read CLAUDE.md for project context if needed
2. Implement the changes required to satisfy ALL acceptance criteria
3. Test your changes to ensure they work correctly
4. Run `ralph mark-complete {story.id} --prd {self.prd_path}` when all criteria are met
5. Update progress.txt with a summary of what you did
6. Output <promise>COMPLETE</promise> to signal completion

## Important

- Focus ONLY on this story - do not implement other stories
- Make sure ALL acceptance criteria are satisfied before marking complete
- Run appropriate checks (mypy, pytest, flutter analyze) if mentioned in acceptance criteria
"""
        return prompt

    def _get_progress_path(self) -> Path:
        """Get the progress file path."""
        return self.base_dir / "progress.txt"

    def _initialize_progress_file(self) -> None:
        """Initialize progress.txt if it doesn't exist."""
        progress_path = self._get_progress_path()
        if not progress_path.exists():
            timestamp = datetime.now().strftime("%Y-%m-%d")
            content = f"# Ralph Progress Log\nStarted: {timestamp}\n---\n"
            progress_path.write_text(content)
            if self.verbose:
                print_info(f"Initialized progress file: {progress_path}")

    def _initialize_log_dir(self) -> None:
        """Create the log directory if logging is enabled."""
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            if self.verbose:
                print_info(f"Log directory: {self.log_dir}")

    def _save_iteration_log(
        self, iteration: int, story_id: str | None, output: str
    ) -> Path | None:
        """
        Save iteration output to a log file.

        Args:
            iteration: The iteration number
            story_id: The story ID being worked on (or None)
            output: The full output from the iteration

        Returns:
            Path to the log file, or None if logging is disabled
        """
        if self.log_dir is None:
            return None

        # Generate filename: iteration-N-STORYID-TIMESTAMP.log
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        story_part = f"-{story_id}" if story_id else ""
        filename = f"iteration-{iteration}{story_part}-{timestamp}.log"
        log_path = self.log_dir / filename

        # Write the log file
        log_path.write_text(output)

        if self.verbose:
            print_info(f"Saved iteration log: {log_path}")

        return log_path

    def _signal_handler(self, signum: int, frame: FrameType | None) -> None:
        """Handle SIGINT and SIGTERM for graceful shutdown."""
        sig_name = signal.Signals(signum).name
        logger.debug(f"Signal {sig_name} received, initiating graceful shutdown...")
        self._interrupted = True
        # Terminate the subprocess if running
        logger.debug("Terminating subprocess...")
        self._managed_process.terminate()
        logger.debug("Subprocess termination initiated")

    def _install_signal_handlers(self) -> None:
        """Install signal handlers for graceful shutdown."""
        self._original_sigint = cast(
            signal.Handlers, signal.signal(signal.SIGINT, self._signal_handler)
        )
        self._original_sigterm = cast(
            signal.Handlers, signal.signal(signal.SIGTERM, self._signal_handler)
        )

    def _restore_signal_handlers(self) -> None:
        """Restore original signal handlers."""
        signal.signal(signal.SIGINT, self._original_sigint)
        signal.signal(signal.SIGTERM, self._original_sigterm)

    def _load_prd(self) -> PRD:
        """Load the PRD from file."""
        return PRD.from_file(self.prd_path)

    def _validate_prd_for_resume(self, state: RunState, prd: PRD) -> None:
        """Validate that PRD hasn't changed significantly since the saved state."""
        # Load the original story count from the state if available
        # For now, we'll compare based on the path match and warn if story count seems off
        state_prd_path = Path(state.prd_path).resolve()
        current_prd_path = self.prd_path.resolve()

        if state_prd_path != current_prd_path:
            print_warning(
                f"PRD path differs from saved state.\n"
                f"  State: {state_prd_path}\n"
                f"  Current: {current_prd_path}"
            )

        # Store story count in state for future validation
        if hasattr(state, "story_count") and state.story_count is not None:
            current_count = len(prd.user_stories)
            if current_count != state.story_count:
                print_warning(
                    f"PRD story count changed since last run.\n"
                    f"  Previous: {state.story_count} stories\n"
                    f"  Current: {current_count} stories\n"
                    f"  Resume may skip or repeat work."
                )

    def _save_run_state(
        self, iteration: int, story_id: str | None, story_count: int | None = None
    ) -> None:
        """Save run state for resume functionality."""
        if self._run_state is None:
            self._run_state = RunState.create(
                prd_path=self.prd_path,
                tool=self.tool.value,
                iteration=iteration,
                story_id=story_id,
                story_count=story_count,
            )
        else:
            self._run_state.update(iteration, story_id)
        save_state(self.base_dir, self._run_state)

    def _create_output_handler(self) -> Callable[[str], None]:
        """Create a handler for streaming output."""
        import os
        reporter = self._reporter
        def handler(text: str) -> None:
            # Write directly to fd 2 (stderr) for unbuffered real-time output
            # This bypasses Python's sys.stderr which can be buffered or intercepted by Rich/Typer
            # This is equivalent to what `tee /dev/stderr` was doing before
            os.write(2, text.encode())
            # Tee to the dashboard tail. The reporter is thread-safe and no-ops
            # when nothing has called start() yet.
            try:
                reporter.on_output_line(text)
            except Exception:
                pass
        return handler

    def _display_tasks(self) -> None:
        """Display Claude tasks if watch_tasks is enabled."""
        if not self.watch_tasks:
            return

        tasks = get_all_tasks_for_project(self.base_dir)
        if tasks:
            print_task_list(tasks, "Current Claude Tasks")

    def _should_run_reeval(self, iteration: int) -> bool:
        """
        Check if re-evaluation should run at this iteration.

        Re-evaluation runs every N iterations (reeval_interval), starting from
        the interval number (e.g., 10, 20, 30 for interval=10).

        Args:
            iteration: Current iteration number

        Returns:
            True if re-evaluation should run
        """
        # Skip if disabled
        if self.no_reeval or self.reeval_interval <= 0:
            return False

        # Check if this is a reeval iteration
        if iteration % self.reeval_interval != 0:
            return False

        # Check if we already ran reeval at this iteration (for resume)
        if self._run_state and self._run_state.last_reeval_iteration == iteration:
            return False

        return True

    def _run_reeval_iteration(self, iteration: int) -> bool:
        """
        Run a re-evaluation iteration.

        This is a "pseudo iteration" that analyzes the PRD for issues and
        may modify pending stories. It does NOT count against max_iterations.

        Args:
            iteration: Current iteration number (for logging)

        Returns:
            True if re-evaluation completed successfully
        """
        from ralph.reeval import (
            REEVAL_COMPLETE_SIGNAL,
            apply_changes,
            append_reeval_to_progress,
            generate_reeval_prompt,
            parse_reeval_response,
            validate_changes,
        )

        print_reeval_header(iteration)

        # Load current PRD
        prd = self._load_prd()
        passing, total = prd.get_progress()
        self._reporter.on_iteration_start(
            iteration, None, phase="reeval",
            prd_passing=passing, prd_total=total,
        )

        # Check if there are any pending stories to evaluate
        pending_count = sum(1 for s in prd.user_stories if not s.passes)
        if pending_count == 0:
            print_reeval_skipped("All stories are complete")
            return True

        # Skip if only 1 pending story (nothing to merge/remove safely)
        if pending_count == 1:
            print_reeval_skipped("Only 1 pending story remaining")
            return True

        # Generate the re-evaluation prompt with iteration context
        progress_path = self._get_progress_path()
        prompt = generate_reeval_prompt(
            prd,
            progress_path,
            current_iteration=iteration,
            max_iterations=self.max_iterations,
        )

        if self.verbose:
            print_info("Running PRD re-evaluation...")

        # Run the tool with a shorter timeout
        self._managed_process.reset()
        output_handler = self._create_output_handler()

        try:
            result = run_tool_with_prompt(
                tool=self.tool,
                prompt=prompt,
                on_output=output_handler,
                cwd=self.base_dir,
                managed_process=self._managed_process,
                timeout=REEVAL_TIMEOUT,
                ccs_profile=self.ccs_profile,
                ccs_args=self.ccs_args,
            )
        except Exception as e:
            print_reeval_error(f"Failed to run re-evaluation: {e}")
            return False

        console.print()  # Newline after output

        # Check for interruption
        if result.interrupted or self._interrupted:
            logger.info("Re-evaluation interrupted")
            return False

        # Check for timeout
        if result.timed_out:
            print_reeval_error("Re-evaluation timed out")
            return False

        # Parse the response
        reeval_result = parse_reeval_response(result.output)

        if reeval_result.error:
            print_reeval_error(reeval_result.error)
            return False

        # Validate proposed changes
        validation = validate_changes(prd, reeval_result.changes)

        # Handle dry-run mode - show what would happen but don't apply
        if self.reeval_dry_run:
            print_reeval_proposed_changes(
                validation.approved_changes,
                validation.rejected_changes,
                reeval_result.summary,
                dry_run=True,
            )
            # Update state so we don't re-run at this iteration on resume
            if self._run_state:
                self._run_state.update_reeval(iteration)
                save_state(self.base_dir, self._run_state)
            return True

        # Apply approved changes
        applied_changes: list[str] = []
        if validation.approved_changes:
            # Show proposed changes and ask for confirmation if enabled
            if self.reeval_confirm:
                print_reeval_proposed_changes(
                    validation.approved_changes,
                    validation.rejected_changes,
                    reeval_result.summary,
                    dry_run=False,
                )
                if not prompt_reeval_confirmation():
                    print_reeval_cancelled()
                    # Log that changes were cancelled
                    append_reeval_to_progress(
                        progress_path,
                        reeval_result,
                        validation,
                        [],  # No changes applied
                    )
                    # Update state
                    if self._run_state:
                        self._run_state.update_reeval(iteration)
                        save_state(self.base_dir, self._run_state)
                    return True

            # Create a backup of the PRD before modifying
            backup_path = self.prd_path.with_suffix(".json.bak")
            try:
                import shutil
                shutil.copy2(self.prd_path, backup_path)
                if self.verbose:
                    print_info(f"Created PRD backup: {backup_path}")
            except Exception as e:
                logger.warning(f"Failed to create PRD backup: {e}")

            prd, applied_changes = apply_changes(prd, validation.approved_changes)
            # Save the modified PRD
            prd.save(self.prd_path)

        # Log to progress.txt
        append_reeval_to_progress(
            progress_path,
            reeval_result,
            validation,
            applied_changes,
        )

        # Print results
        print_reeval_changes(
            applied_changes,
            validation.rejected_changes,
            reeval_result.summary,
        )

        # Update state
        if self._run_state:
            self._run_state.update_reeval(iteration)
            save_state(self.base_dir, self._run_state)

        # Save iteration log
        self._save_iteration_log(iteration, "reeval", result.output)

        return True

    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff delay for retry attempt.

        Uses formula: base * (3 ^ (attempt - 1))
        This gives: 5s, 15s, 45s for attempts 1, 2, 3
        """
        return float(RETRY_BACKOFF_BASE * (3 ** (attempt - 1)))

    def _get_usage_percentage(self) -> float | None:
        """
        Get the current 5-hour window usage percentage.

        When cross-session coordination is enabled, this calculates the
        percentage based on this session's share of the remaining budget.

        Returns:
            Usage percentage (0-100), or None if usage tracking is unavailable.
        """
        try:
            logger.debug("Getting usage percentage...")
            from ralph.config import get_plan, get_plan_limits
            from ralph.usage import get_5hour_window_usage

            # Get the limit to use
            if self.five_hour_limit is not None:
                limit = self.five_hour_limit
            else:
                current_plan = get_plan()
                limits = get_plan_limits(current_plan)
                limit = limits["5hour_tokens"]

            if limit <= 0:
                logger.debug("Limit is 0, returning None")
                return None

            logger.debug("Fetching 5-hour window usage...")
            usage = get_5hour_window_usage()
            total_used = usage.rate_limited_tokens  # Excludes cache reads
            logger.debug(f"Total used: {total_used} tokens")

            # Calculate remaining budget and divide across active sessions
            remaining = max(0, limit - total_used)
            logger.debug("Getting session budget...")
            session_budget, num_sessions = get_budget_for_session(remaining)
            logger.debug(f"Session budget: {session_budget}, num_sessions: {num_sessions}")

            # Calculate this session's effective limit (used + session share of remaining)
            # This ensures each session sees their fair share of the remaining budget
            effective_limit = total_used + session_budget

            if effective_limit <= 0:
                return 100.0  # Fully used

            # Calculate percentage based on effective limit
            percentage = (total_used / effective_limit) * 100
            logger.debug(f"Usage percentage: {percentage:.1f}%")
            return min(percentage, 100.0)
        except Exception as e:
            # If usage tracking fails, don't block the run
            logger.debug(f"Usage tracking failed: {e}")
            return None

    def _build_usage_snapshot(self) -> dict[str, Any]:
        """Compact usage summary for the dashboard. Swallows any failure — the
        dashboard is strictly informational and must never break the run."""
        try:
            from ralph.config import get_plan, get_plan_limits
            from ralph.usage import get_5hour_window_usage

            if self.five_hour_limit is not None:
                limit = self.five_hour_limit
            else:
                current_plan = get_plan()
                limits = get_plan_limits(current_plan)
                limit = limits["5hour_tokens"]

            if limit <= 0:
                return {}

            usage = get_5hour_window_usage()
            total_used = usage.rate_limited_tokens
            remaining = max(0, limit - total_used)
            session_budget, num_sessions = get_budget_for_session(remaining)
            effective_limit = max(1, total_used + session_budget)
            percentage = min(100.0, (total_used / effective_limit) * 100)
            return {
                "tokens_used": total_used,
                "five_hour_limit": limit,
                "allocated_budget": session_budget,
                "num_sessions": num_sessions,
                "percentage": percentage,
            }
        except Exception:
            return {}

    def _calculate_adaptive_delay(self, usage_percentage: float | None) -> tuple[float, float]:
        """
        Calculate the adaptive iteration delay based on usage percentage.

        Args:
            usage_percentage: Current usage percentage (0-100), or None

        Returns:
            Tuple of (adjusted delay in seconds, multiplier applied)
        """
        if usage_percentage is None or not self.adaptive_pacing:
            return self.iteration_delay, 1.0

        # Determine multiplier based on thresholds
        if usage_percentage >= self.pacing_threshold_3:
            multiplier = DEFAULT_PACING_MULTIPLIER_3
        elif usage_percentage >= self.pacing_threshold_2:
            multiplier = DEFAULT_PACING_MULTIPLIER_2
        elif usage_percentage >= self.pacing_threshold_1:
            multiplier = DEFAULT_PACING_MULTIPLIER_1
        else:
            multiplier = 1.0

        adjusted_delay = self.iteration_delay * multiplier
        return adjusted_delay, multiplier

    def _apply_adaptive_pacing(self) -> float:
        """
        Check usage and apply adaptive pacing if needed.

        Returns:
            The delay to use (either base delay or adjusted delay).
        """
        usage_percentage = self._get_usage_percentage()
        adjusted_delay, multiplier = self._calculate_adaptive_delay(usage_percentage)

        # Only log when multiplier changes (to avoid spamming logs)
        if multiplier != self._last_pacing_multiplier and multiplier > 1.0:
            if usage_percentage is not None:
                print_pacing_adjustment(
                    usage_percentage=usage_percentage,
                    base_delay=self.iteration_delay,
                    adjusted_delay=adjusted_delay,
                    multiplier=multiplier,
                )
            self._last_pacing_multiplier = multiplier
        elif multiplier < self._last_pacing_multiplier:
            # Usage has decreased, reset tracking
            if self.verbose and usage_percentage is not None:
                print_info(f"Usage at {usage_percentage:.1f}%, returning to normal pacing")
            self._last_pacing_multiplier = multiplier

        return adjusted_delay

    def _display_iteration_usage(self) -> None:
        """
        Display current usage after each iteration and show warnings if thresholds exceeded.

        Shows:
        - Brief usage summary after every iteration (with cost if enabled)
        - Warning banner at 70% usage
        - Critical warning banner at 90% usage
        - Time until window resets in warnings
        """
        try:
            from datetime import timedelta

            from ralph.config import get_plan, get_plan_limits, load_config
            from ralph.console import _format_time_remaining
            from ralph.usage import get_5hour_window_usage

            # Get config to check if cost tracking is enabled
            config = load_config()
            show_cost = config.enable_cost_tracking

            # Get the limit to use
            if self.five_hour_limit is not None:
                limit = self.five_hour_limit
            else:
                current_plan = get_plan()
                limits = get_plan_limits(current_plan)
                limit = limits["5hour_tokens"]

            if limit <= 0:
                return

            usage = get_5hour_window_usage()
            total_used = usage.rate_limited_tokens  # Excludes cache reads
            percentage = (total_used / limit * 100) if limit > 0 else 0

            # Get cost for display if enabled
            cost_usd = usage.cost_usd if show_cost else None

            # Always display brief usage summary after each iteration
            print_iteration_usage(percentage, total_used, limit, cost_usd=cost_usd)

            # Calculate time until oldest data ages out of the 5-hour window
            from datetime import datetime, timezone

            now = datetime.now(timezone.utc)
            time_remaining = usage.time_until_oldest_ages_out(timedelta(hours=5), now)

            # Format time remaining
            if time_remaining.total_seconds() <= 0:
                time_until_reset = "now"
            else:
                hours = int(time_remaining.total_seconds() // 3600)
                minutes = int((time_remaining.total_seconds() % 3600) // 60)
                if hours > 0:
                    time_until_reset = f"{hours}h {minutes}m"
                else:
                    time_until_reset = f"{minutes}m"

            # Show warning banners based on thresholds
            if percentage >= 90:
                print_usage_critical(percentage, total_used, limit, time_until_reset)
            elif percentage >= 70:
                print_usage_warning(percentage, total_used, limit, time_until_reset)

        except Exception:
            # If usage tracking fails, don't block the run
            pass

    def run_iteration(self, iteration: int) -> tuple[ProcessResult, str | None]:
        """
        Run a single iteration.

        Returns:
            Tuple of (ProcessResult from tool execution, story ID or None)
        """
        logger.debug(f"run_iteration({iteration}) starting")

        # Reload PRD to get latest state
        logger.debug("Loading PRD...")
        prd = self._load_prd()
        next_story = prd.get_next_story()
        story_id = next_story.id if next_story else None
        logger.debug(f"Next story: {story_id}")

        # Print iteration header
        print_iteration_header(iteration, self.max_iterations, next_story)
        passing, total = prd.get_progress()
        self._reporter.on_iteration_start(
            iteration, next_story, phase="single",
            prd_passing=passing, prd_total=total,
        )

        # Display current tasks if enabled
        self._display_tasks()

        # Check if there's a story to work on
        if next_story is None:
            print_info("No incomplete stories found - all done!")
            return ProcessResult(output="", return_code=0, completed=True), None

        # Generate the dynamic prompt for this story
        prompt = self._generate_prompt(next_story, prd, iteration)

        if self.verbose:
            print_info(f"Running {self.tool.value} for story: {next_story.id}")

        # Start conversation watcher for real-time log display
        logger.debug("Starting conversation watcher...")
        watcher = ConversationWatcher(self.base_dir)
        watcher.start()

        # Run the tool with the generated prompt
        logger.debug(f"Running {self.tool.value} with timeout={self.timeout}s...")
        self._managed_process.reset()
        output_handler = self._create_output_handler()
        try:
            result = run_tool_with_prompt(
                tool=self.tool,
                prompt=prompt,
                on_output=output_handler,
                cwd=self.base_dir,
                managed_process=self._managed_process,
                timeout=self.timeout,
                ccs_profile=self.ccs_profile,
                ccs_args=self.ccs_args,
            )
        finally:
            logger.debug("Stopping conversation watcher...")
            watcher.stop()

        console.print()  # Newline after output

        logger.debug(
            f"run_iteration({iteration}) complete: "
            f"returncode={result.return_code}, completed={result.completed}, "
            f"timed_out={result.timed_out}, interrupted={result.interrupted}"
        )

        if self.verbose:
            print_info(f"Tool exited with code: {result.return_code}")
            if result.completed:
                print_info("Completion signal detected!")

        return result, story_id

    def _run_two_phase_iteration(self, iteration: int) -> tuple[ProcessResult, str | None]:
        """
        Run a single two-phase iteration (planning + coding).

        Phase 1: Planning tool analyzes the story and creates a plan
        Phase 2: Coding tool executes the plan

        Returns:
            Tuple of (ProcessResult from final phase, story ID or None)
        """
        from ralph.twophase import (
            extract_plan,
            run_coding_phase,
            run_planning_phase,
        )

        logger.debug(f"_run_two_phase_iteration({iteration}) starting")

        # Reload PRD to get latest state
        prd = self._load_prd()
        next_story = prd.get_next_story()
        story_id = next_story.id if next_story else None

        # Print iteration header
        print_iteration_header(iteration, self.max_iterations, next_story)
        passing, total = prd.get_progress()
        self._reporter.on_iteration_start(
            iteration, next_story, phase="planning",
            prd_passing=passing, prd_total=total,
        )

        # Display current tasks if enabled
        self._display_tasks()

        # Check if there's a story to work on
        if next_story is None:
            print_info("No incomplete stories found - all done!")
            return ProcessResult(output="", return_code=0, completed=True), None

        if self.verbose:
            print_info(f"Two-phase mode: {self.planning_tool.value} (plan) + {self.coding_tool.value} (code)")

        # Phase 1: Planning
        print_phase_header(1, "Planning", self.planning_tool)

        watcher = ConversationWatcher(self.base_dir)
        watcher.start()

        self._managed_process.reset()
        output_handler = self._create_output_handler()

        try:
            planning_result, plan = run_planning_phase(
                story=next_story,
                prd=prd,
                iteration=iteration,
                max_iterations=self.max_iterations,
                prd_path=self.prd_path,
                tool=self.planning_tool,
                cwd=self.base_dir,
                on_output=output_handler,
                managed_process=self._managed_process,
                timeout=self.planning_timeout,
                ccs_profile=self.ccs_profile,
                ccs_args=self.ccs_args,
            )
        finally:
            watcher.stop()

        console.print()  # Newline after output

        # Check for interruption
        if planning_result.interrupted or self._interrupted:
            logger.info("Planning phase interrupted")
            return planning_result, story_id

        # Check for timeout
        if planning_result.timed_out:
            print_timeout(iteration)
            return planning_result, story_id

        # Check if plan was extracted
        print_plan_extraction_result(plan is not None, "No <implementation-plan> tags found" if not plan else None)

        if not plan:
            # Return the planning result so the outer loop can handle retry
            return ProcessResult(
                output=planning_result.output,
                return_code=1,  # Non-zero to trigger retry
                completed=False,
                interrupted=False,
                timed_out=False,
            ), story_id

        # Phase 2: Coding
        print_phase_header(2, "Coding", self.coding_tool)
        passing, total = prd.get_progress()
        self._reporter.on_iteration_start(
            iteration, next_story, phase="coding",
            prd_passing=passing, prd_total=total,
        )

        watcher = ConversationWatcher(self.base_dir)
        watcher.start()

        self._managed_process.reset()
        output_handler = self._create_output_handler()

        try:
            coding_result = run_coding_phase(
                story=next_story,
                prd=prd,
                plan=plan,
                iteration=iteration,
                max_iterations=self.max_iterations,
                prd_path=self.prd_path,
                tool=self.coding_tool,
                cwd=self.base_dir,
                on_output=output_handler,
                managed_process=self._managed_process,
                timeout=self.coding_timeout,
                ccs_profile=self.ccs_profile,
                ccs_args=self.ccs_args,
            )
        finally:
            watcher.stop()

        console.print()  # Newline after output

        # Print summary
        print_two_phase_summary(
            planning_success=True,
            coding_success=coding_result.return_code == 0 or coding_result.completed,
            story_id=story_id,
        )

        logger.debug(
            f"_run_two_phase_iteration({iteration}) complete: "
            f"coding_returncode={coding_result.return_code}, completed={coding_result.completed}"
        )

        return coding_result, story_id

    def _run_iteration_with_retry(
        self, iteration: int
    ) -> tuple[ProcessResult, str | None]:
        """
        Run a single iteration with retry logic for transient failures.

        Retries on non-zero exit code from Claude, but NOT on timeout.
        Uses exponential backoff between retries (5s, 15s, 45s).

        Returns:
            Tuple of (ProcessResult from tool execution, story ID or None)
        """
        # Dispatch to two-phase or single-phase iteration
        if self.two_phase:
            result, story_id = self._run_two_phase_iteration(iteration)
        else:
            result, story_id = self.run_iteration(iteration)

        # Don't retry on success, timeout, interrupt, or completion
        if (
            result.return_code == 0
            or result.timed_out
            or result.interrupted
            or result.completed
        ):
            return result, story_id

        # Retry logic for non-zero exit codes
        for attempt in range(1, self.max_retries + 1):
            # Check for interrupt before retrying
            if self._interrupted:
                return result, story_id

            backoff = self._calculate_backoff(attempt)
            print_retry(iteration, attempt, self.max_retries, result.return_code, backoff)

            # Wait with backoff
            time.sleep(backoff)

            # Check for interrupt after waiting
            if self._interrupted:
                return result, story_id

            # Retry the iteration
            result, story_id = self.run_iteration(iteration)

            # If this attempt succeeded or hit a terminal condition, return
            if (
                result.return_code == 0
                or result.timed_out
                or result.interrupted
                or result.completed
            ):
                return result, story_id

        # All retries exhausted, log and continue to next iteration
        print_retry_exhausted(iteration, self.max_retries, result.return_code)
        return result, story_id

    def run(self) -> bool | int:
        """
        Run the main loop.

        Returns:
            True if all stories were completed
            False if max iterations reached
            130 if interrupted by SIGINT
            143 if interrupted by SIGTERM
        """
        # Install signal handlers
        self._install_signal_handlers()
        logger.debug("Signal handlers installed")

        # Construct the dashboard reporter lazily so tests that build a
        # Runner without invoking run() don't touch ~/.ralph/instances.
        self._reporter = DashboardReporter()
        try:
            # Use SessionManager for cross-session coordination
            # This registers our session and automatically cleans up on exit
            with SessionManager(prd_path=self.prd_path) as session:
                if self.verbose:
                    print_info(f"Registered session (PID={session.pid})")
                return self._run_loop()
        finally:
            logger.debug("Cleaning up...")
            # Ensure subprocess is terminated
            if self._managed_process.process is not None:
                logger.debug("Terminating any remaining subprocess...")
                self._managed_process.terminate()
            try:
                self._reporter.close()
            except Exception:
                logger.debug("Dashboard reporter close() raised", exc_info=True)
            self._restore_signal_handlers()
            logger.debug("Cleanup complete")

    def _run_loop(self) -> bool | int:
        """Internal run loop, separated for signal handler cleanup."""
        logger.debug("_run_loop starting")

        # Load initial PRD
        logger.debug("Loading initial PRD...")
        prd = self._load_prd()

        # Start publishing to the dashboard now that we know the PRD metadata.
        initial_passing, initial_total = prd.get_progress()
        self._reporter.start(
            InstanceMetadata(
                prd_path=self.prd_path,
                prd_project=prd.project or None,
                tool=self.tool.value if not self.two_phase else self.coding_tool.value,
                two_phase=self.two_phase,
                max_iterations=self.max_iterations,
                cwd=self.base_dir,
            ),
            prd_passing=initial_passing,
            prd_total=initial_total,
        )

        # Handle resume if requested
        if self.resume:
            existing_state = load_state(self.base_dir)
            if existing_state:
                # Validate PRD hasn't changed significantly
                self._validate_prd_for_resume(existing_state, prd)

                # Resume from last completed iteration
                self._start_iteration = existing_state.last_iteration + 1
                self._run_state = existing_state
                print_resume_info(
                    existing_state.last_iteration,
                    existing_state.last_story_id,
                    existing_state.started_at,
                )
            else:
                if self.verbose:
                    print_info("No previous run state found, starting fresh")

        # Handle branch change detection and archiving
        archive_path, archived_files = handle_branch_change(
            self.base_dir, prd, self.prd_path
        )

        # Initialize progress file if it doesn't exist
        self._initialize_progress_file()

        # Initialize log directory if logging is enabled
        self._initialize_log_dir()

        if archive_path:
            print_info(f"Previous run archived to: {archive_path}")

        # Show initial status
        if self.verbose:
            print_prd_status(prd)

        # Check if already complete
        if prd.is_complete():
            print_completion()
            clear_state(self.base_dir)
            return True

        # Main loop
        for iteration in range(self._start_iteration, self.max_iterations + 1):
            self._current_iteration = iteration
            logger.debug(f"=== Starting iteration {iteration}/{self.max_iterations} ===")

            # Check for interrupt before starting iteration
            if self._interrupted:
                prd = self._load_prd()
                completed, total = prd.get_progress()
                print_interrupt(iteration, completed, total)
                return 130  # Standard exit code for SIGINT

            # Re-evaluation check BEFORE story iteration
            if self._should_run_reeval(iteration):
                self._run_reeval_iteration(iteration)
                # Check for interrupt after re-eval
                if self._interrupted:
                    prd = self._load_prd()
                    completed, total = prd.get_progress()
                    print_interrupt(iteration, completed, total)
                    return 130

            # Run iteration with retry logic
            result, story_id = self._run_iteration_with_retry(iteration)

            # Check for interrupt after iteration - do this FIRST before any slow operations
            if self._interrupted or result.interrupted:
                prd = self._load_prd()
                completed, total = prd.get_progress()
                print_interrupt(iteration, completed, total)
                return 130  # Standard exit code for SIGINT

            # Save iteration log
            self._save_iteration_log(iteration, story_id, result.output)

            # Save run state for resume functionality
            prd = self._load_prd()
            self._save_run_state(iteration, story_id, len(prd.user_stories))

            # Publish this iteration's outcome + PRD progress to the dashboard.
            passing, total = prd.get_progress()
            self._reporter.on_iteration_end(
                return_code=result.return_code,
                completed=result.completed,
                timed_out=result.timed_out,
                interrupted=result.interrupted,
                prd_passing=passing,
                prd_total=total,
                usage=self._build_usage_snapshot(),
            )

            # Display current usage after each iteration (with warnings if thresholds exceeded)
            # Skip if interrupted to avoid slow usage parsing during shutdown
            if not self._interrupted:
                self._display_iteration_usage()

            # Check for timeout - log and continue to next iteration (no retry on timeout)
            if result.timed_out:
                print_timeout(iteration)
                # Continue to next iteration rather than stopping
                if iteration < self.max_iterations:
                    logger.debug("Calculating adaptive pacing after timeout...")
                    delay = self._apply_adaptive_pacing()
                    logger.debug(f"Sleeping for {delay:.1f}s between iterations...")
                    if self.verbose:
                        print_info(f"Waiting {delay:.1f}s before next iteration...")
                    time.sleep(delay)
                    # Update session heartbeat during delay
                    logger.debug("Updating session heartbeat...")
                    update_heartbeat()
                    logger.debug("Heartbeat updated")
                continue

            # Note: result.completed means the story signaled completion with <promise>COMPLETE</promise>
            # This just means this story is done - we still need to check if ALL stories are done

            # Auto-mark story complete when signal detected
            # This ensures the PRD stays in sync even if Claude forgot to run mark-complete
            if result.completed and story_id:
                prd = self._load_prd()
                story_obj = prd.get_story_by_id(story_id)
                if story_obj and not story_obj.passes:
                    logger.debug(f"Auto-marking {story_id} as complete")
                    prd.mark_story_complete(story_id)
                    prd.save(self.prd_path)
                    if self.verbose:
                        print_info(f"Auto-marked story {story_id} as complete")

            # Reload PRD to check if all stories are done
            prd = self._load_prd()
            if prd.is_complete():
                print_completion()
                clear_state(self.base_dir)
                return True

            # Wait before next iteration (unless this was the last one)
            if iteration < self.max_iterations:
                logger.debug("Calculating adaptive pacing...")
                delay = self._apply_adaptive_pacing()
                logger.debug(f"Sleeping for {delay:.1f}s between iterations...")
                if self.verbose:
                    print_info(f"Waiting {delay:.1f}s before next iteration...")
                time.sleep(delay)
                # Update session heartbeat during delay
                logger.debug("Updating session heartbeat...")
                update_heartbeat()
                self._reporter.heartbeat(usage=self._build_usage_snapshot())
                logger.debug("Heartbeat updated")

        # Max iterations reached without completion
        print_max_iterations_reached(self.max_iterations)
        return False


def run_ralph(
    prd_path: Path,
    tool: Tool = Tool.CLAUDE,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    verbose: bool = False,
    watch_tasks: bool = False,
    timeout: float | None = DEFAULT_TIMEOUT,
    log_dir: Path | None = None,
    resume: bool = False,
    max_retries: int = DEFAULT_MAX_RETRIES,
    adaptive_pacing: bool = True,
    pacing_threshold_1: float = DEFAULT_PACING_THRESHOLD_1,
    pacing_threshold_2: float = DEFAULT_PACING_THRESHOLD_2,
    pacing_threshold_3: float = DEFAULT_PACING_THRESHOLD_3,
    five_hour_limit: int | None = None,
    reeval_interval: int = DEFAULT_REEVAL_INTERVAL,
    no_reeval: bool = False,
    reeval_confirm: bool = True,
    reeval_dry_run: bool = False,
    two_phase: bool = False,
    planning_tool: Tool = Tool.CLAUDE,
    coding_tool: Tool = Tool.OPENCODE,
    planning_timeout: float | None = DEFAULT_PLANNING_TIMEOUT,
    coding_timeout: float | None = DEFAULT_CODING_TIMEOUT,
    ccs_profile: str | None = None,
    ccs_args: str | None = None,
) -> bool | int:
    """
    Convenience function to run Ralph.

    Args:
        prd_path: Path to the PRD file
        tool: Which AI tool to use (ignored when two_phase=True)
        max_iterations: Maximum number of iterations
        verbose: Enable verbose output
        watch_tasks: Show Claude task list during execution
        timeout: Per-iteration timeout in seconds (None = no timeout)
        log_dir: Directory to save iteration logs (None = no logging)
        resume: Resume from previous run state
        max_retries: Per-iteration retry attempts for transient failures
        adaptive_pacing: Enable adaptive pacing based on usage
        pacing_threshold_1: Usage % for 2x delay (default: 70)
        pacing_threshold_2: Usage % for 4x delay (default: 80)
        pacing_threshold_3: Usage % for 8x delay (default: 90)
        five_hour_limit: Override 5-hour token limit (None = use plan limit)
        reeval_interval: Run PRD re-evaluation every N iterations (0 to disable)
        no_reeval: Disable PRD re-evaluation entirely
        reeval_confirm: Require user confirmation before applying re-evaluation changes
        reeval_dry_run: Show what changes would be made without applying them
        two_phase: Enable two-phase orchestration (planning + coding)
        planning_tool: Tool for planning phase (default: Claude)
        coding_tool: Tool for coding phase (default: OpenCode)
        planning_timeout: Planning phase timeout in seconds
        coding_timeout: Coding phase timeout in seconds

    Returns:
        True if all stories were completed
        False if max iterations reached
        130 if interrupted by SIGINT
        143 if interrupted by SIGTERM
    """
    runner = Runner(
        prd_path=prd_path,
        tool=tool,
        max_iterations=max_iterations,
        verbose=verbose,
        watch_tasks=watch_tasks,
        timeout=timeout,
        log_dir=log_dir,
        resume=resume,
        max_retries=max_retries,
        adaptive_pacing=adaptive_pacing,
        pacing_threshold_1=pacing_threshold_1,
        pacing_threshold_2=pacing_threshold_2,
        pacing_threshold_3=pacing_threshold_3,
        five_hour_limit=five_hour_limit,
        reeval_interval=reeval_interval,
        no_reeval=no_reeval,
        reeval_confirm=reeval_confirm,
        reeval_dry_run=reeval_dry_run,
        two_phase=two_phase,
        planning_tool=planning_tool,
        coding_tool=coding_tool,
        planning_timeout=planning_timeout,
        coding_timeout=coding_timeout,
        ccs_profile=ccs_profile,
        ccs_args=ccs_args,
    )
    return runner.run()
