"""Main orchestration loop for Ralph."""

from __future__ import annotations

import signal
import time
from datetime import datetime
from pathlib import Path
from types import FrameType
from typing import Callable, cast

from ralph.archive import handle_branch_change
from ralph.console import (
    console,
    print_completion,
    print_error,
    print_info,
    print_interrupt,
    print_iteration_header,
    print_max_iterations_reached,
    print_prd_status,
    print_resume_info,
    print_task_list,
    print_timeout,
    print_warning,
)
from ralph.conversation import ConversationWatcher
from ralph.prd import PRD, UserStory
from ralph.process import ManagedProcess, ProcessResult, Tool, run_tool_with_prompt
from ralph.state import RunState, clear_state, load_state, save_state
from ralph.tasks import get_all_tasks_for_project, ClaudeTask


DEFAULT_MAX_ITERATIONS = 10
DEFAULT_ITERATION_DELAY = 2.0  # seconds
DEFAULT_TIMEOUT = 30 * 60  # 30 minutes in seconds


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

    def _get_prompt_path(self) -> Path:
        """Get the prompt file path for the current tool."""
        if self.tool == Tool.CLAUDE:
            return self.base_dir / "CLAUDE.md"
        else:
            return self.base_dir / "prompt.md"

    def _generate_prompt(self, story: UserStory, prd: PRD) -> str:
        """Generate a dynamic prompt for the current iteration."""
        # Read the base prompt file (CLAUDE.md or prompt.md) for project context
        base_context = ""
        if self.prompt_path.exists():
            base_context = self.prompt_path.read_text()

        # Format acceptance criteria as a checklist
        criteria_list = "\n".join(f"- [ ] {c}" for c in story.acceptance_criteria)

        prompt = f"""# Task: Implement User Story {story.id}

## Project Context
{base_context}

---

## Your Assignment

You are working on the **{prd.project}** project. Your task is to implement the following user story:

### {story.id}: {story.title}

**Description:** {story.description}

**Acceptance Criteria:**
{criteria_list}

{f"**Notes:** {story.notes}" if story.notes else ""}

## Instructions

1. **Analyze** the codebase to understand the current implementation
2. **Implement** the changes needed to satisfy ALL acceptance criteria
3. **Test** your changes to ensure they work correctly
4. **Update** the prd.json file to set `"passes": true` for story {story.id} once all criteria are met
5. **Update** progress.txt with a summary of what you did

## Important

- Focus ONLY on this story - do not implement other stories
- Make sure ALL acceptance criteria are satisfied before marking complete
- Run `mypy ralph` to ensure type checking passes if the criteria mention it
- Run `pytest` if tests are part of the acceptance criteria

## Completion Signal

When you have successfully implemented this story AND updated prd.json to mark it as passing, output:

<promise>COMPLETE</promise>

This signals to Ralph that the story is done and it should move to the next iteration.
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
        self._interrupted = True
        # Terminate the subprocess if running
        self._managed_process.terminate()

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
        import sys
        def handler(text: str) -> None:
            # Write to stderr for unbuffered real-time output (like bash's tee /dev/stderr)
            # This avoids buffering issues with stdout and Typer/Rich interference
            sys.stderr.write(text)
            sys.stderr.flush()
        return handler

    def _display_tasks(self) -> None:
        """Display Claude tasks if watch_tasks is enabled."""
        if not self.watch_tasks:
            return

        tasks = get_all_tasks_for_project(self.base_dir)
        if tasks:
            print_task_list(tasks, "Current Claude Tasks")

    def run_iteration(self, iteration: int) -> tuple[ProcessResult, str | None]:
        """
        Run a single iteration.

        Returns:
            Tuple of (ProcessResult from tool execution, story ID or None)
        """
        # Reload PRD to get latest state
        prd = self._load_prd()
        next_story = prd.get_next_story()
        story_id = next_story.id if next_story else None

        # Print iteration header
        print_iteration_header(iteration, self.max_iterations, next_story)

        # Display current tasks if enabled
        self._display_tasks()

        # Check if there's a story to work on
        if next_story is None:
            print_info("No incomplete stories found - all done!")
            return ProcessResult(output="", return_code=0, completed=True), None

        # Generate the dynamic prompt for this story
        prompt = self._generate_prompt(next_story, prd)

        if self.verbose:
            print_info(f"Running {self.tool.value} for story: {next_story.id}")

        # Start conversation watcher for real-time log display
        watcher = ConversationWatcher(self.base_dir)
        watcher.start()

        # Run the tool with the generated prompt
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
            )
        finally:
            watcher.stop()

        console.print()  # Newline after output

        if self.verbose:
            print_info(f"Tool exited with code: {result.return_code}")
            if result.completed:
                print_info("Completion signal detected!")

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

        try:
            return self._run_loop()
        finally:
            self._restore_signal_handlers()

    def _run_loop(self) -> bool | int:
        """Internal run loop, separated for signal handler cleanup."""
        # Load initial PRD
        prd = self._load_prd()

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

            # Check for interrupt before starting iteration
            if self._interrupted:
                prd = self._load_prd()
                completed, total = prd.get_progress()
                print_interrupt(iteration, completed, total)
                return 130  # Standard exit code for SIGINT

            result, story_id = self.run_iteration(iteration)

            # Save iteration log
            self._save_iteration_log(iteration, story_id, result.output)

            # Save run state for resume functionality
            prd = self._load_prd()
            self._save_run_state(iteration, story_id, len(prd.user_stories))

            # Check for interrupt after iteration
            if self._interrupted or result.interrupted:
                prd = self._load_prd()
                completed, total = prd.get_progress()
                print_interrupt(iteration, completed, total)
                return 130  # Standard exit code for SIGINT

            # Check for timeout - log and continue to next iteration
            if result.timed_out:
                print_timeout(iteration)
                # Continue to next iteration rather than stopping
                if iteration < self.max_iterations:
                    if self.verbose:
                        print_info(f"Waiting {self.iteration_delay}s before next iteration...")
                    time.sleep(self.iteration_delay)
                continue

            # Check for completion
            if result.completed:
                print_completion()
                clear_state(self.base_dir)
                return True

            # Reload PRD to check if all stories are done
            prd = self._load_prd()
            if prd.is_complete():
                print_completion()
                clear_state(self.base_dir)
                return True

            # Wait before next iteration (unless this was the last one)
            if iteration < self.max_iterations:
                if self.verbose:
                    print_info(f"Waiting {self.iteration_delay}s before next iteration...")
                time.sleep(self.iteration_delay)

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
) -> bool | int:
    """
    Convenience function to run Ralph.

    Args:
        prd_path: Path to the PRD file
        tool: Which AI tool to use
        max_iterations: Maximum number of iterations
        verbose: Enable verbose output
        watch_tasks: Show Claude task list during execution
        timeout: Per-iteration timeout in seconds (None = no timeout)
        log_dir: Directory to save iteration logs (None = no logging)
        resume: Resume from previous run state

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
    )
    return runner.run()
