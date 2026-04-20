"""Typer CLI for Ralph."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from ralph import __version__
from ralph.archive import manual_archive
from ralph.config import (
    LimitMode,
    Plan,
    get_effective_limit,
    get_plan,
    get_plan_limits,
    set_plan as config_set_plan,
)
from ralph.console import (
    print_archive_info,
    print_dry_run_plan,
    print_error,
    print_info,
    print_init_success,
    print_prd_status,
    print_preflight_blocked,
    print_preflight_ok,
    print_preflight_warning,
    print_success,
    print_task_list,
    print_usage_display,
    print_usage_history,
    print_validation_result,
)
from ralph.prd import PRD, validate_prd_file
from ralph.process import Tool
from ralph.runner import run_ralph
from ralph.state import load_state
from ralph.tasks import get_all_tasks_for_project, get_latest_tasks

app = typer.Typer(
    name="ralph",
    help="Autonomous AI agent loop for PRD-driven development",
    add_completion=False,
)
console = Console()


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"ralph version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-V",
        help="Show version and exit",
        callback=version_callback,
        is_eager=True,
    ),
) -> None:
    """Ralph - Autonomous AI agent loop for PRD-driven development."""
    pass


@app.command()
def run(
    max_iterations: int = typer.Option(
        10,
        "--max-iterations",
        "-n",
        help="Maximum number of iterations",
    ),
    tool: str = typer.Option(
        "claude",
        "--tool",
        "-t",
        help="AI tool to use: claude, amp, opencode, or ccs",
    ),
    prd: Path = typer.Option(
        Path("prd.json"),
        "--prd",
        "-p",
        help="Path to prd.json file",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show plan without executing",
    ),
    watch_tasks: bool = typer.Option(
        False,
        "--watch-tasks",
        help="Show Claude task list during execution",
    ),
    timeout: int = typer.Option(
        30,
        "--timeout",
        help="Per-iteration timeout in minutes (default: 30)",
    ),
    log_dir: Optional[Path] = typer.Option(
        None,
        "--log-dir",
        help="Directory to save iteration logs (default: logs/)",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        "-r",
        help="Resume from last completed iteration",
    ),
    max_retries: int = typer.Option(
        2,
        "--max-retries",
        help="Per-iteration retry attempts for transient failures (default: 2)",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Block run if >90% of 5-hour window budget is used",
    ),
    ignore_limits: bool = typer.Option(
        False,
        "--ignore-limits",
        help="Override usage limit warnings and blocking",
    ),
    no_adaptive_pacing: bool = typer.Option(
        False,
        "--no-adaptive-pacing",
        help="Disable adaptive pacing based on usage",
    ),
    pacing_threshold_1: float = typer.Option(
        70.0,
        "--pacing-t1",
        help="Usage %% threshold for 2x delay (default: 70)",
    ),
    pacing_threshold_2: float = typer.Option(
        80.0,
        "--pacing-t2",
        help="Usage %% threshold for 4x delay (default: 80)",
    ),
    pacing_threshold_3: float = typer.Option(
        90.0,
        "--pacing-t3",
        help="Usage %% threshold for 8x delay (default: 90)",
    ),
    limit_mode: str = typer.Option(
        "p90",
        "--limit-mode",
        help="Limit detection mode: plan, p90, or hybrid",
    ),
    cost_tracking: bool = typer.Option(
        True,
        "--cost-tracking/--no-cost-tracking",
        help="Enable/disable cost tracking display",
    ),
    reeval_interval: int = typer.Option(
        10,
        "--reeval-interval",
        help="Run PRD re-evaluation every N iterations (0 to disable)",
    ),
    no_reeval: bool = typer.Option(
        False,
        "--no-reeval",
        help="Disable PRD re-evaluation entirely",
    ),
    reeval_confirm: bool = typer.Option(
        False,
        "--reeval-confirm/--no-reeval-confirm",
        help="Require user confirmation before applying re-evaluation changes (default: auto-accept)",
    ),
    reeval_dry_run: bool = typer.Option(
        False,
        "--reeval-dry-run",
        help="Show what re-evaluation changes would be made without applying them",
    ),
    debug: Optional[Path] = typer.Option(
        None,
        "--debug",
        help="Enable debug logging to specified file (e.g., --debug output.log)",
    ),
    two_phase: bool = typer.Option(
        False,
        "--two-phase",
        help="Enable two-phase orchestration (planning + coding)",
    ),
    planning_tool: str = typer.Option(
        "claude",
        "--planning-tool",
        help="Tool for planning phase: claude, amp, opencode, or ccs",
    ),
    coding_tool: str = typer.Option(
        "opencode",
        "--coding-tool",
        help="Tool for coding phase: claude, amp, opencode, or ccs",
    ),
    planning_timeout: int = typer.Option(
        10,
        "--planning-timeout",
        help="Planning phase timeout in minutes (default: 10)",
    ),
    coding_timeout: int = typer.Option(
        30,
        "--coding-timeout",
        help="Coding phase timeout in minutes (default: 30)",
    ),
    ccs_profile: Optional[str] = typer.Option(
        None,
        "--ccs-profile",
        help="ccs profile/runtime name (e.g. 'personal2'). Only used when tool is ccs.",
    ),
    ccs_args: Optional[str] = typer.Option(
        None,
        "--ccs-args",
        help=(
            "Extra args passed through ccs to the underlying CLI. "
            "Defaults to '--dangerously-skip-permissions --print' "
            "(claude headless mode); pass '' to opt out. "
            "Only used when tool is ccs."
        ),
    ),
) -> None:
    """Run the Ralph autonomous agent loop."""
    import logging
    import time
    from ralph.logging_config import setup_logging
    from ralph.usage import check_usage_before_run

    # Configure debug logging if requested
    setup_logging(debug=debug is not None, log_file=debug)
    logger = logging.getLogger("ralph.cli")

    startup_start = time.monotonic()
    logger.debug("=== Ralph startup beginning ===")

    # Validate PRD path
    logger.debug(f"Resolving PRD path: {prd}")
    prd_path = prd.resolve()
    if not prd_path.exists():
        print_error(f"PRD file not found: {prd_path}")
        raise typer.Exit(1)
    logger.debug(f"PRD path resolved: {prd_path}")

    # Parse tool
    logger.debug(f"Parsing tool: {tool}")
    try:
        tool_enum = Tool(tool.lower())
    except ValueError:
        print_error(f"Invalid tool: {tool}. Must be 'claude', 'amp', 'opencode', or 'ccs'")
        raise typer.Exit(1)

    # Parse two-phase tools if enabled
    planning_tool_enum = Tool.CLAUDE
    coding_tool_enum = Tool.OPENCODE
    if two_phase:
        try:
            planning_tool_enum = Tool(planning_tool.lower())
        except ValueError:
            print_error(f"Invalid planning tool: {planning_tool}. Must be 'claude', 'amp', 'opencode', or 'ccs'")
            raise typer.Exit(1)
        try:
            coding_tool_enum = Tool(coding_tool.lower())
        except ValueError:
            print_error(f"Invalid coding tool: {coding_tool}. Must be 'claude', 'amp', 'opencode', or 'ccs'")
            raise typer.Exit(1)

    # Load PRD
    logger.debug("Loading PRD file...")
    prd_load_start = time.monotonic()
    try:
        prd_obj = PRD.from_file(prd_path)
    except Exception as e:
        print_error(f"Failed to parse PRD: {e}")
        raise typer.Exit(1)
    logger.debug(f"PRD loaded in {time.monotonic() - prd_load_start:.3f}s")

    # Parse limit mode
    logger.debug(f"Parsing limit mode: {limit_mode}")
    try:
        limit_mode_enum = LimitMode(limit_mode.lower())
    except ValueError:
        print_error(f"Invalid limit mode: {limit_mode}. Must be 'plan', 'p90', or 'hybrid'")
        raise typer.Exit(1)

    # Get plan limits based on limit mode
    logger.debug("Getting plan limits...")
    plan_start = time.monotonic()
    current_plan = get_plan()
    logger.debug(f"Current plan: {current_plan}")
    limits = get_effective_limit(plan=current_plan, limit_mode=limit_mode_enum)
    logger.debug(f"Plan limits retrieved in {time.monotonic() - plan_start:.3f}s: {limits}")

    # Pre-flight usage check (unless ignored)
    if not ignore_limits:
        logger.debug("Running preflight usage check...")
        preflight_start = time.monotonic()
        preflight = check_usage_before_run(five_hour_limit=limits["5hour_tokens"])
        logger.debug(f"Preflight check completed in {time.monotonic() - preflight_start:.3f}s")

        if preflight.should_block and strict:
            # Block the run if >90% used and --strict is enabled
            print_preflight_blocked(
                percentage=preflight.percentage,
                tokens_remaining=preflight.tokens_remaining,
                estimated_iterations=preflight.estimated_iterations_remaining,
            )
            raise typer.Exit(1)
        elif preflight.should_warn:
            # Warn if >70% used (but don't block without --strict)
            print_preflight_warning(
                percentage=preflight.percentage,
                tokens_remaining=preflight.tokens_remaining,
                estimated_iterations=preflight.estimated_iterations_remaining,
            )
        elif verbose:
            # Show pre-flight status in verbose mode
            print_preflight_ok(
                percentage=preflight.percentage,
                tokens_remaining=preflight.tokens_remaining,
                estimated_iterations=preflight.estimated_iterations_remaining,
            )

    # Dry run mode
    if dry_run:
        print_dry_run_plan(prd_obj, max_iterations)
        raise typer.Exit(0)

    # Check for existing state and prompt user if not resuming
    if not resume:
        logger.debug("Checking for existing state...")
        state_start = time.monotonic()
        existing_state = load_state(prd_path.parent)
        logger.debug(f"State check completed in {time.monotonic() - state_start:.3f}s")
        if existing_state:
            print_info(
                f"Found previous run state (iteration {existing_state.last_iteration}, "
                f"story {existing_state.last_story_id or 'N/A'})"
            )
            should_resume = typer.confirm(
                "Resume from previous run?",
                default=True,
            )
            if should_resume:
                resume = True
            else:
                # Clear the state file to start fresh
                from ralph.state import clear_state
                clear_state(prd_path.parent)
                print_info("Cleared previous state, starting fresh")

    # Convert timeout from minutes to seconds (None for 0 = no timeout)
    timeout_seconds: float | None = timeout * 60 if timeout > 0 else None

    # Resolve log directory (default to logs/ in PRD directory)
    resolved_log_dir = log_dir if log_dir else prd_path.parent / "logs"

    # Get plan limit for adaptive pacing (always use the limit, even if ignore_limits)
    five_hour_limit_value = limits["5hour_tokens"]

    logger.debug(f"=== Startup completed in {time.monotonic() - startup_start:.3f}s ===")
    logger.debug(f"Starting run_ralph with max_iterations={max_iterations}, timeout={timeout_seconds}s")

    # Run the main loop
    result = run_ralph(
        prd_path=prd_path,
        tool=tool_enum,
        max_iterations=max_iterations,
        verbose=verbose,
        watch_tasks=watch_tasks,
        timeout=timeout_seconds,
        log_dir=resolved_log_dir,
        resume=resume,
        max_retries=max_retries,
        adaptive_pacing=not no_adaptive_pacing,
        pacing_threshold_1=pacing_threshold_1,
        pacing_threshold_2=pacing_threshold_2,
        pacing_threshold_3=pacing_threshold_3,
        five_hour_limit=five_hour_limit_value,
        reeval_interval=reeval_interval,
        no_reeval=no_reeval,
        reeval_confirm=reeval_confirm,
        reeval_dry_run=reeval_dry_run,
        two_phase=two_phase,
        planning_tool=planning_tool_enum,
        coding_tool=coding_tool_enum,
        planning_timeout=planning_timeout * 60 if planning_timeout > 0 else None,
        coding_timeout=coding_timeout * 60 if coding_timeout > 0 else None,
        ccs_profile=ccs_profile,
        ccs_args=ccs_args,
    )

    # Handle return value: True (success), False (max iterations), or int (exit code)
    if result is True:
        raise typer.Exit(0)
    elif result is False:
        raise typer.Exit(1)
    else:
        # result is an int (e.g., 130 for SIGINT)
        raise typer.Exit(result)


@app.command()
def status(
    prd: Path = typer.Option(
        Path("prd.json"),
        "--prd",
        "-p",
        help="Path to prd.json file",
    ),
) -> None:
    """Show PRD progress status."""
    prd_path = prd.resolve()
    if not prd_path.exists():
        print_error(f"PRD file not found: {prd_path}")
        raise typer.Exit(1)

    try:
        prd_obj = PRD.from_file(prd_path)
        print_prd_status(prd_obj)
    except Exception as e:
        print_error(f"Failed to parse PRD: {e}")
        raise typer.Exit(1)


@app.command()
def archive(
    prd: Path = typer.Option(
        Path("prd.json"),
        "--prd",
        "-p",
        help="Path to prd.json file",
    ),
) -> None:
    """Archive the current run."""
    prd_path = prd.resolve()
    base_dir = prd_path.parent

    archive_path, archived_files = manual_archive(base_dir, prd_path)

    if archive_path:
        print_archive_info(str(archive_path), archived_files)
    else:
        print_error("No files to archive")
        raise typer.Exit(1)


@app.command()
def validate(
    prd: Path = typer.Option(
        Path("prd.json"),
        "--prd",
        "-p",
        help="Path to prd.json file",
    ),
) -> None:
    """Validate a PRD file structure."""
    prd_path = prd.resolve()

    errors, warnings = validate_prd_file(prd_path)
    print_validation_result(str(prd_path), errors, warnings)

    if errors:
        raise typer.Exit(1)
    raise typer.Exit(0)


@app.command()
def init(
    project_name: Optional[str] = typer.Option(
        None,
        "--project",
        "-p",
        help="Project name",
    ),
    branch_name: Optional[str] = typer.Option(
        None,
        "--branch",
        "-b",
        help="Branch name for the PRD work",
    ),
    output_dir: Path = typer.Option(
        Path("."),
        "--output",
        "-o",
        help="Output directory (default: current directory)",
    ),
) -> None:
    """Initialize a new PRD project with scaffolded files."""
    resolved_dir = output_dir.resolve()

    # Prompt for project name if not provided
    if not project_name:
        project_name = typer.prompt("Project name")

    # Prompt for branch name if not provided
    if not branch_name:
        branch_name = typer.prompt(
            "Branch name",
            default=f"ralph/{project_name.lower().replace(' ', '-')}",
        )

    # Check if prd.json already exists
    prd_path = resolved_dir / "prd.json"
    if prd_path.exists():
        print_error(f"prd.json already exists at {prd_path}")
        raise typer.Exit(1)

    # Create the output directory if it doesn't exist
    resolved_dir.mkdir(parents=True, exist_ok=True)

    # Create prd.json with example structure
    prd_content = {
        "project": project_name,
        "branchName": branch_name,
        "description": f"User stories for {project_name}",
        "userStories": [
            {
                "id": "US-001",
                "title": "Example user story",
                "description": "As a user, I want to [action] so that [benefit].",
                "acceptanceCriteria": [
                    "Acceptance criterion 1",
                    "Acceptance criterion 2",
                    "Typecheck passes",
                ],
                "priority": 1,
                "passes": False,
                "notes": "Add implementation notes here",
            }
        ],
    }

    import json
    prd_path.write_text(json.dumps(prd_content, indent=2) + "\n")

    files_created = ["prd.json"]

    # Create CLAUDE.md if missing
    claude_md_path = resolved_dir / "CLAUDE.md"
    if not claude_md_path.exists():
        claude_md_content = f"""# CLAUDE.md

Instructions for Claude Code when working on {project_name}.

## Project Overview

[Describe what this project does]

## Commands

```bash
ralph run                    # Run the agent loop
ralph status                 # Show PRD progress
ralph validate               # Validate prd.json
```

## Ralph Agent Instructions

When running autonomously as part of the Ralph loop:

1. Read `prd.json` in this directory
2. Read `progress.txt` for learnings from previous iterations
3. Pick the **highest priority** user story where `passes: false`
4. Implement that single user story
5. Run any required checks (tests, linting, type checking)
6. If checks pass, commit with: `feat: [Story ID] - [Story Title]`
7. Update `prd.json` to set `passes: true` for the completed story
8. Append progress to `progress.txt`

## Progress Report Format

APPEND to progress.txt (never replace):
```
## [Date/Time] - [Story ID]
- What was implemented
- Files changed
- **Learnings:**
  - Patterns discovered
  - Gotchas encountered
---
```

## Stop Condition

After completing a story, check if ALL stories have `passes: true`.

If ALL complete, output:
```
<promise>COMPLETE</promise>
```

If stories remain with `passes: false`, end normally (next iteration continues).

## Important

- Work on ONE story per iteration
- Keep changes minimal and focused
- Follow existing code patterns
"""
        claude_md_path.write_text(claude_md_content)
        files_created.append("CLAUDE.md")

    # Create prompt.md if missing
    prompt_md_path = resolved_dir / "prompt.md"
    if not prompt_md_path.exists():
        prompt_md_content = f"""# {project_name}

## Overview

[Describe the project and its goals]

## Requirements

[List high-level requirements]

## Technical Details

[Add any technical constraints or preferences]
"""
        prompt_md_path.write_text(prompt_md_content)
        files_created.append("prompt.md")

    print_init_success(str(resolved_dir), files_created)


@app.command()
def tasks(
    project: Optional[Path] = typer.Option(
        None,
        "--project",
        "-p",
        help="Project path (defaults to current directory)",
    ),
    all_sources: bool = typer.Option(
        False,
        "--all",
        "-a",
        help="Search all sources including session files",
    ),
) -> None:
    """Show Claude task list."""
    project_path = (project or Path.cwd()).resolve()

    if all_sources:
        task_list = get_all_tasks_for_project(project_path)
    else:
        task_list = get_latest_tasks()

    if not task_list:
        console.print("[dim]No tasks found[/dim]")
        raise typer.Exit(0)

    print_task_list(task_list)


@app.command()
def resume(
    max_iterations: int = typer.Option(
        10,
        "--max-iterations",
        "-n",
        help="Maximum number of iterations",
    ),
    tool: str = typer.Option(
        "claude",
        "--tool",
        "-t",
        help="AI tool to use: claude, amp, opencode, or ccs",
    ),
    prd: Path = typer.Option(
        Path("prd.json"),
        "--prd",
        "-p",
        help="Path to prd.json file",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
    watch_tasks: bool = typer.Option(
        False,
        "--watch-tasks",
        help="Show Claude task list during execution",
    ),
    timeout: int = typer.Option(
        30,
        "--timeout",
        help="Per-iteration timeout in minutes (default: 30)",
    ),
    log_dir: Optional[Path] = typer.Option(
        None,
        "--log-dir",
        help="Directory to save iteration logs (default: logs/)",
    ),
    max_retries: int = typer.Option(
        2,
        "--max-retries",
        help="Per-iteration retry attempts for transient failures (default: 2)",
    ),
    no_adaptive_pacing: bool = typer.Option(
        False,
        "--no-adaptive-pacing",
        help="Disable adaptive pacing based on usage",
    ),
    pacing_threshold_1: float = typer.Option(
        70.0,
        "--pacing-t1",
        help="Usage %% threshold for 2x delay (default: 70)",
    ),
    pacing_threshold_2: float = typer.Option(
        80.0,
        "--pacing-t2",
        help="Usage %% threshold for 4x delay (default: 80)",
    ),
    pacing_threshold_3: float = typer.Option(
        90.0,
        "--pacing-t3",
        help="Usage %% threshold for 8x delay (default: 90)",
    ),
    limit_mode: str = typer.Option(
        "p90",
        "--limit-mode",
        help="Limit detection mode: plan, p90, or hybrid",
    ),
    reeval_interval: int = typer.Option(
        10,
        "--reeval-interval",
        help="Run PRD re-evaluation every N iterations (0 to disable)",
    ),
    no_reeval: bool = typer.Option(
        False,
        "--no-reeval",
        help="Disable PRD re-evaluation entirely",
    ),
    reeval_confirm: bool = typer.Option(
        False,
        "--reeval-confirm/--no-reeval-confirm",
        help="Require user confirmation before applying re-evaluation changes (default: auto-accept)",
    ),
    reeval_dry_run: bool = typer.Option(
        False,
        "--reeval-dry-run",
        help="Show what re-evaluation changes would be made without applying them",
    ),
    debug: Optional[Path] = typer.Option(
        None,
        "--debug",
        help="Enable debug logging to specified file (e.g., --debug output.log)",
    ),
    two_phase: bool = typer.Option(
        False,
        "--two-phase",
        help="Enable two-phase orchestration (planning + coding)",
    ),
    planning_tool: str = typer.Option(
        "claude",
        "--planning-tool",
        help="Tool for planning phase: claude, amp, opencode, or ccs",
    ),
    coding_tool: str = typer.Option(
        "opencode",
        "--coding-tool",
        help="Tool for coding phase: claude, amp, opencode, or ccs",
    ),
    planning_timeout: int = typer.Option(
        10,
        "--planning-timeout",
        help="Planning phase timeout in minutes (default: 10)",
    ),
    coding_timeout: int = typer.Option(
        30,
        "--coding-timeout",
        help="Coding phase timeout in minutes (default: 30)",
    ),
    ccs_profile: Optional[str] = typer.Option(
        None,
        "--ccs-profile",
        help="ccs profile/runtime name (e.g. 'personal2'). Only used when tool is ccs.",
    ),
    ccs_args: Optional[str] = typer.Option(
        None,
        "--ccs-args",
        help=(
            "Extra args passed through ccs to the underlying CLI. "
            "Defaults to '--dangerously-skip-permissions --print' "
            "(claude headless mode); pass '' to opt out. "
            "Only used when tool is ccs."
        ),
    ),
) -> None:
    """Resume a previously interrupted run."""
    from ralph.logging_config import setup_logging

    # Configure debug logging if requested
    setup_logging(debug=debug is not None, log_file=debug)
    # Validate PRD path
    prd_path = prd.resolve()
    if not prd_path.exists():
        print_error(f"PRD file not found: {prd_path}")
        raise typer.Exit(1)

    # Check if there's a state to resume from
    state = load_state(prd_path.parent)
    if not state:
        print_error("No previous run state found. Use 'ralph run' to start a new run.")
        raise typer.Exit(1)

    # Parse tool
    try:
        tool_enum = Tool(tool.lower())
    except ValueError:
        print_error(f"Invalid tool: {tool}. Must be 'claude', 'amp', 'opencode', or 'ccs'")
        raise typer.Exit(1)

    # Parse two-phase tools if enabled
    planning_tool_enum = Tool.CLAUDE
    coding_tool_enum = Tool.OPENCODE
    if two_phase:
        try:
            planning_tool_enum = Tool(planning_tool.lower())
        except ValueError:
            print_error(f"Invalid planning tool: {planning_tool}. Must be 'claude', 'amp', 'opencode', or 'ccs'")
            raise typer.Exit(1)
        try:
            coding_tool_enum = Tool(coding_tool.lower())
        except ValueError:
            print_error(f"Invalid coding tool: {coding_tool}. Must be 'claude', 'amp', 'opencode', or 'ccs'")
            raise typer.Exit(1)

    # Parse limit mode
    try:
        limit_mode_enum = LimitMode(limit_mode.lower())
    except ValueError:
        print_error(f"Invalid limit mode: {limit_mode}. Must be 'plan', 'p90', or 'hybrid'")
        raise typer.Exit(1)

    # Load PRD
    try:
        prd_obj = PRD.from_file(prd_path)
    except Exception as e:
        print_error(f"Failed to parse PRD: {e}")
        raise typer.Exit(1)

    # Convert timeout from minutes to seconds (None for 0 = no timeout)
    timeout_seconds: float | None = timeout * 60 if timeout > 0 else None

    # Resolve log directory (default to logs/ in PRD directory)
    resolved_log_dir = log_dir if log_dir else prd_path.parent / "logs"

    # Get plan limits based on limit mode
    current_plan = get_plan()
    limits = get_effective_limit(plan=current_plan, limit_mode=limit_mode_enum)
    five_hour_limit_value = limits["5hour_tokens"]

    # Run the main loop with resume=True
    result = run_ralph(
        prd_path=prd_path,
        tool=tool_enum,
        max_iterations=max_iterations,
        verbose=verbose,
        watch_tasks=watch_tasks,
        timeout=timeout_seconds,
        log_dir=resolved_log_dir,
        resume=True,
        max_retries=max_retries,
        adaptive_pacing=not no_adaptive_pacing,
        pacing_threshold_1=pacing_threshold_1,
        pacing_threshold_2=pacing_threshold_2,
        pacing_threshold_3=pacing_threshold_3,
        five_hour_limit=five_hour_limit_value,
        reeval_interval=reeval_interval,
        no_reeval=no_reeval,
        reeval_confirm=reeval_confirm,
        reeval_dry_run=reeval_dry_run,
        two_phase=two_phase,
        planning_tool=planning_tool_enum,
        coding_tool=coding_tool_enum,
        planning_timeout=planning_timeout * 60 if planning_timeout > 0 else None,
        coding_timeout=coding_timeout * 60 if coding_timeout > 0 else None,
        ccs_profile=ccs_profile,
        ccs_args=ccs_args,
    )

    # Handle return value: True (success), False (max iterations), or int (exit code)
    if result is True:
        raise typer.Exit(0)
    elif result is False:
        raise typer.Exit(1)
    else:
        # result is an int (e.g., 130 for SIGINT)
        raise typer.Exit(result)


@app.command()
def story(
    story_id: str = typer.Argument(..., help="Story ID to display (e.g., US-001)"),
    prd: Path = typer.Option(
        Path("prd.json"),
        "--prd",
        "-p",
        help="Path to prd.json file",
    ),
) -> None:
    """Display story details and current status."""
    prd_path = prd.resolve()
    if not prd_path.exists():
        print_error(f"PRD file not found: {prd_path}")
        raise typer.Exit(1)

    try:
        prd_obj = PRD.from_file(prd_path)
    except Exception as e:
        print_error(f"Failed to parse PRD: {e}")
        raise typer.Exit(1)

    story_obj = prd_obj.get_story_by_id(story_id)
    if not story_obj:
        print_error(f"Story not found: {story_id}")
        available_ids = [s.id for s in prd_obj.user_stories]
        console.print(f"[dim]Available stories: {', '.join(available_ids)}[/dim]")
        raise typer.Exit(1)

    # Display story details
    status = "[green]PASS[/green]" if story_obj.passes else "[yellow]pending[/yellow]"
    console.print()
    console.print(f"[bold cyan]{story_obj.id}[/bold cyan]: {story_obj.title}")
    console.print(f"Status: {status}")
    console.print(f"Priority: {story_obj.priority}")
    console.print()
    console.print("[bold]Description:[/bold]")
    console.print(f"  {story_obj.description}")
    console.print()
    console.print("[bold]Acceptance Criteria:[/bold]")
    for criterion in story_obj.acceptance_criteria:
        check = "[green]✓[/green]" if story_obj.passes else "[dim]○[/dim]"
        console.print(f"  {check} {criterion}")
    if story_obj.notes:
        console.print()
        console.print("[bold]Notes:[/bold]")
        console.print(f"  {story_obj.notes}")


@app.command(name="mark-complete")
def mark_complete(
    story_id: str = typer.Argument(..., help="Story ID to mark complete (e.g., US-001)"),
    prd: Path = typer.Option(
        Path("prd.json"),
        "--prd",
        "-p",
        help="Path to prd.json file",
    ),
) -> None:
    """Mark a story as complete (passes=true) in the PRD."""
    prd_path = prd.resolve()
    if not prd_path.exists():
        print_error(f"PRD file not found: {prd_path}")
        raise typer.Exit(1)

    try:
        prd_obj = PRD.from_file(prd_path)
    except Exception as e:
        print_error(f"Failed to parse PRD: {e}")
        raise typer.Exit(1)

    story_obj = prd_obj.get_story_by_id(story_id)
    if not story_obj:
        print_error(f"Story not found: {story_id}")
        available_ids = [s.id for s in prd_obj.user_stories]
        console.print(f"[dim]Available stories: {', '.join(available_ids)}[/dim]")
        raise typer.Exit(1)

    if story_obj.passes:
        print_info(f"Story {story_id} is already marked as complete")
        raise typer.Exit(0)

    # Mark as complete and save
    prd_obj.mark_story_complete(story_id)
    prd_obj.save(prd_path)
    print_success(f"Marked story {story_id} as complete")

    # Show progress
    completed, total = prd_obj.get_progress()
    console.print(f"[dim]Progress: {completed}/{total} stories complete[/dim]")


@app.command()
def usage(
    set_plan: Optional[str] = typer.Option(
        None,
        "--set-plan",
        help="Set your Claude plan type (free, pro, max5x, max20x)",
    ),
    history: Optional[int] = typer.Option(
        None,
        "--history",
        "-h",
        help="Show historical usage over past N days with 5-hour windows",
    ),
    show_p90: bool = typer.Option(
        False,
        "--p90",
        help="Show P90-calculated limit based on usage history",
    ),
    no_costs: bool = typer.Option(
        False,
        "--no-costs",
        help="Hide cost information in display",
    ),
) -> None:
    """Show usage statistics and manage plan configuration."""
    from ralph.usage import (
        get_5hour_window_usage,
        get_historical_5hour_windows,
        get_weekly_window_usage,
        parse_all_sessions,
    )
    from datetime import timedelta, timezone, datetime

    # Handle --set-plan option
    if set_plan is not None:
        try:
            plan_enum = Plan.from_string(set_plan)
            config_set_plan(plan_enum)
            print_success(f"Plan set to: {plan_enum.value}")
            raise typer.Exit(0)
        except ValueError as e:
            print_error(str(e))
            raise typer.Exit(1)

    # Get current plan and effective limits (respects limit_mode config)
    current_plan = get_plan()
    limits = get_effective_limit(plan=current_plan)

    # Get P90 limit for display if requested
    p90_limit = None
    if show_p90:
        from ralph.p90 import get_p90_limit
        p90_limit = get_p90_limit()
        if p90_limit is None:
            print_info("Insufficient data for P90 calculation")
        else:
            # Update limits to show P90 in the display
            limits = {"5hour_tokens": p90_limit, "weekly_tokens": p90_limit * 5}

    # Handle --history option
    if history is not None:
        if history < 1:
            print_error("History days must be at least 1")
            raise typer.Exit(1)

        windows = get_historical_5hour_windows(days=history)
        print_usage_history(
            windows=windows,
            five_hour_limit=limits["5hour_tokens"],
            days=history,
            show_costs=not no_costs,
        )
        raise typer.Exit(0)

    # Get usage data
    five_hour_usage = get_5hour_window_usage()
    weekly_usage = get_weekly_window_usage()

    # Calculate tokens by model for the weekly window (larger window for breakdown)
    now = datetime.now(timezone.utc)
    week_ago = now - timedelta(days=7)
    all_records = parse_all_sessions(since=week_ago)

    opus_tokens = sum(r.total_tokens for r in all_records if r.is_opus)
    sonnet_tokens = sum(r.total_tokens for r in all_records if r.is_sonnet)

    # Display the usage information
    print_usage_display(
        plan=current_plan,
        five_hour_usage=five_hour_usage,
        weekly_usage=weekly_usage,
        five_hour_limit=limits["5hour_tokens"],
        weekly_limit=limits["weekly_tokens"],
        opus_tokens=opus_tokens,
        sonnet_tokens=sonnet_tokens,
        show_costs=not no_costs,
        p90_limit=p90_limit,
    )


if __name__ == "__main__":
    app()
