"""Typer CLI for Ralph."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from ralph import __version__
from ralph.archive import manual_archive
from ralph.console import (
    print_archive_info,
    print_dry_run_plan,
    print_error,
    print_info,
    print_init_success,
    print_prd_status,
    print_task_list,
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
        help="AI tool to use: claude or amp",
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
) -> None:
    """Run the Ralph autonomous agent loop."""
    # Validate PRD path
    prd_path = prd.resolve()
    if not prd_path.exists():
        print_error(f"PRD file not found: {prd_path}")
        raise typer.Exit(1)

    # Parse tool
    try:
        tool_enum = Tool(tool.lower())
    except ValueError:
        print_error(f"Invalid tool: {tool}. Must be 'claude' or 'amp'")
        raise typer.Exit(1)

    # Load PRD
    try:
        prd_obj = PRD.from_file(prd_path)
    except Exception as e:
        print_error(f"Failed to parse PRD: {e}")
        raise typer.Exit(1)

    # Dry run mode
    if dry_run:
        print_dry_run_plan(prd_obj, max_iterations)
        raise typer.Exit(0)

    # Convert timeout from minutes to seconds (None for 0 = no timeout)
    timeout_seconds: float | None = timeout * 60 if timeout > 0 else None

    # Resolve log directory (default to logs/ in PRD directory)
    resolved_log_dir = log_dir if log_dir else prd_path.parent / "logs"

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
        help="AI tool to use: claude or amp",
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
) -> None:
    """Resume a previously interrupted run."""
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
        print_error(f"Invalid tool: {tool}. Must be 'claude' or 'amp'")
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
    )

    # Handle return value: True (success), False (max iterations), or int (exit code)
    if result is True:
        raise typer.Exit(0)
    elif result is False:
        raise typer.Exit(1)
    else:
        # result is an int (e.g., 130 for SIGINT)
        raise typer.Exit(result)


if __name__ == "__main__":
    app()
