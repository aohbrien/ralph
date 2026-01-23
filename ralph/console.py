"""Rich console output for Ralph."""

from __future__ import annotations

from typing import Callable

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.text import Text

from ralph.prd import PRD, UserStory
from ralph.tasks import ClaudeTask


console = Console()


def print_header(title: str, subtitle: str = "") -> None:
    """Print a styled header panel."""
    content = title
    if subtitle:
        content = f"{title}\n[dim]{subtitle}[/dim]"
    console.print(Panel(content, style="bold blue"))


def print_iteration_header(iteration: int, max_iterations: int, story: UserStory | None = None) -> None:
    """Print the header for a new iteration."""
    title = f"Iteration {iteration}/{max_iterations}"
    if story:
        subtitle = f"Working on: {story.id} - {story.title}"
    else:
        subtitle = "Checking for work..."
    print_header(title, subtitle)


def print_completion() -> None:
    """Print completion message."""
    console.print()
    console.print(Panel(
        "[bold green]All stories completed![/bold green]",
        style="green",
    ))


def print_max_iterations_reached(iteration: int) -> None:
    """Print message when max iterations is reached."""
    console.print()
    console.print(Panel(
        f"[bold yellow]Max iterations ({iteration}) reached without completion[/bold yellow]",
        style="yellow",
    ))


def print_error(message: str) -> None:
    """Print an error message."""
    console.print(f"[bold red]Error:[/bold red] {message}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    console.print(f"[bold yellow]Warning:[/bold yellow] {message}")


def print_info(message: str) -> None:
    """Print an info message."""
    console.print(f"[dim]{message}[/dim]")


def print_prd_status(prd: PRD) -> None:
    """Print PRD status as a table."""
    completed, total = prd.get_progress()

    console.print()
    console.print(Panel(
        f"[bold]{prd.project}[/bold] - {prd.description}\n"
        f"Branch: [cyan]{prd.branch_name}[/cyan]",
        title="PRD Status",
    ))

    table = Table(show_header=True, header_style="bold")
    table.add_column("ID", style="cyan", width=8)
    table.add_column("Priority", justify="center", width=8)
    table.add_column("Status", justify="center", width=8)
    table.add_column("Title")

    for story in sorted(prd.user_stories, key=lambda s: s.priority):
        status = "[green]PASS[/green]" if story.passes else "[dim]pending[/dim]"
        table.add_row(
            story.id,
            str(story.priority),
            status,
            story.title,
        )

    console.print(table)
    console.print()
    console.print(f"Progress: {completed}/{total} stories completed ({completed/total*100:.0f}%)" if total > 0 else "No stories found")


def print_story_details(story: UserStory) -> None:
    """Print detailed story information."""
    console.print()
    console.print(Panel(
        f"[bold]{story.id}[/bold]: {story.title}\n\n"
        f"{story.description}\n\n"
        f"[bold]Acceptance Criteria:[/bold]\n" +
        "\n".join(f"  - {ac}" for ac in story.acceptance_criteria) +
        (f"\n\n[dim]Notes: {story.notes}[/dim]" if story.notes else ""),
        title=f"Story Details (Priority: {story.priority})",
    ))


def print_task_list(tasks: list[ClaudeTask], title: str = "Claude Tasks") -> None:
    """Print Claude task list."""
    if not tasks:
        console.print(f"[dim]No {title.lower()} found[/dim]")
        return

    table = Table(show_header=True, header_style="bold", title=title)
    table.add_column("Status", justify="center", width=10)
    table.add_column("ID", style="cyan", width=12)
    table.add_column("Subject")
    table.add_column("Owner", width=15)

    status_styles = {
        "pending": "[dim]○ pending[/dim]",
        "in_progress": "[yellow]◐ working[/yellow]",
        "completed": "[green]● done[/green]",
    }

    for task in tasks:
        status = status_styles.get(task.status, task.status)
        table.add_row(
            status,
            task.id[:12] if len(task.id) > 12 else task.id,
            task.subject,
            task.owner or "[dim]-[/dim]",
        )

    console.print()
    console.print(table)


def create_output_handler() -> tuple[Callable[[str], None], list[str]]:
    """
    Create an output handler for streaming process output.

    Returns a tuple of (handler function, output buffer).
    """
    buffer: list[str] = []

    def handler(text: str) -> None:
        buffer.append(text)
        console.print(text, end="", highlight=False, markup=False)

    return handler, buffer


def print_dry_run_plan(prd: PRD, max_iterations: int) -> None:
    """Print what would happen in a dry run."""
    console.print()
    console.print(Panel(
        "[bold]Dry Run Mode[/bold] - No changes will be made",
        style="yellow",
    ))

    print_prd_status(prd)

    next_story = prd.get_next_story()
    if next_story:
        console.print()
        console.print("[bold]Next story to work on:[/bold]")
        print_story_details(next_story)
    else:
        console.print()
        console.print("[green]All stories are already complete![/green]")

    console.print()
    console.print(f"Would run up to {max_iterations} iterations")


def print_archive_info(archive_path: str, files_archived: list[str]) -> None:
    """Print information about archived files."""
    console.print()
    console.print(Panel(
        f"Archived to: [cyan]{archive_path}[/cyan]\n"
        f"Files: {', '.join(files_archived)}",
        title="Archive Complete",
        style="green",
    ))


def print_interrupt(iteration: int, completed: int, total: int) -> None:
    """Print message when interrupted by signal."""
    console.print()
    console.print(Panel(
        f"[bold yellow]Interrupted at iteration {iteration}[/bold yellow]\n"
        f"Progress: {completed}/{total} stories completed",
        title="Shutdown",
        style="yellow",
    ))


def print_timeout(iteration: int) -> None:
    """Print message when iteration times out."""
    console.print()
    console.print(Panel(
        f"[bold yellow]Iteration {iteration} timed out[/bold yellow]\n"
        "Moving to next iteration...",
        title="Timeout",
        style="yellow",
    ))


def print_validation_result(
    prd_path: str,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Print PRD validation results."""
    console.print()

    if not errors and not warnings:
        console.print(Panel(
            f"[bold green]PRD is valid[/bold green]\n"
            f"File: [cyan]{prd_path}[/cyan]",
            title="Validation Passed",
            style="green",
        ))
        return

    # Print errors
    if errors:
        error_text = "\n".join(f"  [red]•[/red] {e}" for e in errors)
        console.print(Panel(
            f"[bold red]Errors ({len(errors)}):[/bold red]\n{error_text}",
            title="Validation Failed",
            style="red",
        ))

    # Print warnings
    if warnings:
        warning_text = "\n".join(f"  [yellow]•[/yellow] {w}" for w in warnings)
        console.print(Panel(
            f"[bold yellow]Warnings ({len(warnings)}):[/bold yellow]\n{warning_text}",
            title="Validation Warnings",
            style="yellow",
        ))


def print_init_success(output_dir: str, files_created: list[str]) -> None:
    """Print success message after initializing a new PRD project."""
    files_list = "\n".join(f"  [green]•[/green] {f}" for f in files_created)
    console.print()
    console.print(Panel(
        f"[bold green]Project initialized successfully![/bold green]\n\n"
        f"Directory: [cyan]{output_dir}[/cyan]\n\n"
        f"Files created:\n{files_list}\n\n"
        f"[dim]Next steps:[/dim]\n"
        f"  1. Edit prd.json to add your user stories\n"
        f"  2. Edit CLAUDE.md with project-specific instructions\n"
        f"  3. Run [cyan]ralph validate[/cyan] to check your PRD\n"
        f"  4. Run [cyan]ralph run[/cyan] to start the agent loop",
        title="Ralph Init",
        style="green",
    ))


def print_resume_info(
    last_iteration: int,
    last_story_id: str | None,
    started_at: str,
) -> None:
    """Print information about resuming a previous run."""
    story_info = f"Story: [cyan]{last_story_id}[/cyan]" if last_story_id else "No story in progress"
    console.print()
    console.print(Panel(
        f"[bold]Resuming from previous run[/bold]\n\n"
        f"Last completed iteration: [cyan]{last_iteration}[/cyan]\n"
        f"{story_info}\n"
        f"Started: [dim]{started_at}[/dim]",
        title="Resume",
        style="blue",
    ))
