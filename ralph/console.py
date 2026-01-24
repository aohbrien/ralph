"""Rich console output for Ralph."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Callable

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.text import Text

from ralph.prd import PRD, UserStory
from ralph.tasks import ClaudeTask

if TYPE_CHECKING:
    from ralph.config import Plan
    from ralph.usage import UsageAggregate


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


def print_success(message: str) -> None:
    """Print a success message."""
    console.print(f"[bold green]✓[/bold green] {message}")


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


def print_retry(
    iteration: int,
    attempt: int,
    max_retries: int,
    return_code: int,
    backoff_seconds: float,
) -> None:
    """Print message when retrying a failed iteration."""
    console.print()
    console.print(Panel(
        f"[bold yellow]Iteration {iteration} failed[/bold yellow] (exit code: {return_code})\n"
        f"Retry {attempt}/{max_retries} in {backoff_seconds:.0f}s...",
        title="Retry",
        style="yellow",
    ))


def print_retry_exhausted(iteration: int, max_retries: int, return_code: int) -> None:
    """Print message when all retries are exhausted."""
    console.print()
    console.print(Panel(
        f"[bold yellow]Iteration {iteration} failed after {max_retries} retries[/bold yellow]\n"
        f"Last exit code: {return_code}\n"
        f"Continuing to next iteration...",
        title="Retries Exhausted",
        style="yellow",
    ))


def _get_usage_color(percentage: float) -> str:
    """
    Get the color based on usage percentage.

    Args:
        percentage: Usage percentage (0-100)

    Returns:
        Color name for Rich formatting
    """
    if percentage < 50:
        return "green"
    elif percentage < 80:
        return "yellow"
    else:
        return "red"


def _format_tokens(tokens: int) -> str:
    """Format token count for display."""
    if tokens >= 1_000_000:
        return f"{tokens / 1_000_000:.1f}M"
    elif tokens >= 1_000:
        return f"{tokens / 1_000:.1f}K"
    return str(tokens)


def _format_cost(cost_usd: float) -> str:
    """
    Format cost in USD for display.

    Args:
        cost_usd: Cost in USD

    Returns:
        Formatted cost string (e.g., "$1.23", "$0.05", "<$0.01")
    """
    if cost_usd < 0.01:
        if cost_usd > 0:
            return "<$0.01"
        return "$0.00"
    elif cost_usd >= 100:
        return f"${cost_usd:.0f}"
    elif cost_usd >= 10:
        return f"${cost_usd:.1f}"
    else:
        return f"${cost_usd:.2f}"


def _format_time_remaining(window_end: datetime, now: datetime | None = None) -> str:
    """
    Format the time remaining until window resets.

    Args:
        window_end: End of the current window
        now: Current time (defaults to now in UTC)

    Returns:
        Human-readable time remaining string
    """
    if now is None:
        now = datetime.now(timezone.utc)

    # For rolling windows, the "reset" happens continuously
    # We show when the oldest data in the window will age out
    remaining = window_end - now
    if remaining.total_seconds() <= 0:
        return "now"

    hours = int(remaining.total_seconds() // 3600)
    minutes = int((remaining.total_seconds() % 3600) // 60)

    if hours > 24:
        days = hours // 24
        hours = hours % 24
        return f"{days}d {hours}h"
    elif hours > 0:
        return f"{hours}h {minutes}m"
    else:
        return f"{minutes}m"


def _create_progress_bar(
    percentage: float,
    label: str,
    used: int,
    limit: int,
    window_name: str,
    time_remaining: str,
) -> Panel:
    """
    Create a styled progress bar panel for usage display.

    Args:
        percentage: Usage percentage (0-100)
        label: Label for the progress bar
        used: Tokens used
        limit: Token limit
        window_name: Name of the time window
        time_remaining: Time remaining until window resets

    Returns:
        Rich Panel with progress bar
    """
    color = _get_usage_color(percentage)

    # Create the progress bar characters
    bar_width = 30
    filled = int(bar_width * min(percentage, 100) / 100)
    empty = bar_width - filled

    if color == "green":
        bar = f"[green]{'█' * filled}[/green][dim]{'░' * empty}[/dim]"
    elif color == "yellow":
        bar = f"[yellow]{'█' * filled}[/yellow][dim]{'░' * empty}[/dim]"
    else:
        bar = f"[red]{'█' * filled}[/red][dim]{'░' * empty}[/dim]"

    # Format the display text
    used_str = _format_tokens(used)
    limit_str = _format_tokens(limit)
    remaining_tokens = max(0, limit - used)
    remaining_str = _format_tokens(remaining_tokens)

    content = (
        f"{bar} [{color}]{percentage:.1f}%[/{color}]\n\n"
        f"Used: [{color}]{used_str}[/{color}] / {limit_str} tokens\n"
        f"Remaining: [cyan]{remaining_str}[/cyan] tokens\n"
        f"Window resets: [dim]rolling ({time_remaining} until oldest data ages out)[/dim]"
    )

    return Panel(
        content,
        title=f"[bold]{label}[/bold]",
        subtitle=f"[dim]{window_name}[/dim]",
        style=color,
    )


def print_usage_display(
    plan: "Plan",
    five_hour_usage: "UsageAggregate",
    weekly_usage: "UsageAggregate",
    five_hour_limit: int,
    weekly_limit: int,
    opus_tokens: int,
    sonnet_tokens: int,
    show_costs: bool = True,
    p90_limit: int | None = None,
) -> None:
    """
    Print the complete usage display with progress bars and breakdown.

    Args:
        plan: Current user plan
        five_hour_usage: Usage aggregate for 5-hour window
        weekly_usage: Usage aggregate for 7-day window
        five_hour_limit: Token limit for 5-hour window
        weekly_limit: Token limit for weekly window
        opus_tokens: Total tokens from Opus models
        sonnet_tokens: Total tokens from Sonnet models
        show_costs: Whether to display cost information
        p90_limit: P90-calculated limit (optional, for display)
    """
    console.print()

    # Header with plan info
    header_content = (
        f"[bold]Claude Code Usage[/bold]\n"
        f"Plan: [cyan]{plan.value.upper()}[/cyan]"
    )
    if p90_limit is not None:
        header_content += f"\nP90 Limit: [green]{_format_tokens(p90_limit)}[/green] tokens"
    console.print(Panel(header_content, style="blue"))

    # Calculate percentages
    five_hour_pct = (five_hour_usage.total_tokens / five_hour_limit * 100) if five_hour_limit > 0 else 0
    weekly_pct = (weekly_usage.total_tokens / weekly_limit * 100) if weekly_limit > 0 else 0

    # Time remaining calculations
    now = datetime.now(timezone.utc)
    five_hour_time_remaining = _format_time_remaining(now + timedelta(hours=5), now)
    weekly_time_remaining = _format_time_remaining(now + timedelta(days=7), now)

    # 5-hour window progress bar
    five_hour_panel = _create_progress_bar(
        percentage=five_hour_pct,
        label="5-Hour Window",
        used=five_hour_usage.total_tokens,
        limit=five_hour_limit,
        window_name="Rolling 5-hour window",
        time_remaining=five_hour_time_remaining,
    )
    console.print(five_hour_panel)

    # Weekly window progress bar
    weekly_panel = _create_progress_bar(
        percentage=weekly_pct,
        label="Weekly Window",
        used=weekly_usage.total_tokens,
        limit=weekly_limit,
        window_name="Rolling 7-day window",
        time_remaining=weekly_time_remaining,
    )
    console.print(weekly_panel)

    # Model breakdown table
    total_tokens = opus_tokens + sonnet_tokens
    table = Table(
        title="Token Breakdown by Model",
        show_header=True,
        header_style="bold",
    )
    table.add_column("Model", style="cyan", width=15)
    table.add_column("Tokens", justify="right", width=12)
    table.add_column("Percentage", justify="right", width=12)

    if total_tokens > 0:
        opus_pct = opus_tokens / total_tokens * 100
        sonnet_pct = sonnet_tokens / total_tokens * 100
    else:
        opus_pct = 0
        sonnet_pct = 0

    table.add_row(
        "Opus",
        _format_tokens(opus_tokens),
        f"{opus_pct:.1f}%",
    )
    table.add_row(
        "Sonnet",
        _format_tokens(sonnet_tokens),
        f"{sonnet_pct:.1f}%",
    )
    table.add_row(
        "[bold]Total[/bold]",
        f"[bold]{_format_tokens(total_tokens)}[/bold]",
        "[bold]100%[/bold]",
    )

    console.print()
    console.print(table)

    # Cost breakdown if enabled
    if show_costs:
        five_hour_cost = five_hour_usage.cost_usd
        weekly_cost = weekly_usage.cost_usd

        console.print()
        cost_table = Table(
            title="Cost Breakdown",
            show_header=True,
            header_style="bold",
        )
        cost_table.add_column("Window", style="cyan", width=15)
        cost_table.add_column("Cost (USD)", justify="right", width=12)

        cost_table.add_row(
            "5-Hour",
            _format_cost(five_hour_cost),
        )
        cost_table.add_row(
            "Weekly",
            _format_cost(weekly_cost),
        )
        console.print(cost_table)

    # Additional stats
    console.print()
    console.print(f"[dim]5-hour window: {five_hour_usage.message_count} messages, {five_hour_usage.request_count} requests[/dim]")
    console.print(f"[dim]Weekly window: {weekly_usage.message_count} messages, {weekly_usage.request_count} requests[/dim]")


def print_preflight_warning(
    percentage: float,
    tokens_remaining: int,
    estimated_iterations: int,
) -> None:
    """
    Print a warning that usage is high (>70%) but not blocking.

    Args:
        percentage: Current usage percentage
        tokens_remaining: Tokens remaining in 5-hour window
        estimated_iterations: Estimated iterations remaining
    """
    console.print()
    console.print(Panel(
        f"[bold yellow]High usage detected![/bold yellow]\n\n"
        f"5-hour window usage: [yellow]{percentage:.1f}%[/yellow]\n"
        f"Remaining capacity: [cyan]{_format_tokens(tokens_remaining)}[/cyan] tokens\n"
        f"Estimated iterations remaining: [cyan]~{estimated_iterations}[/cyan]\n\n"
        f"[dim]Consider waiting for the rolling window to reset,[/dim]\n"
        f"[dim]or continue with --ignore-limits to proceed anyway.[/dim]",
        title="Usage Warning",
        style="yellow",
    ))


def print_preflight_blocked(
    percentage: float,
    tokens_remaining: int,
    estimated_iterations: int,
) -> None:
    """
    Print a blocking message that usage is critical (>90%) with --strict flag.

    Args:
        percentage: Current usage percentage
        tokens_remaining: Tokens remaining in 5-hour window
        estimated_iterations: Estimated iterations remaining
    """
    console.print()
    console.print(Panel(
        f"[bold red]Run blocked due to high usage![/bold red]\n\n"
        f"5-hour window usage: [red]{percentage:.1f}%[/red]\n"
        f"Remaining capacity: [cyan]{_format_tokens(tokens_remaining)}[/cyan] tokens\n"
        f"Estimated iterations remaining: [cyan]~{estimated_iterations}[/cyan]\n\n"
        f"With [bold]--strict[/bold] mode enabled, runs are blocked when usage exceeds 90%.\n\n"
        f"[dim]Wait for the rolling 5-hour window to reset, or use[/dim]\n"
        f"[dim]--ignore-limits to override this check.[/dim]",
        title="Run Blocked",
        style="red",
    ))


def print_preflight_ok(
    percentage: float,
    tokens_remaining: int,
    estimated_iterations: int,
) -> None:
    """
    Print the pre-flight check status (non-warning case).

    Args:
        percentage: Current usage percentage
        tokens_remaining: Tokens remaining in 5-hour window
        estimated_iterations: Estimated iterations remaining
    """
    console.print()
    console.print(
        f"[dim]Pre-flight check: {percentage:.1f}% used, "
        f"~{estimated_iterations} iterations remaining "
        f"({_format_tokens(tokens_remaining)} tokens)[/dim]"
    )


def print_pacing_adjustment(
    usage_percentage: float,
    base_delay: float,
    adjusted_delay: float,
    multiplier: float,
) -> None:
    """
    Print a message when pacing is adjusted due to usage.

    Args:
        usage_percentage: Current usage percentage
        base_delay: Base iteration delay in seconds
        adjusted_delay: Adjusted delay in seconds
        multiplier: Delay multiplier applied
    """
    color = _get_usage_color(usage_percentage)
    console.print()
    console.print(Panel(
        f"[bold {color}]Adaptive pacing activated[/bold {color}]\n\n"
        f"Current usage: [{color}]{usage_percentage:.1f}%[/{color}]\n"
        f"Base delay: [dim]{base_delay:.1f}s[/dim]\n"
        f"Adjusted delay: [cyan]{adjusted_delay:.1f}s[/cyan] ({multiplier:.0f}x)\n\n"
        f"[dim]Slowing down to stay within rate limits.[/dim]",
        title="Pacing Adjusted",
        style=color,
    ))


def print_usage_warning(
    percentage: float,
    tokens_used: int,
    tokens_limit: int,
    time_until_reset: str,
) -> None:
    """
    Print a warning banner when usage exceeds 70% threshold.

    Args:
        percentage: Current usage percentage (70-90%)
        tokens_used: Tokens used in the window
        tokens_limit: Token limit for the window
        time_until_reset: Time remaining until oldest data ages out
    """
    remaining = max(0, tokens_limit - tokens_used)
    console.print()
    console.print(Panel(
        f"[bold yellow]High usage warning[/bold yellow]\n\n"
        f"5-hour window: [yellow]{percentage:.1f}%[/yellow] used\n"
        f"Used: [yellow]{_format_tokens(tokens_used)}[/yellow] / {_format_tokens(tokens_limit)}\n"
        f"Remaining: [cyan]{_format_tokens(remaining)}[/cyan]\n"
        f"Window resets: [dim]rolling ({time_until_reset} until oldest data ages out)[/dim]\n\n"
        f"[dim]Consider slowing down to avoid hitting limits.[/dim]",
        title="Usage Warning",
        style="yellow",
    ))


def print_usage_critical(
    percentage: float,
    tokens_used: int,
    tokens_limit: int,
    time_until_reset: str,
) -> None:
    """
    Print a critical warning banner when usage exceeds 90% threshold.

    Args:
        percentage: Current usage percentage (90%+)
        tokens_used: Tokens used in the window
        tokens_limit: Token limit for the window
        time_until_reset: Time remaining until oldest data ages out
    """
    remaining = max(0, tokens_limit - tokens_used)
    console.print()
    console.print(Panel(
        f"[bold red]Critical usage alert![/bold red]\n\n"
        f"5-hour window: [red]{percentage:.1f}%[/red] used\n"
        f"Used: [red]{_format_tokens(tokens_used)}[/red] / {_format_tokens(tokens_limit)}\n"
        f"Remaining: [cyan]{_format_tokens(remaining)}[/cyan]\n"
        f"Window resets: [dim]rolling ({time_until_reset} until oldest data ages out)[/dim]\n\n"
        f"[bold]Approaching rate limit![/bold] Ralph may be throttled or blocked.",
        title="Critical Usage",
        style="red",
    ))


def print_iteration_usage(
    percentage: float,
    tokens_used: int,
    tokens_limit: int,
    cost_usd: float | None = None,
) -> None:
    """
    Print a brief usage summary after each iteration.

    Args:
        percentage: Current usage percentage
        tokens_used: Tokens used in the window
        tokens_limit: Token limit for the window
        cost_usd: Optional cost for the current window
    """
    color = _get_usage_color(percentage)
    remaining = max(0, tokens_limit - tokens_used)
    usage_str = (
        f"[dim]Usage:[/dim] [{color}]{percentage:.1f}%[/{color}] "
        f"({_format_tokens(tokens_used)} / {_format_tokens(tokens_limit)}, "
        f"[cyan]{_format_tokens(remaining)}[/cyan] remaining)"
    )
    if cost_usd is not None:
        usage_str += f" | Cost: [green]{_format_cost(cost_usd)}[/green]"
    console.print(usage_str)


def print_usage_history(
    windows: list["UsageAggregate"],
    five_hour_limit: int,
    days: int,
    show_costs: bool = True,
) -> None:
    """
    Print historical usage data as a table with 5-hour windows.

    Args:
        windows: List of UsageAggregate objects for each 5-hour window
        five_hour_limit: Token limit for 5-hour window (used for highlighting)
        days: Number of days of history being displayed
        show_costs: Whether to show cost column
    """
    console.print()
    console.print(Panel(
        f"[bold]Historical Usage[/bold]\n"
        f"Showing {days} day{'s' if days != 1 else ''} of usage in 5-hour windows",
        style="blue",
    ))

    if not windows:
        console.print("[dim]No usage data found for this period[/dim]")
        return

    # Calculate threshold for highlighting (windows that exceeded typical limits)
    # We'll highlight windows that used more than 80% of the limit
    threshold = int(five_hour_limit * 0.8)

    table = Table(
        show_header=True,
        header_style="bold",
        title="5-Hour Windows",
    )
    table.add_column("Window Start", style="cyan", width=20)
    table.add_column("Window End", width=20)
    table.add_column("Tokens", justify="right", width=12)
    table.add_column("Messages", justify="right", width=10)
    table.add_column("% of Limit", justify="right", width=10)
    if show_costs:
        table.add_column("Cost", justify="right", width=10)
    table.add_column("Status", justify="center", width=10)

    for window in windows:
        # Format timestamps
        start_str = window.window_start.strftime("%Y-%m-%d %H:%M")
        end_str = window.window_end.strftime("%Y-%m-%d %H:%M")

        # Calculate percentage of limit
        percentage = (window.total_tokens / five_hour_limit * 100) if five_hour_limit > 0 else 0

        # Format tokens
        tokens_str = _format_tokens(window.total_tokens)

        # Determine color and status based on usage
        if window.total_tokens >= five_hour_limit:
            color = "red"
            status = "[red]EXCEEDED[/red]"
        elif window.total_tokens >= threshold:
            color = "yellow"
            status = "[yellow]HIGH[/yellow]"
        elif window.total_tokens > 0:
            color = "green"
            status = "[green]OK[/green]"
        else:
            color = "dim"
            status = "[dim]-[/dim]"

        # Apply color to tokens and percentage
        tokens_display = f"[{color}]{tokens_str}[/{color}]"
        pct_display = f"[{color}]{percentage:.1f}%[/{color}]"

        if show_costs:
            cost_display = _format_cost(window.cost_usd)
            table.add_row(
                start_str,
                end_str,
                tokens_display,
                str(window.message_count),
                pct_display,
                cost_display,
                status,
            )
        else:
            table.add_row(
                start_str,
                end_str,
                tokens_display,
                str(window.message_count),
                pct_display,
                status,
            )

    console.print(table)

    # Summary stats
    total_tokens = sum(w.total_tokens for w in windows)
    total_messages = sum(w.message_count for w in windows)
    total_cost = sum(w.cost_usd for w in windows)
    exceeded_count = sum(1 for w in windows if w.total_tokens >= five_hour_limit)
    high_count = sum(1 for w in windows if threshold <= w.total_tokens < five_hour_limit)

    console.print()
    console.print(f"[bold]Summary:[/bold]")
    console.print(f"  Total tokens: {_format_tokens(total_tokens)}")
    console.print(f"  Total messages: {total_messages}")
    if show_costs:
        console.print(f"  Total cost: [green]{_format_cost(total_cost)}[/green]")
    console.print(f"  Windows with high usage (>80%): [yellow]{high_count}[/yellow]")
    console.print(f"  Windows that exceeded limit: [red]{exceeded_count}[/red]")
