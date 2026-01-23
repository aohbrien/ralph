"""Archive logic for Ralph runs."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

from ralph.prd import PRD


def get_archive_dir(base_dir: Path) -> Path:
    """Get the archive directory."""
    return base_dir / "archive"


def get_last_branch_file(base_dir: Path) -> Path:
    """Get the path to the .last-branch file."""
    return base_dir / ".last-branch"


def get_progress_file(base_dir: Path) -> Path:
    """Get the path to progress.txt."""
    return base_dir / "progress.txt"


def read_last_branch(base_dir: Path) -> str | None:
    """Read the last branch name from .last-branch file."""
    last_branch_file = get_last_branch_file(base_dir)
    if last_branch_file.exists():
        return last_branch_file.read_text().strip()
    return None


def save_last_branch(base_dir: Path, branch_name: str) -> None:
    """Save the current branch name to .last-branch file."""
    last_branch_file = get_last_branch_file(base_dir)
    last_branch_file.write_text(branch_name + "\n")


def check_branch_changed(base_dir: Path, current_branch: str) -> bool:
    """Check if the branch has changed since last run."""
    last_branch = read_last_branch(base_dir)
    if last_branch is None:
        return False
    return last_branch != current_branch


def archive_previous_run(base_dir: Path, prd_path: Path, last_branch: str) -> tuple[Path | None, list[str]]:
    """
    Archive the previous run's files.

    Args:
        base_dir: The base directory for Ralph
        prd_path: Path to the PRD file
        last_branch: The branch name of the previous run

    Returns:
        Tuple of (archive path, list of archived files)
    """
    archive_dir = get_archive_dir(base_dir)
    progress_file = get_progress_file(base_dir)

    # Create archive folder name: YYYY-MM-DD-feature-name
    date_str = datetime.now().strftime("%Y-%m-%d")
    folder_name = last_branch.replace("ralph/", "")  # Remove ralph/ prefix if present
    archive_folder = archive_dir / f"{date_str}-{folder_name}"

    archived_files: list[str] = []

    # Create archive directory
    archive_folder.mkdir(parents=True, exist_ok=True)

    # Archive PRD file
    if prd_path.exists():
        shutil.copy2(prd_path, archive_folder / prd_path.name)
        archived_files.append(prd_path.name)

    # Archive progress file
    if progress_file.exists():
        shutil.copy2(progress_file, archive_folder / progress_file.name)
        archived_files.append(progress_file.name)

    if not archived_files:
        return None, []

    return archive_folder, archived_files


def reset_progress_file(base_dir: Path) -> None:
    """Reset the progress file for a new run."""
    progress_file = get_progress_file(base_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    progress_file.write_text(f"# Ralph Progress Log\nStarted: {timestamp}\n---\n\n")


def handle_branch_change(base_dir: Path, prd: PRD, prd_path: Path) -> tuple[Path | None, list[str]]:
    """
    Handle branch change detection and archiving.

    Args:
        base_dir: The base directory for Ralph
        prd: The current PRD
        prd_path: Path to the PRD file

    Returns:
        Tuple of (archive path if archived, list of archived files)
    """
    current_branch = prd.branch_name
    last_branch = read_last_branch(base_dir)

    archive_path = None
    archived_files: list[str] = []

    # Check if branch has changed
    if last_branch and last_branch != current_branch:
        # Archive previous run
        archive_path, archived_files = archive_previous_run(base_dir, prd_path, last_branch)

        # Reset progress file
        reset_progress_file(base_dir)

    # Update last branch
    save_last_branch(base_dir, current_branch)

    return archive_path, archived_files


def manual_archive(base_dir: Path, prd_path: Path) -> tuple[Path | None, list[str]]:
    """
    Manually archive the current run.

    Args:
        base_dir: The base directory for Ralph
        prd_path: Path to the PRD file

    Returns:
        Tuple of (archive path, list of archived files)
    """
    # Try to get branch name from PRD
    branch_name = "unknown"
    if prd_path.exists():
        try:
            prd = PRD.from_file(prd_path)
            branch_name = prd.branch_name or "unknown"
        except Exception:
            pass

    return archive_previous_run(base_dir, prd_path, branch_name)
