"""Claude task list parser for reading Claude Code's todo state."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ClaudeTask:
    """A task from Claude Code's task list."""
    id: str
    subject: str
    status: str  # "pending", "in_progress", "completed"
    description: str = ""
    owner: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClaudeTask:
        """Create a ClaudeTask from a dictionary."""
        return cls(
            id=data.get("id", ""),
            subject=data.get("subject", ""),
            status=data.get("status", "pending"),
            description=data.get("description", ""),
            owner=data.get("owner", ""),
        )


def get_claude_dir() -> Path:
    """Get the Claude configuration directory."""
    return Path.home() / ".claude"


def get_project_path_encoded(project_path: Path) -> str:
    """
    Encode a project path for Claude's directory structure.

    Claude encodes absolute paths by replacing '/' with '-'.
    e.g., /Users/foo/myproject -> -Users-foo-myproject
    """
    return str(project_path.absolute()).replace("/", "-")


def get_todos_dir() -> Path:
    """Get the Claude todos directory."""
    return get_claude_dir() / "todos"


def get_project_sessions_dir(project_path: Path) -> Path:
    """Get the Claude sessions directory for a project."""
    encoded = get_project_path_encoded(project_path)
    return get_claude_dir() / "projects" / encoded


def list_todo_files() -> list[Path]:
    """List all todo JSON files."""
    todos_dir = get_todos_dir()
    if not todos_dir.exists():
        return []
    return sorted(todos_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)


def parse_todo_file(path: Path) -> list[ClaudeTask]:
    """Parse a single todo JSON file."""
    try:
        data = json.loads(path.read_text())
        if isinstance(data, list):
            return [ClaudeTask.from_dict(t) for t in data]
        elif isinstance(data, dict) and "tasks" in data:
            return [ClaudeTask.from_dict(t) for t in data["tasks"]]
        return []
    except (json.JSONDecodeError, IOError):
        return []


def get_latest_tasks() -> list[ClaudeTask]:
    """Get tasks from the most recent todo file."""
    files = list_todo_files()
    if not files:
        return []
    return parse_todo_file(files[0])


def get_session_files(project_path: Path) -> list[Path]:
    """Get all session JSONL files for a project."""
    sessions_dir = get_project_sessions_dir(project_path)
    if not sessions_dir.exists():
        return []
    return sorted(sessions_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)


def parse_session_for_tasks(path: Path) -> list[ClaudeTask]:
    """
    Parse a session JSONL file for task-related entries.

    Session files contain one JSON object per line with various message types.
    We look for TaskCreate, TaskUpdate, and TaskList tool uses.
    """
    tasks: dict[str, ClaudeTask] = {}

    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    # Look for tool use results that contain task info
                    if isinstance(entry, dict):
                        # Check for tool_result with task data
                        content = entry.get("content", [])
                        if isinstance(content, list):
                            for item in content:
                                if isinstance(item, dict) and item.get("type") == "tool_result":
                                    result = item.get("content", "")
                                    if isinstance(result, str) and "task" in result.lower():
                                        try:
                                            task_data = json.loads(result)
                                            if isinstance(task_data, dict) and "id" in task_data:
                                                task = ClaudeTask.from_dict(task_data)
                                                tasks[task.id] = task
                                        except json.JSONDecodeError:
                                            pass
                except json.JSONDecodeError:
                    continue
    except IOError:
        pass

    return list(tasks.values())


def get_all_tasks_for_project(project_path: Path) -> list[ClaudeTask]:
    """
    Get all tasks for a project from both todos and session files.

    Combines tasks from:
    1. Todo files in ~/.claude/todos/
    2. Session files in ~/.claude/projects/<encoded-path>/
    """
    tasks: dict[str, ClaudeTask] = {}

    # Get from todo files (most recent first)
    for todo_file in list_todo_files()[:5]:  # Check last 5 todo files
        for task in parse_todo_file(todo_file):
            if task.id and task.id not in tasks:
                tasks[task.id] = task

    # Get from session files
    for session_file in get_session_files(project_path)[:3]:  # Check last 3 sessions
        for task in parse_session_for_tasks(session_file):
            if task.id:
                tasks[task.id] = task  # Session tasks override todo tasks

    return list(tasks.values())


def format_task_status(task: ClaudeTask) -> str:
    """Format a task status for display."""
    status_icons = {
        "pending": "○",
        "in_progress": "◐",
        "completed": "●",
    }
    icon = status_icons.get(task.status, "?")
    return f"{icon} [{task.id}] {task.subject}"
