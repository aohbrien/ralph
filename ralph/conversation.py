"""Watch and display Claude conversation logs in real-time."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, TextIO


def get_claude_project_dir(working_dir: Path) -> Path | None:
    """
    Get the Claude project directory for a given working directory.

    Claude stores conversations in ~/.claude/projects/<encoded-path>/
    where encoded-path is the absolute path with / replaced by -
    """
    claude_base = Path.home() / ".claude" / "projects"
    if not claude_base.exists():
        return None

    # Encode the path: /Users/foo/bar -> -Users-foo-bar
    abs_path = working_dir.resolve()
    encoded = str(abs_path).replace("/", "-")

    project_dir = claude_base / encoded
    if project_dir.exists():
        return project_dir
    return None


def get_latest_conversation(project_dir: Path) -> Path | None:
    """Get the most recently modified .jsonl conversation file."""
    jsonl_files = list(project_dir.glob("*.jsonl"))
    if not jsonl_files:
        return None

    # Sort by modification time, newest first
    jsonl_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return jsonl_files[0]


def format_tool_use(content: dict[str, Any]) -> str:
    """Format a tool_use content block."""
    name = content.get("name", "unknown")
    input_data = content.get("input", {})

    # Compact formatting for common tools
    if name == "Read":
        file_path = input_data.get("file_path", "")
        return f"  → Read: {file_path}"
    elif name == "Write":
        file_path = input_data.get("file_path", "")
        return f"  → Write: {file_path}"
    elif name == "Edit":
        file_path = input_data.get("file_path", "")
        return f"  → Edit: {file_path}"
    elif name == "Bash":
        cmd = input_data.get("command", "")
        # Truncate long commands
        if len(cmd) > 80:
            cmd = cmd[:77] + "..."
        return f"  → Bash: {cmd}"
    elif name == "Glob":
        pattern = input_data.get("pattern", "")
        return f"  → Glob: {pattern}"
    elif name == "Grep":
        pattern = input_data.get("pattern", "")
        return f"  → Grep: {pattern}"
    elif name == "Task":
        desc = input_data.get("description", "")
        return f"  → Task: {desc}"
    else:
        return f"  → {name}"


def format_message(entry: dict[str, Any], output: TextIO = sys.stderr) -> None:
    """Format and print a conversation entry."""
    msg_type = entry.get("type")

    if msg_type == "user":
        message = entry.get("message", {})
        content = message.get("content", "")
        if isinstance(content, str) and content:
            # Truncate very long user messages (like CLAUDE.md prompts)
            if len(content) > 200:
                content = content[:197] + "..."
            print(f"\n\033[36m▶ User:\033[0m {content}", file=output)

    elif msg_type == "assistant":
        message = entry.get("message", {})
        content = message.get("content", [])

        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    block_type = block.get("type")
                    if block_type == "text":
                        text = block.get("text", "")
                        if text:
                            print(f"\n\033[32m◀ Claude:\033[0m {text}", file=output)
                    elif block_type == "tool_use":
                        print(f"\033[33m{format_tool_use(block)}\033[0m", file=output)


class ConversationWatcher:
    """Watch a Claude conversation file and display messages in real-time."""

    def __init__(self, working_dir: Path, output: TextIO = sys.stderr):
        self.working_dir = working_dir
        self.output = output
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._seen_uuids: set[str] = set()
        self._initial_file: Path | None = None

    def _find_conversation_file(self) -> Path | None:
        """Find the active conversation file."""
        project_dir = get_claude_project_dir(self.working_dir)
        if not project_dir:
            return None
        return get_latest_conversation(project_dir)

    def _process_line(self, line: str) -> None:
        """Process a single JSONL line."""
        line = line.strip()
        if not line:
            return

        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            return

        # Skip already seen entries
        uuid = entry.get("uuid")
        if uuid:
            if uuid in self._seen_uuids:
                return
            self._seen_uuids.add(uuid)

        # Format and display
        format_message(entry, self.output)
        self.output.flush()

    def _watch_loop(self) -> None:
        """Main watch loop - runs in a thread."""
        project_dir = get_claude_project_dir(self.working_dir)
        if not project_dir:
            return

        current_file: Path | None = None
        file_pos = 0

        while not self._stop_event.is_set():
            # Check for new/changed conversation file
            latest = get_latest_conversation(project_dir)

            if latest and latest != current_file:
                # New conversation started
                if self._initial_file and latest != self._initial_file:
                    # This is a genuinely new file (not the one that existed before we started)
                    current_file = latest
                    file_pos = 0
                    self._seen_uuids.clear()
                elif not self._initial_file:
                    # First time finding a file
                    self._initial_file = latest
                    current_file = latest
                    # Start from current position (don't replay old messages)
                    file_pos = current_file.stat().st_size

            if current_file and current_file.exists():
                try:
                    with open(current_file, "r") as f:
                        f.seek(file_pos)
                        for line in f:
                            self._process_line(line)
                        file_pos = f.tell()
                except (OSError, IOError):
                    pass

            # Small sleep to avoid busy-waiting
            self._stop_event.wait(0.1)

    def start(self) -> None:
        """Start watching in a background thread."""
        if self._thread is not None:
            return

        # Record what file exists before we start (so we can detect new ones)
        project_dir = get_claude_project_dir(self.working_dir)
        if project_dir:
            self._initial_file = get_latest_conversation(project_dir)

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop watching."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
