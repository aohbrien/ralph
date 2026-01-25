"""Watch and display Claude conversation logs in real-time."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("ralph.conversation")


def get_claude_project_dir(working_dir: Path) -> Path | None:
    """
    Get the Claude project directory for a given working directory.

    Claude stores conversations in ~/.claude/projects/<encoded-path>/
    where encoded-path is the absolute path with / and _ replaced by -
    """
    claude_base = Path.home() / ".claude" / "projects"
    if not claude_base.exists():
        return None

    # Encode the path: /Users/foo/bar_baz -> -Users-foo-bar-baz
    # Claude replaces both / and _ with -
    abs_path = working_dir.resolve()
    encoded = str(abs_path).replace("/", "-").replace("_", "-")

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


def format_message(entry: dict[str, Any]) -> None:
    """Format and print a conversation entry.

    Uses os.write(2, ...) directly to bypass Rich/Typer stderr interception.
    """
    msg_type = entry.get("type")

    def write_line(text: str) -> None:
        """Write directly to fd 2 (stderr) bypassing Python buffering."""
        os.write(2, (text + "\n").encode())

    if msg_type == "user":
        message = entry.get("message", {})
        content = message.get("content", "")
        if isinstance(content, str) and content:
            # Truncate very long user messages (like CLAUDE.md prompts)
            if len(content) > 200:
                content = content[:197] + "..."
            write_line(f"\n\033[36m▶ User:\033[0m {content}")

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
                            write_line(f"\n\033[32m◀ Claude:\033[0m {text}")
                    elif block_type == "tool_use":
                        write_line(f"\033[33m{format_tool_use(block)}\033[0m")


class ConversationWatcher:
    """Watch a Claude conversation file and display messages in real-time."""

    def __init__(self, working_dir: Path):
        self.working_dir = working_dir
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

        msg_type = entry.get("type")
        logger.debug(f"Processing message: type={msg_type}, uuid={uuid[:8] if uuid else 'none'}...")

        # Format and display (writes directly to fd 2)
        format_message(entry)

    def _watch_loop(self) -> None:
        """Main watch loop - runs in a thread."""
        project_dir = get_claude_project_dir(self.working_dir)
        if not project_dir:
            logger.debug("No Claude project directory found")
            return

        logger.debug(f"Watching project dir: {project_dir}")
        current_file: Path | None = None
        file_pos = 0

        while not self._stop_event.is_set():
            # Check for new/changed conversation file
            latest = get_latest_conversation(project_dir)

            if latest and latest != current_file:
                # Conversation file changed or first time seeing it
                if self._initial_file and latest != self._initial_file:
                    # This is a genuinely new file - read from the start
                    logger.debug(f"New conversation file detected: {latest.name}")
                    current_file = latest
                    file_pos = 0
                    self._seen_uuids.clear()
                else:
                    # Either no initial file, or Claude is using the same file
                    # Start from current position (don't replay old messages)
                    logger.debug(f"Using conversation file: {latest.name}, starting at pos {latest.stat().st_size}")
                    current_file = latest
                    file_pos = current_file.stat().st_size

            if current_file and current_file.exists():
                try:
                    current_size = current_file.stat().st_size
                    if current_size > file_pos:
                        logger.debug(f"New data in conversation: {current_size - file_pos} bytes")
                    with open(current_file, "r") as f:
                        f.seek(file_pos)
                        for line in f:
                            self._process_line(line)
                        file_pos = f.tell()
                except (OSError, IOError) as e:
                    logger.debug(f"Error reading conversation file: {e}")

            # Small sleep to avoid busy-waiting
            self._stop_event.wait(0.1)

    def start(self) -> None:
        """Start watching in a background thread."""
        logger.debug(f"ConversationWatcher.start() called for {self.working_dir}")

        if self._thread is not None:
            logger.debug("Watcher thread already running")
            return

        # Record what file exists before we start (so we can detect new ones)
        project_dir = get_claude_project_dir(self.working_dir)
        logger.debug(f"Claude project dir: {project_dir}")

        if project_dir:
            self._initial_file = get_latest_conversation(project_dir)
            logger.debug(f"Initial conversation file: {self._initial_file}")
        else:
            logger.debug("No Claude project dir found - watcher may not work")

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()
        logger.debug("Watcher thread started")

    def stop(self) -> None:
        """Stop watching."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
