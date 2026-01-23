"""Subprocess streaming with real-time output using PTY."""

from __future__ import annotations

import errno
import os
import pty
import select
import signal
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable


class Tool(Enum):
    """Supported AI tools."""
    CLAUDE = "claude"
    AMP = "amp"


@dataclass
class ProcessResult:
    """Result from running a subprocess."""
    output: str
    return_code: int
    completed: bool  # True if <promise>COMPLETE</promise> found
    interrupted: bool = False  # True if killed by signal
    timed_out: bool = False  # True if killed by timeout


COMPLETION_SIGNAL = "<promise>COMPLETE</promise>"


class ManagedProcess:
    """
    A managed subprocess that can be terminated externally.

    Used to allow signal handlers to cleanly terminate running processes.
    """

    def __init__(self) -> None:
        self._process: subprocess.Popen[bytes] | None = None
        self._interrupted = False

    @property
    def process(self) -> subprocess.Popen[bytes] | None:
        """Get the current subprocess, if any."""
        return self._process

    @process.setter
    def process(self, proc: subprocess.Popen[bytes] | None) -> None:
        """Set the current subprocess."""
        self._process = proc

    @property
    def interrupted(self) -> bool:
        """Check if the process was interrupted."""
        return self._interrupted

    def terminate(self) -> None:
        """Terminate the subprocess if running."""
        self._interrupted = True
        if self._process is not None and self._process.poll() is None:
            # Try SIGTERM first
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't respond
                self._process.kill()
                self._process.wait()

    def reset(self) -> None:
        """Reset the interrupted state for a new run."""
        self._interrupted = False
        self._process = None


# Global managed process for signal handler access
_current_process: ManagedProcess | None = None


def get_current_process() -> ManagedProcess | None:
    """Get the current managed process, if any."""
    return _current_process


def stream_process(
    cmd: list[str],
    input_text: str | None = None,
    on_output: Callable[[str], None] | None = None,
    cwd: Path | None = None,
    managed_process: ManagedProcess | None = None,
    timeout: float | None = None,
) -> ProcessResult:
    """
    Run a subprocess with real-time streaming output using PTY.

    Uses a pseudo-terminal to get unbuffered output from programs
    that check if they're writing to a TTY.

    Args:
        cmd: Command and arguments to run
        input_text: Optional text to pipe to stdin
        on_output: Callback for each chunk of output (for console display)
        cwd: Working directory for the process
        managed_process: Optional ManagedProcess for external termination control
        timeout: Optional timeout in seconds (None = no timeout)

    Returns:
        ProcessResult with full output, return code, and completion status
    """
    global _current_process

    import time
    start_time = time.monotonic()
    timed_out = False

    # Create a pseudo-terminal for the subprocess
    master_fd, slave_fd = pty.openpty()

    try:
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE if input_text else None,
            stdout=slave_fd,
            stderr=slave_fd,  # Merge stderr into stdout (like 2>&1)
            cwd=cwd,
            close_fds=True,
            # Start in new process group for clean termination
            start_new_session=True,
        )

        # Register with managed process if provided
        if managed_process is not None:
            managed_process.process = process
            _current_process = managed_process

        # Close slave in parent - child has it
        os.close(slave_fd)

        output_buffer: list[str] = []

        # Write input and close stdin
        if input_text and process.stdin:
            try:
                process.stdin.write(input_text.encode())
                process.stdin.close()
            except BrokenPipeError:
                pass

        # Set master to non-blocking
        os.set_blocking(master_fd, False)

        # Read output in real-time
        while True:
            # Check if there's data to read
            try:
                readable, _, _ = select.select([master_fd], [], [], 0.1)
            except (ValueError, OSError):
                break

            if readable:
                try:
                    chunk = os.read(master_fd, 4096)
                    if chunk:
                        text = chunk.decode(errors="replace")
                        output_buffer.append(text)
                        if on_output:
                            on_output(text)
                    else:
                        # EOF
                        break
                except OSError as e:
                    if e.errno == errno.EIO:
                        # EIO means the slave was closed (process ended)
                        break
                    raise

            # Check if externally interrupted
            if managed_process is not None and managed_process.interrupted:
                break

            # Check for timeout
            if timeout is not None and (time.monotonic() - start_time) >= timeout:
                timed_out = True
                # Terminate the process
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                break

            # Check if process has ended
            if process.poll() is not None:
                # Do one more read to get any remaining data
                try:
                    while True:
                        chunk = os.read(master_fd, 4096)
                        if not chunk:
                            break
                        text = chunk.decode(errors="replace")
                        output_buffer.append(text)
                        if on_output:
                            on_output(text)
                except OSError:
                    pass
                break

        # Wait for process to fully terminate
        process.wait()

    finally:
        # Clean up master fd
        try:
            os.close(master_fd)
        except OSError:
            pass
        # Clear global reference
        if managed_process is not None:
            _current_process = None

    full_output = "".join(output_buffer)
    was_interrupted = managed_process is not None and managed_process.interrupted
    return ProcessResult(
        output=full_output,
        return_code=process.returncode or 0,
        completed=COMPLETION_SIGNAL in full_output,
        interrupted=was_interrupted,
        timed_out=timed_out,
    )


def run_tool(
    tool: Tool,
    prompt_path: Path,
    on_output: Callable[[str], None] | None = None,
    cwd: Path | None = None,
    managed_process: ManagedProcess | None = None,
    timeout: float | None = None,
) -> ProcessResult:
    """
    Run an AI tool with a prompt file.

    Args:
        tool: Which AI tool to use (claude or amp)
        prompt_path: Path to the prompt file (CLAUDE.md or prompt.md)
        on_output: Callback for streaming output
        cwd: Working directory
        managed_process: Optional ManagedProcess for external termination control
        timeout: Optional timeout in seconds (None = no timeout)

    Returns:
        ProcessResult with output and completion status
    """
    prompt_text = prompt_path.read_text()

    if tool == Tool.CLAUDE:
        base_cmd = "claude --dangerously-skip-permissions --print"
    else:  # Tool.AMP
        base_cmd = "amp --dangerously-allow-all"

    # Use bash with tee /dev/stderr for real-time streaming display
    # This matches the working bash implementation: command 2>&1 | tee /dev/stderr
    # tee writes to stderr for immediate unbuffered display while stdout captures output
    bash_cmd = f"{base_cmd} 2>&1 | tee /dev/stderr"

    return stream_process(
        cmd=["bash", "-c", bash_cmd],
        input_text=prompt_text,
        on_output=on_output,
        cwd=cwd,
        managed_process=managed_process,
        timeout=timeout,
    )


def default_output_handler(text: str) -> None:
    """Default handler that prints to stdout."""
    print(text, end="", flush=True)
