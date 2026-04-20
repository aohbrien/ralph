"""Subprocess streaming with real-time output using PTY."""

from __future__ import annotations

import errno
import logging
import os
import pty
import select
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

logger = logging.getLogger("ralph.process")


class Tool(Enum):
    """Supported AI tools."""
    CLAUDE = "claude"
    AMP = "amp"
    OPENCODE = "opencode"
    CCS = "ccs"


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
        """Terminate the subprocess and its entire process group."""
        self._interrupted = True
        if self._process is not None and self._process.poll() is None:
            pid = self._process.pid
            logger.debug(f"Terminating process group for PID={pid}")

            try:
                # Send SIGTERM to the entire process group
                # The process was started with start_new_session=True,
                # so its PID is also the process group ID
                os.killpg(pid, signal.SIGTERM)
                logger.debug(f"Sent SIGTERM to process group {pid}")
            except (ProcessLookupError, PermissionError) as e:
                logger.debug(f"Could not send SIGTERM to process group: {e}")
                # Fall back to terminating just the process
                try:
                    self._process.terminate()
                except ProcessLookupError:
                    pass

            try:
                self._process.wait(timeout=5)
                logger.debug(f"Process {pid} terminated gracefully")
            except subprocess.TimeoutExpired:
                logger.debug(f"Process {pid} did not terminate, sending SIGKILL")
                # Force kill the entire process group
                try:
                    os.killpg(pid, signal.SIGKILL)
                    logger.debug(f"Sent SIGKILL to process group {pid}")
                except (ProcessLookupError, PermissionError):
                    # Fall back to killing just the process
                    try:
                        self._process.kill()
                    except ProcessLookupError:
                        pass
                self._process.wait()
                logger.debug(f"Process {pid} killed")

    def reset(self) -> None:
        """Reset the interrupted state for a new run."""
        self._interrupted = False
        self._process = None


# Global managed process for signal handler access
_current_process: ManagedProcess | None = None


def get_current_process() -> ManagedProcess | None:
    """Get the current managed process, if any."""
    return _current_process


def cleanup_current_process() -> None:
    """
    Clean up the current running process, if any.

    This is called on unexpected exits to ensure Claude doesn't keep running.
    """
    global _current_process
    if _current_process is not None:
        logger.debug("cleanup_current_process: terminating orphaned process")
        _current_process.terminate()
        _current_process = None


# Register atexit handler to clean up on unexpected exits
import atexit
atexit.register(cleanup_current_process)


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

    start_time = time.monotonic()
    timed_out = False

    logger.debug(f"stream_process starting: cmd={cmd}, cwd={cwd}, timeout={timeout}")

    # Create a pseudo-terminal for the subprocess
    leader_fd, follower_fd = pty.openpty()
    logger.debug(f"PTY created: leader_fd={leader_fd}, follower_fd={follower_fd}")

    try:
        logger.debug("Spawning subprocess...")
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE if input_text else None,
            stdout=follower_fd,
            stderr=follower_fd,  # Merge stderr into stdout (like 2>&1)
            cwd=cwd,
            close_fds=True,
            # Start in new process group for clean termination
            start_new_session=True,
        )
        logger.debug(f"Subprocess spawned with PID={process.pid}")

        # Register with managed process if provided
        if managed_process is not None:
            managed_process.process = process
            _current_process = managed_process

        # Close follower in parent - child has it
        os.close(follower_fd)
        logger.debug("Closed follower_fd in parent")

        output_buffer: list[str] = []

        # Write input and close stdin
        if input_text and process.stdin:
            logger.debug(f"Writing {len(input_text)} bytes to stdin")
            try:
                process.stdin.write(input_text.encode())
                process.stdin.close()
                logger.debug("Stdin written and closed")
            except BrokenPipeError:
                logger.debug("BrokenPipeError writing to stdin")
                pass

        # Set leader to non-blocking
        os.set_blocking(leader_fd, False)
        logger.debug("Starting read loop...")

        loop_count = 0
        last_log_time = time.monotonic()
        total_bytes_read = 0
        first_byte_time: float | None = None

        # Read output in real-time
        while True:
            loop_count += 1
            elapsed = time.monotonic() - start_time

            # Log progress every 5 seconds to show we're not hung
            if time.monotonic() - last_log_time >= 5.0:
                # Check process state more thoroughly
                poll_result = process.poll()
                try:
                    # Try to get more info about the process
                    import subprocess as sp
                    ps_result = sp.run(
                        ["ps", "-o", "state=,wchan=", "-p", str(process.pid)],
                        capture_output=True, text=True, timeout=1
                    )
                    ps_info = ps_result.stdout.strip() if ps_result.returncode == 0 else "N/A"
                except Exception:
                    ps_info = "N/A"

                logger.debug(
                    f"Read loop: iteration={loop_count}, elapsed={elapsed:.1f}s, "
                    f"bytes_read={total_bytes_read}, process_poll={poll_result}, "
                    f"ps_state={ps_info}, first_byte={'%.1fs' % first_byte_time if first_byte_time else 'waiting'}"
                )
                last_log_time = time.monotonic()
            # Check if there's data to read
            try:
                readable, _, _ = select.select([leader_fd], [], [], 0.1)
            except (ValueError, OSError):
                break

            if readable:
                try:
                    chunk = os.read(leader_fd, 4096)
                    if chunk:
                        if first_byte_time is None:
                            first_byte_time = time.monotonic() - start_time
                            logger.debug(f"First output received after {first_byte_time:.1f}s")
                        total_bytes_read += len(chunk)
                        text = chunk.decode(errors="replace")
                        output_buffer.append(text)
                        if on_output:
                            on_output(text)
                    else:
                        # EOF
                        logger.debug(f"EOF on leader_fd after {total_bytes_read} bytes")
                        break
                except OSError as e:
                    if e.errno == errno.EIO:
                        # EIO means the follower was closed (process ended)
                        logger.debug(f"EIO on leader_fd (follower closed) after {total_bytes_read} bytes")
                        break
                    raise

            # Check if externally interrupted
            if managed_process is not None and managed_process.interrupted:
                logger.debug("External interrupt detected")
                break

            # Check for timeout
            if timeout is not None and (time.monotonic() - start_time) >= timeout:
                logger.debug(f"Timeout reached after {elapsed:.1f}s")
                timed_out = True
                # Terminate the entire process group
                if process.poll() is None:
                    pid = process.pid
                    logger.debug(f"Terminating process group {pid} due to timeout...")
                    try:
                        os.killpg(pid, signal.SIGTERM)
                        logger.debug(f"Sent SIGTERM to process group {pid}")
                    except (ProcessLookupError, PermissionError) as e:
                        logger.debug(f"Could not SIGTERM process group: {e}")
                        try:
                            process.terminate()
                        except ProcessLookupError:
                            pass
                    try:
                        process.wait(timeout=5)
                        logger.debug("Process terminated gracefully")
                    except subprocess.TimeoutExpired:
                        logger.debug("Process did not terminate, sending SIGKILL...")
                        try:
                            os.killpg(pid, signal.SIGKILL)
                        except (ProcessLookupError, PermissionError):
                            try:
                                process.kill()
                            except ProcessLookupError:
                                pass
                        process.wait()
                        logger.debug("Process killed")
                break

            # Check if process has ended
            if process.poll() is not None:
                logger.debug(f"Process ended with code={process.returncode}, draining buffer...")
                # Do one more read to get any remaining data
                try:
                    while True:
                        chunk = os.read(leader_fd, 4096)
                        if not chunk:
                            break
                        total_bytes_read += len(chunk)
                        text = chunk.decode(errors="replace")
                        output_buffer.append(text)
                        if on_output:
                            on_output(text)
                except OSError:
                    pass
                logger.debug(f"Buffer drained, total_bytes_read={total_bytes_read}")
                break

        # Wait for process to fully terminate
        logger.debug("Waiting for process to fully terminate...")
        process.wait()
        logger.debug(f"Process terminated with returncode={process.returncode}")

    finally:
        # Clean up leader fd
        try:
            os.close(leader_fd)
            logger.debug("Closed leader_fd")
        except OSError:
            pass
        # Clear global reference
        if managed_process is not None:
            _current_process = None

    full_output = "".join(output_buffer)
    was_interrupted = managed_process is not None and managed_process.interrupted
    total_time = time.monotonic() - start_time

    logger.debug(
        f"stream_process complete: returncode={process.returncode}, "
        f"output_len={len(full_output)}, interrupted={was_interrupted}, "
        f"timed_out={timed_out}, total_time={total_time:.1f}s"
    )

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
    return run_tool_with_prompt(
        tool=tool,
        prompt=prompt_text,
        on_output=on_output,
        cwd=cwd,
        managed_process=managed_process,
        timeout=timeout,
    )


# Default extra args appended to ccs invocations. `--dangerously-skip-permissions`
# mirrors Tool.CLAUDE's hardcoded flag; `--print` is claude's own headless mode.
# We deliberately use claude's --print (positional prompt) rather than ccs's
# -p/--prompt delegation mode, which requires a separately configured delegation
# profile. Account instances created via `ccs auth create` work with --print but
# NOT with -p. Because ccs forwards unknown flags to the underlying CLI, this
# works uniformly for account profiles, OAuth providers, and other runtimes.
DEFAULT_CCS_ARGS = "--dangerously-skip-permissions --print"


def run_tool_with_prompt(
    tool: Tool,
    prompt: str,
    on_output: Callable[[str], None] | None = None,
    cwd: Path | None = None,
    managed_process: ManagedProcess | None = None,
    timeout: float | None = None,
    ccs_profile: str | None = None,
    ccs_args: str | None = None,
) -> ProcessResult:
    """
    Run an AI tool with a prompt string.

    Args:
        tool: Which AI tool to use (claude, amp, opencode, or ccs)
        prompt: The prompt text to send to the tool
        on_output: Callback for streaming output
        cwd: Working directory
        managed_process: Optional ManagedProcess for external termination control
        timeout: Optional timeout in seconds (None = no timeout)
        ccs_profile: Optional ccs profile/runtime name (e.g. "personal2"). Ignored unless tool is CCS.
        ccs_args: Optional extra args passed through ccs to the underlying CLI
            (shlex-split). Defaults to DEFAULT_CCS_ARGS when None. Pass "" to opt out.
            Ignored unless tool is CCS.

    Returns:
        ProcessResult with output and completion status
    """
    if tool == Tool.CLAUDE:
        cmd = ["claude", "--dangerously-skip-permissions", "--print"]
        input_text: str | None = prompt
    elif tool == Tool.AMP:
        cmd = ["amp", "--dangerously-allow-all"]
        input_text = prompt
    elif tool == Tool.OPENCODE:
        # OpenCode uses -p flag for prompt, -q for quiet/non-interactive
        cmd = ["opencode", "-p", prompt, "-q"]
        input_text = None  # Don't use stdin
    elif tool == Tool.CCS:
        # ccs syntax: `ccs [profile] [passthrough args...] [prompt]`.
        # Profile selects the account/runtime; passthrough args are forwarded to
        # the underlying CLI. The prompt is passed as a positional arg to claude's
        # --print (included in DEFAULT_CCS_ARGS); we avoid ccs's own -p flag
        # because it requires a delegation profile that account instances don't have.
        extras_raw = DEFAULT_CCS_ARGS if ccs_args is None else ccs_args
        cmd = ["ccs"]
        if ccs_profile:
            cmd.append(ccs_profile)
        if extras_raw:
            cmd.extend(shlex.split(extras_raw))
        cmd.append(prompt)
        input_text = None  # Don't use stdin
    else:
        raise ValueError(f"Unknown tool: {tool}")

    logger.debug(f"run_tool_with_prompt: cmd={cmd[:2]}..., prompt_len={len(prompt)}")

    return stream_process(
        cmd=cmd,
        input_text=input_text,
        on_output=on_output,
        cwd=cwd,
        managed_process=managed_process,
        timeout=timeout,
    )


def default_output_handler(text: str) -> None:
    """Default handler that prints to stdout."""
    print(text, end="", flush=True)
