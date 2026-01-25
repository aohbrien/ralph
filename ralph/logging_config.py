"""Logging configuration for Ralph."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Default format for debug logs
DEBUG_FORMAT = "%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s"
DEBUG_DATE_FORMAT = "%H:%M:%S"


def setup_logging(
    debug: bool = False,
    log_file: Path | None = None,
) -> None:
    """
    Configure logging for Ralph.

    Args:
        debug: If True, enable DEBUG level logging
        log_file: If provided, write logs to this file
    """
    # Get root logger for ralph namespace
    ralph_logger = logging.getLogger("ralph")

    # Clear any existing handlers
    ralph_logger.handlers.clear()

    if debug:
        ralph_logger.setLevel(logging.DEBUG)

        # Create formatter
        formatter = logging.Formatter(DEBUG_FORMAT, datefmt=DEBUG_DATE_FORMAT)

        if log_file:
            # File handler for debug output
            file_handler = logging.FileHandler(log_file, mode="w")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            ralph_logger.addHandler(file_handler)
        else:
            # Console handler if no file specified
            console_handler = logging.StreamHandler(sys.stderr)
            console_handler.setLevel(logging.DEBUG)
            console_handler.setFormatter(formatter)
            ralph_logger.addHandler(console_handler)

        # Propagate to child loggers
        ralph_logger.propagate = False
    else:
        # Disable debug logging
        ralph_logger.setLevel(logging.WARNING)
        ralph_logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a Ralph module."""
    return logging.getLogger(f"ralph.{name}")
