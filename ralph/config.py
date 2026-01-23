"""User configuration management for Ralph."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default config directory
DEFAULT_CONFIG_DIR = Path.home() / ".ralph"
DEFAULT_CONFIG_FILE = DEFAULT_CONFIG_DIR / "config.json"


class Plan(str, Enum):
    """Supported Claude plan types."""

    FREE = "free"
    PRO = "pro"
    MAX5X = "max5x"
    MAX20X = "max20x"

    @classmethod
    def from_string(cls, value: str) -> "Plan":
        """
        Parse a plan from a string value.

        Args:
            value: Plan name (case-insensitive)

        Returns:
            Plan enum value

        Raises:
            ValueError: If the plan name is not recognized
        """
        value_lower = value.lower()
        for plan in cls:
            if plan.value == value_lower:
                return plan
        valid_plans = ", ".join(p.value for p in cls)
        raise ValueError(f"Unknown plan: {value}. Valid plans: {valid_plans}")


# Default plan when not configured
DEFAULT_PLAN = Plan.PRO


@dataclass
class Config:
    """User configuration for Ralph."""

    plan: Plan = DEFAULT_PLAN

    def to_dict(self) -> dict[str, Any]:
        """Convert config to a dictionary for JSON serialization."""
        return {"plan": self.plan.value}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        """
        Create a Config from a dictionary.

        Args:
            data: Dictionary with configuration values

        Returns:
            Config instance
        """
        plan = DEFAULT_PLAN
        if "plan" in data:
            try:
                plan = Plan.from_string(data["plan"])
            except ValueError as e:
                logger.warning(f"Invalid plan in config: {e}. Using default: {DEFAULT_PLAN.value}")

        return cls(plan=plan)


def _ensure_config_dir(config_dir: Path) -> None:
    """
    Ensure the config directory exists.

    Args:
        config_dir: Path to the config directory
    """
    config_dir.mkdir(parents=True, exist_ok=True)


def load_config(config_file: Path | None = None) -> Config:
    """
    Load user configuration from file.

    Args:
        config_file: Path to config file. Defaults to ~/.ralph/config.json

    Returns:
        Config instance (defaults if file doesn't exist or is invalid)
    """
    if config_file is None:
        config_file = DEFAULT_CONFIG_FILE

    if not config_file.exists():
        logger.debug(f"Config file not found: {config_file}. Using defaults.")
        return Config()

    try:
        with open(config_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            logger.warning(f"Config file is not a JSON object: {config_file}. Using defaults.")
            return Config()
        return Config.from_dict(data)
    except json.JSONDecodeError as e:
        logger.warning(f"Invalid JSON in config file {config_file}: {e}. Using defaults.")
        return Config()
    except OSError as e:
        logger.warning(f"Could not read config file {config_file}: {e}. Using defaults.")
        return Config()


def save_config(config: Config, config_file: Path | None = None) -> None:
    """
    Save user configuration to file.

    Args:
        config: Config instance to save
        config_file: Path to config file. Defaults to ~/.ralph/config.json

    Raises:
        OSError: If the config file cannot be written
    """
    if config_file is None:
        config_file = DEFAULT_CONFIG_FILE

    _ensure_config_dir(config_file.parent)

    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2)
        f.write("\n")

    logger.debug(f"Saved config to {config_file}")


def get_plan(config_file: Path | None = None) -> Plan:
    """
    Get the configured plan.

    Args:
        config_file: Path to config file. Defaults to ~/.ralph/config.json

    Returns:
        Configured plan or default (pro)
    """
    config = load_config(config_file)
    return config.plan


def set_plan(plan: Plan | str, config_file: Path | None = None) -> None:
    """
    Set the plan in configuration.

    Args:
        plan: Plan enum or string name
        config_file: Path to config file. Defaults to ~/.ralph/config.json

    Raises:
        ValueError: If the plan string is not recognized
        OSError: If the config file cannot be written
    """
    if isinstance(plan, str):
        plan = Plan.from_string(plan)

    config = load_config(config_file)
    config.plan = plan
    save_config(config, config_file)
