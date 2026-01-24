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


class LimitMode(str, Enum):
    """Token limit detection mode."""

    PLAN = "plan"      # Use hardcoded plan limits
    P90 = "p90"        # Auto-detect via P90 calculation
    HYBRID = "hybrid"  # P90 with plan fallback


# Default plan when not configured
DEFAULT_PLAN = Plan.PRO

# Plan rate limits (tokens)
# These are estimates based on publicly available information about Claude plans
# 5-hour window limits (rolling)
# Weekly limits (rolling 7-day window)
PLAN_LIMITS: dict[Plan, dict[str, int]] = {
    Plan.FREE: {
        "5hour_tokens": 30_000,  # Free tier has very limited usage
        "weekly_tokens": 150_000,
    },
    Plan.PRO: {
        "5hour_tokens": 300_000,  # Pro tier baseline
        "weekly_tokens": 1_500_000,
    },
    Plan.MAX5X: {
        "5hour_tokens": 1_500_000,  # 5x Pro limits
        "weekly_tokens": 7_500_000,
    },
    Plan.MAX20X: {
        "5hour_tokens": 6_000_000,  # 20x Pro limits
        "weekly_tokens": 30_000_000,
    },
}


def get_plan_limits(plan: Plan) -> dict[str, int]:
    """
    Get the rate limits for a given plan.

    Args:
        plan: The Plan enum value

    Returns:
        Dictionary with '5hour_tokens' and 'weekly_tokens' limits
    """
    return PLAN_LIMITS.get(plan, PLAN_LIMITS[DEFAULT_PLAN])


@dataclass
class Config:
    """User configuration for Ralph."""

    plan: Plan = DEFAULT_PLAN
    limit_mode: LimitMode = LimitMode.PLAN
    enable_cost_tracking: bool = True
    p90_lookback_days: int = 14

    def to_dict(self) -> dict[str, Any]:
        """Convert config to a dictionary for JSON serialization."""
        return {
            "plan": self.plan.value,
            "limit_mode": self.limit_mode.value,
            "enable_cost_tracking": self.enable_cost_tracking,
            "p90_lookback_days": self.p90_lookback_days,
        }

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
            except (ValueError, AttributeError, TypeError) as e:
                logger.warning(f"Invalid plan in config: {e}. Using default: {DEFAULT_PLAN.value}")

        limit_mode = LimitMode.PLAN
        if "limit_mode" in data:
            try:
                limit_mode = LimitMode(data["limit_mode"])
            except (ValueError, AttributeError, TypeError) as e:
                logger.warning(f"Invalid limit_mode in config: {e}. Using default: plan")

        enable_cost_tracking = True
        if "enable_cost_tracking" in data:
            enable_cost_tracking = bool(data["enable_cost_tracking"])

        p90_lookback_days = 14
        if "p90_lookback_days" in data:
            try:
                p90_lookback_days = int(data["p90_lookback_days"])
            except (ValueError, TypeError):
                pass

        return cls(
            plan=plan,
            limit_mode=limit_mode,
            enable_cost_tracking=enable_cost_tracking,
            p90_lookback_days=p90_lookback_days,
        )


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


def get_limit_mode(config_file: Path | None = None) -> LimitMode:
    """
    Get the configured limit mode.

    Args:
        config_file: Path to config file. Defaults to ~/.ralph/config.json

    Returns:
        Configured limit mode or default (plan)
    """
    config = load_config(config_file)
    return config.limit_mode


def set_limit_mode(mode: LimitMode | str, config_file: Path | None = None) -> None:
    """
    Set the limit mode in configuration.

    Args:
        mode: LimitMode enum or string name
        config_file: Path to config file. Defaults to ~/.ralph/config.json

    Raises:
        ValueError: If the mode string is not recognized
        OSError: If the config file cannot be written
    """
    if isinstance(mode, str):
        mode = LimitMode(mode.lower())

    config = load_config(config_file)
    config.limit_mode = mode
    save_config(config, config_file)


def get_effective_limit(
    plan: Plan | None = None,
    limit_mode: LimitMode | None = None,
    claude_dir: Path | None = None,
) -> dict[str, int]:
    """
    Get effective token limits based on the limit mode.

    Args:
        plan: Plan to use for limits. Defaults to configured plan.
        limit_mode: Limit detection mode. Defaults to configured mode.
        claude_dir: Claude projects directory (for P90 calculation)

    Returns:
        Dictionary with '5hour_tokens' and 'weekly_tokens' limits
    """
    # Load defaults from config if not provided
    if plan is None or limit_mode is None:
        config = load_config()
        if plan is None:
            plan = config.plan
        if limit_mode is None:
            limit_mode = config.limit_mode

    # Get plan-based limits as baseline
    plan_limits = get_plan_limits(plan)

    if limit_mode == LimitMode.PLAN:
        return plan_limits

    # For P90 or HYBRID mode, try to calculate P90
    from ralph.p90 import get_p90_limit

    p90_limit = get_p90_limit(claude_dir=claude_dir)

    if limit_mode == LimitMode.P90:
        if p90_limit is not None:
            # Scale weekly limit proportionally
            five_hour_limit = p90_limit
            # Assume weekly is ~5x the 5-hour limit (based on typical patterns)
            weekly_limit = five_hour_limit * 5
            return {
                "5hour_tokens": five_hour_limit,
                "weekly_tokens": weekly_limit,
            }
        else:
            # No P90 data, fall back to plan limits
            logger.debug("No P90 data available, falling back to plan limits")
            return plan_limits

    # HYBRID mode: use P90 if higher than plan limits
    if p90_limit is not None and p90_limit > plan_limits["5hour_tokens"]:
        five_hour_limit = p90_limit
        weekly_limit = five_hour_limit * 5
        return {
            "5hour_tokens": five_hour_limit,
            "weekly_tokens": weekly_limit,
        }

    return plan_limits
