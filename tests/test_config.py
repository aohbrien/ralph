"""Unit tests for the config.py module.

This module tests all aspects of plan configuration including:
- Saving plan configuration to file
- Loading plan configuration from file
- Default plan behavior when no config exists
- Invalid plan type handling
- Config file permissions and directory creation
- Plan limit lookups for all plan types
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from ralph.config import (
    DEFAULT_CONFIG_DIR,
    DEFAULT_CONFIG_FILE,
    DEFAULT_PLAN,
    PLAN_LIMITS,
    Config,
    LimitMode,
    Plan,
    _ensure_config_dir,
    get_plan,
    get_plan_limits,
    load_config,
    save_config,
    set_plan,
)


# =============================================================================
# Plan Enum Tests
# =============================================================================


class TestPlanEnum:
    """Tests for the Plan enum."""

    def test_plan_values(self):
        """Test that all expected plan values exist."""
        assert Plan.FREE.value == "free"
        assert Plan.PRO.value == "pro"
        assert Plan.MAX5X.value == "max5x"
        assert Plan.MAX20X.value == "max20x"

    def test_from_string_valid_lowercase(self):
        """Test parsing valid plan names in lowercase."""
        assert Plan.from_string("free") == Plan.FREE
        assert Plan.from_string("pro") == Plan.PRO
        assert Plan.from_string("max5x") == Plan.MAX5X
        assert Plan.from_string("max20x") == Plan.MAX20X

    def test_from_string_valid_uppercase(self):
        """Test parsing valid plan names in uppercase."""
        assert Plan.from_string("FREE") == Plan.FREE
        assert Plan.from_string("PRO") == Plan.PRO
        assert Plan.from_string("MAX5X") == Plan.MAX5X
        assert Plan.from_string("MAX20X") == Plan.MAX20X

    def test_from_string_valid_mixed_case(self):
        """Test parsing valid plan names in mixed case."""
        assert Plan.from_string("Free") == Plan.FREE
        assert Plan.from_string("Pro") == Plan.PRO
        assert Plan.from_string("Max5X") == Plan.MAX5X
        assert Plan.from_string("Max20X") == Plan.MAX20X

    def test_from_string_invalid_raises_error(self):
        """Test that invalid plan names raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            Plan.from_string("invalid")

        assert "Unknown plan: invalid" in str(exc_info.value)
        assert "Valid plans:" in str(exc_info.value)

    def test_from_string_invalid_error_message_contains_valid_plans(self):
        """Test that ValueError message lists all valid plans."""
        with pytest.raises(ValueError) as exc_info:
            Plan.from_string("enterprise")

        error_msg = str(exc_info.value)
        for plan in Plan:
            assert plan.value in error_msg

    def test_from_string_empty_string_raises_error(self):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError):
            Plan.from_string("")

    def test_plan_is_str_subclass(self):
        """Test that Plan values can be used as strings."""
        assert isinstance(Plan.PRO.value, str)
        assert Plan.PRO.value == "pro"


# =============================================================================
# Config Dataclass Tests
# =============================================================================


class TestConfigDataclass:
    """Tests for the Config dataclass."""

    def test_default_plan(self):
        """Test that Config has correct default plan."""
        config = Config()
        assert config.plan == DEFAULT_PLAN
        assert config.plan == Plan.PRO

    def test_to_dict(self):
        """Test converting Config to dictionary."""
        config = Config(plan=Plan.MAX5X)
        result = config.to_dict()

        assert result["plan"] == "max5x"
        assert result["limit_mode"] == "plan"
        assert result["enable_cost_tracking"] is True
        assert result["p90_lookback_days"] == 14

    def test_to_dict_with_all_plans(self):
        """Test to_dict with all plan types."""
        for plan in Plan:
            config = Config(plan=plan)
            result = config.to_dict()
            assert result["plan"] == plan.value

    def test_from_dict_valid_plan(self):
        """Test creating Config from valid dictionary."""
        data = {"plan": "max20x"}
        config = Config.from_dict(data)

        assert config.plan == Plan.MAX20X

    def test_from_dict_missing_plan_uses_default(self):
        """Test that missing plan key uses default."""
        data = {}
        config = Config.from_dict(data)

        assert config.plan == DEFAULT_PLAN

    def test_from_dict_invalid_plan_uses_default(self):
        """Test that invalid plan uses default and logs warning."""
        data = {"plan": "invalid_plan"}
        config = Config.from_dict(data)

        assert config.plan == DEFAULT_PLAN

    def test_from_dict_with_extra_fields(self):
        """Test that extra fields are ignored."""
        data = {
            "plan": "pro",
            "extra_field": "ignored",
            "another_field": 123,
        }
        config = Config.from_dict(data)

        assert config.plan == Plan.PRO

    def test_round_trip_conversion(self):
        """Test that to_dict and from_dict are inverse operations."""
        for plan in Plan:
            original = Config(plan=plan)
            data = original.to_dict()
            restored = Config.from_dict(data)
            assert restored.plan == original.plan


# =============================================================================
# Save Config Tests
# =============================================================================


class TestSaveConfig:
    """Tests for saving plan to config file."""

    def test_save_config_creates_file(self):
        """Test that save_config creates the config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.json"
            config = Config(plan=Plan.PRO)

            save_config(config, config_file)

            assert config_file.exists()

    def test_save_config_writes_correct_content(self):
        """Test that save_config writes correct JSON content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.json"
            config = Config(plan=Plan.MAX5X)

            save_config(config, config_file)

            with open(config_file) as f:
                data = json.load(f)

            assert data["plan"] == "max5x"
            assert data["limit_mode"] == "plan"
            assert data["enable_cost_tracking"] is True
            assert data["p90_lookback_days"] == 14

    def test_save_config_all_plan_types(self):
        """Test saving all plan types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for plan in Plan:
                config_file = Path(tmpdir) / f"config_{plan.value}.json"
                config = Config(plan=plan)

                save_config(config, config_file)

                with open(config_file) as f:
                    data = json.load(f)
                assert data["plan"] == plan.value

    def test_save_config_overwrites_existing(self):
        """Test that save_config overwrites existing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.json"

            # Write initial config
            config1 = Config(plan=Plan.FREE)
            save_config(config1, config_file)

            # Overwrite with new config
            config2 = Config(plan=Plan.MAX20X)
            save_config(config2, config_file)

            with open(config_file) as f:
                data = json.load(f)

            assert data["plan"] == "max20x"

    def test_save_config_uses_default_path_when_none(self):
        """Test that save_config uses default path when None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            default_file = Path(tmpdir) / ".ralph" / "config.json"

            with patch("ralph.config.DEFAULT_CONFIG_FILE", default_file):
                config = Config(plan=Plan.PRO)
                save_config(config, None)

                assert default_file.exists()

    def test_save_config_creates_parent_directory(self):
        """Test that save_config creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "nested" / "dir" / "config.json"
            config = Config(plan=Plan.PRO)

            save_config(config, config_file)

            assert config_file.exists()
            assert config_file.parent.exists()

    def test_save_config_json_formatting(self):
        """Test that saved JSON is properly formatted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.json"
            config = Config(plan=Plan.PRO)

            save_config(config, config_file)

            content = config_file.read_text()
            # Should be indented
            assert "  " in content or "\t" in content
            # Should end with newline
            assert content.endswith("\n")


# =============================================================================
# Load Config Tests
# =============================================================================


class TestLoadConfig:
    """Tests for loading plan from config file."""

    def test_load_config_returns_saved_plan(self):
        """Test that load_config returns the saved plan."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.json"

            # Save config first
            config = Config(plan=Plan.MAX5X)
            save_config(config, config_file)

            # Load and verify
            loaded = load_config(config_file)
            assert loaded.plan == Plan.MAX5X

    def test_load_config_all_plan_types(self):
        """Test loading all plan types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for plan in Plan:
                config_file = Path(tmpdir) / f"config_{plan.value}.json"

                # Save and load
                save_config(Config(plan=plan), config_file)
                loaded = load_config(config_file)

                assert loaded.plan == plan

    def test_load_config_nonexistent_file_returns_default(self):
        """Test that non-existent file returns default config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "nonexistent.json"

            loaded = load_config(config_file)

            assert loaded.plan == DEFAULT_PLAN

    def test_load_config_uses_default_path_when_none(self):
        """Test that load_config uses default path when None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            default_file = Path(tmpdir) / ".ralph" / "config.json"
            default_file.parent.mkdir(parents=True)

            # Write config to default location
            with open(default_file, "w") as f:
                json.dump({"plan": "max20x"}, f)

            with patch("ralph.config.DEFAULT_CONFIG_FILE", default_file):
                loaded = load_config(None)
                assert loaded.plan == Plan.MAX20X

    def test_load_config_invalid_json_returns_default(self):
        """Test that invalid JSON returns default config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.json"
            config_file.write_text("not valid json {")

            loaded = load_config(config_file)

            assert loaded.plan == DEFAULT_PLAN

    def test_load_config_empty_file_returns_default(self):
        """Test that empty file returns default config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.json"
            config_file.write_text("")

            loaded = load_config(config_file)

            assert loaded.plan == DEFAULT_PLAN

    def test_load_config_non_object_json_returns_default(self):
        """Test that non-object JSON returns default config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.json"
            config_file.write_text('["array", "not", "object"]')

            loaded = load_config(config_file)

            assert loaded.plan == DEFAULT_PLAN

    def test_load_config_null_json_returns_default(self):
        """Test that null JSON returns default config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.json"
            config_file.write_text("null")

            loaded = load_config(config_file)

            assert loaded.plan == DEFAULT_PLAN


# =============================================================================
# Default Plan Tests
# =============================================================================


class TestDefaultPlan:
    """Tests for default plan when no config exists."""

    def test_default_plan_is_pro(self):
        """Test that default plan is PRO."""
        assert DEFAULT_PLAN == Plan.PRO

    def test_load_config_returns_default_for_missing_file(self):
        """Test load_config returns default for missing file."""
        nonexistent = Path("/tmp/definitely_does_not_exist_12345/config.json")
        config = load_config(nonexistent)
        assert config.plan == DEFAULT_PLAN

    def test_config_default_initialization(self):
        """Test Config default initialization."""
        config = Config()
        assert config.plan == DEFAULT_PLAN

    def test_from_dict_empty_uses_default(self):
        """Test Config.from_dict with empty dict uses default."""
        config = Config.from_dict({})
        assert config.plan == DEFAULT_PLAN

    def test_get_plan_returns_default_for_missing_config(self):
        """Test get_plan returns default for missing config."""
        nonexistent = Path("/tmp/definitely_does_not_exist_12345/config.json")
        plan = get_plan(nonexistent)
        assert plan == DEFAULT_PLAN


# =============================================================================
# Invalid Plan Type Handling Tests
# =============================================================================


class TestInvalidPlanHandling:
    """Tests for invalid plan type handling."""

    def test_plan_from_string_invalid_raises_valueerror(self):
        """Test that Plan.from_string raises ValueError for invalid plans."""
        invalid_plans = ["invalid", "basic", "premium", "enterprise", "123", "", " "]

        for invalid in invalid_plans:
            with pytest.raises(ValueError):
                Plan.from_string(invalid)

    def test_config_from_dict_invalid_plan_uses_default(self):
        """Test that Config.from_dict uses default for invalid plan."""
        data = {"plan": "not_a_real_plan"}
        config = Config.from_dict(data)
        assert config.plan == DEFAULT_PLAN

    def test_load_config_with_invalid_plan_in_file(self):
        """Test loading config file with invalid plan value."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.json"
            config_file.write_text('{"plan": "invalid_plan_type"}')

            config = load_config(config_file)
            assert config.plan == DEFAULT_PLAN

    def test_set_plan_invalid_string_raises_valueerror(self):
        """Test that set_plan raises ValueError for invalid string."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.json"

            with pytest.raises(ValueError):
                set_plan("invalid_plan", config_file)

    def test_load_config_with_null_plan_uses_default(self):
        """Test loading config with null plan value uses default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.json"
            config_file.write_text('{"plan": null}')

            config = load_config(config_file)
            # null will be converted to "None" string which is invalid
            assert config.plan == DEFAULT_PLAN

    def test_load_config_with_numeric_plan_uses_default(self):
        """Test loading config with numeric plan value uses default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.json"
            config_file.write_text('{"plan": 123}')

            config = load_config(config_file)
            assert config.plan == DEFAULT_PLAN

    def test_load_config_corrupted_json_uses_default(self):
        """Test loading config with corrupted JSON uses default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.json"
            config_file.write_text('{"plan": "pro"')  # Missing closing brace

            config = load_config(config_file)
            assert config.plan == DEFAULT_PLAN


# =============================================================================
# Config File Permissions and Directory Creation Tests
# =============================================================================


class TestConfigPermissionsAndDirectoryCreation:
    """Tests for config file permissions and directory creation."""

    def test_ensure_config_dir_creates_directory(self):
        """Test that _ensure_config_dir creates directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = Path(tmpdir) / "new_config_dir"
            assert not new_dir.exists()

            _ensure_config_dir(new_dir)

            assert new_dir.exists()
            assert new_dir.is_dir()

    def test_ensure_config_dir_creates_nested_directories(self):
        """Test that _ensure_config_dir creates nested directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = Path(tmpdir) / "level1" / "level2" / "level3"
            assert not nested_dir.exists()

            _ensure_config_dir(nested_dir)

            assert nested_dir.exists()

    def test_ensure_config_dir_existing_directory(self):
        """Test that _ensure_config_dir handles existing directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            existing_dir = Path(tmpdir) / "existing"
            existing_dir.mkdir()

            # Should not raise
            _ensure_config_dir(existing_dir)

            assert existing_dir.exists()

    def test_save_config_creates_directory_tree(self):
        """Test that save_config creates full directory tree."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "a" / "b" / "c" / "config.json"
            config = Config(plan=Plan.PRO)

            save_config(config, config_file)

            assert config_file.exists()

    def test_load_config_handles_permission_error(self):
        """Test that load_config handles permission errors gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.json"
            config_file.write_text('{"plan": "pro"}')

            # Make file unreadable (skip on Windows)
            if os.name != "nt":
                os.chmod(config_file, 0o000)
                try:
                    config = load_config(config_file)
                    # Should return default on permission error
                    assert config.plan == DEFAULT_PLAN
                finally:
                    # Restore permissions for cleanup
                    os.chmod(config_file, stat.S_IRUSR | stat.S_IWUSR)

    def test_save_config_raises_on_permission_error(self):
        """Test that save_config raises OSError on permission issues."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a directory that we'll try to write inside
            protected_dir = Path(tmpdir) / "protected"
            protected_dir.mkdir()
            config_file = protected_dir / "config.json"

            # Make directory unwritable (skip on Windows)
            if os.name != "nt":
                os.chmod(protected_dir, stat.S_IRUSR | stat.S_IXUSR)
                try:
                    with pytest.raises(OSError):
                        save_config(Config(plan=Plan.PRO), config_file)
                finally:
                    # Restore permissions for cleanup
                    os.chmod(protected_dir, stat.S_IRWXU)

    def test_default_config_directory(self):
        """Test default config directory is in home."""
        assert DEFAULT_CONFIG_DIR == Path.home() / ".ralph"

    def test_default_config_file(self):
        """Test default config file is in default directory."""
        assert DEFAULT_CONFIG_FILE == DEFAULT_CONFIG_DIR / "config.json"


# =============================================================================
# Plan Limit Lookups Tests
# =============================================================================


class TestPlanLimitLookups:
    """Tests for plan limit lookups for all plan types."""

    def test_get_plan_limits_free(self):
        """Test plan limits for FREE tier."""
        limits = get_plan_limits(Plan.FREE)

        assert "5hour_tokens" in limits
        assert "weekly_tokens" in limits
        assert limits["5hour_tokens"] == 30_000
        assert limits["weekly_tokens"] == 150_000

    def test_get_plan_limits_pro(self):
        """Test plan limits for PRO tier."""
        limits = get_plan_limits(Plan.PRO)

        assert limits["5hour_tokens"] == 300_000
        assert limits["weekly_tokens"] == 1_500_000

    def test_get_plan_limits_max5x(self):
        """Test plan limits for MAX5X tier."""
        limits = get_plan_limits(Plan.MAX5X)

        assert limits["5hour_tokens"] == 1_500_000
        assert limits["weekly_tokens"] == 7_500_000

    def test_get_plan_limits_max20x(self):
        """Test plan limits for MAX20X tier."""
        limits = get_plan_limits(Plan.MAX20X)

        assert limits["5hour_tokens"] == 6_000_000
        assert limits["weekly_tokens"] == 30_000_000

    def test_all_plans_have_limits(self):
        """Test that all plan types have defined limits."""
        for plan in Plan:
            limits = get_plan_limits(plan)
            assert "5hour_tokens" in limits
            assert "weekly_tokens" in limits
            assert limits["5hour_tokens"] > 0
            assert limits["weekly_tokens"] > 0

    def test_plan_limits_dict_has_all_plans(self):
        """Test that PLAN_LIMITS contains all Plan enum values."""
        for plan in Plan:
            assert plan in PLAN_LIMITS

    def test_higher_tiers_have_higher_limits(self):
        """Test that higher tiers have higher limits."""
        free_limits = get_plan_limits(Plan.FREE)
        pro_limits = get_plan_limits(Plan.PRO)
        max5x_limits = get_plan_limits(Plan.MAX5X)
        max20x_limits = get_plan_limits(Plan.MAX20X)

        # 5-hour limits should increase
        assert free_limits["5hour_tokens"] < pro_limits["5hour_tokens"]
        assert pro_limits["5hour_tokens"] < max5x_limits["5hour_tokens"]
        assert max5x_limits["5hour_tokens"] < max20x_limits["5hour_tokens"]

        # Weekly limits should increase
        assert free_limits["weekly_tokens"] < pro_limits["weekly_tokens"]
        assert pro_limits["weekly_tokens"] < max5x_limits["weekly_tokens"]
        assert max5x_limits["weekly_tokens"] < max20x_limits["weekly_tokens"]

    def test_max5x_is_5x_pro(self):
        """Test that MAX5X limits are approximately 5x PRO limits."""
        pro_limits = get_plan_limits(Plan.PRO)
        max5x_limits = get_plan_limits(Plan.MAX5X)

        assert max5x_limits["5hour_tokens"] == pro_limits["5hour_tokens"] * 5
        assert max5x_limits["weekly_tokens"] == pro_limits["weekly_tokens"] * 5

    def test_max20x_is_20x_pro(self):
        """Test that MAX20X limits are approximately 20x PRO limits."""
        pro_limits = get_plan_limits(Plan.PRO)
        max20x_limits = get_plan_limits(Plan.MAX20X)

        assert max20x_limits["5hour_tokens"] == pro_limits["5hour_tokens"] * 20
        assert max20x_limits["weekly_tokens"] == pro_limits["weekly_tokens"] * 20


# =============================================================================
# Integration Tests: get_plan and set_plan
# =============================================================================


class TestGetAndSetPlan:
    """Integration tests for get_plan and set_plan functions."""

    def test_get_plan_returns_configured_value(self):
        """Test that get_plan returns the configured plan."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.json"

            # Set a plan
            set_plan(Plan.MAX5X, config_file)

            # Get and verify
            plan = get_plan(config_file)
            assert plan == Plan.MAX5X

    def test_set_plan_with_enum(self):
        """Test set_plan with Plan enum value."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.json"

            set_plan(Plan.MAX20X, config_file)

            assert get_plan(config_file) == Plan.MAX20X

    def test_set_plan_with_string(self):
        """Test set_plan with string value."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.json"

            set_plan("free", config_file)

            assert get_plan(config_file) == Plan.FREE

    def test_set_plan_with_uppercase_string(self):
        """Test set_plan with uppercase string value."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.json"

            set_plan("MAX5X", config_file)

            assert get_plan(config_file) == Plan.MAX5X

    def test_set_plan_preserves_existing_config(self):
        """Test that set_plan loads and preserves existing config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.json"

            # Initial set
            set_plan(Plan.FREE, config_file)
            assert get_plan(config_file) == Plan.FREE

            # Update
            set_plan(Plan.PRO, config_file)
            assert get_plan(config_file) == Plan.PRO

    def test_round_trip_all_plans(self):
        """Test set_plan and get_plan round-trip for all plans."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.json"

            for plan in Plan:
                set_plan(plan, config_file)
                result = get_plan(config_file)
                assert result == plan

    def test_get_plan_uses_default_path(self):
        """Test that get_plan uses default path when None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            default_file = Path(tmpdir) / ".ralph" / "config.json"
            default_file.parent.mkdir(parents=True)

            with open(default_file, "w") as f:
                json.dump({"plan": "max5x"}, f)

            with patch("ralph.config.DEFAULT_CONFIG_FILE", default_file):
                plan = get_plan(None)
                assert plan == Plan.MAX5X

    def test_set_plan_uses_default_path(self):
        """Test that set_plan uses default path when None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            default_file = Path(tmpdir) / ".ralph" / "config.json"

            with patch("ralph.config.DEFAULT_CONFIG_FILE", default_file):
                set_plan(Plan.MAX20X, None)

                with open(default_file) as f:
                    data = json.load(f)
                assert data["plan"] == "max20x"
                # Check that other config fields are also present
                assert "limit_mode" in data


class TestLimitModeConfig:
    """Tests for LimitMode configuration."""

    def test_limit_mode_values(self):
        """Test that all expected limit mode values exist."""
        assert LimitMode.PLAN.value == "plan"
        assert LimitMode.P90.value == "p90"
        assert LimitMode.HYBRID.value == "hybrid"

    def test_config_default_limit_mode(self):
        """Test that Config has correct default limit mode."""
        config = Config()
        assert config.limit_mode == LimitMode.PLAN

    def test_config_from_dict_limit_mode(self):
        """Test creating Config with limit_mode from dictionary."""
        data = {"plan": "pro", "limit_mode": "p90"}
        config = Config.from_dict(data)
        assert config.limit_mode == LimitMode.P90

    def test_config_from_dict_invalid_limit_mode_uses_default(self):
        """Test that invalid limit_mode uses default."""
        data = {"plan": "pro", "limit_mode": "invalid"}
        config = Config.from_dict(data)
        assert config.limit_mode == LimitMode.PLAN

    def test_config_round_trip_limit_mode(self):
        """Test that limit_mode survives round-trip."""
        for mode in LimitMode:
            original = Config(plan=Plan.PRO, limit_mode=mode)
            data = original.to_dict()
            restored = Config.from_dict(data)
            assert restored.limit_mode == original.limit_mode


class TestCostTrackingConfig:
    """Tests for cost tracking configuration."""

    def test_config_default_enable_cost_tracking(self):
        """Test that Config defaults to cost tracking enabled."""
        config = Config()
        assert config.enable_cost_tracking is True

    def test_config_from_dict_cost_tracking_disabled(self):
        """Test loading config with cost tracking disabled."""
        data = {"plan": "pro", "enable_cost_tracking": False}
        config = Config.from_dict(data)
        assert config.enable_cost_tracking is False

    def test_config_default_p90_lookback_days(self):
        """Test that Config defaults to 14 lookback days."""
        config = Config()
        assert config.p90_lookback_days == 14

    def test_config_from_dict_custom_lookback_days(self):
        """Test loading config with custom lookback days."""
        data = {"plan": "pro", "p90_lookback_days": 7}
        config = Config.from_dict(data)
        assert config.p90_lookback_days == 7
