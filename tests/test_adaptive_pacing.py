"""Unit tests for the adaptive pacing logic that adjusts iteration delays.

This module tests the adaptive pacing functionality:
- Delay multiplier at each threshold (70%, 80%, 90%)
- No adjustment below 70%
- Delay calculation with custom base delay
- Pacing state updates during iteration loop
- Configurable threshold overrides
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ralph.runner import (
    DEFAULT_ITERATION_DELAY,
    DEFAULT_PACING_MULTIPLIER_1,
    DEFAULT_PACING_MULTIPLIER_2,
    DEFAULT_PACING_MULTIPLIER_3,
    DEFAULT_PACING_THRESHOLD_1,
    DEFAULT_PACING_THRESHOLD_2,
    DEFAULT_PACING_THRESHOLD_3,
    Runner,
)


# =============================================================================
# Test default constants
# =============================================================================


class TestDefaultConstants:
    """Tests for default pacing constants."""

    def test_default_threshold_values(self):
        """Test that default threshold values are correct."""
        assert DEFAULT_PACING_THRESHOLD_1 == 70.0
        assert DEFAULT_PACING_THRESHOLD_2 == 80.0
        assert DEFAULT_PACING_THRESHOLD_3 == 90.0

    def test_default_multiplier_values(self):
        """Test that default multiplier values are correct."""
        assert DEFAULT_PACING_MULTIPLIER_1 == 2.0
        assert DEFAULT_PACING_MULTIPLIER_2 == 4.0
        assert DEFAULT_PACING_MULTIPLIER_3 == 8.0

    def test_default_iteration_delay(self):
        """Test that default iteration delay is set."""
        assert DEFAULT_ITERATION_DELAY == 2.0


# =============================================================================
# Test delay multiplier at each threshold
# =============================================================================


class TestDelayMultiplierThresholds:
    """Tests for delay multiplier at each threshold (70%, 80%, 90%)."""

    def create_runner(
        self,
        prd_path: Path,
        iteration_delay: float = DEFAULT_ITERATION_DELAY,
        adaptive_pacing: bool = True,
        pacing_threshold_1: float = DEFAULT_PACING_THRESHOLD_1,
        pacing_threshold_2: float = DEFAULT_PACING_THRESHOLD_2,
        pacing_threshold_3: float = DEFAULT_PACING_THRESHOLD_3,
    ) -> Runner:
        """Create a Runner instance for testing."""
        return Runner(
            prd_path=prd_path,
            iteration_delay=iteration_delay,
            adaptive_pacing=adaptive_pacing,
            pacing_threshold_1=pacing_threshold_1,
            pacing_threshold_2=pacing_threshold_2,
            pacing_threshold_3=pacing_threshold_3,
        )

    def test_2x_multiplier_at_70_percent(self):
        """Test 2x delay multiplier when usage is at 70%."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prd_path = Path(tmpdir) / "prd.json"
            prd_content = {
                "project": "Test",
                "branchName": "test",
                "description": "Test",
                "userStories": [],
            }
            prd_path.write_text(json.dumps(prd_content))

            runner = self.create_runner(prd_path, iteration_delay=2.0)

            # Test at exactly 70% - should get 2x multiplier
            delay, multiplier = runner._calculate_adaptive_delay(70.0)

            assert multiplier == 2.0
            assert delay == 4.0  # 2.0 * 2.0

    def test_2x_multiplier_at_75_percent(self):
        """Test 2x delay multiplier when usage is at 75% (between 70% and 80%)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prd_path = Path(tmpdir) / "prd.json"
            prd_content = {
                "project": "Test",
                "branchName": "test",
                "description": "Test",
                "userStories": [],
            }
            prd_path.write_text(json.dumps(prd_content))

            runner = self.create_runner(prd_path, iteration_delay=2.0)

            delay, multiplier = runner._calculate_adaptive_delay(75.0)

            assert multiplier == 2.0
            assert delay == 4.0

    def test_4x_multiplier_at_80_percent(self):
        """Test 4x delay multiplier when usage is at 80%."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prd_path = Path(tmpdir) / "prd.json"
            prd_content = {
                "project": "Test",
                "branchName": "test",
                "description": "Test",
                "userStories": [],
            }
            prd_path.write_text(json.dumps(prd_content))

            runner = self.create_runner(prd_path, iteration_delay=2.0)

            delay, multiplier = runner._calculate_adaptive_delay(80.0)

            assert multiplier == 4.0
            assert delay == 8.0  # 2.0 * 4.0

    def test_4x_multiplier_at_85_percent(self):
        """Test 4x delay multiplier when usage is at 85% (between 80% and 90%)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prd_path = Path(tmpdir) / "prd.json"
            prd_content = {
                "project": "Test",
                "branchName": "test",
                "description": "Test",
                "userStories": [],
            }
            prd_path.write_text(json.dumps(prd_content))

            runner = self.create_runner(prd_path, iteration_delay=2.0)

            delay, multiplier = runner._calculate_adaptive_delay(85.0)

            assert multiplier == 4.0
            assert delay == 8.0

    def test_8x_multiplier_at_90_percent(self):
        """Test 8x delay multiplier when usage is at 90%."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prd_path = Path(tmpdir) / "prd.json"
            prd_content = {
                "project": "Test",
                "branchName": "test",
                "description": "Test",
                "userStories": [],
            }
            prd_path.write_text(json.dumps(prd_content))

            runner = self.create_runner(prd_path, iteration_delay=2.0)

            delay, multiplier = runner._calculate_adaptive_delay(90.0)

            assert multiplier == 8.0
            assert delay == 16.0  # 2.0 * 8.0

    def test_8x_multiplier_at_95_percent(self):
        """Test 8x delay multiplier when usage is at 95% (above 90%)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prd_path = Path(tmpdir) / "prd.json"
            prd_content = {
                "project": "Test",
                "branchName": "test",
                "description": "Test",
                "userStories": [],
            }
            prd_path.write_text(json.dumps(prd_content))

            runner = self.create_runner(prd_path, iteration_delay=2.0)

            delay, multiplier = runner._calculate_adaptive_delay(95.0)

            assert multiplier == 8.0
            assert delay == 16.0

    def test_8x_multiplier_at_100_percent(self):
        """Test 8x delay multiplier when usage is at 100%."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prd_path = Path(tmpdir) / "prd.json"
            prd_content = {
                "project": "Test",
                "branchName": "test",
                "description": "Test",
                "userStories": [],
            }
            prd_path.write_text(json.dumps(prd_content))

            runner = self.create_runner(prd_path, iteration_delay=2.0)

            delay, multiplier = runner._calculate_adaptive_delay(100.0)

            assert multiplier == 8.0
            assert delay == 16.0


# =============================================================================
# Test no adjustment below 70%
# =============================================================================


class TestNoAdjustmentBelowThreshold:
    """Tests for no adjustment when usage is below 70%."""

    def create_runner(self, prd_path: Path, iteration_delay: float = 2.0) -> Runner:
        """Create a Runner instance for testing."""
        return Runner(
            prd_path=prd_path,
            iteration_delay=iteration_delay,
            adaptive_pacing=True,
        )

    def test_no_adjustment_at_0_percent(self):
        """Test no delay adjustment at 0% usage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prd_path = Path(tmpdir) / "prd.json"
            prd_content = {
                "project": "Test",
                "branchName": "test",
                "description": "Test",
                "userStories": [],
            }
            prd_path.write_text(json.dumps(prd_content))

            runner = self.create_runner(prd_path, iteration_delay=2.0)

            delay, multiplier = runner._calculate_adaptive_delay(0.0)

            assert multiplier == 1.0
            assert delay == 2.0  # No change

    def test_no_adjustment_at_50_percent(self):
        """Test no delay adjustment at 50% usage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prd_path = Path(tmpdir) / "prd.json"
            prd_content = {
                "project": "Test",
                "branchName": "test",
                "description": "Test",
                "userStories": [],
            }
            prd_path.write_text(json.dumps(prd_content))

            runner = self.create_runner(prd_path, iteration_delay=2.0)

            delay, multiplier = runner._calculate_adaptive_delay(50.0)

            assert multiplier == 1.0
            assert delay == 2.0

    def test_no_adjustment_at_69_percent(self):
        """Test no delay adjustment at 69% usage (just below threshold)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prd_path = Path(tmpdir) / "prd.json"
            prd_content = {
                "project": "Test",
                "branchName": "test",
                "description": "Test",
                "userStories": [],
            }
            prd_path.write_text(json.dumps(prd_content))

            runner = self.create_runner(prd_path, iteration_delay=2.0)

            delay, multiplier = runner._calculate_adaptive_delay(69.0)

            assert multiplier == 1.0
            assert delay == 2.0

    def test_no_adjustment_at_69_9_percent(self):
        """Test no delay adjustment at 69.9% usage (just below threshold)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prd_path = Path(tmpdir) / "prd.json"
            prd_content = {
                "project": "Test",
                "branchName": "test",
                "description": "Test",
                "userStories": [],
            }
            prd_path.write_text(json.dumps(prd_content))

            runner = self.create_runner(prd_path, iteration_delay=2.0)

            delay, multiplier = runner._calculate_adaptive_delay(69.9)

            assert multiplier == 1.0
            assert delay == 2.0

    def test_no_adjustment_when_adaptive_pacing_disabled(self):
        """Test no delay adjustment when adaptive pacing is disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prd_path = Path(tmpdir) / "prd.json"
            prd_content = {
                "project": "Test",
                "branchName": "test",
                "description": "Test",
                "userStories": [],
            }
            prd_path.write_text(json.dumps(prd_content))

            runner = Runner(
                prd_path=prd_path,
                iteration_delay=2.0,
                adaptive_pacing=False,  # Disabled
            )

            # Even at 90%, should not adjust when disabled
            delay, multiplier = runner._calculate_adaptive_delay(90.0)

            assert multiplier == 1.0
            assert delay == 2.0

    def test_no_adjustment_with_none_usage(self):
        """Test no delay adjustment when usage percentage is None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prd_path = Path(tmpdir) / "prd.json"
            prd_content = {
                "project": "Test",
                "branchName": "test",
                "description": "Test",
                "userStories": [],
            }
            prd_path.write_text(json.dumps(prd_content))

            runner = self.create_runner(prd_path, iteration_delay=2.0)

            delay, multiplier = runner._calculate_adaptive_delay(None)

            assert multiplier == 1.0
            assert delay == 2.0


# =============================================================================
# Test delay calculation with custom base delay
# =============================================================================


class TestCustomBaseDelay:
    """Tests for delay calculation with custom base delay."""

    def test_custom_base_delay_5_seconds(self):
        """Test delay calculation with 5 second base delay."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prd_path = Path(tmpdir) / "prd.json"
            prd_content = {
                "project": "Test",
                "branchName": "test",
                "description": "Test",
                "userStories": [],
            }
            prd_path.write_text(json.dumps(prd_content))

            runner = Runner(
                prd_path=prd_path,
                iteration_delay=5.0,  # Custom base delay
                adaptive_pacing=True,
            )

            # Below threshold
            delay_low, multiplier_low = runner._calculate_adaptive_delay(50.0)
            assert multiplier_low == 1.0
            assert delay_low == 5.0

            # At 70%
            delay_70, multiplier_70 = runner._calculate_adaptive_delay(70.0)
            assert multiplier_70 == 2.0
            assert delay_70 == 10.0  # 5.0 * 2.0

            # At 80%
            delay_80, multiplier_80 = runner._calculate_adaptive_delay(80.0)
            assert multiplier_80 == 4.0
            assert delay_80 == 20.0  # 5.0 * 4.0

            # At 90%
            delay_90, multiplier_90 = runner._calculate_adaptive_delay(90.0)
            assert multiplier_90 == 8.0
            assert delay_90 == 40.0  # 5.0 * 8.0

    def test_custom_base_delay_10_seconds(self):
        """Test delay calculation with 10 second base delay."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prd_path = Path(tmpdir) / "prd.json"
            prd_content = {
                "project": "Test",
                "branchName": "test",
                "description": "Test",
                "userStories": [],
            }
            prd_path.write_text(json.dumps(prd_content))

            runner = Runner(
                prd_path=prd_path,
                iteration_delay=10.0,
                adaptive_pacing=True,
            )

            delay, multiplier = runner._calculate_adaptive_delay(90.0)

            assert multiplier == 8.0
            assert delay == 80.0  # 10.0 * 8.0

    def test_custom_base_delay_1_second(self):
        """Test delay calculation with 1 second base delay."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prd_path = Path(tmpdir) / "prd.json"
            prd_content = {
                "project": "Test",
                "branchName": "test",
                "description": "Test",
                "userStories": [],
            }
            prd_path.write_text(json.dumps(prd_content))

            runner = Runner(
                prd_path=prd_path,
                iteration_delay=1.0,
                adaptive_pacing=True,
            )

            delay, multiplier = runner._calculate_adaptive_delay(80.0)

            assert multiplier == 4.0
            assert delay == 4.0  # 1.0 * 4.0

    def test_custom_base_delay_0_seconds(self):
        """Test delay calculation with 0 second base delay (edge case)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prd_path = Path(tmpdir) / "prd.json"
            prd_content = {
                "project": "Test",
                "branchName": "test",
                "description": "Test",
                "userStories": [],
            }
            prd_path.write_text(json.dumps(prd_content))

            runner = Runner(
                prd_path=prd_path,
                iteration_delay=0.0,
                adaptive_pacing=True,
            )

            delay, multiplier = runner._calculate_adaptive_delay(90.0)

            assert multiplier == 8.0
            assert delay == 0.0  # 0.0 * 8.0 = 0.0

    def test_custom_base_delay_fractional(self):
        """Test delay calculation with fractional base delay."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prd_path = Path(tmpdir) / "prd.json"
            prd_content = {
                "project": "Test",
                "branchName": "test",
                "description": "Test",
                "userStories": [],
            }
            prd_path.write_text(json.dumps(prd_content))

            runner = Runner(
                prd_path=prd_path,
                iteration_delay=1.5,
                adaptive_pacing=True,
            )

            delay, multiplier = runner._calculate_adaptive_delay(70.0)

            assert multiplier == 2.0
            assert delay == 3.0  # 1.5 * 2.0


# =============================================================================
# Test pacing state updates during iteration loop
# =============================================================================


class TestPacingStateUpdates:
    """Tests for pacing state updates during iteration loop."""

    def test_initial_pacing_multiplier_is_1(self):
        """Test that initial pacing multiplier is 1.0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prd_path = Path(tmpdir) / "prd.json"
            prd_content = {
                "project": "Test",
                "branchName": "test",
                "description": "Test",
                "userStories": [],
            }
            prd_path.write_text(json.dumps(prd_content))

            runner = Runner(
                prd_path=prd_path,
                adaptive_pacing=True,
            )

            assert runner._last_pacing_multiplier == 1.0

    def test_multiplier_state_tracks_changes(self):
        """Test that multiplier state is tracked between calls."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prd_path = Path(tmpdir) / "prd.json"
            prd_content = {
                "project": "Test",
                "branchName": "test",
                "description": "Test",
                "userStories": [],
            }
            prd_path.write_text(json.dumps(prd_content))

            runner = Runner(
                prd_path=prd_path,
                iteration_delay=2.0,
                adaptive_pacing=True,
            )

            # Initial state
            assert runner._last_pacing_multiplier == 1.0

            # Simulate pacing update at 70%
            with patch.object(runner, "_get_usage_percentage", return_value=70.0):
                with patch("ralph.runner.print_pacing_adjustment"):
                    runner._apply_adaptive_pacing()

            # State should be updated
            assert runner._last_pacing_multiplier == 2.0

    def test_multiplier_state_updates_on_increase(self):
        """Test that state updates when multiplier increases."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prd_path = Path(tmpdir) / "prd.json"
            prd_content = {
                "project": "Test",
                "branchName": "test",
                "description": "Test",
                "userStories": [],
            }
            prd_path.write_text(json.dumps(prd_content))

            runner = Runner(
                prd_path=prd_path,
                iteration_delay=2.0,
                adaptive_pacing=True,
            )

            # Start at 70%
            with patch.object(runner, "_get_usage_percentage", return_value=70.0):
                with patch("ralph.runner.print_pacing_adjustment"):
                    runner._apply_adaptive_pacing()

            assert runner._last_pacing_multiplier == 2.0

            # Increase to 80%
            with patch.object(runner, "_get_usage_percentage", return_value=80.0):
                with patch("ralph.runner.print_pacing_adjustment"):
                    runner._apply_adaptive_pacing()

            assert runner._last_pacing_multiplier == 4.0

            # Increase to 90%
            with patch.object(runner, "_get_usage_percentage", return_value=90.0):
                with patch("ralph.runner.print_pacing_adjustment"):
                    runner._apply_adaptive_pacing()

            assert runner._last_pacing_multiplier == 8.0

    def test_multiplier_state_updates_on_decrease(self):
        """Test that state updates when multiplier decreases."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prd_path = Path(tmpdir) / "prd.json"
            prd_content = {
                "project": "Test",
                "branchName": "test",
                "description": "Test",
                "userStories": [],
            }
            prd_path.write_text(json.dumps(prd_content))

            runner = Runner(
                prd_path=prd_path,
                iteration_delay=2.0,
                adaptive_pacing=True,
            )

            # Start at 90%
            runner._last_pacing_multiplier = 8.0

            # Decrease to 70%
            with patch.object(runner, "_get_usage_percentage", return_value=70.0):
                with patch("ralph.runner.print_info"):
                    runner._apply_adaptive_pacing()

            assert runner._last_pacing_multiplier == 2.0

    def test_apply_adaptive_pacing_returns_correct_delay(self):
        """Test that _apply_adaptive_pacing returns the correct delay."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prd_path = Path(tmpdir) / "prd.json"
            prd_content = {
                "project": "Test",
                "branchName": "test",
                "description": "Test",
                "userStories": [],
            }
            prd_path.write_text(json.dumps(prd_content))

            runner = Runner(
                prd_path=prd_path,
                iteration_delay=2.0,
                adaptive_pacing=True,
            )

            # Test at 80%
            with patch.object(runner, "_get_usage_percentage", return_value=80.0):
                with patch("ralph.runner.print_pacing_adjustment"):
                    delay = runner._apply_adaptive_pacing()

            assert delay == 8.0  # 2.0 * 4.0

    def test_apply_adaptive_pacing_handles_none_usage(self):
        """Test that _apply_adaptive_pacing handles None usage percentage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prd_path = Path(tmpdir) / "prd.json"
            prd_content = {
                "project": "Test",
                "branchName": "test",
                "description": "Test",
                "userStories": [],
            }
            prd_path.write_text(json.dumps(prd_content))

            runner = Runner(
                prd_path=prd_path,
                iteration_delay=2.0,
                adaptive_pacing=True,
            )

            # Return None (usage tracking unavailable)
            with patch.object(runner, "_get_usage_percentage", return_value=None):
                delay = runner._apply_adaptive_pacing()

            assert delay == 2.0  # Base delay, no adjustment


# =============================================================================
# Test configurable threshold overrides
# =============================================================================


class TestConfigurableThresholds:
    """Tests for configurable threshold overrides."""

    def test_custom_threshold_70_changed_to_50(self):
        """Test custom first threshold at 50%."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prd_path = Path(tmpdir) / "prd.json"
            prd_content = {
                "project": "Test",
                "branchName": "test",
                "description": "Test",
                "userStories": [],
            }
            prd_path.write_text(json.dumps(prd_content))

            runner = Runner(
                prd_path=prd_path,
                iteration_delay=2.0,
                adaptive_pacing=True,
                pacing_threshold_1=50.0,  # Custom threshold
            )

            # At 50%, should get 2x multiplier
            delay, multiplier = runner._calculate_adaptive_delay(50.0)
            assert multiplier == 2.0
            assert delay == 4.0

            # At 49%, should get no multiplier
            delay, multiplier = runner._calculate_adaptive_delay(49.0)
            assert multiplier == 1.0
            assert delay == 2.0

    def test_custom_threshold_80_changed_to_60(self):
        """Test custom second threshold at 60%."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prd_path = Path(tmpdir) / "prd.json"
            prd_content = {
                "project": "Test",
                "branchName": "test",
                "description": "Test",
                "userStories": [],
            }
            prd_path.write_text(json.dumps(prd_content))

            runner = Runner(
                prd_path=prd_path,
                iteration_delay=2.0,
                adaptive_pacing=True,
                pacing_threshold_1=50.0,
                pacing_threshold_2=60.0,  # Custom threshold
            )

            # At 60%, should get 4x multiplier
            delay, multiplier = runner._calculate_adaptive_delay(60.0)
            assert multiplier == 4.0
            assert delay == 8.0

            # At 55%, should get 2x multiplier (between thresholds)
            delay, multiplier = runner._calculate_adaptive_delay(55.0)
            assert multiplier == 2.0
            assert delay == 4.0

    def test_custom_threshold_90_changed_to_75(self):
        """Test custom third threshold at 75%."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prd_path = Path(tmpdir) / "prd.json"
            prd_content = {
                "project": "Test",
                "branchName": "test",
                "description": "Test",
                "userStories": [],
            }
            prd_path.write_text(json.dumps(prd_content))

            runner = Runner(
                prd_path=prd_path,
                iteration_delay=2.0,
                adaptive_pacing=True,
                pacing_threshold_1=50.0,
                pacing_threshold_2=60.0,
                pacing_threshold_3=75.0,  # Custom threshold
            )

            # At 75%, should get 8x multiplier
            delay, multiplier = runner._calculate_adaptive_delay(75.0)
            assert multiplier == 8.0
            assert delay == 16.0

            # At 70%, should get 4x multiplier (between thresholds)
            delay, multiplier = runner._calculate_adaptive_delay(70.0)
            assert multiplier == 4.0
            assert delay == 8.0

    def test_all_custom_thresholds(self):
        """Test with all custom thresholds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prd_path = Path(tmpdir) / "prd.json"
            prd_content = {
                "project": "Test",
                "branchName": "test",
                "description": "Test",
                "userStories": [],
            }
            prd_path.write_text(json.dumps(prd_content))

            runner = Runner(
                prd_path=prd_path,
                iteration_delay=1.0,
                adaptive_pacing=True,
                pacing_threshold_1=40.0,
                pacing_threshold_2=50.0,
                pacing_threshold_3=60.0,
            )

            # Below all thresholds
            delay, multiplier = runner._calculate_adaptive_delay(39.0)
            assert multiplier == 1.0
            assert delay == 1.0

            # At first threshold
            delay, multiplier = runner._calculate_adaptive_delay(40.0)
            assert multiplier == 2.0
            assert delay == 2.0

            # At second threshold
            delay, multiplier = runner._calculate_adaptive_delay(50.0)
            assert multiplier == 4.0
            assert delay == 4.0

            # At third threshold
            delay, multiplier = runner._calculate_adaptive_delay(60.0)
            assert multiplier == 8.0
            assert delay == 8.0

    def test_runner_stores_custom_thresholds(self):
        """Test that Runner stores custom thresholds correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prd_path = Path(tmpdir) / "prd.json"
            prd_content = {
                "project": "Test",
                "branchName": "test",
                "description": "Test",
                "userStories": [],
            }
            prd_path.write_text(json.dumps(prd_content))

            runner = Runner(
                prd_path=prd_path,
                pacing_threshold_1=45.0,
                pacing_threshold_2=55.0,
                pacing_threshold_3=65.0,
            )

            assert runner.pacing_threshold_1 == 45.0
            assert runner.pacing_threshold_2 == 55.0
            assert runner.pacing_threshold_3 == 65.0


# =============================================================================
# Test edge cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases in adaptive pacing."""

    def test_exactly_at_boundary_70(self):
        """Test behavior at exactly 70% boundary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prd_path = Path(tmpdir) / "prd.json"
            prd_content = {
                "project": "Test",
                "branchName": "test",
                "description": "Test",
                "userStories": [],
            }
            prd_path.write_text(json.dumps(prd_content))

            runner = Runner(
                prd_path=prd_path,
                iteration_delay=2.0,
                adaptive_pacing=True,
            )

            # At exactly 70%, should trigger 2x multiplier
            delay, multiplier = runner._calculate_adaptive_delay(70.0)
            assert multiplier == 2.0

    def test_exactly_at_boundary_80(self):
        """Test behavior at exactly 80% boundary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prd_path = Path(tmpdir) / "prd.json"
            prd_content = {
                "project": "Test",
                "branchName": "test",
                "description": "Test",
                "userStories": [],
            }
            prd_path.write_text(json.dumps(prd_content))

            runner = Runner(
                prd_path=prd_path,
                iteration_delay=2.0,
                adaptive_pacing=True,
            )

            # At exactly 80%, should trigger 4x multiplier
            delay, multiplier = runner._calculate_adaptive_delay(80.0)
            assert multiplier == 4.0

    def test_exactly_at_boundary_90(self):
        """Test behavior at exactly 90% boundary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prd_path = Path(tmpdir) / "prd.json"
            prd_content = {
                "project": "Test",
                "branchName": "test",
                "description": "Test",
                "userStories": [],
            }
            prd_path.write_text(json.dumps(prd_content))

            runner = Runner(
                prd_path=prd_path,
                iteration_delay=2.0,
                adaptive_pacing=True,
            )

            # At exactly 90%, should trigger 8x multiplier
            delay, multiplier = runner._calculate_adaptive_delay(90.0)
            assert multiplier == 8.0

    def test_usage_over_100_percent(self):
        """Test behavior when usage exceeds 100%."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prd_path = Path(tmpdir) / "prd.json"
            prd_content = {
                "project": "Test",
                "branchName": "test",
                "description": "Test",
                "userStories": [],
            }
            prd_path.write_text(json.dumps(prd_content))

            runner = Runner(
                prd_path=prd_path,
                iteration_delay=2.0,
                adaptive_pacing=True,
            )

            # At 150% (over limit), should still use 8x multiplier
            delay, multiplier = runner._calculate_adaptive_delay(150.0)
            assert multiplier == 8.0
            assert delay == 16.0

    def test_negative_usage_percentage(self):
        """Test behavior with negative usage percentage (edge case)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prd_path = Path(tmpdir) / "prd.json"
            prd_content = {
                "project": "Test",
                "branchName": "test",
                "description": "Test",
                "userStories": [],
            }
            prd_path.write_text(json.dumps(prd_content))

            runner = Runner(
                prd_path=prd_path,
                iteration_delay=2.0,
                adaptive_pacing=True,
            )

            # Negative percentage (shouldn't happen, but test anyway)
            delay, multiplier = runner._calculate_adaptive_delay(-10.0)
            assert multiplier == 1.0
            assert delay == 2.0

    def test_very_small_percentage(self):
        """Test behavior with very small percentage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prd_path = Path(tmpdir) / "prd.json"
            prd_content = {
                "project": "Test",
                "branchName": "test",
                "description": "Test",
                "userStories": [],
            }
            prd_path.write_text(json.dumps(prd_content))

            runner = Runner(
                prd_path=prd_path,
                iteration_delay=2.0,
                adaptive_pacing=True,
            )

            delay, multiplier = runner._calculate_adaptive_delay(0.001)
            assert multiplier == 1.0
            assert delay == 2.0


# =============================================================================
# Test _get_usage_percentage integration
# =============================================================================


class TestGetUsagePercentage:
    """Tests for _get_usage_percentage method."""

    def test_returns_none_on_exception(self):
        """Test that _get_usage_percentage returns None on exception."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prd_path = Path(tmpdir) / "prd.json"
            prd_content = {
                "project": "Test",
                "branchName": "test",
                "description": "Test",
                "userStories": [],
            }
            prd_path.write_text(json.dumps(prd_content))

            runner = Runner(
                prd_path=prd_path,
                adaptive_pacing=True,
            )

            # Mock to raise exception - patch at the source module
            with patch("ralph.usage.get_5hour_window_usage", side_effect=Exception("Test error")):
                result = runner._get_usage_percentage()

            assert result is None

    def test_returns_percentage_with_custom_limit(self):
        """Test that _get_usage_percentage uses custom five_hour_limit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prd_path = Path(tmpdir) / "prd.json"
            prd_content = {
                "project": "Test",
                "branchName": "test",
                "description": "Test",
                "userStories": [],
            }
            prd_path.write_text(json.dumps(prd_content))

            runner = Runner(
                prd_path=prd_path,
                adaptive_pacing=True,
                five_hour_limit=100000,  # Custom limit
            )

            # Mock usage data
            mock_usage = MagicMock()
            mock_usage.total_tokens = 50000  # 50% of custom limit

            # Patch at the source module and session module
            with patch("ralph.usage.get_5hour_window_usage", return_value=mock_usage):
                with patch("ralph.session.get_budget_for_session", return_value=(50000, 1)):
                    result = runner._get_usage_percentage()

            # Should return a percentage based on the effective limit
            assert result is not None
            assert isinstance(result, float)
