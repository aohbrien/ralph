"""Unit tests for the PRD re-evaluation feature.

This module tests the re-evaluation functionality:
- Parsing re-evaluation responses from Claude
- Validating proposed changes against safeguards
- Applying approved changes to the PRD
- Prompt generation
- Integration with the Runner class
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ralph.prd import PRD, UserStory
from ralph.reeval import (
    MAX_CHANGES_PER_REEVAL,
    MIN_PENDING_STORIES,
    REEVAL_COMPLETE_SIGNAL,
    ChangeAction,
    ReEvalChange,
    ReEvalResult,
    ValidationResult,
    apply_changes,
    generate_reeval_prompt,
    parse_reeval_response,
    validate_changes,
)
from ralph.runner import DEFAULT_REEVAL_INTERVAL, REEVAL_TIMEOUT, Runner


# =============================================================================
# Test fixtures
# =============================================================================


def create_test_prd(stories: list[dict] | None = None) -> PRD:
    """Create a test PRD with optional stories."""
    if stories is None:
        stories = [
            {
                "id": "US-001",
                "title": "First story",
                "description": "First story description",
                "acceptanceCriteria": ["AC1", "AC2"],
                "priority": 1,
                "passes": True,  # Completed
            },
            {
                "id": "US-002",
                "title": "Second story",
                "description": "Second story description",
                "acceptanceCriteria": ["AC1", "AC2", "AC3"],
                "priority": 2,
                "passes": False,  # Pending
            },
            {
                "id": "US-003",
                "title": "Third story",
                "description": "Third story description",
                "acceptanceCriteria": ["AC1"],
                "priority": 3,
                "passes": False,  # Pending
            },
        ]
    return PRD.from_dict({
        "project": "Test Project",
        "branchName": "test-branch",
        "description": "Test project description",
        "userStories": stories,
    })


# =============================================================================
# Test ReEvalChange dataclass
# =============================================================================


class TestReEvalChange:
    """Tests for ReEvalChange dataclass."""

    def test_from_dict_remove_action(self):
        """Test creating a remove action from dict."""
        data = {
            "action": "remove",
            "story_id": "US-002",
            "reason": "Story is obsolete",
        }
        change = ReEvalChange.from_dict(data)

        assert change.action == ChangeAction.REMOVE
        assert change.story_id == "US-002"
        assert change.reason == "Story is obsolete"
        assert change.merge_into is None
        assert change.new_data is None

    def test_from_dict_merge_action(self):
        """Test creating a merge action from dict."""
        data = {
            "action": "merge",
            "story_id": "US-003",
            "merge_into": "US-002",
            "reason": "Duplicate functionality",
        }
        change = ReEvalChange.from_dict(data)

        assert change.action == ChangeAction.MERGE
        assert change.story_id == "US-003"
        assert change.merge_into == "US-002"
        assert change.reason == "Duplicate functionality"

    def test_from_dict_modify_action(self):
        """Test creating a modify action from dict."""
        data = {
            "action": "modify",
            "story_id": "US-002",
            "reason": "Clarify description",
            "new_data": {
                "title": "Updated title",
                "description": "Updated description",
            },
        }
        change = ReEvalChange.from_dict(data)

        assert change.action == ChangeAction.MODIFY
        assert change.story_id == "US-002"
        assert change.new_data["title"] == "Updated title"

    def test_from_dict_invalid_action(self):
        """Test that invalid action raises ValueError."""
        data = {
            "action": "invalid",
            "story_id": "US-002",
            "reason": "Test",
        }
        with pytest.raises(ValueError, match="Invalid action"):
            ReEvalChange.from_dict(data)


# =============================================================================
# Test parse_reeval_response
# =============================================================================


class TestParseReEvalResponse:
    """Tests for parsing re-evaluation responses."""

    def test_parse_json_in_code_block(self):
        """Test parsing JSON in markdown code block."""
        output = """
Some intro text...

```json
{
  "changes": [
    {"action": "remove", "story_id": "US-002", "reason": "Obsolete"}
  ],
  "summary": "Removed one obsolete story"
}
```

<reeval-complete>
"""
        result = parse_reeval_response(output)

        assert result.completed is True
        assert len(result.changes) == 1
        assert result.changes[0].action == ChangeAction.REMOVE
        assert result.summary == "Removed one obsolete story"
        assert result.error is None

    def test_parse_json_without_code_block(self):
        """Test parsing raw JSON without code block."""
        output = """
Here's my analysis:
{"changes": [], "summary": "No changes needed"}
<reeval-complete>
"""
        result = parse_reeval_response(output)

        assert result.completed is True
        assert len(result.changes) == 0
        assert result.summary == "No changes needed"

    def test_parse_without_completion_signal(self):
        """Test parsing when completion signal is missing."""
        output = """
```json
{"changes": [], "summary": "Test"}
```
"""
        result = parse_reeval_response(output)

        assert result.completed is False
        assert result.error is None

    def test_parse_invalid_json(self):
        """Test parsing invalid JSON."""
        output = """
```json
{invalid json}
```
<reeval-complete>
"""
        result = parse_reeval_response(output)

        assert result.error is not None
        assert "Invalid JSON" in result.error

    def test_parse_no_json_found(self):
        """Test when no JSON is found in output."""
        output = "Just some text without any JSON"
        result = parse_reeval_response(output)

        assert result.error is not None
        assert "Could not find JSON" in result.error

    def test_parse_multiple_changes(self):
        """Test parsing multiple changes."""
        output = """
```json
{
  "changes": [
    {"action": "remove", "story_id": "US-002", "reason": "Obsolete"},
    {"action": "merge", "story_id": "US-003", "merge_into": "US-004", "reason": "Duplicate"},
    {"action": "modify", "story_id": "US-005", "reason": "Clarify", "new_data": {"title": "New"}}
  ],
  "summary": "Multiple changes"
}
```
<reeval-complete>
"""
        result = parse_reeval_response(output)

        assert result.completed is True
        assert len(result.changes) == 3
        assert result.changes[0].action == ChangeAction.REMOVE
        assert result.changes[1].action == ChangeAction.MERGE
        assert result.changes[2].action == ChangeAction.MODIFY

    def test_parse_skips_invalid_changes(self):
        """Test that invalid changes are skipped."""
        output = """
```json
{
  "changes": [
    {"action": "remove", "story_id": "US-002", "reason": "Valid"},
    {"action": "invalid", "story_id": "US-003", "reason": "Invalid action"}
  ],
  "summary": "Test"
}
```
<reeval-complete>
"""
        result = parse_reeval_response(output)

        assert len(result.changes) == 1
        assert result.changes[0].story_id == "US-002"

    def test_parse_handles_multiple_json_blocks(self):
        """Test that parser correctly handles output with multiple JSON-like blocks."""
        output = """
Let me think about this... Here's some analysis: {"note": "thinking"}

Now here's the actual response:

```json
{
  "changes": [{"action": "remove", "story_id": "US-002", "reason": "Obsolete"}],
  "summary": "Removed one"
}
```
<reeval-complete>
"""
        result = parse_reeval_response(output)

        # Should parse the correct JSON block
        assert len(result.changes) == 1
        assert result.changes[0].story_id == "US-002"

    def test_parse_handles_nested_json(self):
        """Test parsing JSON with nested objects."""
        output = """
```json
{
  "changes": [
    {
      "action": "modify",
      "story_id": "US-002",
      "reason": "Update",
      "new_data": {
        "title": "New Title",
        "acceptanceCriteria": ["AC1", "AC2"]
      }
    }
  ],
  "summary": "Modified one"
}
```
<reeval-complete>
"""
        result = parse_reeval_response(output)

        assert len(result.changes) == 1
        assert result.changes[0].new_data is not None
        assert result.changes[0].new_data["title"] == "New Title"


# =============================================================================
# Test validate_changes - Safeguards
# =============================================================================


class TestValidateChanges:
    """Tests for validating proposed changes against safeguards."""

    def test_reject_modify_completed_story(self):
        """Test that modifying a completed story is rejected."""
        prd = create_test_prd()
        changes = [
            ReEvalChange(
                action=ChangeAction.MODIFY,
                story_id="US-001",  # Completed story
                reason="Try to modify completed",
                new_data={"title": "New title"},
            )
        ]

        result = validate_changes(prd, changes)

        assert result.valid is False
        assert len(result.approved_changes) == 0
        assert len(result.rejected_changes) == 1
        assert "completed story" in result.rejected_changes[0][1].lower()

    def test_reject_remove_completed_story(self):
        """Test that removing a completed story is rejected."""
        prd = create_test_prd()
        changes = [
            ReEvalChange(
                action=ChangeAction.REMOVE,
                story_id="US-001",  # Completed story
                reason="Try to remove completed",
            )
        ]

        result = validate_changes(prd, changes)

        assert result.valid is False
        assert len(result.rejected_changes) == 1

    def test_approve_remove_pending_story(self):
        """Test that removing a pending story is approved."""
        prd = create_test_prd()
        changes = [
            ReEvalChange(
                action=ChangeAction.REMOVE,
                story_id="US-002",  # Pending story
                reason="Story is obsolete",
            )
        ]

        result = validate_changes(prd, changes)

        assert result.valid is True
        assert len(result.approved_changes) == 1
        assert len(result.rejected_changes) == 0

    def test_reject_nonexistent_story(self):
        """Test that modifying a non-existent story is rejected."""
        prd = create_test_prd()
        changes = [
            ReEvalChange(
                action=ChangeAction.MODIFY,
                story_id="US-999",  # Non-existent
                reason="Doesn't exist",
                new_data={"title": "New"},
            )
        ]

        result = validate_changes(prd, changes)

        assert result.valid is False
        assert "not found" in result.rejected_changes[0][1].lower()

    def test_reject_merge_into_completed_story(self):
        """Test that merging into a completed story is rejected."""
        prd = create_test_prd()
        changes = [
            ReEvalChange(
                action=ChangeAction.MERGE,
                story_id="US-002",  # Pending
                merge_into="US-001",  # Completed
                reason="Try to merge into completed",
            )
        ]

        result = validate_changes(prd, changes)

        assert result.valid is False
        assert "completed story" in result.rejected_changes[0][1].lower()

    def test_approve_merge_pending_stories(self):
        """Test that merging pending stories is approved."""
        prd = create_test_prd()
        changes = [
            ReEvalChange(
                action=ChangeAction.MERGE,
                story_id="US-003",  # Pending
                merge_into="US-002",  # Also pending
                reason="Duplicate functionality",
            )
        ]

        result = validate_changes(prd, changes)

        assert result.valid is True
        assert len(result.approved_changes) == 1

    def test_reject_merge_missing_target(self):
        """Test that merge without target is rejected."""
        prd = create_test_prd()
        changes = [
            ReEvalChange(
                action=ChangeAction.MERGE,
                story_id="US-002",
                merge_into=None,  # Missing target
                reason="No target",
            )
        ]

        result = validate_changes(prd, changes)

        assert result.valid is False
        assert "merge_into" in result.rejected_changes[0][1].lower()

    def test_reject_expand_acceptance_criteria(self):
        """Test that expanding acceptance criteria is rejected."""
        prd = create_test_prd()
        changes = [
            ReEvalChange(
                action=ChangeAction.MODIFY,
                story_id="US-002",  # Has 3 AC
                reason="Try to expand",
                new_data={
                    "acceptanceCriteria": ["AC1", "AC2", "AC3", "AC4"],  # 4 AC (expanded)
                },
            )
        ]

        result = validate_changes(prd, changes)

        assert result.valid is False
        assert "expand" in result.rejected_changes[0][1].lower()

    def test_approve_reduce_acceptance_criteria(self):
        """Test that reducing acceptance criteria is approved."""
        prd = create_test_prd()
        changes = [
            ReEvalChange(
                action=ChangeAction.MODIFY,
                story_id="US-002",  # Has 3 AC
                reason="Simplify",
                new_data={
                    "acceptanceCriteria": ["AC1", "AC2"],  # 2 AC (reduced)
                },
            )
        ]

        result = validate_changes(prd, changes)

        assert result.valid is True
        assert len(result.approved_changes) == 1

    def test_reject_change_story_id(self):
        """Test that changing story ID is rejected."""
        prd = create_test_prd()
        changes = [
            ReEvalChange(
                action=ChangeAction.MODIFY,
                story_id="US-002",
                reason="Try to change ID",
                new_data={"id": "US-NEW"},  # Changed ID
            )
        ]

        result = validate_changes(prd, changes)

        assert result.valid is False
        assert "story ID" in result.rejected_changes[0][1]

    def test_approve_modify_title_description(self):
        """Test that modifying title and description is approved."""
        prd = create_test_prd()
        changes = [
            ReEvalChange(
                action=ChangeAction.MODIFY,
                story_id="US-002",
                reason="Clarify description",
                new_data={
                    "title": "Updated title",
                    "description": "Updated description",
                },
            )
        ]

        result = validate_changes(prd, changes)

        assert result.valid is True
        assert len(result.approved_changes) == 1

    def test_mixed_approved_and_rejected(self):
        """Test that some changes can be approved while others rejected."""
        prd = create_test_prd()
        changes = [
            ReEvalChange(
                action=ChangeAction.REMOVE,
                story_id="US-002",  # Pending - should be approved
                reason="Obsolete",
            ),
            ReEvalChange(
                action=ChangeAction.MODIFY,
                story_id="US-001",  # Completed - should be rejected
                reason="Try to modify completed",
                new_data={"title": "New"},
            ),
        ]

        result = validate_changes(prd, changes)

        assert result.valid is False  # Has rejections
        assert len(result.approved_changes) == 1
        assert len(result.rejected_changes) == 1

    def test_reject_removing_all_pending(self):
        """Test that removing all pending stories is rejected."""
        # Create PRD with only 2 pending stories
        prd = create_test_prd([
            {
                "id": "US-001",
                "title": "Completed",
                "description": "Done",
                "acceptanceCriteria": ["AC1"],
                "priority": 1,
                "passes": True,
            },
            {
                "id": "US-002",
                "title": "Pending 1",
                "description": "Todo",
                "acceptanceCriteria": ["AC1"],
                "priority": 2,
                "passes": False,
            },
            {
                "id": "US-003",
                "title": "Pending 2",
                "description": "Todo",
                "acceptanceCriteria": ["AC1"],
                "priority": 3,
                "passes": False,
            },
        ])

        # Try to remove both pending stories
        changes = [
            ReEvalChange(action=ChangeAction.REMOVE, story_id="US-002", reason="R1"),
            ReEvalChange(action=ChangeAction.REMOVE, story_id="US-003", reason="R2"),
        ]

        result = validate_changes(prd, changes)

        # Should only approve one removal (to keep MIN_PENDING_STORIES)
        assert len(result.approved_changes) == 1
        assert len(result.rejected_changes) == 1
        assert "limit" in result.rejected_changes[0][1].lower()

    def test_reject_merge_into_removed_story(self):
        """Test that merging into a story that's being removed is rejected."""
        # Create PRD with enough pending stories to allow operations
        prd = create_test_prd([
            {"id": "US-001", "title": "Completed", "description": "D",
             "acceptanceCriteria": ["AC"], "priority": 1, "passes": True},
            {"id": "US-002", "title": "Pending 1", "description": "D",
             "acceptanceCriteria": ["AC"], "priority": 2, "passes": False},
            {"id": "US-003", "title": "Pending 2", "description": "D",
             "acceptanceCriteria": ["AC"], "priority": 3, "passes": False},
            {"id": "US-004", "title": "Pending 3", "description": "D",
             "acceptanceCriteria": ["AC"], "priority": 4, "passes": False},
        ])
        changes = [
            ReEvalChange(action=ChangeAction.REMOVE, story_id="US-002", reason="Remove"),
            ReEvalChange(action=ChangeAction.MERGE, story_id="US-003", merge_into="US-002", reason="Merge"),
        ]

        result = validate_changes(prd, changes)

        # Remove should be approved, merge should be rejected due to target being removed
        approved_actions = [c.action for c in result.approved_changes]
        assert ChangeAction.REMOVE in approved_actions
        assert any("being removed" in r[1] for r in result.rejected_changes)

    def test_max_changes_limit(self):
        """Test that changes are limited to MAX_CHANGES_PER_REEVAL."""
        # Create PRD with many pending stories
        stories = [{"id": f"US-{i:03d}", "title": f"Story {i}", "description": "D",
                    "acceptanceCriteria": ["AC"], "priority": i, "passes": False}
                   for i in range(1, 15)]
        prd = create_test_prd(stories)

        # Propose more changes than allowed
        changes = [
            ReEvalChange(action=ChangeAction.MODIFY, story_id=f"US-{i:03d}",
                         reason="Modify", new_data={"title": "New"})
            for i in range(1, 10)
        ]

        result = validate_changes(prd, changes)

        # Should have warning about too many changes
        assert any("too many" in w.lower() for w in result.warnings)


# =============================================================================
# Test apply_changes
# =============================================================================


class TestApplyChanges:
    """Tests for applying validated changes to the PRD."""

    def test_apply_remove_story(self):
        """Test applying a remove action."""
        prd = create_test_prd()
        initial_count = len(prd.user_stories)
        changes = [
            ReEvalChange(
                action=ChangeAction.REMOVE,
                story_id="US-002",
                reason="Obsolete story",
            )
        ]

        prd, applied = apply_changes(prd, changes)

        assert len(prd.user_stories) == initial_count - 1
        assert prd.get_story_by_id("US-002") is None
        assert len(applied) == 1
        assert "Removed" in applied[0]

    def test_apply_merge_stories(self):
        """Test applying a merge action."""
        prd = create_test_prd()
        changes = [
            ReEvalChange(
                action=ChangeAction.MERGE,
                story_id="US-003",
                merge_into="US-002",
                reason="Duplicate functionality",
            )
        ]

        prd, applied = apply_changes(prd, changes)

        # Source story should be removed
        assert prd.get_story_by_id("US-003") is None
        # Target story should have merge note
        target = prd.get_story_by_id("US-002")
        assert target is not None
        assert "Merged from US-003" in target.notes

    def test_apply_merge_combines_acceptance_criteria(self):
        """Test that merge combines acceptance criteria from both stories."""
        prd = create_test_prd([
            {
                "id": "US-001",
                "title": "Target",
                "description": "Target story",
                "acceptanceCriteria": ["AC-Target-1", "AC-Shared"],
                "priority": 1,
                "passes": False,
            },
            {
                "id": "US-002",
                "title": "Source",
                "description": "Source story",
                "acceptanceCriteria": ["AC-Source-1", "AC-Shared"],  # AC-Shared is duplicate
                "priority": 2,
                "passes": False,
            },
        ])

        changes = [
            ReEvalChange(
                action=ChangeAction.MERGE,
                story_id="US-002",
                merge_into="US-001",
                reason="Merge test",
            )
        ]

        prd, applied = apply_changes(prd, changes)

        target = prd.get_story_by_id("US-001")
        # Should have original ACs plus unique ones from source
        assert "AC-Target-1" in target.acceptance_criteria
        assert "AC-Source-1" in target.acceptance_criteria
        assert "AC-Shared" in target.acceptance_criteria
        # Should not have duplicates
        assert target.acceptance_criteria.count("AC-Shared") == 1

    def test_apply_modify_story(self):
        """Test applying a modify action."""
        prd = create_test_prd()
        changes = [
            ReEvalChange(
                action=ChangeAction.MODIFY,
                story_id="US-002",
                reason="Update description",
                new_data={
                    "title": "New Title",
                    "description": "New Description",
                    "priority": 5,
                },
            )
        ]

        prd, applied = apply_changes(prd, changes)

        story = prd.get_story_by_id("US-002")
        assert story.title == "New Title"
        assert story.description == "New Description"
        assert story.priority == 5
        assert len(applied) == 1

    def test_apply_multiple_changes(self):
        """Test applying multiple changes."""
        prd = create_test_prd()
        changes = [
            ReEvalChange(
                action=ChangeAction.MODIFY,
                story_id="US-002",
                reason="Update",
                new_data={"title": "Updated Title"},
            ),
            ReEvalChange(
                action=ChangeAction.REMOVE,
                story_id="US-003",
                reason="Obsolete",
            ),
        ]

        prd, applied = apply_changes(prd, changes)

        assert len(applied) == 2
        assert prd.get_story_by_id("US-002").title == "Updated Title"
        assert prd.get_story_by_id("US-003") is None


# =============================================================================
# Test generate_reeval_prompt
# =============================================================================


class TestGenerateReEvalPrompt:
    """Tests for prompt generation."""

    def test_prompt_contains_project_info(self):
        """Test that prompt contains project information."""
        prd = create_test_prd()
        with tempfile.TemporaryDirectory() as tmpdir:
            progress_path = Path(tmpdir) / "progress.txt"
            progress_path.write_text("Some progress notes")

            prompt = generate_reeval_prompt(prd, progress_path)

            assert "Test Project" in prompt
            assert "Test project description" in prompt

    def test_prompt_separates_completed_and_pending(self):
        """Test that prompt separates completed and pending stories."""
        prd = create_test_prd()
        with tempfile.TemporaryDirectory() as tmpdir:
            progress_path = Path(tmpdir) / "progress.txt"
            progress_path.write_text("")

            prompt = generate_reeval_prompt(prd, progress_path)

            assert "Completed Stories (DO NOT MODIFY)" in prompt
            assert "Pending Stories (may be modified)" in prompt
            # US-001 is completed
            assert "US-001" in prompt
            # US-002 and US-003 are pending
            assert "US-002" in prompt
            assert "US-003" in prompt

    def test_prompt_includes_progress_content(self):
        """Test that prompt includes progress.txt content."""
        prd = create_test_prd()
        with tempfile.TemporaryDirectory() as tmpdir:
            progress_path = Path(tmpdir) / "progress.txt"
            progress_path.write_text("Learnings from iteration 1\nMore learnings")

            prompt = generate_reeval_prompt(prd, progress_path)

            assert "Learnings from iteration 1" in prompt

    def test_prompt_handles_missing_progress_file(self):
        """Test that prompt handles missing progress.txt."""
        prd = create_test_prd()
        progress_path = Path("/nonexistent/progress.txt")

        prompt = generate_reeval_prompt(prd, progress_path)

        assert "No progress.txt file found" in prompt

    def test_prompt_includes_completion_signal(self):
        """Test that prompt instructs to output completion signal."""
        prd = create_test_prd()
        with tempfile.TemporaryDirectory() as tmpdir:
            progress_path = Path(tmpdir) / "progress.txt"
            progress_path.write_text("")

            prompt = generate_reeval_prompt(prd, progress_path)

            assert REEVAL_COMPLETE_SIGNAL in prompt


# =============================================================================
# Test Runner integration
# =============================================================================


class TestRunnerReEvalIntegration:
    """Tests for Runner re-evaluation integration."""

    def create_test_prd_file(self, tmpdir: str) -> Path:
        """Create a test PRD file."""
        prd_path = Path(tmpdir) / "prd.json"
        prd_content = {
            "project": "Test",
            "branchName": "test",
            "description": "Test project",
            "userStories": [
                {
                    "id": "US-001",
                    "title": "Story 1",
                    "description": "First story",
                    "acceptanceCriteria": ["AC1"],
                    "priority": 1,
                    "passes": False,
                },
                {
                    "id": "US-002",
                    "title": "Story 2",
                    "description": "Second story",
                    "acceptanceCriteria": ["AC1"],
                    "priority": 2,
                    "passes": False,
                },
            ],
        }
        prd_path.write_text(json.dumps(prd_content))
        return prd_path

    def test_default_reeval_interval(self):
        """Test that default reeval interval is 10."""
        assert DEFAULT_REEVAL_INTERVAL == 10

    def test_reeval_timeout(self):
        """Test that reeval timeout is 5 minutes."""
        assert REEVAL_TIMEOUT == 5 * 60

    def test_runner_stores_reeval_config(self):
        """Test that Runner stores reeval configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prd_path = self.create_test_prd_file(tmpdir)

            runner = Runner(
                prd_path=prd_path,
                reeval_interval=5,
                no_reeval=True,
            )

            assert runner.reeval_interval == 5
            assert runner.no_reeval is True

    def test_should_run_reeval_at_interval(self):
        """Test _should_run_reeval returns True at interval iterations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prd_path = self.create_test_prd_file(tmpdir)

            runner = Runner(
                prd_path=prd_path,
                reeval_interval=10,
                no_reeval=False,
            )

            # Should NOT run at iteration 1, 5, 9
            assert runner._should_run_reeval(1) is False
            assert runner._should_run_reeval(5) is False
            assert runner._should_run_reeval(9) is False

            # Should run at iteration 10, 20, 30
            assert runner._should_run_reeval(10) is True
            assert runner._should_run_reeval(20) is True
            assert runner._should_run_reeval(30) is True

    def test_should_run_reeval_disabled(self):
        """Test _should_run_reeval returns False when disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prd_path = self.create_test_prd_file(tmpdir)

            runner = Runner(
                prd_path=prd_path,
                reeval_interval=10,
                no_reeval=True,  # Disabled
            )

            assert runner._should_run_reeval(10) is False

    def test_should_run_reeval_interval_zero(self):
        """Test _should_run_reeval returns False when interval is 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prd_path = self.create_test_prd_file(tmpdir)

            runner = Runner(
                prd_path=prd_path,
                reeval_interval=0,  # Disabled via interval
                no_reeval=False,
            )

            assert runner._should_run_reeval(10) is False

    def test_should_run_reeval_custom_interval(self):
        """Test _should_run_reeval with custom interval."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prd_path = self.create_test_prd_file(tmpdir)

            runner = Runner(
                prd_path=prd_path,
                reeval_interval=3,  # Custom interval
                no_reeval=False,
            )

            # Should run at iterations 3, 6, 9
            assert runner._should_run_reeval(1) is False
            assert runner._should_run_reeval(2) is False
            assert runner._should_run_reeval(3) is True
            assert runner._should_run_reeval(4) is False
            assert runner._should_run_reeval(6) is True
            assert runner._should_run_reeval(9) is True

    def test_should_run_reeval_skips_if_already_ran(self):
        """Test _should_run_reeval skips if already ran at this iteration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prd_path = self.create_test_prd_file(tmpdir)

            runner = Runner(
                prd_path=prd_path,
                reeval_interval=10,
                no_reeval=False,
            )

            # Simulate that we already ran reeval at iteration 10
            from ralph.state import RunState
            runner._run_state = RunState.create(
                prd_path=prd_path,
                tool="claude",
                iteration=10,
                story_id=None,
            )
            runner._run_state.last_reeval_iteration = 10

            # Should NOT run again at 10
            assert runner._should_run_reeval(10) is False
            # Should still run at 20
            assert runner._should_run_reeval(20) is True


# =============================================================================
# Test constants
# =============================================================================


class TestConstants:
    """Tests for reeval constants."""

    def test_reeval_complete_signal(self):
        """Test that reeval complete signal is correct."""
        assert REEVAL_COMPLETE_SIGNAL == "<reeval-complete>"

    def test_default_reeval_interval_value(self):
        """Test default reeval interval value."""
        from ralph.reeval import DEFAULT_REEVAL_INTERVAL as REEVAL_DEFAULT
        assert REEVAL_DEFAULT == 10


# =============================================================================
# Test state tracking
# =============================================================================


class TestStateTracking:
    """Tests for reeval state tracking."""

    def test_run_state_has_reeval_field(self):
        """Test that RunState has last_reeval_iteration field."""
        from ralph.state import RunState

        state = RunState.create(
            prd_path=Path("test.json"),
            tool="claude",
            iteration=1,
            story_id="US-001",
        )

        assert hasattr(state, "last_reeval_iteration")
        assert state.last_reeval_iteration is None

    def test_run_state_update_reeval(self):
        """Test RunState.update_reeval method."""
        from ralph.state import RunState

        state = RunState.create(
            prd_path=Path("test.json"),
            tool="claude",
            iteration=1,
            story_id="US-001",
        )

        state.update_reeval(10)

        assert state.last_reeval_iteration == 10

    def test_run_state_serialization_with_reeval(self):
        """Test that last_reeval_iteration is serialized."""
        from ralph.state import RunState

        state = RunState.create(
            prd_path=Path("test.json"),
            tool="claude",
            iteration=1,
            story_id="US-001",
        )
        state.last_reeval_iteration = 10

        data = state.to_dict()
        restored = RunState.from_dict(data)

        assert restored.last_reeval_iteration == 10
