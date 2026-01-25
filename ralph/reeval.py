"""PRD re-evaluation logic for periodic health checks."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from ralph.prd import PRD, UserStory

logger = logging.getLogger("ralph.reeval")

# Constants
DEFAULT_REEVAL_INTERVAL = 10
REEVAL_TIMEOUT = 5 * 60  # 5 minutes
REEVAL_COMPLETE_SIGNAL = "<reeval-complete>"

# Safeguard limits
MAX_CHANGES_PER_REEVAL = 5  # Maximum changes allowed per re-evaluation
MIN_PENDING_STORIES = 1  # Must leave at least this many pending stories


class ChangeAction(Enum):
    """Types of changes that can be made to the PRD."""
    REMOVE = "remove"
    MERGE = "merge"
    MODIFY = "modify"


@dataclass
class ReEvalChange:
    """A single change proposed by re-evaluation."""
    action: ChangeAction
    story_id: str
    reason: str
    merge_into: str | None = None  # For merge actions
    new_data: dict[str, Any] | None = None  # For modify actions

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReEvalChange:
        """Create a ReEvalChange from a dictionary."""
        action_str = data.get("action", "").lower()
        try:
            action = ChangeAction(action_str)
        except ValueError:
            raise ValueError(f"Invalid action: {action_str}")

        return cls(
            action=action,
            story_id=data.get("story_id", ""),
            reason=data.get("reason", ""),
            merge_into=data.get("merge_into"),
            new_data=data.get("new_data"),
        )


@dataclass
class ReEvalResult:
    """Result of a re-evaluation."""
    changes: list[ReEvalChange] = field(default_factory=list)
    summary: str = ""
    raw_output: str = ""
    completed: bool = False
    error: str | None = None


@dataclass
class ValidationResult:
    """Result of validating proposed changes."""
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    approved_changes: list[ReEvalChange] = field(default_factory=list)
    rejected_changes: list[tuple[ReEvalChange, str]] = field(default_factory=list)


def generate_reeval_prompt(
    prd: PRD,
    progress_path: Path,
    current_iteration: int = 0,
    max_iterations: int = 0,
) -> str:
    """
    Generate the re-evaluation prompt for Claude.

    Args:
        prd: The current PRD object
        progress_path: Path to progress.txt file
        current_iteration: Current iteration number (for context)
        max_iterations: Maximum iterations configured

    Returns:
        The formatted prompt string
    """
    # Get progress stats
    completed_count = sum(1 for s in prd.user_stories if s.passes)
    total_count = len(prd.user_stories)

    # Separate completed and pending stories
    completed_stories = [s for s in prd.user_stories if s.passes]
    pending_stories = [s for s in prd.user_stories if not s.passes]

    # Format completed stories (read-only)
    completed_section = ""
    if completed_stories:
        completed_lines = []
        for story in sorted(completed_stories, key=lambda s: s.priority):
            completed_lines.append(f"- **{story.id}** (P{story.priority}): {story.title}")
        completed_section = "\n".join(completed_lines)
    else:
        completed_section = "_No stories completed yet_"

    # Format pending stories (may be modified)
    pending_section = ""
    if pending_stories:
        pending_lines = []
        for story in sorted(pending_stories, key=lambda s: s.priority):
            ac_list = "\n".join(f"    - {ac}" for ac in story.acceptance_criteria)
            notes_line = f"\n  - Notes: {story.notes}" if story.notes else ""
            pending_lines.append(
                f"### {story.id} (Priority {story.priority})\n"
                f"**Title:** {story.title}\n"
                f"**Description:** {story.description}\n"
                f"**Acceptance Criteria:**\n{ac_list}{notes_line}"
            )
        pending_section = "\n\n".join(pending_lines)
    else:
        pending_section = "_All stories completed!_"

    # Read progress.txt if it exists
    progress_content = ""
    if progress_path.exists():
        try:
            progress_content = progress_path.read_text()
            # Truncate if too long
            if len(progress_content) > 5000:
                progress_content = progress_content[-5000:]
                progress_content = "... (truncated)\n" + progress_content
        except Exception as e:
            progress_content = f"_Error reading progress.txt: {e}_"
    else:
        progress_content = "_No progress.txt file found_"

    # Build iteration context
    iteration_context = ""
    if current_iteration > 0 and max_iterations > 0:
        remaining = max_iterations - current_iteration
        iteration_context = f"""
- **Current Iteration:** {current_iteration} of {max_iterations}
- **Iterations Remaining:** {remaining}"""

    prompt = f"""# PRD Health Check

## Context
- **Project:** {prd.project}
- **Original Goals:** {prd.description}
- **Progress:** {completed_count}/{total_count} stories complete{iteration_context}

## Completed Stories (DO NOT MODIFY)
{completed_section}

## Pending Stories (may be modified)
{pending_section}

## Progress Learnings
```
{progress_content}
```

## Your Task

Analyze the pending stories for:

1. **Duplicates**: Stories that implement the same functionality
   - GOOD to merge: "Add login button" and "Implement login UI" (same feature)
   - BAD to merge: "Add login" and "Add logout" (different features)

2. **Obsolete (OBE)**: Stories no longer needed based on progress learnings
   - GOOD to remove: Story asks for X, but progress shows X was already built as part of another story
   - BAD to remove: Story still needed, just hasn't been started yet

3. **Clarification**: Descriptions that are ambiguous or need updating based on learnings
   - GOOD to modify: Update description to reflect architectural decisions made
   - BAD to modify: Expand scope beyond original intent

## Hard Constraints

1. **DO NOT modify completed stories** (passes=true) - they are locked
2. **DO NOT add new stories** - scope creep prevention
3. **DO NOT expand acceptance criteria** - can only reduce or clarify
4. **Maximum {MAX_CHANGES_PER_REEVAL} changes allowed** - be conservative
5. **Leave at least {MIN_PENDING_STORIES} pending story** - never remove all work
6. **Preserve story IDs** - never change the ID field
7. **When in doubt, make no changes** - prefer stability over optimization

## Response Format

Respond with ONLY a JSON object (no other text before it):

```json
{{
  "changes": [
    {{"action": "remove", "story_id": "US-XXX", "reason": "Specific reason why this is obsolete..."}},
    {{"action": "merge", "story_id": "US-XXX", "merge_into": "US-YYY", "reason": "These overlap because..."}},
    {{"action": "modify", "story_id": "US-XXX", "reason": "Clarifying based on...", "new_data": {{"description": "Updated description"}}}}
  ],
  "summary": "Brief summary"
}}
```

If no changes needed (the most common case):
```json
{{
  "changes": [],
  "summary": "All pending stories are valid and well-defined. No changes needed."
}}
```

Output `{REEVAL_COMPLETE_SIGNAL}` after the JSON to signal completion.
"""
    return prompt


def parse_reeval_response(output: str) -> ReEvalResult:
    """
    Parse the response from a re-evaluation run.

    Args:
        output: The raw output from Claude

    Returns:
        ReEvalResult with parsed changes
    """
    result = ReEvalResult(raw_output=output)

    # Check for completion signal
    result.completed = REEVAL_COMPLETE_SIGNAL in output

    # Try to extract JSON from the output using multiple strategies
    json_str = None

    # Strategy 1: Look for JSON in markdown code blocks (most reliable)
    # Use a more precise pattern that matches balanced braces
    code_block_match = re.search(r'```(?:json)?\s*(\{[^`]*\})\s*```', output)
    if code_block_match:
        json_str = code_block_match.group(1)

    # Strategy 2: Look for a JSON object that starts with {"changes"
    if not json_str:
        # Find all potential JSON objects starting with {"changes"
        # Use a simple brace-counting approach for better accuracy
        start_pattern = r'\{\s*"changes"\s*:'
        for match in re.finditer(start_pattern, output):
            start_idx = match.start()
            # Count braces to find the matching closing brace
            brace_count = 0
            end_idx = start_idx
            for i, char in enumerate(output[start_idx:], start=start_idx):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break
            if end_idx > start_idx:
                candidate = output[start_idx:end_idx]
                # Validate it's actually JSON
                try:
                    json.loads(candidate)
                    json_str = candidate
                    break
                except json.JSONDecodeError:
                    continue

    if not json_str:
        result.error = "Could not find JSON response in output"
        logger.warning(f"Failed to parse reeval response: {result.error}")
        return result

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        result.error = f"Invalid JSON: {e}"
        logger.warning(f"Failed to parse reeval JSON: {result.error}")
        return result

    # Validate the structure
    if not isinstance(data, dict):
        result.error = "JSON response is not an object"
        return result

    if "changes" not in data:
        result.error = "JSON response missing 'changes' field"
        return result

    # Parse changes
    changes_data = data.get("changes", [])
    if not isinstance(changes_data, list):
        result.error = "'changes' field is not an array"
        return result

    for change_data in changes_data:
        try:
            change = ReEvalChange.from_dict(change_data)
            result.changes.append(change)
        except (ValueError, KeyError) as e:
            logger.warning(f"Skipping invalid change: {e}")
            continue

    result.summary = data.get("summary", "")
    return result


def validate_changes(prd: PRD, changes: list[ReEvalChange]) -> ValidationResult:
    """
    Validate proposed changes against safeguards.

    Allowed:
    - Remove obsolete stories
    - Merge duplicate stories
    - Clarify descriptions/criteria
    - Adjust priorities of incomplete stories

    Blocked:
    - Modify completed stories (passes=true)
    - Add new stories
    - Change story IDs
    - Expand acceptance criteria count
    - Too many changes at once
    - Removing all pending stories

    Args:
        prd: The current PRD
        changes: List of proposed changes

    Returns:
        ValidationResult with approved/rejected changes
    """
    result = ValidationResult(valid=True)

    # Count pending stories
    pending_stories = [s for s in prd.user_stories if not s.passes]
    pending_count = len(pending_stories)

    # Limit total number of changes
    if len(changes) > MAX_CHANGES_PER_REEVAL:
        result.warnings.append(
            f"Too many changes proposed ({len(changes)}). "
            f"Only first {MAX_CHANGES_PER_REEVAL} will be considered."
        )
        changes = changes[:MAX_CHANGES_PER_REEVAL]

    # Count how many stories would be removed (including merges)
    removal_count = sum(
        1 for c in changes
        if c.action in (ChangeAction.REMOVE, ChangeAction.MERGE)
    )

    # Ensure we leave at least MIN_PENDING_STORIES
    if removal_count >= pending_count:
        max_removals = max(0, pending_count - MIN_PENDING_STORIES)
        if max_removals == 0:
            result.errors.append(
                f"Cannot remove any stories: only {pending_count} pending, "
                f"must keep at least {MIN_PENDING_STORIES}"
            )
            result.valid = False
            return result
        result.warnings.append(
            f"Too many removals proposed ({removal_count}). "
            f"Limiting to {max_removals} to preserve minimum pending stories."
        )

    # Build a map of story IDs to stories
    story_map = {s.id: s for s in prd.user_stories}

    # Track which stories are being removed to detect conflicts
    stories_being_removed: set[str] = set()
    # Track which stories are merge targets (protected from removal)
    stories_being_merged_into: set[str] = set()
    approved_count = 0

    for change in changes:
        story = story_map.get(change.story_id)

        # Check if story exists
        if story is None:
            result.rejected_changes.append(
                (change, f"Story '{change.story_id}' not found in PRD")
            )
            continue

        # Block modifications to completed stories
        if story.passes:
            result.rejected_changes.append(
                (change, f"Cannot modify completed story '{change.story_id}'")
            )
            continue

        # Validate based on action type
        if change.action == ChangeAction.REMOVE:
            # Check if we've hit the removal limit
            current_removals = len([
                c for c in result.approved_changes
                if c.action in (ChangeAction.REMOVE, ChangeAction.MERGE)
            ])
            max_removals = max(0, pending_count - MIN_PENDING_STORIES)
            if current_removals >= max_removals:
                result.rejected_changes.append(
                    (change, f"Removal limit reached (max {max_removals} to preserve {MIN_PENDING_STORIES} pending)")
                )
                continue

            # Check if this story is a merge target (protected)
            if change.story_id in stories_being_merged_into:
                result.rejected_changes.append(
                    (change, f"Cannot remove '{change.story_id}' which is a merge target")
                )
                continue

            # Allow removal of incomplete stories
            result.approved_changes.append(change)
            stories_being_removed.add(change.story_id)
            logger.info(f"Approved: Remove story '{change.story_id}' - {change.reason}")

        elif change.action == ChangeAction.MERGE:
            # Check if we've hit the removal limit (merge removes the source story)
            current_removals = len([
                c for c in result.approved_changes
                if c.action in (ChangeAction.REMOVE, ChangeAction.MERGE)
            ])
            max_removals = max(0, pending_count - MIN_PENDING_STORIES)
            if current_removals >= max_removals:
                result.rejected_changes.append(
                    (change, f"Removal limit reached (max {max_removals} to preserve {MIN_PENDING_STORIES} pending)")
                )
                continue

            # Validate merge target exists and is incomplete
            if not change.merge_into:
                result.rejected_changes.append(
                    (change, "Merge action missing 'merge_into' field")
                )
                continue

            target_story = story_map.get(change.merge_into)
            if target_story is None:
                result.rejected_changes.append(
                    (change, f"Merge target '{change.merge_into}' not found")
                )
                continue

            if target_story.passes:
                result.rejected_changes.append(
                    (change, f"Cannot merge into completed story '{change.merge_into}'")
                )
                continue

            # Check if target is already being removed by another change
            if change.merge_into in stories_being_removed:
                result.rejected_changes.append(
                    (change, f"Cannot merge into '{change.merge_into}' which is being removed")
                )
                continue

            result.approved_changes.append(change)
            stories_being_removed.add(change.story_id)
            # Protect the merge target from subsequent removal
            stories_being_merged_into.add(change.merge_into)
            logger.info(
                f"Approved: Merge story '{change.story_id}' into '{change.merge_into}' - {change.reason}"
            )

        elif change.action == ChangeAction.MODIFY:
            # Validate modification doesn't expand scope
            if not change.new_data:
                result.rejected_changes.append(
                    (change, "Modify action missing 'new_data' field")
                )
                continue

            # Check for blocked modifications
            new_data = change.new_data

            # Cannot change story ID
            if "id" in new_data and new_data["id"] != story.id:
                result.rejected_changes.append(
                    (change, "Cannot change story ID")
                )
                continue

            # Cannot expand acceptance criteria
            if "acceptanceCriteria" in new_data:
                new_ac = new_data["acceptanceCriteria"]
                if len(new_ac) > len(story.acceptance_criteria):
                    result.rejected_changes.append(
                        (change, f"Cannot expand acceptance criteria (was {len(story.acceptance_criteria)}, proposed {len(new_ac)})")
                    )
                    continue

            result.approved_changes.append(change)
            logger.info(f"Approved: Modify story '{change.story_id}' - {change.reason}")

        else:
            result.rejected_changes.append(
                (change, f"Unknown action: {change.action}")
            )

    # Set valid flag based on whether there were any rejections
    if result.rejected_changes:
        result.valid = False
        for change, reason in result.rejected_changes:
            result.errors.append(f"Rejected: {change.action.value} {change.story_id} - {reason}")
            logger.warning(f"Rejected change: {reason}")

    return result


def apply_changes(prd: PRD, changes: list[ReEvalChange]) -> tuple[PRD, list[str]]:
    """
    Apply validated changes to the PRD.

    Args:
        prd: The PRD to modify
        changes: List of validated changes to apply

    Returns:
        Tuple of (modified PRD, list of applied change descriptions)
    """
    applied: list[str] = []

    # Build story map for quick lookup
    story_map = {s.id: s for s in prd.user_stories}
    stories_to_remove: set[str] = set()

    for change in changes:
        if change.action == ChangeAction.REMOVE:
            stories_to_remove.add(change.story_id)
            applied.append(f"Removed story '{change.story_id}': {change.reason}")

        elif change.action == ChangeAction.MERGE:
            # Mark source story for removal
            stories_to_remove.add(change.story_id)

            # Update target story to incorporate source content
            if change.merge_into is None:
                continue  # Should not happen after validation
            target = story_map.get(change.merge_into)
            source = story_map.get(change.story_id)
            if target and source:
                # Merge acceptance criteria (add any unique ones from source)
                existing_ac = set(target.acceptance_criteria)
                for ac in source.acceptance_criteria:
                    if ac not in existing_ac:
                        target.acceptance_criteria.append(ac)

                # Add merge note with context
                merge_note = f"Merged from {change.story_id}: {source.title}"
                if source.description and source.description != source.title:
                    merge_note += f"\nOriginal description: {source.description}"
                if target.notes:
                    target.notes = f"{target.notes}\n{merge_note}"
                else:
                    target.notes = merge_note

            applied.append(
                f"Merged story '{change.story_id}' into '{change.merge_into}': {change.reason}"
            )

        elif change.action == ChangeAction.MODIFY:
            story = story_map.get(change.story_id)
            if story and change.new_data:
                # Apply modifications
                if "title" in change.new_data:
                    story.title = change.new_data["title"]
                if "description" in change.new_data:
                    story.description = change.new_data["description"]
                if "acceptanceCriteria" in change.new_data:
                    story.acceptance_criteria = change.new_data["acceptanceCriteria"]
                if "priority" in change.new_data:
                    story.priority = change.new_data["priority"]
                if "notes" in change.new_data:
                    story.notes = change.new_data["notes"]

                applied.append(f"Modified story '{change.story_id}': {change.reason}")

    # Remove stories marked for removal
    if stories_to_remove:
        prd.user_stories = [s for s in prd.user_stories if s.id not in stories_to_remove]

    return prd, applied


def append_reeval_to_progress(
    progress_path: Path,
    result: ReEvalResult,
    validation: ValidationResult,
    applied_changes: list[str],
) -> None:
    """
    Append re-evaluation results to progress.txt.

    Args:
        progress_path: Path to progress.txt
        result: The re-evaluation result
        validation: The validation result
        applied_changes: List of applied change descriptions
    """
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"\n## {timestamp} - PRD Re-Evaluation\n",
    ]

    if result.summary:
        lines.append(f"**Summary:** {result.summary}\n")

    if applied_changes:
        lines.append("\n**Applied Changes:**\n")
        for change_desc in applied_changes:
            lines.append(f"- {change_desc}\n")
    else:
        lines.append("\n_No changes applied_\n")

    if validation.rejected_changes:
        lines.append("\n**Rejected Changes:**\n")
        for change, reason in validation.rejected_changes:
            lines.append(f"- {change.action.value} {change.story_id}: {reason}\n")

    lines.append("\n---\n")

    # Append to progress file
    try:
        with open(progress_path, "a") as f:
            f.writelines(lines)
    except Exception as e:
        logger.error(f"Failed to append to progress.txt: {e}")
