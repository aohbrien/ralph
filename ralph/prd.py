"""PRD (Product Requirements Document) parsing and management."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class UserStory:
    """A single user story in the PRD."""
    id: str
    title: str
    description: str
    acceptance_criteria: list[str]
    priority: int
    passes: bool = False
    notes: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UserStory:
        """Create a UserStory from a dictionary."""
        return cls(
            id=data["id"],
            title=data["title"],
            description=data["description"],
            acceptance_criteria=data.get("acceptanceCriteria", []),
            priority=data.get("priority", 999),
            passes=data.get("passes", False),
            notes=data.get("notes", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "acceptanceCriteria": self.acceptance_criteria,
            "priority": self.priority,
            "passes": self.passes,
            "notes": self.notes,
        }


@dataclass
class PRD:
    """Product Requirements Document."""
    project: str
    branch_name: str
    description: str
    user_stories: list[UserStory] = field(default_factory=list)

    @classmethod
    def from_file(cls, path: Path) -> PRD:
        """Load a PRD from a JSON file."""
        data = json.loads(path.read_text())
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PRD:
        """Create a PRD from a dictionary."""
        stories = [UserStory.from_dict(s) for s in data.get("userStories", [])]
        return cls(
            project=data.get("project", ""),
            branch_name=data.get("branchName", ""),
            description=data.get("description", ""),
            user_stories=stories,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "project": self.project,
            "branchName": self.branch_name,
            "description": self.description,
            "userStories": [s.to_dict() for s in self.user_stories],
        }

    def save(self, path: Path) -> None:
        """Save the PRD to a JSON file."""
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    def get_next_story(self) -> UserStory | None:
        """
        Get the next story to work on.

        Returns the highest priority story (lowest number) where passes=false.
        Returns None if all stories are complete.
        """
        pending = [s for s in self.user_stories if not s.passes]
        if not pending:
            return None
        return min(pending, key=lambda s: s.priority)

    def mark_story_complete(self, story_id: str) -> bool:
        """
        Mark a story as complete (passes=true).

        Returns True if the story was found and updated.
        """
        for story in self.user_stories:
            if story.id == story_id:
                story.passes = True
                return True
        return False

    def is_complete(self) -> bool:
        """Check if all stories have been completed."""
        return all(s.passes for s in self.user_stories)

    def get_progress(self) -> tuple[int, int]:
        """
        Get progress as (completed, total).
        """
        completed = sum(1 for s in self.user_stories if s.passes)
        return completed, len(self.user_stories)

    def get_story_by_id(self, story_id: str) -> UserStory | None:
        """Get a story by its ID."""
        for story in self.user_stories:
            if story.id == story_id:
                return story
        return None

    def validate(self) -> tuple[list[str], list[str]]:
        """
        Validate the PRD structure.

        Returns a tuple of (errors, warnings).
        Errors indicate invalid structure, warnings indicate potential issues.
        """
        errors: list[str] = []
        warnings: list[str] = []

        # Check required PRD fields
        if not self.project:
            errors.append("Missing required field: project")
        if not self.branch_name:
            errors.append("Missing required field: branchName")
        if not self.user_stories:
            errors.append("No user stories defined (userStories array is empty)")

        # Validate each user story
        for i, story in enumerate(self.user_stories):
            story_prefix = f"Story {i + 1}"
            if story.id:
                story_prefix = f"Story '{story.id}'"

            # Required story fields
            if not story.id:
                errors.append(f"{story_prefix}: Missing required field 'id'")
            if not story.title:
                errors.append(f"{story_prefix}: Missing required field 'title'")
            if not story.acceptance_criteria:
                errors.append(f"{story_prefix}: Missing or empty 'acceptanceCriteria'")

        # Check for duplicate story IDs
        story_ids = [s.id for s in self.user_stories if s.id]
        seen_ids: set[str] = set()
        for sid in story_ids:
            if sid in seen_ids:
                errors.append(f"Duplicate story ID: '{sid}'")
            seen_ids.add(sid)

        # Check priorities are sequential (warning only)
        if self.user_stories:
            priorities = sorted(s.priority for s in self.user_stories)
            expected = list(range(1, len(priorities) + 1))
            if priorities != expected:
                warnings.append(
                    f"Priorities are not sequential 1..{len(priorities)}: "
                    f"found {priorities}"
                )

        return errors, warnings


def validate_prd_file(path: Path) -> tuple[list[str], list[str]]:
    """
    Validate a PRD file.

    Returns a tuple of (errors, warnings).
    This function catches JSON parsing errors as well as structural errors.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # Check file exists
    if not path.exists():
        return [f"File not found: {path}"], []

    # Try to parse JSON
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        return [f"Invalid JSON: {e}"], []

    # Check it's a dict
    if not isinstance(data, dict):
        return ["PRD must be a JSON object, not an array or primitive"], []

    # Validate raw structure before parsing
    # Check required PRD fields
    if not data.get("project"):
        errors.append("Missing required field: project")
    if not data.get("branchName"):
        errors.append("Missing required field: branchName")

    user_stories = data.get("userStories", [])
    if not isinstance(user_stories, list):
        errors.append("userStories must be an array")
        return errors, warnings

    if not user_stories:
        errors.append("No user stories defined (userStories array is empty)")

    # Validate each story structure
    story_ids: list[str] = []
    priorities: list[int] = []

    for i, story in enumerate(user_stories):
        if not isinstance(story, dict):
            errors.append(f"Story {i + 1}: Must be an object")
            continue

        story_id = story.get("id", "")
        story_prefix = f"Story '{story_id}'" if story_id else f"Story {i + 1}"

        # Required story fields
        if not story_id:
            errors.append(f"{story_prefix}: Missing required field 'id'")
        else:
            story_ids.append(story_id)

        if not story.get("title"):
            errors.append(f"{story_prefix}: Missing required field 'title'")

        ac = story.get("acceptanceCriteria")
        if not ac or not isinstance(ac, list) or len(ac) == 0:
            errors.append(f"{story_prefix}: Missing or empty 'acceptanceCriteria'")

        # Track priority for sequential check
        priority = story.get("priority")
        if isinstance(priority, int):
            priorities.append(priority)

    # Check for duplicate story IDs
    seen_ids: set[str] = set()
    for sid in story_ids:
        if sid in seen_ids:
            errors.append(f"Duplicate story ID: '{sid}'")
        seen_ids.add(sid)

    # Check priorities are sequential (warning only)
    if priorities:
        sorted_priorities = sorted(priorities)
        expected = list(range(1, len(sorted_priorities) + 1))
        if sorted_priorities != expected:
            warnings.append(
                f"Priorities are not sequential 1..{len(sorted_priorities)}: "
                f"found {sorted_priorities}"
            )

    return errors, warnings
