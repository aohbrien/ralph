"""Unit tests for the two-phase orchestration feature.

This module tests the two-phase functionality:
- Plan extraction from output
- Planning prompt generation
- Coding prompt generation
- Phase execution functions
"""

from __future__ import annotations

from pathlib import Path

import pytest

from unittest.mock import patch

from ralph.prd import PRD, UserStory
from ralph.process import Tool, run_tool_with_prompt
from ralph.twophase import (
    IMPLEMENTATION_PLAN_END,
    IMPLEMENTATION_PLAN_START,
    PLANNING_COMPLETE_SIGNAL,
    extract_plan,
    generate_coding_prompt,
    generate_planning_prompt,
)


# =============================================================================
# Test fixtures
# =============================================================================


def create_test_prd() -> PRD:
    """Create a test PRD."""
    return PRD.from_dict({
        "project": "Test Project",
        "branchName": "test-branch",
        "description": "Test project description",
        "userStories": [
            {
                "id": "US-001",
                "title": "First story",
                "description": "First story description",
                "acceptanceCriteria": ["AC1", "AC2"],
                "priority": 1,
                "passes": False,
            },
        ],
    })


def create_test_story() -> UserStory:
    """Create a test user story."""
    return UserStory(
        id="US-001",
        title="Add authentication",
        description="As a user, I want to log in securely",
        acceptance_criteria=["Login form works", "Session persists"],
        priority=1,
        passes=False,
        notes="Use OAuth2",
    )


# =============================================================================
# Test extract_plan
# =============================================================================


class TestExtractPlan:
    """Tests for extract_plan function."""

    def test_extract_valid_plan(self):
        """Test extracting a valid plan from output."""
        output = f"""
Some output before the plan...

{IMPLEMENTATION_PLAN_START}
## Summary
This is the implementation plan summary.

## Files to Modify
- src/auth.py - Add login function

## Step-by-Step Implementation
1. **Add login route**: Create the /login endpoint
2. **Add session handling**: Store session in cookie
{IMPLEMENTATION_PLAN_END}

More output after the plan...
{PLANNING_COMPLETE_SIGNAL}
"""
        plan = extract_plan(output)
        assert plan is not None
        assert "## Summary" in plan
        assert "## Files to Modify" in plan
        assert "Add login route" in plan

    def test_extract_plan_no_tags(self):
        """Test that None is returned when no plan tags are present."""
        output = "Some output without any plan tags"
        plan = extract_plan(output)
        assert plan is None

    def test_extract_plan_empty_content(self):
        """Test that None is returned when plan content is empty."""
        output = f"""
{IMPLEMENTATION_PLAN_START}

{IMPLEMENTATION_PLAN_END}
"""
        plan = extract_plan(output)
        assert plan is None

    def test_extract_plan_whitespace_only(self):
        """Test that None is returned when plan is whitespace only."""
        output = f"""
{IMPLEMENTATION_PLAN_START}


{IMPLEMENTATION_PLAN_END}
"""
        plan = extract_plan(output)
        assert plan is None

    def test_extract_plan_multiline(self):
        """Test extracting a multiline plan."""
        output = f"""
{IMPLEMENTATION_PLAN_START}
## Summary
Line 1
Line 2
Line 3

## Details
More details here
{IMPLEMENTATION_PLAN_END}
"""
        plan = extract_plan(output)
        assert plan is not None
        assert "Line 1" in plan
        assert "Line 2" in plan
        assert "Line 3" in plan
        assert "More details" in plan

    def test_extract_plan_strips_whitespace(self):
        """Test that plan content is stripped of leading/trailing whitespace."""
        output = f"""
{IMPLEMENTATION_PLAN_START}

  Content here

{IMPLEMENTATION_PLAN_END}
"""
        plan = extract_plan(output)
        assert plan is not None
        assert plan == "Content here"

    def test_extract_plan_with_special_characters(self):
        """Test extracting plan with code blocks and special characters."""
        output = f"""
{IMPLEMENTATION_PLAN_START}
## Code Example
```python
def hello():
    print("Hello, world!")
```

Special chars: < > & " '
{IMPLEMENTATION_PLAN_END}
"""
        plan = extract_plan(output)
        assert plan is not None
        assert "```python" in plan
        assert 'print("Hello, world!")' in plan
        assert "< > & \" '" in plan


# =============================================================================
# Test generate_planning_prompt
# =============================================================================


class TestGeneratePlanningPrompt:
    """Tests for generate_planning_prompt function."""

    def test_basic_prompt_generation(self):
        """Test that basic prompt includes required elements."""
        story = create_test_story()
        prd = create_test_prd()
        prd_path = Path("/test/prd.json")

        prompt = generate_planning_prompt(
            story=story,
            prd=prd,
            iteration=1,
            max_iterations=10,
            prd_path=prd_path,
        )

        # Check story details are included
        assert "US-001" in prompt
        assert "Add authentication" in prompt
        assert "As a user, I want to log in securely" in prompt

        # Check acceptance criteria are included
        assert "Login form works" in prompt
        assert "Session persists" in prompt

        # Check notes are included
        assert "Use OAuth2" in prompt

        # Check iteration context
        assert "1 of 10" in prompt

        # Check planning phase markers
        assert IMPLEMENTATION_PLAN_START in prompt
        assert IMPLEMENTATION_PLAN_END in prompt
        assert PLANNING_COMPLETE_SIGNAL in prompt

        # Check instruction to NOT implement
        assert "do NOT implement" in prompt.lower() or "do not implement" in prompt.lower()

    def test_prompt_includes_prd_context(self):
        """Test that prompt includes PRD context."""
        story = create_test_story()
        prd = create_test_prd()
        prd_path = Path("/test/prd.json")

        prompt = generate_planning_prompt(
            story=story,
            prd=prd,
            iteration=1,
            max_iterations=10,
            prd_path=prd_path,
        )

        assert "Test Project" in prompt
        assert str(prd_path) in prompt

    def test_prompt_without_notes(self):
        """Test prompt generation when story has no notes."""
        story = UserStory(
            id="US-002",
            title="Simple story",
            description="A simple story",
            acceptance_criteria=["AC1"],
            priority=1,
            passes=False,
            notes=None,
        )
        prd = create_test_prd()

        prompt = generate_planning_prompt(
            story=story,
            prd=prd,
            iteration=1,
            max_iterations=10,
            prd_path=Path("/test/prd.json"),
        )

        # Should not have a Notes section if notes is None
        assert "**Notes:**" not in prompt


# =============================================================================
# Test generate_coding_prompt
# =============================================================================


class TestGenerateCodingPrompt:
    """Tests for generate_coding_prompt function."""

    def test_basic_coding_prompt(self):
        """Test that coding prompt includes required elements."""
        story = create_test_story()
        prd = create_test_prd()
        prd_path = Path("/test/prd.json")
        plan = """## Summary
Implement authentication.

## Files to Modify
- src/auth.py

## Steps
1. Add login function
"""

        prompt = generate_coding_prompt(
            story=story,
            prd=prd,
            plan=plan,
            iteration=1,
            max_iterations=10,
            prd_path=prd_path,
        )

        # Check story details are included
        assert "US-001" in prompt
        assert "Add authentication" in prompt

        # Check plan is included
        assert "## Summary" in prompt
        assert "Implement authentication" in prompt
        assert "Add login function" in prompt

        # Check completion instructions
        assert "<promise>COMPLETE</promise>" in prompt
        assert "ralph mark-complete" in prompt

    def test_coding_prompt_includes_plan(self):
        """Test that the full plan is embedded in the coding prompt."""
        story = create_test_story()
        prd = create_test_prd()
        plan = """## Detailed Plan
Step 1: Do this
Step 2: Do that
Step 3: Do the other thing
"""

        prompt = generate_coding_prompt(
            story=story,
            prd=prd,
            plan=plan,
            iteration=5,
            max_iterations=20,
            prd_path=Path("/test/prd.json"),
        )

        assert "Step 1: Do this" in prompt
        assert "Step 2: Do that" in prompt
        assert "Step 3: Do the other thing" in prompt
        assert "5 of 20" in prompt

    def test_coding_prompt_has_phase_context(self):
        """Test that coding prompt indicates it's the CODING phase."""
        story = create_test_story()
        prd = create_test_prd()

        prompt = generate_coding_prompt(
            story=story,
            prd=prd,
            plan="Simple plan",
            iteration=1,
            max_iterations=10,
            prd_path=Path("/test/prd.json"),
        )

        assert "CODING" in prompt or "Coding" in prompt


# =============================================================================
# Test Tool enum
# =============================================================================


class TestToolEnum:
    """Tests for the Tool enum with OPENCODE."""

    def test_opencode_tool_exists(self):
        """Test that OPENCODE tool is available."""
        assert Tool.OPENCODE.value == "opencode"

    def test_ccs_tool_exists(self):
        """Test that CCS tool is available."""
        assert Tool.CCS.value == "ccs"

    def test_all_tools(self):
        """Test all tools are available."""
        tools = [t.value for t in Tool]
        assert "claude" in tools
        assert "amp" in tools
        assert "opencode" in tools
        assert "ccs" in tools

    def test_ccs_dispatch_defaults_include_dangerously_skip_permissions_and_print(self):
        """By default ccs invocations pass --dangerously-skip-permissions --print through."""
        with patch("ralph.process.stream_process") as mock_stream:
            run_tool_with_prompt(Tool.CCS, "hello world")

        assert mock_stream.call_count == 1
        kwargs = mock_stream.call_args.kwargs
        assert kwargs["cmd"] == [
            "ccs",
            "--dangerously-skip-permissions",
            "--print",
            "hello world",
        ]
        assert kwargs["input_text"] is None

    def test_ccs_dispatch_with_profile_and_extra_args(self):
        """ccs_profile is positional; ccs_args is shlex-split; prompt trails."""
        with patch("ralph.process.stream_process") as mock_stream:
            run_tool_with_prompt(
                Tool.CCS,
                "hi",
                ccs_profile="personal2",
                ccs_args="--dangerously-skip-permissions --print --foo bar",
            )

        kwargs = mock_stream.call_args.kwargs
        assert kwargs["cmd"] == [
            "ccs",
            "personal2",
            "--dangerously-skip-permissions",
            "--print",
            "--foo",
            "bar",
            "hi",
        ]

    def test_ccs_dispatch_empty_ccs_args_opts_out_of_defaults(self):
        """Passing ccs_args='' suppresses the default passthrough flags."""
        with patch("ralph.process.stream_process") as mock_stream:
            run_tool_with_prompt(Tool.CCS, "hi", ccs_profile="personal2", ccs_args="")

        kwargs = mock_stream.call_args.kwargs
        assert kwargs["cmd"] == ["ccs", "personal2", "hi"]


# =============================================================================
# Integration-style tests
# =============================================================================


class TestTwoPhaseIntegration:
    """Integration-style tests for two-phase orchestration."""

    def test_planning_prompt_produces_extractable_plan_format(self):
        """Test that the planning prompt template includes correct tags for extraction."""
        story = create_test_story()
        prd = create_test_prd()

        prompt = generate_planning_prompt(
            story=story,
            prd=prd,
            iteration=1,
            max_iterations=10,
            prd_path=Path("/test/prd.json"),
        )

        # The prompt should show the format for the agent to use
        assert IMPLEMENTATION_PLAN_START in prompt
        assert IMPLEMENTATION_PLAN_END in prompt

    def test_extracted_plan_can_be_passed_to_coding_prompt(self):
        """Test the flow: extract plan -> pass to coding prompt."""
        # Simulate a planning phase output
        planning_output = f"""
Analyzing story US-001...
Reading codebase...

{IMPLEMENTATION_PLAN_START}
## Summary
Add OAuth2 authentication.

## Files to Modify
- src/auth.py - Add OAuth2 client

## Steps
1. Install oauth2 library
2. Create OAuth2 client
3. Add login endpoint
{IMPLEMENTATION_PLAN_END}

{PLANNING_COMPLETE_SIGNAL}
"""

        # Extract the plan
        plan = extract_plan(planning_output)
        assert plan is not None

        # Use the extracted plan in coding prompt
        story = create_test_story()
        prd = create_test_prd()

        coding_prompt = generate_coding_prompt(
            story=story,
            prd=prd,
            plan=plan,
            iteration=1,
            max_iterations=10,
            prd_path=Path("/test/prd.json"),
        )

        # Verify the plan content is in the coding prompt
        assert "Add OAuth2 authentication" in coding_prompt
        assert "Install oauth2 library" in coding_prompt
        assert "Create OAuth2 client" in coding_prompt
