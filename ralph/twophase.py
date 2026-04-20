"""Two-phase orchestration for Ralph (planning + coding)."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ralph.prd import PRD, UserStory
from ralph.process import ManagedProcess, ProcessResult, Tool, run_tool_with_prompt

logger = logging.getLogger("ralph.twophase")

# Signals for two-phase orchestration
PLANNING_COMPLETE_SIGNAL = "<planning-complete>"
IMPLEMENTATION_PLAN_START = "<implementation-plan>"
IMPLEMENTATION_PLAN_END = "</implementation-plan>"


@dataclass
class PlanResult:
    """Result from the planning phase."""
    plan: str | None
    output: str
    success: bool
    error: str | None = None


@dataclass
class TwoPhaseResult:
    """Result from a two-phase iteration."""
    planning_result: ProcessResult
    coding_result: ProcessResult | None
    plan: str | None
    story_id: str | None
    completed: bool


def extract_plan(output: str) -> str | None:
    """
    Extract the implementation plan from the planning phase output.

    Looks for content between <implementation-plan> and </implementation-plan> tags.

    Args:
        output: Full output from the planning phase

    Returns:
        Extracted plan text, or None if no valid plan found
    """
    pattern = re.compile(
        rf"{re.escape(IMPLEMENTATION_PLAN_START)}\s*(.*?)\s*{re.escape(IMPLEMENTATION_PLAN_END)}",
        re.DOTALL,
    )
    match = pattern.search(output)
    if match:
        plan = match.group(1).strip()
        if plan:
            return plan
    return None


def generate_planning_prompt(
    story: UserStory,
    prd: PRD,
    iteration: int,
    max_iterations: int,
    prd_path: Path,
) -> str:
    """
    Generate the prompt for the planning phase.

    This prompt instructs the planning tool (typically Claude) to analyze
    the story and create a detailed implementation plan.

    Args:
        story: The user story to plan
        prd: The PRD containing the story
        iteration: Current iteration number
        max_iterations: Maximum iterations allowed
        prd_path: Path to the PRD file

    Returns:
        The planning prompt text
    """
    criteria_list = "\n".join(f"- {c}" for c in story.acceptance_criteria)
    notes_section = f"\n**Notes:** {story.notes}" if story.notes else ""

    return f"""# Planning Phase: User Story {story.id}

## Context
- **Iteration:** {iteration} of {max_iterations}
- **PRD Path:** {prd_path}
- **Project:** {prd.project}
- **Phase:** PLANNING (do NOT implement yet)

## Story Details
**ID:** {story.id}
**Title:** {story.title}
**Description:** {story.description}

**Acceptance Criteria:**
{criteria_list}
{notes_section}

## Your Task

You are in the PLANNING phase. Your goal is to analyze this story and create a detailed implementation plan. Do NOT write any code or make any changes yet.

### Steps to Follow:

1. **Read CLAUDE.md** for project context and coding conventions
2. **Explore the codebase** to understand existing patterns and architecture
3. **Identify all files** that need to be created or modified
4. **Plan the implementation** step by step
5. **List verification commands** to run after implementation

### Output Format

After your analysis, output your plan in the following format:

{IMPLEMENTATION_PLAN_START}
## Summary
Brief overview of the implementation approach.

## Files to Modify
- path/to/file.py - Description of changes

## Files to Create (if any)
- path/to/new_file.py - Purpose of new file

## Step-by-Step Implementation
1. **Step Title**: Detailed instructions for what to change/add
2. **Step Title**: Detailed instructions for what to change/add
(continue as needed)

## Verification Commands
- command1  # What this verifies
- command2  # What this verifies
{IMPLEMENTATION_PLAN_END}

After outputting the plan, output:
{PLANNING_COMPLETE_SIGNAL}

## Important Rules

- Do NOT implement anything yet - only plan
- Be specific about file paths and function/class names
- Include enough detail that another agent can execute the plan without additional context
- Reference specific line numbers or code patterns where helpful
- Consider edge cases and error handling in your plan
"""


def generate_coding_prompt(
    story: UserStory,
    prd: PRD,
    plan: str,
    iteration: int,
    max_iterations: int,
    prd_path: Path,
) -> str:
    """
    Generate the prompt for the coding phase.

    This prompt provides the extracted plan and instructs the coding tool
    (typically OpenCode) to execute the implementation.

    Args:
        story: The user story to implement
        prd: The PRD containing the story
        plan: The extracted implementation plan from planning phase
        iteration: Current iteration number
        max_iterations: Maximum iterations allowed
        prd_path: Path to the PRD file

    Returns:
        The coding prompt text
    """
    criteria_list = "\n".join(f"- [ ] {c}" for c in story.acceptance_criteria)
    notes_section = f"\n**Notes:** {story.notes}" if story.notes else ""

    return f"""# Coding Phase: User Story {story.id}

## Context
- **Iteration:** {iteration} of {max_iterations}
- **PRD Path:** {prd_path}
- **Project:** {prd.project}
- **Phase:** CODING (execute the plan)

## Story Details
**ID:** {story.id}
**Title:** {story.title}
**Description:** {story.description}

**Acceptance Criteria:**
{criteria_list}
{notes_section}

## Implementation Plan

A planning phase has analyzed this story and created the following implementation plan.
Follow it step-by-step:

---
{plan}
---

## Your Task

Execute the implementation plan above:

1. Follow each step in the plan exactly
2. Make the code changes as specified
3. Run the verification commands listed in the plan
4. Mark the story complete when all criteria are met

## Available CLI Tools

- `ralph story {story.id} --prd {prd_path}` - View this story's details
- `ralph mark-complete {story.id} --prd {prd_path}` - Mark this story as complete
- `ralph status --prd {prd_path}` - View overall PRD progress

## Completion

When all acceptance criteria are satisfied:
1. Run `ralph mark-complete {story.id} --prd {prd_path}`
2. Update progress.txt with a summary of what you did
3. Output <promise>COMPLETE</promise> to signal completion

## Important

- Follow the plan step by step
- Run all verification commands before marking complete
- Ensure ALL acceptance criteria are satisfied
"""


def run_planning_phase(
    story: UserStory,
    prd: PRD,
    iteration: int,
    max_iterations: int,
    prd_path: Path,
    tool: Tool,
    cwd: Path,
    on_output: Callable[[str], None] | None = None,
    managed_process: ManagedProcess | None = None,
    timeout: float | None = None,
) -> tuple[ProcessResult, str | None]:
    """
    Run the planning phase of two-phase orchestration.

    Args:
        story: The user story to plan
        prd: The PRD containing the story
        iteration: Current iteration number
        max_iterations: Maximum iterations
        prd_path: Path to the PRD file
        tool: Planning tool to use (typically Claude)
        cwd: Working directory
        on_output: Output callback
        managed_process: For external termination control
        timeout: Planning phase timeout in seconds

    Returns:
        Tuple of (ProcessResult, extracted plan or None)
    """
    prompt = generate_planning_prompt(
        story=story,
        prd=prd,
        iteration=iteration,
        max_iterations=max_iterations,
        prd_path=prd_path,
    )

    logger.debug(f"Running planning phase with {tool.value}")
    result = run_tool_with_prompt(
        tool=tool,
        prompt=prompt,
        on_output=on_output,
        cwd=cwd,
        managed_process=managed_process,
        timeout=timeout,
    )

    # Extract plan from output
    plan = extract_plan(result.output)

    if plan:
        logger.debug(f"Successfully extracted plan ({len(plan)} chars)")
    else:
        logger.debug("Failed to extract plan from output")

    return result, plan


def run_coding_phase(
    story: UserStory,
    prd: PRD,
    plan: str,
    iteration: int,
    max_iterations: int,
    prd_path: Path,
    tool: Tool,
    cwd: Path,
    on_output: Callable[[str], None] | None = None,
    managed_process: ManagedProcess | None = None,
    timeout: float | None = None,
) -> ProcessResult:
    """
    Run the coding phase of two-phase orchestration.

    Args:
        story: The user story to implement
        prd: The PRD containing the story
        plan: The implementation plan from planning phase
        iteration: Current iteration number
        max_iterations: Maximum iterations
        prd_path: Path to the PRD file
        tool: Coding tool to use (typically OpenCode)
        cwd: Working directory
        on_output: Output callback
        managed_process: For external termination control
        timeout: Coding phase timeout in seconds

    Returns:
        ProcessResult from the coding phase
    """
    prompt = generate_coding_prompt(
        story=story,
        prd=prd,
        plan=plan,
        iteration=iteration,
        max_iterations=max_iterations,
        prd_path=prd_path,
    )

    logger.debug(f"Running coding phase with {tool.value}")
    return run_tool_with_prompt(
        tool=tool,
        prompt=prompt,
        on_output=on_output,
        cwd=cwd,
        managed_process=managed_process,
        timeout=timeout,
    )
