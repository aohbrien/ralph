# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Ralph is an autonomous AI agent loop for PRD-driven development. It orchestrates Claude Code (or Amp) in an iterative loop to implement user stories defined in a `prd.json` file. Each iteration picks the highest priority incomplete story, implements it, runs checks, and marks it complete.

## Commands

```bash
# Install in development mode
pip install -e ".[dev]"

# Run the agent loop
ralph run --prd path/to/prd.json

# Check PRD progress
ralph status --prd path/to/prd.json

# Validate PRD structure
ralph validate --prd path/to/prd.json

# Initialize a new PRD project
ralph init --project "My Project"

# Resume an interrupted run
ralph resume --prd path/to/prd.json

# Archive current run state
ralph archive --prd path/to/prd.json

# Run tests
pytest

# Type checking
mypy ralph
```

## Architecture

**Entry Point**: `ralph/cli.py` - Typer CLI with commands: `run`, `status`, `validate`, `init`, `resume`, `archive`, `tasks`

**Core Loop**: `ralph/runner.py`
- `Runner` class orchestrates the main loop
- Iterates up to max_iterations, picking highest priority incomplete story
- Runs the AI tool (claude/amp) with the prompt file
- Detects completion via `<promise>COMPLETE</promise>` signal in output
- Handles SIGINT/SIGTERM gracefully with state persistence

**PRD Model**: `ralph/prd.py`
- `PRD` and `UserStory` dataclasses parse `prd.json`
- Stories have: id, title, description, acceptanceCriteria, priority, passes (bool)
- `get_next_story()` returns lowest-priority-number story where `passes=false`

**Process Execution**: `ralph/process.py`
- Uses PTY for unbuffered real-time streaming output
- `run_tool()` pipes prompt file to `claude --dangerously-skip-permissions --print` or `amp --dangerously-allow-all`
- `ManagedProcess` allows external termination (for signal handlers)

**State Management**: `ralph/state.py`
- Persists run state to `.ralph-state.json` for resume functionality
- Tracks last iteration, story ID, timestamps

**Branch/Archive**: `ralph/archive.py`
- Detects branch changes via `.last-branch` file
- Archives previous run's `prd.json` and `progress.txt` to `archive/YYYY-MM-DD-branchname/`

## PRD Format

```json
{
  "project": "Project Name",
  "branchName": "ralph/feature-name",
  "description": "Project description",
  "userStories": [
    {
      "id": "US-001",
      "title": "Story title",
      "description": "As a user...",
      "acceptanceCriteria": ["Criterion 1", "Criterion 2"],
      "priority": 1,
      "passes": false,
      "notes": "Optional notes"
    }
  ]
}
```

## Completion Protocol

When running inside Ralph's loop, output `<promise>COMPLETE</promise>` after all stories pass to signal completion. Ralph detects this in the output stream and exits successfully.
