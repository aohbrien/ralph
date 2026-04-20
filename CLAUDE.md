# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Ralph is an autonomous AI agent loop for PRD-driven development. It orchestrates Claude Code (or Amp) in an iterative loop to implement user stories defined in a `prd.json` file. Each iteration picks the highest priority incomplete story, implements it, runs checks, and marks it complete.

## Commands

```bash
# Install in development mode
pip install -e ".[dev]"

# Run the agent loop (single-phase: one tool does everything)
ralph run --prd path/to/prd.json

# Run with two-phase orchestration (Claude plans, OpenCode codes)
ralph run --two-phase --prd path/to/prd.json

# Two-phase with custom tools
ralph run --two-phase --planning-tool claude --coding-tool amp

# Check PRD progress
ralph status --prd path/to/prd.json

# Validate PRD structure
ralph validate --prd path/to/prd.json

# Initialize a new PRD project
ralph init --project "My Project"

# Resume an interrupted run
ralph resume --prd path/to/prd.json

# Resume in two-phase mode
ralph resume --two-phase --prd path/to/prd.json

# Archive current run state
ralph archive --prd path/to/prd.json

# Run tests (parallel, excludes slow tests by default)
pytest

# Run all tests including slow integration tests
pytest -m ""

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
- `run_tool()` pipes prompt file to `claude --dangerously-skip-permissions --print`, `amp --dangerously-allow-all`, `opencode -p <prompt> -q`, or `ccs [profile] --dangerously-skip-permissions --print <prompt>`
- `ManagedProcess` allows external termination (for signal handlers)
- Supports four tools: `claude`, `amp`, `opencode`, `ccs`

**Two-Phase Orchestration**: `ralph/twophase.py`
- Separates planning and coding into two phases
- Phase 1 (Planning): Analyzes story and creates implementation plan
- Phase 2 (Coding): Executes the plan step-by-step
- Plan extraction via `<implementation-plan>` tags
- Configurable tools and timeouts per phase

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

## Two-Phase Orchestration

Two-phase mode separates planning from implementation, leveraging different AI models for each phase:

```bash
ralph run --two-phase --prd path/to/prd.json
```

### How It Works

1. **Phase 1 (Planning)**: The planning tool (default: Claude) analyzes the story, explores the codebase, and outputs a detailed implementation plan within `<implementation-plan>` tags.

2. **Phase 2 (Coding)**: The coding tool (default: OpenCode) receives the extracted plan and executes it step-by-step.

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--two-phase` | false | Enable two-phase orchestration |
| `--planning-tool` | claude | Tool for planning: claude, amp, opencode, ccs |
| `--coding-tool` | opencode | Tool for coding: claude, amp, opencode, ccs |
| `--planning-timeout` | 10 min | Planning phase timeout |
| `--coding-timeout` | 30 min | Coding phase timeout |

### Plan Format

The planning phase must output a plan in this format:

```markdown
<implementation-plan>
## Summary
Brief overview of the implementation approach.

## Files to Modify
- path/to/file.py - Description of changes

## Step-by-Step Implementation
1. **Step Title**: Detailed instructions
2. **Step Title**: Detailed instructions

## Verification Commands
- pytest tests/
- mypy ralph/
</implementation-plan>
```

### OpenCode Configuration

OpenCode uses Gemini 3 Pro by default. Configure via `.opencode.json`:

```json
{
  "provider": "gemini",
  "model": "gemini-3-pro"
}
```

### CCS (Claude Code Switch)

`ccs` is a wrapper that switches between Claude accounts and other runtimes (Codex, GLM, OpenRouter, etc.) under a stable command surface. Install and configure separately:

```bash
npm install -g @kaitranntt/ccs
ccs config
```

Ralph invokes it as `ccs [profile] [passthrough args...] <prompt>`:

- `--ccs-profile <name>` — ccs profile/account name (e.g. `personal2`). Passed as the first positional arg.
- `--ccs-args "<flags>"` — extra args forwarded through ccs to the underlying CLI. Defaults to `--dangerously-skip-permissions --print` — claude's own headless mode. We deliberately avoid ccs's `-p` delegation flag because it requires a separately configured delegation profile; account instances (`ccs auth create <name>`) only work with claude's `--print`. Pass `--ccs-args ""` to suppress the default.

Example matching the typical `ccs personal2 --dangerously-skip-permissions` invocation:

```bash
ralph run --tool ccs --ccs-profile personal2 --prd prd.json
```

Both flags are also available on `ralph resume` and work with `--planning-tool ccs` / `--coding-tool ccs` in two-phase mode.

## Completion Protocol

When running inside Ralph's loop, output `<promise>COMPLETE</promise>` after all stories pass to signal completion. Ralph detects this in the output stream and exits successfully.
