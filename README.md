# Ralph

Autonomous AI agent loop for PRD-driven development.

Ralph orchestrates Claude Code (or Amp) to iteratively implement user stories defined in a `prd.json` file. Each iteration picks the highest priority incomplete story, implements it, and marks it complete.

## Installation

```bash
pip install -e .
```

## Quick Start

1. Initialize a new project:
   ```bash
   ralph init --project "My Feature"
   ```

2. Edit `prd.json` with your user stories

3. Run the agent loop:
   ```bash
   ralph run
   ```

## Commands

| Command | Description |
|---------|-------------|
| `ralph run` | Run the agent loop |
| `ralph status` | Show PRD progress |
| `ralph validate` | Validate PRD structure |
| `ralph init` | Initialize a new PRD project |
| `ralph resume` | Resume an interrupted run |
| `ralph archive` | Archive current run state |
| `ralph tasks` | Show Claude task list |
| `ralph usage` | Show usage statistics and costs |

## PRD Format

```json
{
  "project": "My Feature",
  "branchName": "ralph/my-feature",
  "description": "Feature description",
  "userStories": [
    {
      "id": "US-001",
      "title": "User story title",
      "description": "As a user, I want to...",
      "acceptanceCriteria": [
        "Criterion 1",
        "Criterion 2"
      ],
      "priority": 1,
      "passes": false
    }
  ]
}
```

Stories are processed in priority order (lowest number first). Set `passes: true` when a story is complete.

## Options

```bash
ralph run --help
```

Key options:
- `--max-iterations, -n` - Maximum iterations (default: 10)
- `--tool, -t` - AI tool: `claude`, `amp`, `opencode`, or `ccs` (default: claude)
- `--timeout` - Per-iteration timeout in minutes (default: 30)
- `--verbose, -v` - Enable verbose output
- `--dry-run` - Show plan without executing
- `--limit-mode` - Limit detection: `plan`, `p90`, or `hybrid` (default: p90)
- `--cost-tracking/--no-cost-tracking` - Enable/disable cost display (default: enabled)

## Usage Tracking & Cost Management

Ralph tracks Claude API usage and calculates costs based on model-specific pricing.

### View Usage

```bash
ralph usage              # Show current usage with costs
ralph usage --p90        # Show P90-calculated limit
ralph usage --history 7  # Show 7 days of usage history
ralph usage --no-costs   # Hide cost information
```

### Limit Detection Modes

Ralph supports three modes for determining token limits:

| Mode | Description |
|------|-------------|
| `p90` | Auto-detect limits from usage history (90th percentile) **(default)** |
| `plan` | Use hardcoded limits based on your Claude plan |
| `hybrid` | Use P90 if higher than plan limits, otherwise use plan |

```bash
ralph run --limit-mode p90      # Use auto-detected limits
ralph run --limit-mode hybrid   # Use whichever is higher
```

### Set Your Plan

```bash
ralph usage --set-plan pro      # Set plan: free, pro, max5x, max20x
```

### Cost Tracking

Costs are calculated using official Anthropic pricing for all Claude models including:
- Claude Opus 4.5, Opus 4
- Claude Sonnet 4, Sonnet 3.5
- Claude Haiku 3.5, Haiku 3

Costs are displayed after each iteration and in usage summaries.

## How It Works

1. Ralph reads `prd.json` and finds the highest priority story where `passes: false`
2. Runs the AI tool with the prompt file (`CLAUDE.md` for Claude, `prompt.md` for Amp)
3. The AI implements the story and updates `prd.json`
4. Ralph checks if all stories are complete or continues to the next iteration
5. Completion is signaled by `<promise>COMPLETE</promise>` in the output

## License

MIT
