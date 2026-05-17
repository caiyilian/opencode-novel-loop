---
description: Show Ralph Loop help and available commands
---

# Ralph Loop Help

## Available Commands

- `/ralph-loop <task>` - start an auto-continuation loop for the given task
- `/cancel-ralph` - stop an active Ralph Loop
- `/ralph-help` - show this help

## Quick Start

```text
/ralph-loop Build a REST API with user authentication
```

The AI will keep working until it truthfully emits `<promise>DONE</promise>`.

## How It Works

1. Creates state file at `.opencode/ralph-loop.local.md`
2. Works on the task until idle
3. If no `<promise>DONE</promise>` is found, the plugin auto-continues the same session
4. Repeats until complete or max iterations are reached
