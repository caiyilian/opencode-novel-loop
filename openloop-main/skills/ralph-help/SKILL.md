---
name: ralph-help
description: Explain Ralph Loop plugin and available commands
---

# Ralph Loop Help

The Ralph Loop plugin provides auto-continuation for longer OpenCode tasks.

## Available Commands

### `/ralph-loop <task>`

Start a loop that automatically continues the current session until the task is complete.

### `/cancel-ralph`

Cancel an active Ralph Loop.

### `/ralph-help`

Show this quick reference.

## Completion Signal

The loop stops when the assistant truthfully outputs:

```text
<promise>DONE</promise>
```

## State File

The plugin stores loop state in `.opencode/ralph-loop.local.md`.
