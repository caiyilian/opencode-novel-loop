---
description: Start Ralph Loop - auto-continues until task completion
---

# Ralph Loop

Start an iterative development loop that automatically continues until the task is complete.

## Setup

Create the state file in the project directory:

```bash
mkdir -p .opencode && cat > .opencode/ralph-loop.local.md << 'EOF'
---
active: true
iteration: 0
maxIterations: 100
---

$ARGUMENTS
EOF
```

## Task

Now begin working on the task: **$ARGUMENTS**

## Completion

When the task is fully complete, signal completion by outputting:

```text
<promise>DONE</promise>
```

Only output that promise when the work is completely and verifiably done.

## Cancellation

Use `/cancel-ralph` to stop early.
