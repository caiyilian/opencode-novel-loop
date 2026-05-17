---
name: ralph-loop
description: Start Ralph Loop - auto-continues until task completion
---

# Ralph Loop

Start an iterative development loop that automatically continues until the task is complete.

## How It Works

The plugin keeps a small state file in the project and watches for `session.idle` events.

1. You work on the task until you go idle.
2. The plugin checks whether your latest assistant message contains `<promise>DONE</promise>`.
3. If not, it injects a continuation prompt into the same session.
4. This repeats until the task is actually complete, cancelled, or the iteration cap is reached.

## Starting the Loop

When invoked, create the state file in the project directory:

```bash
mkdir -p .opencode && cat > .opencode/ralph-loop.local.md << 'EOF'
---
active: true
iteration: 0
maxIterations: 100
---

[The user's task prompt goes here]
EOF
```

Then begin working on the task.

## Completion Promise

When the task is fully complete, output:

```text
<promise>DONE</promise>
```

Only emit that promise when it is completely and verifiably true.

## Cancellation

The user can stop the loop with `/cancel-ralph`.

## State File

The loop state lives at `.opencode/ralph-loop.local.md`.

Add that path to `.gitignore`.
