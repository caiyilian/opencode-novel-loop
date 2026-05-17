---
name: cancel-ralph
description: Cancel active Ralph Loop
---

# Cancel Ralph

Stop an active Ralph Loop before completion.

## How to Use

1. Check whether a loop is active:

```bash
test -f .opencode/ralph-loop.local.md && echo "Loop is active" || echo "No active loop"
```

2. If active, read the current iteration count:

```bash
grep '^iteration:' .opencode/ralph-loop.local.md
```

3. Delete the state file:

```bash
rm -f .opencode/ralph-loop.local.md
```

4. Tell the user the loop was cancelled and how far it got.
