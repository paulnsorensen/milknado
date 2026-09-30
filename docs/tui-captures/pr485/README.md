# PR 485: stop-scheduling help with a terminal run selected

This pair covers the one presentation change from the PR 485 review fixes.
The run help overlay now shows `s stop scheduling` when stop scheduling is available.
Before the fix, the hint showed only when an active run was selected.

## Revisions and environment

- Before: `1a8f305d16e74844236829c39d0d59eb7a52c72a`, the PR head before review fixes.
- After: `b8237dca3f92e5a265f7f1daac2cd6d5fa28d45c`, the PR head with review fixes.
- Terminal dimensions: standard `120x40`; compact `80x24`.
- Theme: Textual `textual-dark`; header clock: `12:00:00`.
- Data: the five-node snapshot from [the agent-steering fixture](../agent-steering/fixture.py).
- Renderer: the [PR 483 capture harness](../pr483/capture.py) with the real `ExecutionApp`.

## State

Mount the run surface, select terminal run `run-13`, and press F1.
The fixture keeps active run `run-12`, so stop scheduling is available.

## Reproduce

```bash
repo=$(git rev-parse --show-toplevel)
work=$(mktemp -d)
git worktree add --detach "$work/before" 1a8f305d16e74844236829c39d0d59eb7a52c72a
PYTHON=${PYTHON:-"$repo/.venv/bin/python"}
PYTHONPATH="$work/before/src" "$PYTHON" "$repo/docs/tui-captures/pr485/capture.py" \
  --source-root "$work/before" --output "$repo/docs/tui-captures/pr485/before"
PYTHONPATH="$repo/src" "$PYTHON" "$repo/docs/tui-captures/pr485/capture.py" \
  --source-root "$repo" --output "$repo/docs/tui-captures/pr485/after"
```

## Inspection results

- Both layouts show the help overlay with focus on the help scroll and the Esc/F1 close hint.
- The after frames add `s stop scheduling`; no line clips at 120x40 or 80x24.
- The before frames omit the hint, although `s` still opens stop confirmation.

## Limits

These are synthetic presentation captures, not live worker evidence.
The other PR 485 review fixes change durable data or the run loop, not layout.
The fixture has no database, so the stopped-run totals fix does not appear here.

## PNG pairs

| Surface | State | 120x40 | 80x24 |
|---|---|---|---|
| run | help-terminal | [before](./before/run-help-terminal-120x40.png) / [after](./after/run-help-terminal-120x40.png) | [before](./before/run-help-terminal-80x24.png) / [after](./after/run-help-terminal-80x24.png) |
