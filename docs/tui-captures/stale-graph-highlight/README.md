# Stale graph-highlight TUI evidence

These frames compare pinned baseline `a7c2bfa02121faf88c4d9fce8b8df6219eae135e` with the fixed worktree.
The fixed capture used uncommitted source changes on top of that baseline commit.
The [before manifest](./before/manifest.json) and [after manifest](./after/manifest.json) record the navigation-source hashes.

## Reproduce

Use an environment with Textual, Rich, Playwright, and Playwright Chromium installed.
Run the same script with source imports pinned to each checkout.
The script checks the imported application and fixture roots.

```bash
evidence=$(git rev-parse --show-toplevel)
baseline=$(mktemp -d)
git worktree add --detach "$baseline" a7c2bfa02121faf88c4d9fce8b8df6219eae135e
python="${PYTHON:?Set PYTHON to an installed Python environment}"
script="$evidence/docs/tui-captures/stale-graph-highlight/capture.py"
PYTHONPATH="$baseline:$baseline/src" "$python" "$script" \
  --source-root "$baseline" --output "$evidence/docs/tui-captures/stale-graph-highlight/before" \
  --phase before
PYTHONPATH="$evidence:$evidence/src" "$python" "$script" \
  --source-root "$evidence" --output "$evidence/docs/tui-captures/stale-graph-highlight/after" \
  --phase after
```

The fixture starts with a two-node graph and one run on node 1.
It then removes the graph and replaces the active run with `run-2` on node 2.
It delivers the old node-1 `Tree.NodeHighlighted` event after the transition.
The baseline ends at `(selected_node_id, selected_run_id) == (1, None)`.
The fixed worktree ends at `(2, "run-2")`.
The manifests record these IDs for each capture.
Both phases use the same fixture, dark theme, fixed `12:00:00` clock, viewport, and event order.
The script waits up to 20 refresh cycles for rendered footer hints before it exports each frame.

## Reviewer-accessible pairs

| Surface | Compact 80×24 | Standard 120×40 |
| --- | --- | --- |
| `milknado run` | [before](./before/run-80x24.png) / [after](./after/run-80x24.png) | [before](./before/run-120x40.png) / [after](./after/run-120x40.png) |
| `milknado watch` | [before](./before/watch-80x24.png) / [after](./after/watch-80x24.png) | [before](./before/watch-120x40.png) / [after](./after/watch-120x40.png) |

## Inspection

All eight PNGs were opened.
Each frame shows one active run on node 2, a running status, a highlighted row, and footer key hints.
No visible label, status, or key hint clips in the standard layouts.
The compact layouts truncate the long header summary in both phases; the totals line remains visible.
The standard baseline detail panel says `No run selected.`.
The standard fixed panel shows node 2 and `run-2`.
The compact detail panel is hidden, so the selection difference appears only in the manifest.
The fixed run footer shows more controls because the corrected selection restores available run actions.
The watch footer remains read-only in both phases.
The focused widget is recorded in each manifest.

## Limits

The fixture uses synthetic snapshots and direct event delivery.
It does not run live workers, a controller, the CLI, a provider, or a database.
The PNGs come from Textual SVG screenshots rendered by Chromium.
They do not prove terminal palette fidelity.
