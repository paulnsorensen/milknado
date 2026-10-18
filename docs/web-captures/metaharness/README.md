# Metaharness view captures

These captures show the metaharness web view against a real `milknado web` server. The data is a seeded demo project. The `00-*` images show the claude.ai/design "Tern v4" mockup for comparison.

| Capture | State |
|---|---|
| `00-mockup.png`, `00-mockup-popover.png` | The mockup: thread view and decision popover |
| `01-thread.png` | A goal thread from the coordinator, with a pending goal review |
| `02-popover.png` | The goal review decision popover |
| `03-diff.png` | The diff view: real worktree changes with a permission decision bar |
| `04-details.png` | The worker thread, gate pill, reviewer verdict, and details panel |
| `05-failed.png` | A failed node with the Retry, Retry with note, and Skip popover |
| `06-expanded.png` | The expanded graph (`G`) |
| `07-rail.png` | The edge-hover rail |
| `09-stats.png` | The session stats panel |
| `11-projects.png` | The project list (root goals) |
| `e1-hover-plus.png` | A hovered card shows its **+** (add a prerequisite) control |
| `e2-drag-over.png` | A card dragged over another card: the drop target has a dashed lime outline |
| `e3-after-drop.png` | After the drop: the node moved under the target, with a confirmation toast |
| `e4-add-dialog.png` | The Add node dialog with the parent preset to the clicked card |
| `e5-details-actions.png` | The details panel **Actions**: Add prerequisite, Edit, Move, Archive |
| `e6-projects-new-goal.png` | The project list with the **New goal** entry |
| `e7-new-goal-dialog.png` | The New goal dialog with the coordinator choice |

## Reproduce

Run these commands from the repository root. The viewport is 1440x900 with the dark theme.

```bash
npm --prefix web run build
uv run python docs/web-captures/metaharness/seed_demo.py /tmp/mh-demo
uv run milknado web --project-root /tmp/mh-demo --port 8765 --no-open > /tmp/mh-web.log 2>&1 &
uv run python docs/web-captures/metaharness/capture.py /tmp/mh-web.log /tmp/mh-shots
uv run python docs/web-captures/metaharness/capture_edits.py /tmp/mh-web.log /tmp/mh-shots
```

## Fixture limits

- The server runs as a read-only observer. In this mode, Pause/Resume and goal-review decisions return the server's 409 reason. Steering needs published owner capabilities.
- No live worker runs. The session events, runs, and reviews are seeded rows in the graph database.
- `tests/browser/test_metaharness.py` tests the owner-side actions end to end against a real graph database.
- `capture_edits.py` moves one node in the demo database. Seed a fresh project before each run.
- `tests/browser/test_metaharness_edits.py` tests drag-to-move, add, edit, archive, cancel, and New goal against a real graph database.
