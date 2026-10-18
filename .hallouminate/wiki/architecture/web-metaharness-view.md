# Web metaharness view

The metaharness view is a second wide layout in the web dashboard: Mikado graph | per-node chat | details. It ports the "Tern v4" claude.ai/design mockup and reads only real backend data. Source: `web/src/features/metaharness/`.

## How it mounts

- The `narrow` feature owns the single `layout` slot. `features/narrow/WideLayout.tsx` renders `MetaharnessView` when `viewMode.ts` says `metaharness`, else `DefaultLayout`.
- The classic header gets a **Metaharness** button (`header-control` slot). The rail has **Classic view**. The choice persists in `localStorage['milknado.view']`; `#metaharness` in the URL also selects it.
- Classic stays the default because the existing browser tests assume it.

## Mockup element → backend source

| Mockup | Source |
|---|---|
| Projects | root goals in `snapshot.graph.root_ids` |
| Graph, fold, focus | `snapshot.graph` parent tree (`model/tree.ts`, `model/layout.ts`) |
| Chat thread | `GET /api/nodes/{id}` runs, reviews, session events + live `active_runs[].session.events`; goal nodes add coordinator records from `/api/coordinators/{sid}/snapshot` |
| Operator messages | `Operator note: ` paragraphs appended to the node description |
| Composer | running + `steer` action → `POST /api/runs/{id}/session-input`; else `POST /api/nodes/{id}/note` |
| Needs you | permission events (`session.permissions`), `pending_goal_reviews`, failed nodes (`model/needs.ts`) |
| Decisions | approve/deny → session-input; accept/reject → `/api/reviews/{id}/decision`; retry → `/api/nodes/{id}/retry`; skip → `/api/nodes/{id}/skip` |
| Diff | `/api/runs/{id}/changes` + `/diff` |
| Details | node detail, `branch_name`/`worktree_path`, flavor gates from `GET /api/metaharness` |
| Knowledge | `GET /api/metaharness` `knowledge` (wiki pages, `domains/wiki/knowledge.py`) |
| Issues | listener errors, stalled runs, blocked nodes with no open prerequisites; Queue fix → `POST /api/nodes` |
| Stats, workers | snapshot run totals + `concurrency` |
| Pause/resume | `POST /api/scheduling/pause` / `resume` (owner host only; reversible, unlike `/api/scheduling/stop`) |

## Gotchas

- **Skip means done.** `VALID_TRANSITIONS` lets a failed node go only to pending, and archive accepts only done nodes. `/skip` steps failed → pending → done and appends `Operator note: Skipped by the operator.`
- **A pending goal review pauses steering under that goal.** Session input on a run below it returns 409 `goal N review pending; execution paused.` This is backend policy, not a UI bug.
- **Goal review decisions need a controller.** Without `graph.register_controller_master()` the decision returns 409. Read-only `milknado web` has no review handler at all.
- **Observer mode** (`milknado web`) cannot pause the loop; the UI surfaces the 409 reason as a notice.
- **`set_worktree` is fenced** by the node's `run_id`; seeding a branch for a demo needs the node's `run_id` set first.

## Tests and evidence

- `tests/browser/test_metaharness.py` + `metaharness_db.py`: real `MikadoGraph`, real git worktree, polled snapshot source; each test asserts the backend effect.
- `web/src/features/metaharness/model/model.test.ts`: tree, layout, needs, thread, insights, diff parsing.
