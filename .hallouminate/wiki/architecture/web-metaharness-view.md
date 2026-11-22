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
| Composer | live run + `steer` → session-input; failed node + retry → `/api/nodes/{id}/retry`; otherwise → `/api/nodes/{id}/note` |
| Needs you | permission events (`session.permissions`), `pending_goal_reviews`, failed nodes (`model/needs.ts`) |
| Decisions | approve/deny → session-input; accept/reject → `/api/reviews/{id}/decision`; retry → `/api/nodes/{id}/retry`; skip → `/api/nodes/{id}/skip` |
| Diff | `/api/runs/{id}/changes` + `/diff` |
| Details | node detail, `branch_name`/`worktree_path`, flavor gates from `GET /api/metaharness` |
| Knowledge | `GET /api/metaharness` `knowledge` (wiki pages, `domains/wiki/knowledge.py`) |
| Issues | listener errors, stalled runs, blocked nodes with no open prerequisites; Queue fix → `POST /api/nodes` |
| Stats, workers | snapshot run totals + `concurrency` |
| Pause/resume | `POST /api/scheduling/pause` / `resume` (owner host only; reversible, unlike `/api/scheduling/stop`) |
| Drag a card onto a card or crumb | `POST /api/nodes/{id}/move`; the dragged node becomes a prerequisite of the target, or a top-level node on **Projects** (`ui/edits.ts`, `model/tree.ts` `canAdopt`) |
| Card **+**, `A` key, brand **+** | the shared graph-edits Add node dialog with the parent preset → `POST /api/nodes` |
| Details **Actions** | Edit → `PATCH /api/nodes/{id}`; Move, Archive → the shared graph-edits dialogs; Cancel run → `POST /api/runs/{run_id}/cancel` after a confirm |
| **New goal** | `POST /api/coordinators/commands` `start_goal` (creates a root goal and a coordinator session) |

## Gotchas

- **Skip means done.** `VALID_TRANSITIONS` lets a failed node go only to pending, and archive accepts only done nodes. `/skip` steps failed → pending → done and appends `Operator note: Skipped by the operator.`
- **A pending goal review pauses steering under that goal.** Session input on a run below it returns 409 `goal N review pending; execution paused.` This is backend policy, not a UI bug.
- **Goal review decisions need a controller.** Without `graph.register_controller_master()` the decision returns 409. Read-only `milknado web` has no review handler at all.
- **Observer mode** (`milknado web`) cannot pause the loop; the UI surfaces the 409 reason as a notice.
- **Graph-edit dialogs are shared.** The add, edit, move, archive, and cancel-confirm dialogs belong to `graph-edits` and `run-controls` and render in the shell `dialog` slot, outside `.mh`. The view opens them with `openDialog` and `requestConfirm`; `openDialog('add', parentId)` presets the parent. They keep the classic dialog style.
- **Card drop targets need a live drag id.** HTML5 `dragover` cannot read `dataTransfer`, so the dragged id lives in `viewState.drag`. `canAdopt` refuses the current parent, the node itself, and its own prerequisites; the server still validates the move.
- **Cancel is per run.** The view cancels the node's live `run_id`, not the owner loop `run_id` the classic header uses.
- **`set_worktree` is fenced** by the node's `run_id`; seeding a branch for a demo needs the node's `run_id` set first.

## Tests and evidence

- `tests/browser/test_metaharness.py` + `metaharness_db.py`: real `MikadoGraph`, real git worktree, polled snapshot source; each test asserts the backend effect.
- `tests/browser/test_metaharness_edits.py`: drag-to-move (card and **Projects** crumb), add with a preset parent, edit and archive, cancel with a confirm, and New goal through a real `CoordinatorControl`.
- `web/src/features/metaharness/model/model.test.ts`: tree, layout, needs, thread, insights, diff parsing.

## T3-backed goal-first web orchestration

The web metaharness can be evaluated as a goal-first interface above [T3 Code](../sources/t3-code.md).
T3's outside-MCP thread control offers a possible runtime boundary, not a complete browser integration.
T3 rejects browser-origin MCP calls, so this option needs a server-side adapter.
Its pending-request MCP tool cannot approve provider permissions.
MCP credentials also cannot access ordinary HTTP or WebSocket routes.[^t3-web]

A unified permission inbox therefore needs T3's own human-facing surface or another authorized integration.
Using internal client contracts adds a separate maintenance and authentication boundary.
The T3 contracts and client-runtime packages are private workspace packages.[^t3-client]

The current Milknado graph pane renders a parent tree rather than every dependency edge.
Its conversation pane projects run events, reviews, coordinator records, and notes.
Its composer steers a live run, retries a failed node, or saves a note.[^view-model]
A goal-first format could emphasize prerequisites, blocking decisions, and accepted-result evidence.
This remains a design option, not an approved replacement or a proven usability improvement.

The inspected browser tests exercise graph effects, but their cancel fixture records the requested run ID.
They do not prove real T3 provider cancellation.
No live T3 integration check runs in this comparison.[^fixture-limit]

## Pluggable execution backend and state ownership proposal

Milknado would own goal state when T3 runs its workers through a pluggable execution backend.
The goal-first web metaharness would use this runtime interface through its server.
This is a design proposal, not implemented behavior or an approved replacement.
[T3 Code](../sources/t3-code.md) would be the first runtime implementation, not a second owner of goal state.
The existing browser and permission restrictions still apply.[^t3-web]

| Layer | Proposed responsibility |
| --- | --- |
| Web metaharness | Goals, dependency edges, blocking decisions, attempts, and accepted-result evidence |
| Goal controller | Reviewed plans, dependency admission, file claims, budgets, review policy, and acceptance |
| Execution backend interface | Start, inspect, wait, request cancellation, and report supported operations |
| T3 adapter and T3 | Map domain attempts to threads/runs; execute provider work and own runtime lifecycle |
| Planning and intelligence | Produce validated plans; obtain external code-graph facts independently of runtime choice |

This interface is not a drop-in implementation of today's `LoopPort`.
That port includes local paths, completion callbacks, worker-stop operations, review, and specification verification.
The executor also writes local loop artifacts before starting work.
These couplings need an explicit redesign, not remote objects disguised as local run handles.[^runtime-port]

The goal controller would persist an attempt, backend identity, and stable request key before dispatch.
The adapter would bind the returned environment, thread, and run identifiers to that attempt.
A lost reply would reuse the same request key and reconcile state, rather than create another attempt.
T3 launch supports request-key retries and explicit workspace bases; uncommitted edits do not transfer.[^runtime-launch]

One runtime would own each attempt for its lifetime.
A disconnected T3 attempt would remain unknown until reconciled; it would not silently fail over to another backend.
The controller would retain accepted artifacts and decisions, rather than depend on complete historical event replay.[^runtime-history]
The runtime would own workspace creation and cleanup; the controller would select the base and enforce integration policy.
Neither side would delete an unresolved attempt's workspace.
Review, acceptance, and merge policy would remain outside provider-turn success.[^runtime-complete]

Optional operations would be explicit capabilities: steering, restart continuation, isolated workspaces, and permission decisions.
Unsupported operations would be disabled or routed to T3's own interface, not simulated as successful.
User-question answers would remain separate from provider permission approval.[^t3-web]
Cancellation would distinguish the named run from stopping the whole thread and its delegated work.
T3's broad `stop` option ignores `runId`; a safe adapter must not apply it to a reused thread accidentally.[^runtime-control]
An interrupt acknowledgement would not prove that all workers have exited.

The planning extraction would start with the existing batch-plan return boundary, not the whole `Planner`.
Milknado's external code-review-graph dependency would remain separate; the reusable value is validation, batching, and context policy.
See [planning pipeline extraction](./planning.md#planning-pipeline-extraction-for-runtime-reuse).

A first trial would run two dependent tasks through T3 and retain the existing goal acceptance gate.
It would test lost replies, restart, reconnect, a permission pause, cancellation, and a late result from an old attempt.
The trial must reject duplicate dispatch, premature dependent work, stale acceptance, and cleanup without confirmed ownership.
No live trial runs in this research.
A second backend would be added only for a demonstrated need; the interface alone does not justify maintaining two runtimes.

[^runtime-port]: `src/milknado/domains/common/protocols.py:169-230`; `src/milknado/domains/execution/executor.py:615-704`; checked at c9250b8f039e4c06b38df728c3aade180a489614.
[^runtime-launch]: https://github.com/pingdotgg/t3code/blob/6ab633f6d6e8273f08a060137ab64463638787ce/apps/server/src/mcp/toolkits/project/tools.ts#L129-L169.
[^runtime-control]: https://github.com/pingdotgg/t3code/blob/6ab633f6d6e8273f08a060137ab64463638787ce/packages/contracts/src/orchestratorMcp.ts#L539-L613.
[^runtime-complete]: `src/milknado/domains/execution/executor.py:1372-1410` checks cleanup, completion target, review, and rebase before finalization.
[^runtime-history]: [T3 Code recovery and audit limits](../sources/t3-code.md#recovery-and-audit-limits).

[^t3-web]: https://github.com/pingdotgg/t3code/blob/6ab633f6d6e8273f08a060137ab64463638787ce/apps/server/src/mcp/McpHttpServer.ts (lines 283-294); https://github.com/pingdotgg/t3code/blob/6ab633f6d6e8273f08a060137ab64463638787ce/apps/server/src/mcp/toolkits/thread/tools.ts (lines 129-147); https://github.com/pingdotgg/t3code/blob/6ab633f6d6e8273f08a060137ab64463638787ce/docs/internals/environment-auth.md (lines 55-72); verified 2026-11-17.
[^t3-client]: https://github.com/pingdotgg/t3code/blob/6ab633f6d6e8273f08a060137ab64463638787ce/packages/contracts/package.json; https://github.com/pingdotgg/t3code/blob/6ab633f6d6e8273f08a060137ab64463638787ce/packages/client-runtime/package.json; verified 2026-11-17.
[^view-model]: web/src/features/metaharness/model/tree.ts:11-26; web/src/features/metaharness/model/thread.ts:151-216; web/src/features/metaharness/ui/Composer.tsx:14-32; verified at c9250b8f039e4c06b38df728c3aade180a489614.
[^fixture-limit]: tests/browser/test_metaharness_edits.py:127-164.

_Source: pinned Milknado implementation and T3 Code comparison · Updated: 2026-11-18 · Supersedes: incomplete composer mapping; adds an unapproved backend proposal, not a runtime decision._
