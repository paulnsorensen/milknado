# Adapters — External-System Boundary

Adapters are the infrastructure edge of Milknado's hexagonal architecture. Each
one wraps a single external system and implements a `Protocol` ("port") defined in
`src/milknado/domains/common/protocols.py`. Domain logic depends on the port, never
on the adapter — so the engine can be tested with fakes and the concrete subprocess/
library calls stay quarantined in `src/milknado/adapters/`.

The four ports and their adapters:

| Port (protocols.py) | Adapter | External system |
|---|---|---|
| `GitPort` | `git.GitAdapter` | `git` CLI + `mergiraf` |
| `RalphPort` | `ralphify.RalphifyAdapter` | `ralphify` library (subprocess agent loops) |
| `TilthPort` | `tilth.TilthAdapter` | `tilth` CLI (code intelligence) |
| `CrgPort` | `crg.CrgAdapter` | `code-review-graph` library + CLI |

## git.py — GitPort

Wraps the `git` CLI via `subprocess` for worktree lifecycle (create/remove/prune),
branch query, commit, squash-and-commit, and rebase. Worker isolation runs each
ralph in its own worktree+branch, so this is the boundary for that.

The interesting part is `rebase`: on conflict it parses conflicted paths out of git
output (`_CONFLICT_FILE_RE`), then attempts `mergiraf solve` per file
(`_try_mergiraf_resolve`) for AST-aware auto-resolution. If mergiraf is absent
(`shutil.which`) or any file fails, it falls through. On success it stages and runs
`rebase --continue` with `GIT_EDITOR=true` to skip the editor; on any failure it
aborts. Failure modes are explicit: a successful abort returns
`RebaseResult(success=False, conflicting_files, detail)` (caller decides), but an
abort that *itself* fails raises `RebaseAbortError` — fail-loud, since the worktree
is now in an unknown state. `squash_and_commit` swallows a `merge-base` failure
(soft-reset is best-effort) and only commits when there are staged changes.

## ralphify.py — RalphPort

The largest adapter (~272 lines). Wraps the `ralphify` library, which spawns coding-
agent subprocesses ("ralph loops") that iterate until a completion signal. This is
where Milknado's parallel execution actually happens.

- **Run lifecycle**: `create_run` builds a `RunConfig` keyed on the constant
  completion signal `MILKNADO_NODE_COMPLETE`; `stop_on_completion_signal=True` makes
  the loop self-terminate when the worker emits `<promise>MILKNADO_NODE_COMPLETE</promise>`.
  If `<project_root>/.mcp.json` exists, the agent command is augmented with
  `--mcp-config` so the worker inherits the same MCP servers.
- **Event plumbing**: a single `RunManager` plus a `QueueEmitter` feeding a
  `queue.Queue`. `wait_for_next_completion` is the executor's blocking wait — it
  drains events, buffers `PROGRESS` events (thread-safe under `_progress_lock`,
  later flushed by `poll_progress_events`), and returns `(run_id, success)` only on a
  `RUN_STOPPED` event for a run in the caller's active set. Success = final status
  `RunStatus.COMPLETED`. A `timeout` raises `CompletionTimeout` (a deadline is
  computed once via `time.monotonic()`); this is the guard against a hung worker
  stalling the whole batch.
- **`verify_spec`**: spins up a *separate, throwaway* `RunManager` in a tempdir to ask
  an agent whether a spec is fully implemented, parsing `<result>done|gaps</result>`
  (+ optional `<goal_delta>`) out of iteration text. Edge handling is deliberately
  lenient: no agent configured, a 120s timeout, unparseable output, or an exception
  all degrade to a safe default (`done`/`gaps`) rather than crashing the run.
- **`generate_ralph_md`**: writes the per-node `ralph.md` prompt (description,
  context, quality gates, the Mikado "register follow-ups rather than widen scope"
  instruction, and the completion-signal contract). An `OSError` becomes
  `RalphMarkdownWriteError` — fail-loud on a write the loop depends on.

## tilth.py — TilthPort

Wraps the `tilth` CLI for structural code maps, symbol search, and section reads.
Distinct failure philosophy from the others: tilth is *optional* code intelligence,
so missing/broken tilth must degrade gracefully, not abort a run.

- `structural_map` returns a `TilthMap | DegradationMarker` union — a missing binary
  (`shutil.which`), timeout, non-zero exit, or invalid/non-object JSON each yields a
  typed `DegradationMarker(source, reason, detail)` the domain can reason about
  instead of an exception.
- `search_symbol` / `read_section` degrade silently to `[]` / `""` on any failure.
  All calls are `check=False` with a 30s timeout. `_parse_symbol_headers` regex-parses
  `## path:start-end [kind]` headers into `SymbolLocation`.

## crg.py — CrgPort

Wraps `code-review-graph` for graph intelligence: impact radius, architecture
overview, communities, flows, bridge/hub nodes. Uses the library directly
(`GraphStore` + analysis functions) for queries, but the `build`/`update` graph
operations shell out to the `code-review-graph` CLI via `_run_crg`.

- **Lazy + cached store**: `_get_store` instantiates `GraphStore` once against
  `.code-review-graph/graph.db`.
- **`ensure_graph`** is the freshness gate: builds if the db is absent, runs `update`
  if `_is_stale` (any source file mtime newer than the db). Staleness walks the tree
  via `_walk_sources`, filtered by `_SOURCE_EXTENSIONS` and skipping `_SKIP_DIRS`
  (vcs, caches, build dirs, `ralphs`).
- **Failure mode**: `_run_crg` runs `check=True` and re-raises `CalledProcessError`
  as a `RuntimeError` carrying exit code + stderr (finding #79) so callers can
  surface *why* the build failed rather than a bare non-zero.

## Gotchas

- Every adapter calls external processes; all timed/error paths matter. Note the
  split: git/crg/ralphify **fail loud** on operations the engine depends on (raise),
  while tilth **degrades** because it's advisory. Match this when extending.
- `RalphPort.create_run` / `get_run` / `list_runs` return `Any` — the `ralphify`
  run object is not typed at the port boundary.
- `verify_spec` builds its own `RunManager` rather than reusing the adapter's, so its
  events never reach the shared completion queue.
