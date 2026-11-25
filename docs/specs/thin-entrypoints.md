---
slug: thin-entrypoints
status: draft
source: mold-handshake
created: 2026-10-18
confidence: medium
leverage: ["contract", "cross-slice-dep"]
gates_overridden: []
execution_holds: ["typed-planner: no validated PlannerResult or CurdPlan yet; run the typed planner before bound curd execution", "taste-test: the round-2 verdict failed on one naming contradiction (derive_goal home); the draft fixes it after the verdict, so a fresh taste verdict is required before Cook"]
agent_introduced_scope: ["app.node_ops", "app.goal", "app.run_control", "app.roadmap", "app.github_roadmap", "app.local_run", "app.coordinator", "app.toolchain", "app.graph_view", "app.shutdown", "entrypoint-boundary contract", "service-slice contract", "app type re-exports", "private-name check"]
goal_coverage: {"G-1": "covered", "G-2": "covered", "G-3": "covered", "G-4": "covered", "G-5": "covered"}
entity_referent_bindings: [{"noun": "node", "verdict": "Bound", "referent": "MikadoGraph node (add_node / archive_node)", "citation": "src/milknado/domains/graph/graph.py", "note": "a graph node, created by CLI add-node and MCP create_todo"}, {"noun": "run", "verdict": "Bound", "referent": "worker run row (graph.runs)", "citation": "src/milknado/mcp/run.py:155", "note": "one dispatch of one node"}, {"noun": "goal claim", "verdict": "Bound", "referent": "claim_goal_row / claim_or_reclaim_goal", "citation": "src/milknado/domains/graph/_goal_claims.py:185", "note": "ownership of one goal by one coordinator"}, {"noun": "coordinator", "verdict": "Bound", "referent": "CoordinatorControl", "citation": "src/milknado/cli/web.py:65-119", "note": "web coordinator control plane"}]
agent_resolution: []
landing:
  shape: diamond_stack
  layers: [["c1-node-operations", "c2-run-and-loop-control", "c3-roadmap-and-github", "c4-cli-composition-roots"], ["c5-private-names-aliases-launch"], ["c6-boundary-checks"]]
  per_layer_green: required
  review_fixes: fold
---

# Thin CLI and MCP entry points

## Problem

Goal: thin CLI and MCP entry points — every CLI command and MCP tool only parses input, calls a public app.* function, and formats the result, per ADR sliced-bread-crust-003.

ADR sliced-bread-crust-003 (`docs/adr/sliced-bread-crust-003.md`) says that CLI and MCP modules only parse, call `app.*`, and format. `app/node.py:1-7` repeats it: "entry module constructs no adapters and holds no policy inline". The code does not follow it:

- CLI and MCP implement node creation differently. CLI `add-node` (`cli/__init__.py:291-306`) blocks the parent but never parses the flavor. MCP `create_todo` (`mcp/_todo_creation.py:30-58`) parses the flavor and kind but never blocks the parent. Archive and unarchive report a missing node differently (`mcp/todo_mutate.py:112,130` and `cli/graph.py:97,120`). This is issue #335.
- MCP tools construct adapters and run policy inline: `mcp/run.py:182` (cancel), `:146` (inline poll), `:189-241` (deposits), `mcp/loop.py:93-115` (loop poll), `mcp/github.py:36,59,82`, `mcp/wiki.py:53-56,93`, `mcp/node.py:76-169`, `mcp/todo_mutate.py:43-238`, and `mcp/server.py:34`.
- CLI modules do the same: `cli/web.py:65-119` is a full composition root, `cli/run.py:186-294` holds branch protection, dispatch exclusions, and orphan reconciliation, and `cli/__init__.py:110,135,154,253`, `cli/tools.py:34,45`, `cli/github_roadmap.py:48,73,97`, `cli/roadmap.py:127`, and `cli/graph.py:49-80` construct adapters or hold policy.
- Entry modules import private app names. `mcp/loop.py:28-32` imports `_DEFAULT_RUNNER`, `_remove_reclaimed_worktree`, and `_resolve_runner_cmd` with pyright suppressions and re-exports them. `mcp/server.py:9` imports `_solve_manifest`. `mcp/node.py:35-39` imports five private `app.node` names. `cli/plan.py:39-63` aliases and re-exports five private names from `milknado.app.plan` (imported as `plan_policy`), and `cli/run.py:195` imports the private module `app._shutdown`. This is issue #334.
- `cli/_helpers.py:123-131` keeps seven "historical private name" aliases, listed in `__all__` at `:147-153`. Five have no consumer. Two have only test consumers: `_open_project` (`tests/test_cli.py:2023,2031`) and `_ensure_plugins_loaded` (`tests/test_cli.py:2000,2012`). The project rules forbid compatibility layers.
- `milknado launch` (`cli/web.py:178-186`) duplicates `milknado web` but has no `--allow-host` option.

Users feel the divergence directly: the same node operation gives different graph state or a different error depending on the entry point.

## Goals

- G-1: Every operation that CLI and MCP share runs through one public `app.*` function, with one behavior and one error path.
- G-2: No CLI command or MCP tool constructs an adapter or runs application policy inline, including the composition roots in `cli/run.py` and `cli/web.py`.
- G-3: No CLI or MCP module imports a private `app` name, and no entry module re-exports a private name for tests.
- G-4: The dead alias block in `cli/_helpers.py` is removed.
- G-5: One web command remains: `milknado web`. `milknado launch` is removed.

## Non-goals

- Splitting `app/run.py` (662 lines) or `cli/__init__.py` (328 lines) below the 300-line cap, beyond what the moves in this spec need (issue #341). [AGENT-INTRODUCED]
- Changing the MCP boundary parsers in `mcp/_core.py`. ADR sliced-bread-crust-003 keeps them in MCP. [AGENT-INTRODUCED]
- Removing the module-local `as _x` rebinds in CLI modules that tests patch (`cli/run.py`, `cli/plan.py`). They are test seams, not compatibility aliases. [AGENT-INTRODUCED]
- Replacing raw dicts with typed results across the app surface (issue #337). New app functions return typed values, but existing ones keep their return types. [AGENT-INTRODUCED]
- Fixing the `runner_cmd` allowlist blocker at `mcp/loop.py:65`. It is a separate security fix. [AGENT-INTRODUCED]

## Deferred follow-ups

None. The non-goals map to open issues #337 and #341 and to the audit blocker at `src/milknado/mcp/loop.py:65`.

## Grounding

| Probe | Outcome | Evidence |
| --- | --- | --- |
| wiki | hit | `docs/adr/sliced-bread-crust-003.md` and `.hallouminate/wiki/architecture/entry-points.md:21-39`: app owns policy; CLI and MCP parse, call `app.*`, and format; `lint-imports` enforces the existing directions. |
| explorer | hit | Theme-2 explorer digest (this session): private-import table, adapter-construction table for `mcp/` and `cli/`, the #335 divergence, the alias consumers, the `launch` references, and an 18-row operation table. |

## Approach

Move every piece of policy and adapter construction from `cli/` and `mcp/` into public `app` functions. Then add checks that keep it there. The user chose full ADR-003 compliance (F-1), so the CLI composition roots move too.

Ship six curds. c1, c3, and c4 are independent and can start now. c2 starts after execution-path-consolidation c2-runner-to-app lands, because it edits the `mcp/loop.py` runner imports that the move changes. c5 follows c2, because it removes the private re-exports that c2 replaces. c6 adds the checks after the other five land.

1. c1-node-operations: Add `app.node_ops` with `create_node`, `edit_node`, `archive_node`, `unarchive_node`, `delete_node`, `move_node`, and `set_node_status`. Add `app.goal` with `claim_goal`, `release_goal`, and `claim_todo`. `create_node` runs the union of both current paths (F-2). CLI `add-node`, `archive`, and `unarchive`, and the MCP todo and node tools, call these functions. (F-1, F-2)
2. c2-run-and-loop-control: Add `app.run_control` with `cancel_run`, `poll_inline_run`, `list_runs`, `deposit_result`, and `deposit_review`. Add `poll_loop_run` and a public `default_runner_command` to `app.loop`. Make `app.plan_batches.apply_plan_manifest` public. Add `app.graph_view.graph_summary`. The MCP run, loop, server, and summary tools call these functions. (F-1)
3. c3-roadmap-and-github: Add `app.roadmap` (`import_roadmap`, `export_roadmap`, `render_roadmap`) and `app.github_roadmap` (`import_github_roadmap`, `bind_github_roadmap`, `export_github_roadmap`). They construct `HallouminateIndexer` and `GhProjectAdapter`. The CLI and MCP roadmap and GitHub commands call them. (F-1)
4. c4-cli-composition-roots: Add `app.local_run` with the branch-protection, dispatch-exclusion, controller-choice, and orphan-reconcile policy from `cli/run.py:186-294`. Add `app.coordinator.build_coordinator_control` for the web composition root in `cli/web.py:65-119`. Add `app.toolchain` for `cli/tools.py`. Move the `CrgAdapter` and `LoopAdapter` uses in `cli/__init__.py` and the review gating in `cli/graph.py:49-80` behind app functions. (F-1)
5. c5-private-names-aliases-launch: Remove every private app import and every private re-export from `cli/` and `mcp/`. Tests import from the owning `app` module. Rename `app/_shutdown.py` to `app/shutdown.py`. Remove the `_derive_goal` re-export from `cli/__init__.py`; c6 moves its behavior behind `app.plan.derive_goal`, and the CLI keeps only the `UnicodeDecodeError` to `typer.BadParameter` mapping (`cli/plan.py:226-230`). Delete the five unused `cli/_helpers.py` aliases, and point the two test consumers at `open_project_for_cli` and `ensure_plugins_loaded` before you delete those aliases too (F-4). Delete `milknado launch` and change `milknado launch` to `milknado web` in `docs/tui-captures/cockpit-polish/README.md:34` and `docs/tui-captures/cockpit-polish/capture.py:7`. The wiki page `.hallouminate/wiki/gotchas/coordinator-live-smoke-2026-10.md` records a past run, so it stays as is (F-3). Tests that patch `milknado.mcp.loop.TmuxAdapter` patch the app module instead (F-6).
6. c6-boundary-checks: First, remove each remaining `cli/` and `mcp/` import from a domain service slice. Types that an entry point uses only to parse or format are re-exported by the owning `app` module: `DUMB_ZONE_BUDGET` from `app.plan_batches`, `CoordinatorControl` and `ReviewDecisionPort` from `app.coordinator`, `RunLoopResult` from `app.local_run`, and `PlanResult` from `app.plan`. Behavior that `cli/plan.py` reaches directly (`Planner`, `resolve_plan_spec`, `derive_goal` from `domains.planning`) moves behind `app.plan` functions. Then add the import-linter contract `cli and mcp must not import adapters` (sources `milknado.cli` and `milknado.mcp`, forbidden `milknado.adapters`) and a contract that forbids the same sources from importing the domain service slices (dispatch, execution, coordinator, github, wiki, planning, batching). Add an architecture test that fails when a module in `cli/` or `mcp/` imports from `milknado.app` a name that starts with an underscore, or imports a `milknado.app._*` module (F-5).

Rejected options:

- Do nothing: CLI and MCP keep different node semantics and duplicate adapter wiring.
- Duplicated operations only (F-1, option B): MCP-only policy stays inline in MCP.
- All MCP plus shared CLI operations (F-1, option A): leaves the `cli/run.py` and `cli/web.py` composition roots. The user chose full compliance.
- Per-caller flags for node creation (F-2, option B): keeps the divergence as configuration.
- MCP semantics for node creation (F-2, option C): CLI stops blocking the parent, which breaks the Mikado flow for CLI users.

## Decisions

- F-1: full ADR-003 compliance, including the `cli/run.py` and `cli/web.py` composition roots. — User choice.
- F-2: `app.node_ops.create_node` validates and normalizes hint paths, parses flavor and kind, adds the node, claims files, and blocks the parent. MCP `create_todo` now blocks the parent, and CLI `add-node` now rejects an unknown flavor. — User choice.
- F-3: delete `milknado launch`; keep `milknado web`. — The project is pre-release and forbids compatibility shims. A `git grep` for `milknado launch` outside `docs/audits` and `docs/specs` finds only `cli/web.py:178,296`, two lines in `docs/tui-captures/cockpit-polish/`, and one historical wiki page. [AGENT-DECIDED]
- F-4: delete all seven `cli/_helpers.py` aliases and their `__all__` entries; tests use the public names. — Five have no consumer, and two have only the test consumers cited in Problem. [AGENT-DECIDED]
- F-5: enforce the boundary with two import-linter contracts (no adapters, no domain service slices) plus one architecture test that owns the private-name rule (underscore names and `milknado.app._*` modules). — import-linter checks modules, not names, so the private-name rule needs a test, and one owner for that rule avoids a duplicate contract. The service-slice contract is the mechanical check for "no policy inline"; graph method calls on an opened graph stay a review item. [AGENT-DECIDED]
- F-6: tests patch adapter seams in `app` modules, not in `mcp` modules. — After the move, the MCP modules no longer hold adapter references. [AGENT-DECIDED]
- _Minor decisions:_ new app functions return typed values (frozen dataclasses or domain types) and raise `ValueError` or the existing typed errors, following `app/rebalance.py` [AGENT-DECIDED]; new behavior lands in new `app` modules so that `app/run.py` and `app/node.py` do not grow [AGENT-DECIDED]; archive and unarchive raise `ValueError("node N not found")` from one place, and each entry point formats it [AGENT-DECIDED].

## Acceptance

- AC-1: WHEN a client creates a child node under a parent through MCP `milknado_todo_add` or through CLI `milknado add-node` THE SYSTEM SHALL produce the same graph state in both cases: the hint paths are normalized and claimed, the flavor is validated, and the parent is blocked. (F-2, G-1)
- AC-2: WHEN a client archives or unarchives a node that does not exist through MCP or CLI THE SYSTEM SHALL report the same "node N not found" message: a structured tool error in MCP, and exit code 1 in the CLI. (F-1, G-1)
- AC-3: WHEN CLI and MCP run the same roadmap or GitHub roadmap operation THE SYSTEM SHALL run it through one `app.roadmap` or `app.github_roadmap` function. (F-1, G-1)
- AC-4: WHEN `just check-llm` runs `lint-imports` THE SYSTEM SHALL enforce a contract that forbids `milknado.cli` and `milknado.mcp` from importing `milknado.adapters`, and the contract SHALL pass with no ignored imports. (F-1, F-5, F-6, G-2)
- AC-5: WHEN `milknado run` or `milknado web` starts THE SYSTEM SHALL build its adapters and coordinator control inside `app.local_run` and `app.coordinator`, and the `milknado run` and `milknado web` tests SHALL keep the same assertions on observable behavior; only their patch targets may move. (F-1, F-6, G-2)
- AC-6: WHEN `just check-llm` runs the test suite THE SYSTEM SHALL run an architecture test that fails if any module under `src/milknado/cli/` or `src/milknado/mcp/` imports from `milknado.app` a name that starts with an underscore, or imports a `milknado.app._*` module, and the test SHALL pass. A local rebind such as `from milknado.app.x import y as _y` is allowed. (F-5, G-3)
- AC-10: WHEN `just check-llm` runs `lint-imports` THE SYSTEM SHALL enforce a contract that forbids `milknado.cli` and `milknado.mcp` from importing the service slices `milknado.domains.dispatch`, `.execution`, `.coordinator`, `.github`, `.wiki`, `.planning`, and `.batching`, so that domain services are reached only through `app`; the contract SHALL pass with no ignored imports. (F-1, F-5, G-2)
- AC-7: WHEN a reader opens `src/milknado/cli/_helpers.py` THE SYSTEM SHALL contain no alias block and no underscore names in `__all__`. (F-4, G-4)
- AC-8: WHEN a user runs `milknado launch` THE SYSTEM SHALL exit with Typer's "No such command" error, and `milknado web --allow-host` SHALL keep working. (F-3, G-5)
- AC-9: WHEN each curd lands THE SYSTEM SHALL pass `just check-llm`, including the `real_process` and browser lanes. (F-1, F-2, F-3, F-4, F-5, F-6, G-1, G-2, G-3, G-4, G-5)

## Acceptance demonstration

| Field | Value |
| --- | --- |
| User task | Add a prerequisite under a goal from an MCP client and from the CLI, and get the same Mikado graph from both. |
| Input | Two copies of one temporary project with goal node 1. In copy A, call MCP `milknado_todo_add` with parent 1, flavor `implement`, and hint path `src/a.py`. In copy B, run `milknado add-node "same" --parent 1 --flavor implement --files src/a.py`. Then archive node 99 in both. |
| Entrypoint | The real `milknado-mcp` stdio server and the real `milknado` CLI. |
| Expected result | Both graphs show node 1 blocked, a new child with flavor `implement`, and a claim on `src/a.py`. Both archive calls report "node 99 not found". |
| Real boundaries | The real MCP stdio transport, the real CLI process, and real SQLite graph files. |
| False success | Each entry point is tested alone against its own old behavior. The check compares the two graph dumps field by field, so an unblocked parent on the MCP side fails it. Run it on `9bc29e6` first: it must fail there. |
| Evidence | The two `milknado_graph_summary` dumps and the two archive error messages. |

## Interface sketches

```text
slice:            app (application policy and composition); cli and mcp shrink to parse, call, format
spine step:       workflow (app), entry (cli, mcp)
public interface: app.node_ops.create_node(root: Path, request: NodeCreateRequest) -> Node  (F-2)
public interface: app.node_ops.edit_node / archive_node / unarchive_node / delete_node / move_node / set_node_status(root: Path, ...) -> Node  (F-1)
public interface: app.goal.claim_goal / release_goal / claim_todo(root: Path, request) -> typed result  (F-1)
public interface: app.run_control.cancel_run / poll_inline_run / list_runs / deposit_result / deposit_review(root: Path, ...) -> typed result  (F-1)
public interface: app.loop.poll_loop_run(root: Path, run_id: str) -> typed result; app.loop.default_runner_command() -> tuple[str, ...]  (F-1)
public interface: app.plan_batches.apply_plan_manifest(root: Path, request) -> typed result; app.graph_view.graph_summary(root: Path, ...) -> typed result  (F-1)
public interface: app.roadmap.import_roadmap / export_roadmap / render_roadmap; app.github_roadmap.import_github_roadmap / bind_github_roadmap / export_github_roadmap  (F-1)
public interface: app.local_run.prepare_local_run(root: Path, options) -> LocalRunPlan; app.coordinator.build_coordinator_control(root: Path) -> CoordinatorControl; app.toolchain.toolchain_status(root: Path) -> typed result  (F-1)
public interface: app.shutdown (renamed from app._shutdown); app.plan.derive_goal  (F-1)
removed:          milknado launch; cli/_helpers.py alias block; private re-exports in cli/plan.py, cli/__init__.py, mcp/loop.py  (F-3, F-4)
private:          adapter construction and policy inside the new app modules; test patch seams for adapters live in app modules  (F-6)
crust delta:      new app modules and exports, including type re-exports for entry-point formatting; contracts "cli and mcp must not import adapters" and "cli and mcp must not import domain service slices"; architecture test for underscore names and milknado.app._* modules  (F-5)
arrows:           removed: cli -> adapters, mcp -> adapters, cli/mcp -> app private names; unchanged: cli/mcp -> app, app -> adapters, app -> domains
```

## Risks

- AC-1 changes MCP behavior: `milknado_todo_add` now blocks the parent. Coordinators that add children and expect the parent to stay ready see a blocked parent. This is the Mikado rule that CLI already follows. Call it out in the c1 PR.
- AC-1 also changes CLI behavior: `add-node` rejects an unknown flavor.
- Tests patch many entry-module names (`milknado.mcp.loop.TmuxAdapter`, `milknado.cli.run._ensure_db`, and others). The c1 to c5 curds move patch targets with the code. A missed patch target makes a test pass against the real adapter, so the curds check each patch target still resolves.
- `cli/web.py` composition touches the coordinator recovery adapters. The browser lane covers it; run it on each curd that touches `cli/web.py`.
- The c6 contract fails until `mcp/_loop_node_runner.py` leaves `mcp` (execution-path-consolidation c2-runner-to-app).
- The issues #334 and #335 cite old module names (`mcp/ralph.py`, `app.ralph`). Close them against this spec, not against their text.

## Open questions

- [TBD] Does any test invoke `milknado launch` through the Typer app? `tests/web/test_cli_web.py` was not read in full. The c5 curd checks this first.
- [TBD] Which policy in `mcp/server.py:73-130` (`plan_apply`) and `mcp/node.py:120-225` was not itemized? The c1 and c2 curds itemize it before they move it.
- [TBD] Does any other `cli/` or `mcp/` import from a service slice exist beyond the ones that c6 names? The c6 curd runs `lint-imports` with the new contract before it removes the imports, and handles each extra hit the same way: a type becomes an `app` re-export, and behavior moves behind an `app` function.

## Quality gates

- `just check-llm`: prints the PASS line after each curd.
- `uv run lint-imports`: the two new contracts pass with no ignored imports after c6.
- Review checklist for each curd: no CLI command or MCP tool calls a `MikadoGraph` mutation method directly; it calls an `app` function.

## Curds

| Curd | Covers | Depends on | State |
| --- | --- | --- | --- |
| c1-node-operations | G-1 (AC-1, AC-2) | none | planned |
| c2-run-and-loop-control | G-2, G-3 | execution-path-consolidation c2-runner-to-app | planned |
| c3-roadmap-and-github | G-1 (AC-3), G-2 | none | planned |
| c4-cli-composition-roots | G-2 (AC-5) | none | planned |
| c5-private-names-aliases-launch | G-3, G-4 (AC-7), G-5 (AC-8) | c2 | planned |
| c6-boundary-checks | G-2 (AC-4, AC-10), G-3 (AC-6) | c1, c2, c3, c4, c5, execution-path-consolidation c2-runner-to-app | planned |

The typed planner has not run. See `execution_holds`.
