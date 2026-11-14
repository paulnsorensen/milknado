---
slug: execution-path-consolidation
status: draft
source: mold-handshake
created: 2026-10-18
confidence: medium
leverage: ["contract", "cross-slice-dep"]
gates_overridden: []
execution_holds: ["typed-planner: no validated PlannerResult or CurdPlan yet; run the typed planner before bound curd execution"]
agent_introduced_scope: ["config_for_profile", "milknado.loop.process", "milknado.loop.events", "lifeline_command", "run_types", "provider_capabilities"]
goal_coverage: {"G-1": "covered", "G-2": "covered", "G-3": "covered", "G-4": "covered"}
entity_referent_bindings: [{"noun": "run", "verdict": "Bound", "referent": "RunStatus / RunConfig (one worker loop run)", "citation": "src/milknado/loop/_run_types.py:37", "note": "run is one worker loop run, not a coordinator session"}, {"noun": "worker", "verdict": "Bound", "referent": "WorkerProcess", "citation": "src/milknado/loop/_process_gate.py", "note": "a spawned agent process"}, {"noun": "lifeline", "verdict": "Bound", "referent": "run_lifeline", "citation": "src/milknado/loop/_lifeline.py", "note": "helper process that watches one worker"}]
agent_resolution: []
landing:
  shape: stacked_linear
  layers: [["c1-profile-builder"], ["c2-runner-to-app"], ["c3-domain-types-to-common"], ["c4-loop-public-surface"]]
  per_layer_green: required
  review_fixes: fold
---

# One execution path

## Problem

Goal: one execution path — a single flavor-profile to ExecutionConfig builder, the detached loop runner in app, no domain imports of milknado.loop, and adapters that use only a public loop surface, with import-linter enforcement.

Today the worker execution path has three owners for one mapping and four boundary leaks:

- Three copies map configuration and flavor profiles to `ExecutionConfig`: `src/milknado/app/run.py:83-93` (`build_exec_config`), `src/milknado/domains/execution/run_loop/__init__.py:532-551` (`dataclasses.replace`), and `src/milknado/mcp/_loop_node_runner.py:201-220`. The copies disagree. `build_exec_config` never sets `commit_footer`, and the run-loop copy does not set it either. A local `milknado run` therefore sends workers no commit footer, but a detached MCP loop run does (`executor.py:654` reads `config.commit_footer`).
- `mcp/_loop_node_runner.py` is application composition. It wires `GitAdapter`, `LoopAdapter`, `CrgAdapter`, `FlockSlotPool`, `Executor`, and `RunLoop` (`:171-227`). `app/loop.py:53` launches it by the module path `milknado.mcp._loop_node_runner`, which is an `app` to `mcp` reverse arrow.
- Seven domain files import `milknado.loop`: `coordinator/projection.py:29`, `execution/completion.py:12`, `execution/executor.py:89`, and `execution/run_loop/{__init__,_completion,_projection,state}.py`. Each import is a pure type or a static table, but it loads the worker runtime (`loop/__init__.py` imports `manager` and `sessions`).
- Twelve adapter files import 17 private `milknado.loop._*` names. `loop/__init__.py` `__all__` covers none of them.
- `loop/_process_helper.py:110` starts `milknado.adapters._loop_lifeline` by module name. That module imports `loop._lifeline`. This cycle is invisible to import-linter.

The gap is visible to users: local runs and detached runs build different worker configurations from the same `milknado.toml`.

## Goals

- G-1: One function maps `MilknadoConfig` plus a flavor profile to `ExecutionConfig`, and every execution entry uses it.
- G-2: The detached loop runner lives in `milknado.app`; `milknado.mcp` keeps only tool registration.
- G-3: No module under `milknado.domains` imports `milknado.loop`, and an import-linter contract enforces it.
- G-4: Adapters import only a public `milknado.loop` surface, an import-linter contract enforces it, and the lifeline launch has no hidden reverse dependency.

## Non-goals

- Splitting `domains/execution/executor.py` or `loop/_agent.py` (issues #332 and #333). [AGENT-INTRODUCED]
- Moving execution policy out of `adapters/loop.py` (issue #339). [AGENT-INTRODUCED]
- Moving subprocess mechanics out of `domains/execution/completion.py` (issue #213). [AGENT-INTRODUCED]
- Changing the `MILKNADO_LOOP_RUNNER_CMD` override or the `runner_cmd` MCP argument. The audit blocker on `runner_cmd` is a separate fix. [AGENT-INTRODUCED]

## Deferred follow-ups

None. The non-goals map to open issues #213, #332, #333, and #339, and to the audit blocker at `src/milknado/mcp/loop.py:65`.

## Grounding

| Probe | Outcome | Evidence |
| --- | --- | --- |
| wiki | hit | `.hallouminate/wiki/history/loop-vendor-in-decision.md`: since 2026-09-02, `src/milknado/loop/` is owned source with a small public crust. `docs/adr/sliced-bread-crust-002.md`: ship structural moves as stacked curds that each pass `just check-llm`. |
| explorer | hit | Theme-1 explorer digest (this session): three builder copies, 7 domain imports of pure loop types, 17 private adapter imports, the string-launched lifeline cycle, and the current import-linter contracts at `pyproject.toml:51-140`. |

## Approach

Ship four stacked curds. Each curd is behavior-preserving except c1, which fixes the missing `commit_footer` in local runs. Local runs also gain the flavor review settings when `RunLoop` has no `_milknado_config`; the first `[TBD]` decides whether that path exists in production.

1. c1-profile-builder: Add one public builder in the execution slice. `app.run.build_exec_config` sets every base field that `MilknadoConfig` owns, including `commit_footer`. `config_for_profile` applies a flavor profile to a base config. `RunLoop`, the detached runner, and `milknado run` use only these two functions. (F-1-builder)
2. c2-runner-to-app: Move `mcp/_loop_node_runner.py` to `app/loop_runner.py`. Point `app/loop.py` `_DEFAULT_RUNNER` at `milknado.app.loop_runner`. Update the tests that name the module path. (F-1-runner)
3. c3-domain-types-to-common: Move `RunStatus` and `CompletionVerdict` to `domains/common/run_types.py`. Move the provider capability table to `domains/common/provider_capabilities.py`: `ProviderFamily` (from `loop/sessions/_protocol.py:9`), `LifecycleOperation`, `CapabilitySupport`, `RuntimeCapabilities`, the floor and action tables, and `runtime_capabilities` (from `loop/sessions/_capabilities.py:11-61`). `loop/_run_types.py`, `loop/sessions/_protocol.py`, and `loop/sessions/_capabilities.py` import these names from `domains.common`, which keeps the existing `loop` to `common` direction. `snapshot` and the other session helpers stay in `loop.sessions`. Add the contract `domains must not import milknado.loop`. (F-2)
4. c4-loop-public-surface: Add the public submodules `milknado.loop.process` and `milknado.loop.events`. They re-export the 17 names that adapters use. Point every adapter import at them. `ProtectionContext` carries the lifeline launch command (`lifeline_command`), and the adapter that builds the context supplies it. `loop/_process_helper.py` no longer names an adapter module. Add the contract `adapters must not import milknado.loop._*`. (F-3, F-4-lifeline)

Rejected options:

- Do nothing: local and detached runs keep different worker configurations, and the domain keeps loading the runtime.
- Ports for `RunStatus` and `CompletionVerdict` in `common.protocols` (F-2, option B): adds a translation layer for two plain value types.
- Allowlisted contract exceptions (F-2, option C): the domain still loads the runtime at import.
- Widen `loop/__init__.py` (F-3, option B): every adapter import would pay the eager `manager` and `sessions` import cost.
- Move adapter process code into `loop` (F-3, option C): the largest diff, and it overlaps #333.

## Decisions

- F-1-builder: one builder pair (`build_exec_config` plus `config_for_profile`) replaces three copies. Local `milknado run` workers gain `commit_footer`, and gain the flavor review settings on the path where `RunLoop` has no `_milknado_config`. — This is the fix for the divergence. The user chose the consolidation; the behavior change is a bug fix. [AGENT-DECIDED]
- F-1-runner: the runner moves to `app/loop_runner.py`. — It is a composition root, and ADR sliced-bread-crust-003 puts composition in `app`. [AGENT-DECIDED]
- F-2: `RunStatus`, `CompletionVerdict`, and the provider capability table move to `domains/common`. — User choice. It is the smallest change that lets the contract pass, and `loop` already imports `domains.common` at 26 sites.
- F-3: adapters use the new public submodules `milknado.loop.process` and `milknado.loop.events`. — User choice. `loop/__init__.py` stays lean.
- F-4-lifeline: the lifeline launch command is injected through `ProtectionContext`, and the entry module stays in `adapters`. — The entry module composes `domains.graph.WorkerEvidenceStore` with `loop._lifeline`, so moving it into `loop` would add a `loop` to `domains.graph` arrow. [AGENT-DECIDED]
- _Minor decisions:_ stacked curds per ADR sliced-bread-crust-002 [AGENT-DECIDED]; `config_for_profile` lives in `domains/execution` next to `ExecutionConfig` [AGENT-DECIDED]; the run loop keeps per-node profile resolution [AGENT-DECIDED]; `completion_timeout_seconds` keeps the `int` type that `ExecutionConfig` declares [AGENT-DECIDED].

## Acceptance

- AC-1: WHEN `milknado run` dispatches a node in a project whose `milknado.toml` sets `commit_footer` THE SYSTEM SHALL pass that footer to the worker run, the same as a detached loop run does. (F-1-builder, G-1)
- AC-2: WHEN any execution entry (local run loop, detached loop runner, inline run) builds a worker configuration for a node with a flavor profile THE SYSTEM SHALL produce it through `build_exec_config` and `config_for_profile` only, and a search of `src/` SHALL find no other `ExecutionConfig(` construction or `replace(` call on an `ExecutionConfig`. (F-1-builder, G-1)
- AC-3: WHEN `milknado_run_loop_start` starts a detached loop run without a runner override THE SYSTEM SHALL launch `python -m milknado.app.loop_runner`, and no module under `src/milknado/mcp/` SHALL construct an adapter for the runner. (F-1-runner, G-2)
- AC-4: WHEN `just check-llm` runs `lint-imports` THE SYSTEM SHALL enforce a contract that forbids `milknado.domains` from importing `milknado.loop`, and the contract SHALL pass with no ignored imports. (F-2, G-3)
- AC-5: WHEN `just check-llm` runs `lint-imports` THE SYSTEM SHALL enforce a contract that forbids `milknado.adapters` from importing any `milknado.loop._*` module, and the contract SHALL pass with no ignored imports. (F-3, G-4)
- AC-6: WHEN a protected worker starts its lifeline helper THE SYSTEM SHALL run the command that `ProtectionContext` carries, and no string in `src/milknado/loop/` SHALL name a module under `milknado.adapters`. (F-4-lifeline, G-4)
- AC-7: WHEN each curd lands THE SYSTEM SHALL pass `just check-llm`, including the `real_process` lane. (F-1-builder, F-1-runner, F-2, F-3, F-4-lifeline, G-1, G-2, G-3, G-4)

## Acceptance demonstration

| Field | Value |
| --- | --- |
| User task | Run one graph node locally and get a worker commit that carries the configured footer. |
| Input | A temporary repository with `milknado.toml` that sets `commit_footer = "Co-Authored-By: Test <t@example.com>"`, one task node, and a fake worker command that commits and prints the completion signal. |
| Entrypoint | `milknado run` (headless) |
| Expected result | The worker prompt contains the footer instruction, and the merged commit message ends with the footer. |
| Real boundaries | The real CLI process, a real git repository and worktree, the real `loop` engine, and a real fake-worker subprocess. |
| False success | The test passes because it reads `ExecutionConfig` in-process. The check reads the prompt that the fake worker received and the commit on the target branch, so a missing footer fails it. Run it on `9bc29e6` first: it must fail there. |
| Evidence | The test output and the target-branch commit message from the run. |

## Interface sketches

```text
slice:            domains/execution (builder), app (runner, base config), domains/common (types), loop (public submodules)
spine step:       workflow (builder, runner), domain (types), infra (loop surface)
public interface: app.run.build_exec_config(config: MilknadoConfig, project_root: Path) -> ExecutionConfig  (F-1-builder)
public interface: domains.execution.config_for_profile(base: ExecutionConfig, profile: FlavorProfile) -> ExecutionConfig  (F-1-builder)
public interface: python -m milknado.app.loop_runner  (F-1-runner)
public interface: domains.common.RunStatus, domains.common.CompletionVerdict, domains.common.ProviderFamily, LifecycleOperation, CapabilitySupport, RuntimeCapabilities, runtime_capabilities(family: ProviderFamily) -> RuntimeCapabilities  (F-2)
public interface: milknado.loop.process: SpawnOptions, StartEvidence, WorkerProcess, ProtectedWorker, spawn_protected, ProtectionContext, WorkerRegistry, Descendant, observe_descendants, terminate_verified, terminate_verified_result, run_lifeline, AgentRunSpec, ManagedRun, translate_argv  (F-3)
public interface: milknado.loop.events: Event, EventData  (F-3)
public interface: ProtectionContext.lifeline_command: tuple[str, ...]  (F-4-lifeline)
private:          loop._* modules keep their bodies; the new submodules only re-export
crust delta:      new exports in domains.common, domains.execution, milknado.loop.process, milknado.loop.events; app.loop_runner replaces mcp._loop_node_runner
arrows:           removed: domains -> loop, app -> mcp (runner path), loop -> adapters (lifeline string); unchanged: loop -> domains.common, adapters -> loop (public only)
```

## Risks

- The runner module path is a string in `app/loop.py:53` and in at least 4 test files. A missed string fails only at run time. Search for `_loop_node_runner` after c2.
- AC-1 changes local-run commit messages for every project that sets `commit_footer`. This is intended. Call it out in the c1 PR.
- `RunStatus` has 14 test importers and `CompletionVerdict` has 5. Test churn is mechanical but wide.
- `tests/loop` imports many `_process_*` private names. Those tests can keep private imports, because the new contract covers only `milknado.adapters`.
- The import-linter section sets `include_external_packages = true`. A new contract can need `allow_indirect_imports` to avoid transitive hits through `domains.common`.

## Open questions

- [TBD] Does `RunLoop` apply review settings when `_milknado_config` is `None`? The c1 curd traces this before it changes the call site.
- [TBD] Does any `real_process` test pin `milknado.adapters._loop_lifeline` as a literal argv? The c4 curd checks this first.

## Quality gates

- `just check-llm`: prints the PASS line after each curd.
- `uv run lint-imports`: the two new contracts pass with no ignored imports.

## Curds

| Curd | Covers | Depends on | State |
| --- | --- | --- | --- |
| c1-profile-builder | G-1 (AC-1, AC-2) | none | planned |
| c2-runner-to-app | G-2 (AC-3) | c1 | planned |
| c3-domain-types-to-common | G-3 (AC-4) | c2 | planned |
| c4-loop-public-surface | G-4 (AC-5, AC-6) | c3 | planned |

The typed planner has not run. See `execution_holds`.
