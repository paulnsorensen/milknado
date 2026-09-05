# Dead-code coverage gate — decision

Spec: `dead-code-coverage-gate` (durable specs corpus). Related issues: #363 (class 4), #364 (class 3).

### ADR-dead-code-coverage-gate-001: coverage.xml whole-symbol zero-hit cross-check over skylos / deadcode / mypy  [status: accepted]

- **Context:** `just check-llm`'s `dead-code` step is `vulture`, which matches bare names ignoring scope. A live canary (2026-08-16) proved a planted dead `run()` passed silently because `run` is used elsewhere in `src/` (name-collision blindness); dead reference cycles unreachable from any entry point are equally invisible. The 95% coverage floor lets ~5% of `src/` be dead and unnoticed. A prior `/briesearch` (three parallel researchers) found no single tool closes all four false-negative classes.
- **Decision:** Add a stdlib-only coverage cross-check that flags whole symbols (functions/methods) whose every measured `coverage.xml` line has zero hits, honoring vulture `ignore_decorators`, `[project.scripts]` targets, and a reviewed `[tool.milknado.dead-code].allow` allowlist. Vulture `ignore_names` applies only to vulture because its bare-name wildcard behavior would recreate the false negatives this gate closes. Posture: whole-symbol gate + allowlist. One mechanism closes class 1 (name-collision blindness) and class 2 (dead reference cycles), since both surface as zero-hit code.
- **Alternatives:** *skylos* — purpose-built entry-point reachability, but a heavyweight dep with an unconfirmed findings exit-code contract. *deadcode* — dotted-path class-1 mechanism, but talk-only evidence and overlaps vulture (its AGPLv3 licence is *not* a blocker for a subprocess CLI gate — no import, link, or redistribution — so licence was not the reason to decline). *mypy `--warn-unreachable`* — targets class 4 only and adds a second type checker beside `ty`. All declined to avoid a new dependency, since `coverage.xml` is already emitted by the gate.
- **Consequences:** Zero new dependency; reuses `coverage.xml` and the existing `# pragma: no cover` discipline (excluded lines are absent from the report — a free false-positive filter). Costs an AST + coverage-correlation script to maintain and a one-time adoption triage of the current uncovered tail (delete / test / pragma / allowlist). Class 4 semantically dead branches stay out of scope (tracked in #363).

### Context: why `ty`, not `mypy`, and what it costs here

> **Superseded 2026-09:** `ty` was removed in favour of basedpyright, which now runs in
> `check-llm` and `build-ci` — see `basedpyright-typecheck-gate-decision.md`. The claims
> below (`ty` as the dev dep, no type checker in the gate) describe the toolchain as it
> stood when this ADR was written, not the current state.

milknado's toolchain was Astral end-to-end (`ruff`, `uv`, `ty>=0.0.38` as a dev dep); `mypy` is not a dependency. Neither type checker ran in the `check-llm` gate at the time. `ty` was pre-1.0 and shipped no unreachable diagnostic (astral-sh/ty#784 open), which is exactly why closing class 4 would have forced a reach back to `mypy --warn-unreachable` — the cost of the toolchain-cohesion bet.

### ADR-dead-code-coverage-gate-004: CI and local gates run the same dead-code checks  [status: accepted]

- **Decision:** Run the whole-symbol coverage check immediately after coverage generation in both `check-llm` and `build-ci`; keep the vulture step in both paths.
- **Consequence:** A green local PR gate and CI evaluate the same dead-code contracts against the same generated report. Invalid, missing, unreadable, or structurally empty coverage XML fails with its path and parser or filesystem cause.

### ADR-dead-code-coverage-gate-002: owner-qualified exemptions and owned loop code  [status: accepted]

- **Context:** Vulture's global `ignore_names` accepts every same-named symbol regardless of owner. Textual framework members still need owner-qualified checks. The loop package entered the repository from ralphify, but Milknado now owns its maintenance.
- **Decision:** Remove unused loop APIs instead of preserving a compatibility surface. Keep one exact Vulture exception for `IterationEndedData.echo_stdout`, because `LoopAdapter` reads that field through a string key.[^5] Keep the two Windows whole-symbol coverage exceptions because the Windows job path calls them on that platform.[^6]
- **Turn cap:** Preserve `RunConfig.max_turns`, hard-cap enforcement, turn counting, cap events, and Claude/Codex soft wind-down. No current Milknado production constructor supplies `max_turns`, but the user retained this cost-control capability explicitly.[^7]
- **Removed surface:** Delete lifecycle hooks, pause and resume state, unused manager operations, frontmatter serialization, presentation-only events and fields, and unused adapter flags. Model soft wind-down by the structural `SoftWindDownAdapter` protocol instead of false capability fields and unsupported stubs.[^8]
- **Consequence:** The loop has no broad compatibility allowlist. New dead loop code fails the gate. The exact false-positive set contains only `echo_stdout`.[^9]

#### 2026-09-02 liveness audit outcome

The audit found no Milknado-owned activation for lifecycle hooks or manager convenience operations. Those features were removed.

The audit also found that `echo_stdout` is live. Review runs use it when structured result text is unavailable.[^5]

The retained turn cap is an explicit product decision, not an inference from current callers. This decision supersedes the prior verbatim-compatibility policy.

[^5]: `src/milknado/adapters/loop.py:341-383`; `src/milknado/loop/_events.py:111-118`
[^6]: `src/milknado/loop/_agent.py:160-219`; `pyproject.toml:312-315`
[^7]: `src/milknado/loop/_run_types.py:125`; `src/milknado/loop/engine.py:246-252`; `tests/loop/test_turn_cap.py`
[^8]: `src/milknado/loop/adapters/_protocol.py:135-151`; `src/milknado/loop/manager.py`
[^9]: `scripts/check_dead_code.py:103-108`

_Source: user decision and PR #405 loop liveness cleanup · Updated: 2026-09-02 · Supersedes: the 33-item vendored-loop compatibility allowlist · 2026-09-02_

### ADR-dead-code-coverage-gate-003: production-only barrel liveness follows public contracts  [status: accepted]

- **Decision:** Parse literal barrel `__all__` declarations statically and index only production `src/`. A directly used export keeps its public signature annotation types and recursively keeps types referenced by exported model fields. Tests and barrel bookkeeping do not establish production liveness.
- **Audit outcome:** Keep `TrustedGlobalPath`, `HeadlessOutcome`, `GithubFieldOption`, `HarvestSummary`, `RoadmapDocument`, and the five GitHub/wiki result contracts because live production APIs expose them. Remove only the crust exports for `global_config_path`, `cancel_path`, `WorktreeManager`, `summarize`, and `normalize_plan_identifier`; their implementations remain internal. Delete the test-only `extract_section`. Connect `reconcile_run_window` at the terminal successful MCP poll seam rather than deleting the documented lifecycle hook. `GraphSummary` remains internal with `summarize`.
- **Consequence:** A test-only consumer cannot preserve an export, while DTO and return-contract types remain live without ad hoc exemptions.
