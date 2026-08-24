# Dead-code coverage gate — decision

Spec: `dead-code-coverage-gate` (durable specs corpus). Related issues: #363 (class 4), #364 (class 3).

### ADR-dead-code-coverage-gate-001: coverage.xml whole-symbol zero-hit cross-check over skylos / deadcode / mypy  [status: accepted]

- **Context:** `just check-llm`'s `dead-code` step is `vulture`, which matches bare names ignoring scope. A live canary (2026-08-16) proved a planted dead `run()` passed silently because `run` is used elsewhere in `src/` (name-collision blindness); dead reference cycles unreachable from any entry point are equally invisible. The 95% coverage floor lets ~5% of `src/` be dead and unnoticed. A prior `/briesearch` (three parallel researchers) found no single tool closes all four false-negative classes.
- **Decision:** Add a stdlib-only coverage cross-check that flags whole symbols (functions/methods) whose every measured `coverage.xml` line has zero hits, honoring vulture `ignore_decorators`, `[project.scripts]` targets, and a reviewed `[tool.milknado.dead-code].allow` allowlist. Vulture `ignore_names` applies only to vulture because its bare-name wildcard behavior would recreate the false negatives this gate closes. Posture: whole-symbol gate + allowlist. One mechanism closes class 1 (name-collision blindness) and class 2 (dead reference cycles), since both surface as zero-hit code.
- **Alternatives:** *skylos* — purpose-built entry-point reachability, but a heavyweight dep with an unconfirmed findings exit-code contract. *deadcode* — dotted-path class-1 mechanism, but talk-only evidence and overlaps vulture (its AGPLv3 licence is *not* a blocker for a subprocess CLI gate — no import, link, or redistribution — so licence was not the reason to decline). *mypy `--warn-unreachable`* — targets class 4 only and adds a second type checker beside `ty`. All declined to avoid a new dependency, since `coverage.xml` is already emitted by the gate.
- **Consequences:** Zero new dependency; reuses `coverage.xml` and the existing `# pragma: no cover` discipline (excluded lines are absent from the report — a free false-positive filter). Costs an AST + coverage-correlation script to maintain and a one-time adoption triage of the current uncovered tail (delete / test / pragma / allowlist). Class 4 semantically dead branches stay out of scope (tracked in #363).

### Context: why `ty`, not `mypy`, and what it costs here

milknado's toolchain is Astral end-to-end (`ruff`, `uv`, `ty>=0.0.38` as a dev dep); `mypy` is not a dependency. Neither type checker runs in the `check-llm` gate today. `ty` is pre-1.0 and ships no unreachable diagnostic (astral-sh/ty#784 open), which is exactly why closing class 4 would force a reach back to `mypy --warn-unreachable` — the cost of the toolchain-cohesion bet.

### ADR-dead-code-coverage-gate-004: CI and local gates run the same dead-code checks  [status: accepted]

- **Decision:** Run the whole-symbol coverage check immediately after coverage generation in both `check-llm` and `build-ci`; keep the vulture step in both paths.
- **Consequence:** A green local PR gate and CI evaluate the same dead-code contracts against the same generated report. Invalid, missing, unreadable, or structurally empty coverage XML fails with its path and parser or filesystem cause.
