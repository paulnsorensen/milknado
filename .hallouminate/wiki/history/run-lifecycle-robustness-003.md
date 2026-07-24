# ADR — Graph-layer self-heal: reconnect, startup check, automigrate, quarantine (run-lifecycle-robustness-003)

Date: 2026-07-23 · Status: accepted · Spec: run-lifecycle-robustness

## Decision

`MikadoGraph` self-heals at its single chokepoint (`__init__` / `_conn` access): (1) `_conn` becomes a health-checked property that reopens a closed connection, re-applies WAL/foreign_keys/busy_timeout, and re-runs schema setup; (2) `close()` records its caller stack, and an unexpected reopen logs both the close and heal stacks; (3) every open runs `PRAGMA quick_check`, quarantining a corrupt db (+ sidecars) to `*.corrupt-<ts>` and recreating; (4) `PRAGMA user_version` + an ordered `MIGRATIONS` list automigrate on open. The in-flight statement at close time fails loud — no transparent retry.

## Rationale

- User direction: "if the DB is down we should be fixing it, not ignoring it … a refresh mechanism or automigrater … and a check so we can self-heal over time."
- Every graph method delegates through `self._conn` to `_persistence` functions — one property covers all call sites without per-site whack-a-mole.
- `MikadoGraph.__init__` is the one construction chokepoint for MCP per-call opens, CLI, and TUI — no per-entry-point wiring.
- Schema management was additive-only `create_tables`; install drift (the observed 03:47 stale-install crash class) had no healing path.
- Loud logging on unexpected reopen keeps the underlying lifetime bug visible instead of masking it; the close-stack audit names the culprit pair.

## Alternatives

- Per-site guards (`try/except` around `get_execution_overview` etc.) — symptom-level, per-call-site whack-a-mole.
- Lifecycle ordering fix only (no publish after close) — correct only if teardown ordering is the bug, which is unproven; heals nothing for drift or corruption.
- Fail loud on corruption — safer, but the MCP stays broken until manual repair; rejected in favor of quarantine-and-recreate.

## Consequences

Local sqlite becomes self-healing across reconnect, corruption, and version drift. New machinery (MIGRATIONS, quarantine) gains its first real consumer immediately (migration v2, run-lifecycle-robustness-004). Quarantine destroys run history on corruption — accepted, as a corrupt DB is unusable and node state rebuilds from the wiki.
