---
created: 2026-06-30
---
# Milestone 0.3.0 — Harden the Native Backend & Lock Distribution

Close the open seams the two most-recent decisions left dangling — the native
"ultracode" execution backend (PR #145) and the portable plugin / PyPI
distribution work — then take the next architectural step (a real
coordinator-run identity) and pay down two standing debts (worker↔coordinator
messaging, batching-heuristic calibration). Every goal here is sourced from a
deferred item already recorded in the wiki, not net-new scope.

Rough order: lock the distribution layout first (it decides where the ultracode
workflow ships), make ultracode installable and broadly tested, then replace the
pid-proxy goal-claim model with a real coordinator-run entity, then build the
bidirectional messaging that entity enables. Batching calibration runs
independently against accumulated telemetry.

<!-- HALLOUMINATE:INDEX-START -->
<!-- HALLOUMINATE:INDEX-END -->
