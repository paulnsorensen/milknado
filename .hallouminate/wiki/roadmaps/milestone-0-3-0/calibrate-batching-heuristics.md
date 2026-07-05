---
kind: goal
slug: calibrate-batching-heuristics
roadmap: milestone-0-3-0
created: 2026-06-30
prereqs: []
---
# Calibrate the batching heuristics from telemetry

## Intent

`MEGA_BATCH_THRESHOLD = 5` is, by the wiki's own admission, an **uncalibrated
heuristic** (`architecture/planning.md` §Mega-batch detection): "one batch = one
ralph loop = one review unit", with an explicit invitation to "tune it in
`change.py` if calibration data appears". Calibration data *does* appear —
`record_batch_snapshot` (`planning.md` §Telemetry) appends a JSONL line per plan
to `.milknado/calibration.jsonl` with change/batch/oversized counts, solver
status, and symbol-spread max/mean. The threshold (and the `DUMB_ZONE_BUDGET`
ceiling it sits alongside) have never been revisited against that record.

Done looks like: an analysis of accumulated `calibration.jsonl` data that either
justifies the current `MEGA_BATCH_THRESHOLD` / budget values with evidence or
proposes tuned values, with the analysis recorded so the next reviewer doesn't
start from zero. This is an independent goal — it touches only the planning /
batching slice and depends on nothing else in this milestone.

## Acceptance

- The accumulated `.milknado/calibration.jsonl` records are analyzed (counts,
  distributions of change/batch/oversized counts, symbol-spread, solver status)
  — computed, not eyeballed (Rule 2).
- A decision is recorded: either `MEGA_BATCH_THRESHOLD` (and/or
  `DUMB_ZONE_BUDGET`) is retuned in `change.py` with the data backing the new
  value, or the current value is confirmed with evidence and the "uncalibrated"
  caveat in `planning.md` is downgraded accordingly.
- The mega-batch vs oversized distinction is preserved — the change-count signal
  (`mega_batch_change_count`) and the token-budget signal (`Batch.oversized`)
  must not be folded together (the wiki flags this as a "must not fold").
- If the threshold changes, the CLI-reports / MCP-aborts reaction split stays
  intact (detection domain-owned, reaction per-entrypoint), and tests covering
  the boundary are updated.
- `architecture/planning.md` §Mega-batch detection is updated with the
  calibration outcome.
