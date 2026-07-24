# ADR — Deliver rejection findings via regenerated RALPH.md (run-lifecycle-robustness-002)

Date: 2026-07-23 · Status: accepted · Spec: run-lifecycle-robustness

## Decision

`_handle_review_rejection` passes its in-memory findings string through `_redispatch_review_round` → `_create_ralph_run` → `LoopAdapter._build_ralph_content`, which emits a `## Prior review findings (round N)` section before `## Task` when non-empty. The resume path keeps the default empty value and never emits the section.

## Rationale

- Redispatched workers re-derived fixes from the static brief and burned review rounds without converging (#298).
- RALPH.md is the document the ralph loop feeds the worker every iteration — putting findings there makes them required reading with no timing dependency.
- The findings string is already in memory at the rejection site; threading it adds no I/O and no DB dependency (critical while #297's closed-DB class exists).

## Alternatives

- `RunManager.queue_guidance` to the new run — guidance is a mid-run steering channel consumed at an iteration boundary; round 2's first iteration can run before it lands. Wrong semantics for initial conditions.
- Brief references the `.cheese/age/<slug>.md` path — relies on the worker bothering to read it; the failure mode already observed.
- Do Nothing — contradicts the filed bug.

## Consequences

`_build_ralph_content` gains a `prior_findings: str = ""` parameter; two dispatch call sites thread it. Workers cannot miss findings. The guidance channel remains reserved for operator steering.
