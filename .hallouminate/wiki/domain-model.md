# Milknado domain model

## Execution ownership

These lifecycle distinctions retain the selected design responsibilities and now map to the implemented runtime.[^design][^implementation]

**Supervisor** — The host runtime owns the run and the actual worker handles.
_Avoid_: using worker identity as supervisor identity.
_Code_: `src/milknado/domains/common/process.py:21-27`; `src/milknado/loop/_process_lifecycle.py:27-49`.

**Worker invocation** — One actual agent process and its execution identity. Replacing a lifeline does not create another invocation.
_Avoid_: using a lifeline PID as the worker PID.
_Code_: `src/milknado/domains/common/process.py:13-18`; `src/milknado/loop/_process_lifecycle.py:85-138`.

**Lifeline** — A per-worker monitor for supervisor liveness and verified cleanup. The supervisor retains agent streams and exit collection.
_Avoid_: stream proxy, shared guardian.
_Code_: `src/milknado/loop/_lifeline.py:78-124`; `src/milknado/domains/common/process.py:30-35`.

**Covered targets** — Owned groups and identity-verified observed descendants included in bounded cleanup.
_Avoid_: every possible descendant.
_Code_: `src/milknado/loop/_process_lifecycle.py:202-238`; F-8/F-9.

**Confirmed cleanup** — Exit confirmation for every covered target. Signal delivery, leader exit, or closed stdout does not establish it.
_Code_: `src/milknado/loop/_process_lifecycle.py:222-234`; `src/milknado/loop/_lifeline.py:119-124`.

See [execution architecture](./architecture/execution.md) for failure policies, verification evidence, and accepted containment limits.

[^design]: Durable spec `reap-orphaned-loop-workers.md`, Decisions F-7–F-12 and Acceptance AC-3/AC-13–AC-18.
[^implementation]: P0 implementation through `ad0110b`; current source `7e06824`. Real-process and durable-record tests pass with the complete project gate.

_Source: Approved lifecycle design and verified P0 implementation · Updated: 2026-09-30 · Supersedes: proposed-only status and pre-implementation code references; canonical terms remain unchanged._
