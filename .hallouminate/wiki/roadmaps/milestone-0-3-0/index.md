---
kind: roadmap
slug: milestone-0-3-0
created: 2026-06-30
---
# Milestone 0.3.0 — Harden the Native Backend & Lock Distribution

Close the open seams the two most-recent decisions left dangling — the native
"ultracode" execution backend (PR #145) and the portable plugin / PyPI
distribution work — then take the next architectural step (a real
coordinator-run identity) and pay down two standing debts (worker↔coordinator
messaging, batching-heuristic calibration). Every goal here is sourced from a
deferred item already recorded in the wiki, not net-new scope.

A second track adopts what makes firstmate more usable, so milknado becomes
the golden path over it — ralph trees plus firstmate's casual sub-agent feel
plus full higher-level roadmaps: tmux as a first-class run primitive
(visibility), non-headless interactive steering built on it, and zero-token
event-driven supervision replacing poll loops (built on the messaging
channel). Two supporting goals round out the track: fail-closed worktree
teardown (today every removal is `--force`, destroying unlanded work — unsafe
once humans steer runs) and a golden-path README walkthrough that tells the
story firstmate's transcript tells, after install ergonomics land. Each of
these goals carries its open design questions explicitly in its Intent — to
be resolved at design time, not silently.

Rough order: lock the distribution layout first (it decides where the ultracode
workflow ships), make ultracode installable and broadly tested, then replace the
pid-proxy goal-claim model with a real coordinator-run entity, then build the
bidirectional messaging that entity enables. Batching calibration runs
independently against accumulated telemetry. On the adoption track, tmux lands
first (no prereqs), interactive steering follows it, and zero-token supervision
follows worker↔coordinator messaging.

<!-- HALLOUMINATE:INDEX-START -->
<!-- HALLOUMINATE:INDEX-END -->
