---
kind: goal
slug: golden-path-walkthrough
roadmap: milestone-0-3-0
created: 2026-07-01
prereqs: [ultracode-install-ergonomics]
---
# Golden-path walkthrough in the README

## Intent

The README today is installation and configuration reference (channel
matrix, `quality_gates` tables) — there is no narrative showing what using
milknado feels like. Firstmate's README converts evaluators with a worked
transcript: three shell commands, then a plain-English conversation ending
in a PR. An evaluator choosing between the two sees firstmate's story and
milknado's config tables.

This goal writes milknado's story: a single worked walkthrough from zero to
harvested outcome — install, `milknado init`, author or import a wiki
roadmap, `plan` a goal into tasks, `run` the parallel ralph loops, watch
them (attach, once [[tmux-run-primitive]] lands — the walkthrough should be
written to absorb that when available, not blocked on it), and harvest the
outcome back into the wiki. The framing is the golden-path pitch: ralph
trees, casual sub-agent supervision, and full higher-level roadmaps — the
things firstmate cannot offer — presented in firstmate's approachable
register.

Depends on [[ultracode-install-ergonomics]] because the walkthrough's first
step is the install, and it must document the ergonomic path, not the
current one — writing it earlier bakes in steps that goal is about to
change.

Open questions to resolve during design (not silently):

- One walkthrough or two: single CLI-driven story, or CLI + MCP-coordinator
  variants (the MCP path is how agent users actually drive it)?
- Where does the full walkthrough live if the README carries only a
  condensed version — docs/ tree (new) or wiki page linked from README?
- Is the transcript hand-written or generated from a real session (real is
  more honest, but rots; decide the refresh mechanism).

## Acceptance

- The README contains a worked walkthrough (or a condensed version linking
  to the full one) covering install → init → roadmap → plan → run → harvest
  with real command output shapes, not placeholders.
- Every command in the walkthrough is verified against the released install
  path (fresh-environment run), not just the dev checkout.
- The golden-path framing is explicit: the walkthrough names what the graph,
  batching, and verification gates buy over a flat orchestrator.
- The walkthrough stays truthful as features land: attach/steering steps are
  included only once their goals ship, with a stated place to slot them in.
