---
kind: goal
slug: ultracode-install-ergonomics
roadmap: milestone-0-3-0
created: 2026-06-30
prereqs: [lock-distribution-layout]
---
# Make ultracode installable in one step

## Intent

The native-backend decision (`history/workflow-executor-decision.md`) records two
ergonomics gaps that make ultracode harder to adopt than it should be:

1. **`workflows/` is not a recognized plugin component.** `node-runner.js` ships
   in `plugins/milknado/workflows/` as the source of truth, but installing the
   plugin/MCP does *not* make it loadable — the user must hand-copy it to
   `.claude/workflows/` or invoke it by explicit `scriptPath`. Installing
   milknado should make running ultracode a documented one-step action.
2. **`node-runner.js` defers change-id→node-id resolution** to the orchestrator
   (`milknado_plan_batches` returns change-id-keyed batches; the node-id arrays
   must be resolved before fan-out). That resolution step is currently implicit
   and easy to get wrong.

Done looks like: a documented, friction-minimal path from "milknado plugin
installed" to "ultracode workflow runs", and a resolution step that's either
automated or spelled out so the orchestrator can't skip it. This goal depends on
[[lock-distribution-layout]] because where `node-runner.js` ships and how it's
discovered is part of the locked payload layout.

## Acceptance

- A documented one-step install/run path for the ultracode workflow exists
  (e.g. a `just` recipe or a copy-on-install step that lands `node-runner.js` in
  `.claude/workflows/`), and the manual-copy caveat in
  `workflow-executor-decision.md` is updated to reflect it.
- The change-id→node-id resolution is either implemented as a helper the runner
  calls or documented as an explicit orchestrator step with a worked example —
  no longer left as undocumented tribal knowledge.
- A fresh-install smoke test confirms: install plugin → reconnect `/mcp` →
  invoke the workflow → it loads and fans out at least one `agent()` without a
  manual file copy beyond the documented step.
- Constraint preserved: the Workflow script still does `agent()` fan-out only and
  calls no MCP tool at script scope (the script-scope `undefined` constraint from
  the decision page must not regress).
