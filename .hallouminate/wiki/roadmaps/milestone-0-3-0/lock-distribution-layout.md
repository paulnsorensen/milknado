---
kind: goal
slug: lock-distribution-layout
roadmap: milestone-0-3-0
created: 2026-06-30
prereqs: []
---
# Lock the portable plugin layout

## Intent

The portable-distribution design (`architecture/distribution.md`) shipped a
two-file marketplace split (`.claude-plugin/marketplace.json` for Claude+Codex
legacy, `.agents/plugins/marketplace.json` for Codex native) plus a shared
`plugins/milknado/` payload — but it closes with five explicitly *unverified*
questions under "Open questions (verify before locking the layout)". Until those
are answered, the layout is provisional and the duplication can't be justified or
simplified with confidence.

Done looks like: each of the five open questions answered with a citation, a
command output, or a throwaway-install test — and the distribution page updated
so the layout decision is *locked* rather than provisional. Where an answer lets
us simplify (e.g. Codex's native reader tolerates Claude schema → the two-file
split is not strictly required), make the simplification; where it confirms the
split is required, record the proof so it isn't re-litigated.

## Acceptance

- All five `distribution.md` open questions have a recorded answer with evidence:
  (1) does Codex's native `.agents/plugins` reader tolerate Claude-schema
  marketplace.json; (2) does Claude Code follow a symlinked `.claude-plugin/` on
  git-clone installs; (3) can Claude's `plugin.json` `skills` field reference
  paths outside the plugin dir; (4) is repo-level Codex marketplace pickup gated
  by `trust_level`; (5) does HTTP-transport MCP work inside a Claude Code plugin.
- Each answer cites a primary source (doc URL + quoted line), a command output,
  or a reproducible install test — not an unverified assertion (Rule 12).
- `architecture/distribution.md` is updated: the "Open questions" section is
  replaced by the resolved decisions, and any layout simplification an answer
  unlocks is either applied (with the marketplace/payload files changed) or
  explicitly recorded as "confirmed required, kept as-is".
- The plugin still installs cleanly on Claude Code and Codex after any layout
  change (manual install test from the repo-as-marketplace path).
