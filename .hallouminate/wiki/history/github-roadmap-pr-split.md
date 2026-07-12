# GitHub Projects v2 roadmap integration — PR split map

The peer-spoke monolith (`claude/roadmap`, PR #193, closed superseded) was split
into a Graphite stack so each slice reviews in isolation:

1. **#194** (merged) — `github_ref` binding on the node model
2. **#195** (merged) — `feat/roadmap-2-gh-adapter`: gh adapter, harvest builder, wiki locate/export refactor
3. **#196** (open, base `main`) — `feat/roadmap-3-github-domain`: Projects v2 crossover slice (bind/export/import)
4. **#197** (open, base roadmap-3) — `feat/roadmap-4-cli-mcp`: CLI + MCP surface

The stack versions supersede the monolith — age-review fixes (passes 1–3, see
`.cheese/age/github-roadmap-integration.md` in the roadmap worktree) were applied
on the stack branches only, so never merge `claude/roadmap` directly.

## Gotcha — worktrees pin merged branches during `gt sync`

`gt sync` cannot delete a merged branch checked out in a `.worktrees/*` worktree
(it warns and skips). Fix: `git -C .worktrees/<name> checkout --detach` (after
confirming the worktree is clean), then re-run `gt sync --no-interactive --force`
— `--force` is also needed non-interactively when a child's local parent differs
from the remote parent after a mid-stack merge.
