# Decision — PR94 Refactor Split

PR #94 had grown to 50 files (+3698/-1380) by mixing two unrelated kinds of
change: pure structural refactors and new execution features (pr_stack mode,
crash recovery / resume). It was unreviewable. The decision: carve the
refactors out into their own PR, leaving #94 as features-only.

## What was done

- **PR #99** — the 7 pure-refactor commits, extracted as a contiguous prefix
  (`main..1fdb9b3`), based on `main`. Branch `ralphify/03a-refactors`.
- **PR #94** — retargeted from base `main` to base `ralphify/03a-refactors`.
  The shared refactor prefix then drops out of #94's diff automatically
  (50→42 files, deletions -1380→-259). Retitled to
  `feat(execution): pr_stack mode + crash recovery (resume)`.
- Merge order is **bottom-up**: #99 first, then #94.

## Why this approach

- **Base-retargeting, not history rewrite.** The 7 refactors are a clean
  contiguous prefix and a true ancestor of #94's head, so retargeting #94's
  base collapses them out of its diff with zero conflict, zero force-push, and
  full reversibility. Rewriting #94's history was rejected as fragile — its
  feature commits include a merge commit (pulling #97/#98) plus follow-ups that
  interleave refactor polish with feature polish.
- **`--mcp-config` headless fix stays in #94.** Extracting it as a third PR was
  rejected: its tests (`TestMcpConfigInjection`) were repointed to an
  allowlisted worker command in a later entangled commit, so a clean extraction
  would have forced exactly the fragile rebase the split was avoiding.

## Gotchas for future contributors

- **`gh pr edit --base` fails** on this repo with exit 1 (a deprecated
  projects-classic GraphQL call). Retarget a PR base via the REST API instead:
  `gh api -X PATCH repos/<owner>/<repo>/pulls/<n> -f base=<branch>`.
- A stacked feature PR transiently shows merge content from PRs its base
  predates (here #94 showed #97/#98's copilot docs + batching). This is cosmetic
  and drops out once the stack merges and the PR rebases — do not "fix" it.
- Watch `solver.py` at merge time: #99's `SolverConfig` refactor and #97's
  budget-default change touch the same file. Merge #99 (bottom of stack) first
  and resolve once if GitHub flags it.

The durable rule: when one PR mixes structural refactors with features, split
by base-retargeting on a clean commit prefix, not by rewriting history.
