# Research: what milknado uses ralphify for, and what's vendorable

Date: 2026-06-06 · Question: honest build-vs-depend assessment of the `ralphify` dependency
Sources: tilth search over milknado worktree, explorer agents (adapter usage map + installed-package inventory), `gh` repo/compare API. `artifact-path` CLI not present — fallback path `.cheese/research/`.

## Synthesis

Milknado uses ralphify for exactly one thing: headless lifecycle management of coding-agent subprocess loops — create/start/stop/observe runs and drain their event stream. The consumption surface is five symbols (`RunManager`, `RunConfig`, `QueueEmitter`, `EventType`, `RunStatus`) imported in a single adapter file. That surface transitively requires ~3,400 of ralphify's ~5,700 LOC (engine, subprocess agent layer, per-CLI adapters, event types); the other ~2,500 LOC (44%) is the `ralph` CLI, Rich TUI renderer, and keypress handling that milknado never executes. The dependency is a beta (`0.4.0b3`) **fork** of `computerlovetech/ralphify`, pinned to `@main`, 10 commits ahead / 78 behind upstream, whose fork-only commits include the library-embedding API milknado depends on. Milknado is the only consumer of the fork found in `~/Dev`.

## Claim table

| # | Claim | Evidence | Confidence |
|---|---|---|---|
| 1 | Single importer: `src/milknado/adapters/ralphify.py` (271 lines) + its test file; nothing else in `src/` imports ralphify | tilth content search, all 10 non-adapter hits are tests | certain |
| 2 | Imported surface: `EventType, QueueEmitter, RunConfig, RunManager, RunStatus` | `adapters/ralphify.py:10` | certain |
| 3 | Members used: `RunManager.{create_run,start_run,stop_run,list_runs,get_run}`; `RunConfig(agent, ralph_dir, ralph_file, project_root, completion_signal, stop_on_completion_signal, log_dir)`; `EventType.{RUN_STOPPED, PROGRESS, ITERATION_COMPLETED, ITERATION_FAILED}`; `RunStatus.COMPLETED`; `run.state.{run_id,status}`, `run.stdout`, `emitter.queue` | adapter lines 24–222 (explorer digest) | certain |
| 4 | No ralphify type crosses the `RalphPort` boundary — port methods use milknado-native `ProgressEvent`/`VerifySpecResult` or `Any` | `domains/common/protocols.py:15–26`; run objects typed `Any` | certain |
| 5 | No CLI/subprocess invocation of `ralphify` anywhere in `src/` — `doctor.py:67` only probes package presence | explorer sweep of `src/` | certain |
| 6 | Used transitive closure: 17 modules ≈ 3,400 LOC (`_events`, `_run_types`, `manager`, `engine`, `_agent` 797, `_frontmatter`, `_output`, `_resolver`, `_runner`, `adapters/*` incl. claude/codex/copilot, `_promise`) | installed wheel at `.venv/.../ralphify/`, wc -l | certain |
| 7 | Unused remainder: 4 modules ≈ 2,500 LOC (44%) — `cli.py` 688, `_console_emitter.py` 1,589, `_keypress.py` 222, `_brand.py` 10 | same inventory | certain |
| 8 | ralphify deps (`pyyaml`, `rich`, `typer`) are all already direct milknado deps → vendoring adds zero new dependencies | ralphify METADATA:14–16 vs milknado `pyproject.toml:13–20` | certain |
| 9 | `paulnsorensen/ralphify` is a fork of `computerlovetech/ralphify`; 10 ahead / 78 behind; fork-only commits include `feat: support embedding ralphify as a library (#16)` — the API milknado uses | `gh repo view` + `gh api .../compare` | certain |
| 10 | Milknado is the sole consumer of ralphify across `~/Dev` (grep of all `pyproject.toml`) | explorer sweep | certain |
| 11 | Dependency pinned to `@main` of the fork — uv.lock freezes a commit (`8a68022`) until next `uv lock -U`, but the pyproject pin itself is floating | `pyproject.toml:12`, `direct_url.json` | certain |
| 12 | Installed version `0.4.0b3` (beta); no test suite ships in the wheel | dist-info RECORD/METADATA | certain |
| 13 | Fork's own commit log shows ongoing per-CLI adapter churn (opencode schema fix #18, Crush adapter #19, max_turns #20) — this maintenance happens regardless of where the code lives | compare API commit list | certain |

## What's vendorable

- **Minimal honest vendor = the full used closure (~3,400 LOC, 17 modules).** The five symbols are not separable from the engine: `RunManager → engine → _agent + adapters/*`. You cannot vendor "just the types" — `EventType`/`RunConfig` without the engine is useless to the adapter.
- **What you'd drop at the door**: `cli.py`, `_console_emitter.py`, `_keypress.py`, `_brand.py` (~2,500 LOC of TUI/CLI dead weight at milknado runtime).
- **Zero new deps** (claim 8). Coverage gate note: vendored code would fall under milknado's 95% diff-coverage gate and ships with no tests of its own (claim 12) — vendoring implies writing or porting tests, the largest hidden cost.

## Open questions (alternatives, not recommendations)

1. Does the `ralph` standalone CLI still matter to you as a separate tool? If yes, vendoring the engine forks it into two diverging copies. If no, the fork repo exists solely to serve milknado and the two-repo dance (PR in ralphify → bump in milknado) buys nothing. — speculating
2. Any intent to upstream the library-embedding API to `computerlovetech/ralphify`? 78-behind drift makes that window close over time. — don't know
3. If keeping the dep: pin to a tag/commit in `pyproject.toml` rather than `@main`? — speculating

## Confidence

**Certain** on usage surface, closure size, and fork status — all from reading installed code and GitHub API. The only speculative items are intent questions only the user can answer.

## Verdict (user-confirmed 2026-06-06)

Open questions resolved by the user:

1. Standalone `ralph` CLI does **not** matter — the fork exists solely for milknado.
2. Upstream merge path is dead: user's PRs to computerlovetech — #46 (`<promise>` completion tag / completion gate) open since 2026-04-17 unmerged; #47 (adapter layer + lifecycle hooks) and #48 (cache stats) closed unmerged. Fork PR #11 (upstream sync) sits open — sync cost already visible. Verified via `gh pr list`.
3. **Hard constraint**: milknado is headed to PyPI, and PyPI rejects uploads whose metadata contains PEP 508 direct-URL dependencies (`ralphify @ git+...`). Vendoring is required, not optional.

**Decision: vendor the used closure (~3,400 LOC, 17 modules) into milknado; drop CLI/TUI modules (~2,500 LOC); delete the git dep; archive the fork.** Zero new deps. Main work item: the engine ships no tests and milknado's 95% diff-coverage gate will demand them. Next: `/mold` the vendor-in spec (landing slice, RalphPort seam fate, test strategy, fork archival).
