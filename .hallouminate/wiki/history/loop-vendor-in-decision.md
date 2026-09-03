# Decision — Vendor the Ralphify Engine into milknado.loop

Date: 2026-06-06 · Status: executed in PR #137 · See [pr137-loop-vendor-in](./pr137-loop-vendor-in.md) for the execution record.

## Why vendoring was required, not optional

Milknado depended on a **fork** of `computerlovetech/ralphify` (`paulnsorensen/ralphify`, beta `0.4.0b3`, pinned `@main` as a `git+` direct-URL dep). Three facts closed the decision:

1. **PyPI rejects direct-URL deps.** Milknado is headed to PyPI, and PyPI refuses uploads whose metadata contains PEP 508 direct-URL dependencies (`ralphify @ git+...`). No pin or lockfile trick fixes this.
2. **The upstream merge path was dead.** Fork-only commits include the library-embedding API milknado depends on (`feat: support embedding ralphify as a library (#16)`). PRs to upstream went nowhere: #46 open unmerged since 2026-04-17, #47/#48 closed unmerged. The fork sat 10 ahead / 78 behind upstream, with sync cost already visible.
3. **The fork existed solely for milknado.** User-confirmed: the standalone `ralph` CLI does not matter; milknado was the only consumer of the fork across `~/Dev`. The two-repo dance (PR in ralphify → bump in milknado) bought nothing.

## What milknado actually used

One adapter file (`adapters/ralphify.py`, now `adapters/loop.py`) imported exactly five symbols: `RunManager`, `RunConfig`, `QueueEmitter`, `EventType`, `RunStatus` — headless lifecycle management of coding-agent subprocess loops (create/start/stop/observe runs, drain the event stream). No ralphify type crossed the port boundary (port methods use milknado-native types or `Any`); nothing invoked the `ralph` CLI.

The five symbols are **not separable from the engine**: `RunManager → engine → _agent + adapters/*`. The minimal honest vendor was the full used closure — **17 modules, ~3,400 LOC** (engine, subprocess agent layer, events, run types, manager, per-CLI adapters, hooks, frontmatter/promise/resolver/runner utilities).

## What was dropped at the door

~2,500 LOC (44% of the package) that milknado never executed: `cli.py` (688), `_console_emitter.py` (1,589 — Rich TUI), `_keypress.py` (222), `_brand.py` (10).

## Cost/benefit notes that shaped the spec

- **Zero new dependencies** — ralphify's deps (`pyyaml`, `rich`, `typer`) were already direct milknado deps.
- **The largest hidden cost was tests**: the engine shipped no test suite in the wheel, and milknado's 95% diff-coverage gate would demand coverage. Resolved by vendoring the fork's 529-test closure suite alongside the code.
- The fork's own commit log showed ongoing per-CLI adapter churn (opencode schema fix, Crush adapter, max_turns) — maintenance that happens regardless of where the code lives; vendoring moves it in-repo rather than creating it.
- **Drain process output before closing pipes.** Cleanup first gives reader threads a bounded drain window, which preserves final buffered output. If a detached descendant holds a pipe open, cleanup closes parent descriptors and drains again. Raw descriptor closure does not wake a blocked reader on every operating system. Cleanup must skip Python-level pipe finalization while any reader remains alive. The regression test must detach the descendant; otherwise process-group termination hides the forced-close fallback.[^4]

[^4]: `src/milknado/loop/_agent.py:990-1004`; `tests/loop/test_agent.py:1837-1885`; PR #405 commit `f539b21`.

## Decision

Vendor the used closure (~3,400 LOC, 17 modules) into `src/milknado/loop/`; drop the CLI/TUI modules; delete the `git+` dep; archive the fork (manual post-merge step). Vendored internals land **verbatim** — byte-identical fork behaviour, formatting-only divergence allowed — so the vendor-in is auditable against fork SHA `ec494875`.
