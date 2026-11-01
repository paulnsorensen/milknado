# Build Gate — just check-llm

`just check-llm` is **the** PR gate. Run it — and only it — before opening any PR.
It bundles file-size, import-contracts, dead-code, the architecture gates, lint, format, web and browser suites, tests, project
coverage, diff coverage, and typecheck in one shot, so you never need to chain the
individual recipes to gate. Source: `justfile`, `AGENTS.md` (`CLAUDE.md` is a symlink
to it), `codecov.yml`.

## What it checks

The gate is a Python script inside the `check-llm` recipe (`justfile`). The coverage
threshold is `COVERAGE_THRESHOLD := "95"`. The recipe groups steps into lanes. The
lanes run at the same time. Steps inside one lane run in order.

Concurrent lanes:

- **file-size** — `scripts/check_file_lengths.py`
- **import-contracts** — `lint-imports`
- **dead-code** — `scripts/check_dead_code.py`
- **slice-guard, then depth-ratchet** — the architecture gates. See
  [[architecture-gates-decision]].
- **lint+format** — `just lint`
- **web-typecheck, web-lint, web-test** — the `web/` dashboard suites (need Node)
- **tests** — pytest with `-m "not browser and not real_process"` on an xdist pool
- **browser** — `tests/browser` (Playwright, needs Chromium)
- **typecheck** — `basedpyright` at the recommended tier

Serial steps run only after every lane passes:

1. **tests-real-process** — pytest with `-m "real_process and not browser"`, run alone
   with `--cov-append`. It also enforces `--cov-fail-under=95` and writes `coverage.xml`.
2. **dead-code-coverage** — `scripts/check_dead_code_coverage.py`
3. **diff-coverage** — `diff-cover coverage.xml --compare-branch=origin/main --fail-under=95`

The recipe runs `git fetch -q origin main` first so the diff base is current.

## Why it's shaped this way

- **Token-efficient by design.** On success it prints a single line:
  `✅ check:llm PASS …`. On failure it prints `❌ check:llm FAILED at: <step>`
  followed by **only that step's** captured output, then exits non-zero. You never
  pay for green logs, and a red run hands you exactly the failing step. This is why
  the project mandates `check-llm` over running `lint`/`test`/`coverage-check`
  separately — same coverage, a fraction of the context.
- **Lanes isolate failures.** The gate reports every failing lane in declared order.
  A lint failure never hides a test failure. The serial steps are skipped when any
  lane fails.
- **Real-process tests run alone.** A test module that starts process trees (tmux,
  PTYs, lifeline helpers, fake workers) carries `pytest.mark.real_process`. Those
  tests would compete with the xdist pool, so the gate runs them after the lanes.
  Project and diff coverage count both runs. `REAL_PROCESS_WORKERS` is an upper
  bound. A numeric `TEST_WORKERS` lowers it.
- **Non-mutating.** `check-llm` never autofixes. If it fails on **lint** or
  **format**, run `just lint-fix` (ruff check `--fix` + ruff format), then re-run
  the gate. Do not hand-patch formatting.
- **Diff coverage mirrors `codecov/patch`.** `codecov.yml` sets both `project` and
  `patch` targets to 95% (1% threshold). The local `diff-cover` step computes the
  same patch metric: of the lines **this branch changes vs `origin/main`** (staged
  and uncommitted edits included), 95% must be covered. This catches a
  `codecov/patch` CI failure locally, before you push — the usual reason a PR goes
  red even when the suite is green.

## Interpreting the result

- **Green** (PASS line) → open the PR.
- **Red** → do not open a PR. Fix the reported failing step, re-run `just check-llm`.
  Lint/format red → `just lint-fix` then re-gate.

## Key recipes (justfile)

```bash
just install        # uv sync — install all dependencies
just lint           # ruff check + format --check (no changes; CI uses this)
just lint-fix       # ruff check --fix + ruff format (autofix)
just test [args]    # pytest tests/ (e.g. just test -k pattern)
just test-file <f>  # pytest on a single file
just check-llm      # THE PR GATE — quiet on success, failing step only on failure
just build          # autofix pipeline: lint-fix → coverage-check → typecheck
just build-ci       # no-autofix pipeline: lint → file-size → dead-code → slice-guard → depth-ratchet → coverage → dead-code-coverage → typecheck (CI uses this)
just slice-guard    # check src/ against slices.toml (also a check-llm step)
just depth-ratchet  # compare slice crust depth with depth-ratchet.json (also a check-llm step)
just mcp-dev        # run milknado-mcp under watchfiles, restart on src/ changes
just clean          # remove .pytest_cache, .ruff_cache, htmlcov, .coverage, *.pyc
```

`build` vs `build-ci`: `build` starts with `lint-fix` (mutating), while `build-ci`
runs the check-only lint, file-size, dead-code, slice-guard, depth-ratchet, coverage,
and dead-code-coverage stages. Both end with `typecheck` last. CI runs `build-ci`.

## mcp-dev caveat

`just mcp-dev` runs `watchfiles "uv run milknado-mcp" src/milknado` — it restarts the
MCP server on every change under `src/milknado/`. The server speaks stdio, so a
restart **drops the existing connection**. After each restart the client must `/mcp`
reconnect to pick up the new process.
