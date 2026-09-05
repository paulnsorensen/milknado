# Build Gate — just check-llm

`just check-llm` is **the** PR gate. Run it — and only it — before opening any PR.
It bundles file-size, import-contracts, dead-code, lint, format, tests, project
coverage, diff coverage, and typecheck in one shot, so you never need to chain the
individual recipes to gate. Source: `justfile`, `AGENTS.md` (`CLAUDE.md` is a symlink
to it), `codecov.yml`.

## What it checks, in order

Defined in the `check-llm` recipe (`justfile`), threshold pinned to `COVERAGE_THRESHOLD := "95"`:

1. **file-size** — `python scripts/check_file_lengths.py`
2. **import-contracts** — `lint-imports`
3. **dead-code** — `python scripts/check_dead_code.py`
4. **lint+format** — `just lint`
5. **tests+coverage** — `pytest tests/ -q --cov=src/milknado --cov-fail-under=95`, writing `coverage.xml`
6. **dead-code-coverage** — `python scripts/check_dead_code_coverage.py`
7. **diff-coverage** — `diff-cover coverage.xml --compare-branch=origin/main --fail-under=95`
8. **typecheck** — `basedpyright` at the recommended tier, run LAST so a type-check
   failure never suppresses lint, test, or coverage signal from this sole gate command

It runs `git fetch -q origin main` first so the diff base is current.

## Why it's shaped this way

- **Token-efficient by design.** On success it prints a single line:
  `✅ check:llm PASS …`. On failure it prints `❌ check:llm FAILED at: <step>`
  followed by **only that step's** captured output, then exits non-zero. You never
  pay for green logs, and a red run hands you exactly the failing step. This is why
  the project mandates `check-llm` over running `lint`/`test`/`coverage-check`
  separately — same coverage, a fraction of the context.
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
just build-ci       # no-autofix pipeline: lint → file-size → dead-code → coverage → dead-code-coverage → typecheck (CI uses this)
just mcp-dev        # run milknado-mcp under watchfiles, restart on src/ changes
just clean          # remove .pytest_cache, .ruff_cache, htmlcov, .coverage, *.pyc
```

`build` vs `build-ci`: `build` starts with `lint-fix` (mutating), while `build-ci`
runs the check-only lint, file-size, dead-code, coverage, and dead-code-coverage
stages. Both end with `typecheck` last. CI runs `build-ci`.

## mcp-dev caveat

`just mcp-dev` runs `watchfiles "uv run milknado-mcp" src/milknado` — it restarts the
MCP server on every change under `src/milknado/`. The server speaks stdio, so a
restart **drops the existing connection**. After each restart the client must `/mcp`
reconnect to pick up the new process.
