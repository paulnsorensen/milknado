set dotenv-load := true

# Matches codecov.yml project/patch target (95%)
COVERAGE_THRESHOLD := "95"
# A number pins a pool. "auto" sizes pytest (tests/worker_pool.py) and vitest
# (web/vitest.workers.ts) from free cores; TEST_WORKERS_MAX caps that size.
# Browser auto sizing stops at 4 workers.
TEST_WORKERS := env("TEST_WORKERS", "auto")
BROWSER_WORKERS := env("BROWSER_WORKERS", "auto")
# Tests marked real_process start real process trees. The gate runs them last, alone,
# with at most this many workers. TEST_WORKERS and free cores can lower the count.
REAL_PROCESS_WORKERS := env("REAL_PROCESS_WORKERS", "3")

# Show all available recipes
default:
    @just --list

# Install dependencies using uv, npm, and the browser tests' Chromium build
install:
    uv sync
    npm --prefix web ci
    uv run playwright install chromium

# Run lint and format checks concurrently (both are read-only).
lint:
    #!/usr/bin/env python3
    import subprocess
    import sys

    commands = [
        ["uv", "run", "ruff", "check", "src/", "tests/", "scripts/", "--preview"],
        ["uv", "run", "ruff", "format", "--check", "src/", "tests/", "scripts/"],
        ["uv", "run", "lint-imports"],
    ]
    processes = [subprocess.Popen(command) for command in commands]
    if any(process.wait() for process in processes):
        sys.exit(1)

# Run linters with autofix
lint-fix:
    uv run ruff check src/ tests/ scripts/ --fix --preview
    uv run ruff format src/ tests/ scripts/

# Run the test suite
test *args:
    uv run pytest tests/ -n {{TEST_WORKERS}} {{args}}

# Run individual test file
test-file file *args:
    uv run pytest {{file}} {{args}}

# Run tests with verbose output
test-verbose *args:
    uv run pytest tests/ -vv -n {{TEST_WORKERS}} {{args}}

# Inner loop only, never a gate: run tests affected since the last run (pytest-testmon).
# Excludes browser tests by path because testmon disables selection under -m.
# testmon misses data-file, subprocess, and dynamic-import dependencies; gate with check-llm.
test-affected *args:
    uv run pytest tests/ -n {{TEST_WORKERS}} --ignore=tests/browser --testmon {{args}}

# Run tests with coverage report
test-coverage:
    uv run pytest tests/ -n {{TEST_WORKERS}} --cov=src/milknado --cov-report=term-missing --cov-report=html

# Check coverage meets threshold
coverage-check:
    #!/usr/bin/env python3
    import subprocess
    import sys

    # Streamed (not captured): a captured, buffered run prints nothing until
    # the process exits, which hides all progress if pytest ever hangs.
    # Excludes tests/browser: those need a built web/ bundle and a browser,
    # and are run as their own CI/check-llm step instead.
    result = subprocess.run(
        [
            "uv", "run", "pytest", "tests/",
            "-n", "{{TEST_WORKERS}}",
            "-m", "not browser",
            "--cov=src/milknado",
            "--cov-report=term",
            "--cov-report=xml:coverage.xml",
            "--cov-fail-under={{COVERAGE_THRESHOLD}}",
        ],
    )

    if result.returncode != 0:
        sys.exit(result.returncode)

# Full build with autofix: lint-fix → coverage check → typecheck (for agents/developers)
build: lint-fix coverage-check typecheck
    @echo "✅ Build passed — ready for PR"

# Enforce the source file-size budget (mirrors the file-size step in check-llm)
file-size:
    uv run python scripts/check_file_lengths.py

# Report unreachable code (mirrors the dead-code step in check-llm)
dead-code:
    uv run python scripts/check_dead_code.py

# Hold src/ to the slice map in slices.toml (mirrors the slice-guard step in check-llm)
slice-guard:
    uv run python -m scripts.check_slices check --strict

# Compare each slice's crust depth with depth-ratchet.json (mirrors the depth-ratchet step in check-llm)
depth-ratchet:
    uv run python -m scripts.check_depth check

# Report whole-symbol zero-coverage dead code (requires coverage.xml from coverage-check)
dead-code-coverage:
    uv run python scripts/check_dead_code_coverage.py

# Type-check all Python (basedpyright recommended tier; fails on warnings too).
typecheck:
    uv run basedpyright

# Full build no autofix: lint → file-size → dead-code → slice guard → depth ratchet → coverage check → dead-code-coverage → typecheck (for CI validation)
build-ci: lint file-size dead-code slice-guard depth-ratchet coverage-check dead-code-coverage typecheck
    @echo "✅ CI build passed"

# Agent gate: lint + format + dead code + tests + project coverage + diff coverage + typecheck.
# Quiet on success (one line), full output only on failing steps. Non-mutating.
# Independent lanes run concurrently; every failing lane is reported, so one
# failure never suppresses another lane's signal.
# diff-coverage mirrors codecov/patch: it fails if the lines THIS branch changes
# (vs origin/main, including staged/uncommitted edits) aren't covered to threshold.
check-llm:
    #!/usr/bin/env python3
    import subprocess
    import sys
    from concurrent.futures import ThreadPoolExecutor

    threshold = "{{COVERAGE_THRESHOLD}}"
    base = "origin/main"

    # Best-effort refresh of the base ref so diff coverage matches codecov/patch.
    subprocess.run(["git", "fetch", "-q", "origin", "main"], capture_output=True, text=True)

    # Each lane runs its steps in order; lanes run concurrently. Every step is
    # read-only against the tree, and each writes only its own artifacts
    # (coverage.xml is produced and consumed by the serial steps below).
    lanes = [
        [("file-size", ["uv", "run", "python", "scripts/check_file_lengths.py"])],
        [("import-contracts", ["uv", "run", "lint-imports"])],
        [("dead-code", ["uv", "run", "python", "scripts/check_dead_code.py"])],
        [
            ("slice-guard", ["just", "slice-guard"]),
            ("depth-ratchet", ["just", "depth-ratchet"]),
        ],
        [("lint+format", ["just", "lint"])],
        [
            ("web-typecheck", ["npm", "--prefix", "web", "run", "typecheck"]),
            ("web-lint", ["npm", "--prefix", "web", "run", "lint"]),
            ("web-test", ["npm", "--prefix", "web", "run", "test"]),
        ],
        [
            (
                "tests",
                [
                    "uv", "run", "pytest", "tests/", "-q",
                    "-n", "{{TEST_WORKERS}}", "--dist", "worksteal",
                    "-m", "not browser and not real_process",
                    "--cov=src/milknado", "--cov-report=",
                ],
            ),
        ],
        [
            (
                "browser",
                [
                    "uv", "run", "pytest", "tests/browser", "-m", "browser", "-q",
                    "-n", "{{BROWSER_WORKERS}}",
                    # Own cache dir: the coverage lane writes .pytest_cache concurrently.
                    "-o", "cache_dir=.pytest_cache/browser",
                ],
            ),
        ],
        [("typecheck", ["just", "typecheck"])],
    ]

    # Runs alone after the concurrent lanes pass, so real-process tests do not compete
    # with the xdist pool or the other lanes. It extends the coverage data from "tests".
    # REAL_PROCESS_WORKERS is an upper bound: a numeric TEST_WORKERS pin lowers it, and
    # "auto" sizes from free cores through TEST_WORKERS_CEILING (tests/worker_pool.py).
    test_workers = "{{TEST_WORKERS}}"
    real_process_workers = "{{REAL_PROCESS_WORKERS}}"
    if test_workers.isdecimal():
        real_process_workers = str(min(int(test_workers), int(real_process_workers)))
    else:
        real_process_workers = "auto"
    serial = [
        (
            "tests-real-process",
            [
                "env", "TEST_WORKERS_CEILING={{REAL_PROCESS_WORKERS}}",
                "uv", "run", "pytest", "tests/", "-q",
                "-n", real_process_workers, "--dist", "worksteal",
                "-m", "real_process and not browser",
                "--cov=src/milknado", "--cov-append",
                "--cov-report=term-missing",
                "--cov-report=xml:coverage.xml",
                f"--cov-fail-under={threshold}",
            ],
        ),
        (
            "dead-code-coverage",
            ["uv", "run", "python", "scripts/check_dead_code_coverage.py"],
        ),
        (
            "diff-coverage",
            [
                "uv", "run", "diff-cover", "coverage.xml",
                f"--compare-branch={base}",
                f"--fail-under={threshold}",
            ],
        ),
    ]


    def run_lane(lane):
        for name, cmd in lane:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                return name, result
        return None


    with ThreadPoolExecutor(max_workers=len(lanes)) as pool:
        failures = [failure for failure in pool.map(run_lane, lanes) if failure]
    if not failures:
        failures = [failure for failure in [run_lane(serial)] if failure]

    # Report every failing lane in declared order so one failure never hides another.
    for name, result in failures:
        print(f"❌ check:llm FAILED at: {name}\n")
        print((result.stdout + result.stderr).strip() + "\n")
    if failures:
        sys.exit(failures[0][1].returncode)

    print(
        f"✅ check:llm PASS — lint+format clean, no dead code, "
        f"tests green, project+diff coverage ≥{threshold}%, typecheck clean"
    )

# Run the CLI for manual testing
run *args:
    uv run milknado {{args}}

# Launch interactive planning demo with local mock planner (no external agent)
plan-mock keep="0":
    #!/usr/bin/env python3
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path

    keep_workspace = "{{keep}}" == "1"
    repo_root = Path.cwd()
    project_root = Path(tempfile.mkdtemp(prefix="milknado-plan-mock-"))
    (project_root / "spec.md").write_text(
        "# Mock planning goal\n\nUse interactive planning loop with local mock planner.\n",
        encoding="utf-8",
    )
    planner_path = (repo_root / "scripts" / "mock_planner.py").resolve()
    (project_root / "milknado.toml").write_text(
        "\n".join(
            [
                "[milknado]",
                'agent_family = "claude"',
                f'planning_agent = "python {planner_path}"',
                'db_path = ".milknado/milknado.db"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Mock workspace: {project_root}")
    print("Tips: enter 2 to revise, then feedback, then 1 to accept.")
    subprocess.run(
        [
            "uv",
            "run",
            "milknado",
            "plan",
            "--interactive",
            "--spec",
            str(project_root / "spec.md"),
            "--project-root",
            str(project_root),
        ],
        check=False,
    )
    if keep_workspace:
        print(f"Keeping workspace: {project_root}")
    else:
        shutil.rmtree(project_root, ignore_errors=True)

# Run the MCP server
mcp-server:
    uv run python -m milknado.mcp.server

# Run the MCP server under a file watcher that restarts it on src/ changes
mcp-dev:
    uv run watchfiles "uv run milknado-mcp" src/milknado

# Clean build artifacts and caches
clean:
    rm -rf .pytest_cache .ruff_cache __pycache__ htmlcov .coverage
    find . -type d -name __pycache__ -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete

# Format code without linting (subset of lint-fix)
fmt:
    uv run ruff format src/ tests/

# Open HTML coverage report
coverage-html: test-coverage
    @if command -v open &> /dev/null; then open htmlcov/index.html; else echo "htmlcov/index.html ready"; fi

# Audit runtime dependencies for known CVEs
audit:
    uv run --with pip-audit pip-audit
