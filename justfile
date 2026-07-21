set dotenv-load := true

# Matches codecov.yml project/patch target (95%)
COVERAGE_THRESHOLD := "95"

# Show all available recipes
default:
    @just --list

# Install dependencies using uv
install:
    uv sync

# Run linters without modifying files (for CI/build validation)
lint:
    uv run ruff check src/ tests/ --preview
    uv run ruff format --check src/ tests/

# Run linters with autofix
lint-fix:
    uv run ruff check src/ tests/ --fix --preview
    uv run ruff format src/ tests/

# Run the test suite
test *args:
    uv run pytest tests/ {{args}}

# Run individual test file
test-file file *args:
    uv run pytest {{file}} {{args}}

# Run tests with verbose output
test-verbose *args:
    uv run pytest tests/ -vv {{args}}

# Run tests with coverage report
test-coverage:
    uv run pytest tests/ --cov=src/milknado --cov-report=term-missing --cov-report=html

# Check coverage meets threshold
coverage-check:
    #!/usr/bin/env python3
    import subprocess
    import sys

    result = subprocess.run(
        [
            "uv", "run", "pytest", "tests/",
            "--cov=src/milknado",
            "--cov-report=term",
            "--cov-report=xml:coverage.xml",
            "--cov-fail-under={{COVERAGE_THRESHOLD}}",
        ],
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr
    print(output, end="")

    if result.returncode != 0:
        sys.exit(result.returncode)

# Full build with autofix: lint-fix → test → coverage check (for agents/developers)
build: lint-fix test coverage-check
    @echo "✅ Build passed — ready for PR"

# Full build no autofix: lint → test → coverage check (for CI validation)
build-ci: lint test coverage-check
    @echo "✅ CI build passed"

# Agent gate: lint + format + tests + project coverage + diff coverage in one shot.
# Quiet on success (one line), full output only on the failing step. Non-mutating.
# diff-coverage mirrors codecov/patch: it fails if the lines THIS branch changes
# (vs origin/main, including staged/uncommitted edits) aren't covered to threshold.
check-llm:
    #!/usr/bin/env python3
    import subprocess
    import sys

    threshold = "{{COVERAGE_THRESHOLD}}"
    base = "origin/main"

    # Best-effort refresh of the base ref so diff coverage matches codecov/patch.
    subprocess.run(["git", "fetch", "-q", "origin", "main"], capture_output=True, text=True)

    steps = [
        ("lint", ["uv", "run", "ruff", "check", "src/", "tests/", "--preview"]),
        ("format", ["uv", "run", "ruff", "format", "--check", "src/", "tests/"]),
        (
            "tests+coverage",
            [
                "uv", "run", "pytest", "tests/", "-q",
                "--cov=src/milknado",
                "--cov-report=term-missing",
                "--cov-report=xml:coverage.xml",
                f"--cov-fail-under={threshold}",
            ],
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

    for name, cmd in steps:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ check:llm FAILED at: {name}\n")
            print((result.stdout + result.stderr).strip())
            sys.exit(result.returncode)

    print(f"✅ check:llm PASS — lint+format clean, tests green, project+diff coverage ≥{threshold}%")

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
