set dotenv-load := true

COVERAGE_THRESHOLD := "90"

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

# Run the CLI for manual testing
run *args:
    uv run milknado {{args}}

# Run the MCP server
mcp-server:
    uv run python -m milknado.mcp_server

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
