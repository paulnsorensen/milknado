set dotenv-load := true

# Project name and Python version
PROJECT := "milknado"
PYTHON := "3.11"
COVERAGE_THRESHOLD := "90"

# Show all available recipes
default:
    @just --list

# Install dependencies using uv
install:
    uv sync

# Run linters with autofix (ruff)
lint:
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
    result = subprocess.run(
        ["uv", "run", "pytest", "tests/", "--cov=src/milknado", "--cov-report=term"],
        capture_output=True,
        text=True
    )
    output = result.stdout + result.stderr
    print(output)

    # Extract coverage percentage from output
    import re
    match = re.search(r'TOTAL.*?(\d+)%', output)
    if match:
        coverage = int(match.group(1))
        threshold = int("{{COVERAGE_THRESHOLD}}")
        if coverage < threshold:
            print(f"❌ Coverage {coverage}% is below threshold {threshold}%")
            exit(1)
        else:
            print(f"✅ Coverage {coverage}% meets threshold {threshold}%")
    else:
        print("⚠️  Could not parse coverage from pytest output")
        exit(1)

# Full build: lint → test → coverage check (must pass before PR)
build: lint test coverage-check
    @echo "✅ Build passed — ready for PR"

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

# Format code without linting (subset of lint)
fmt:
    uv run ruff format src/ tests/

# Open HTML coverage report
coverage-html: test-coverage
    @if command -v open &> /dev/null; then open htmlcov/index.html; else echo "htmlcov/index.html ready"; fi

# Watch mode: run tests on file changes (requires watchfiles)
watch *args:
    uv run pytest-watch tests/ {{args}}
