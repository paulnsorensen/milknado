"""RED oracle for the dead-code-coverage-gate spec (AC-1..AC-4).

Each case invokes the not-yet-existing ``scripts/check_dead_code_coverage.py``
as a subprocess against a fixture directory (coverage.xml + pyproject.toml +
source files) and asserts on exit code / stdout. Pre-implementation the
script is absent, so every subprocess call fails and each assertion below
fails red for its own reason.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE_SCRIPT = REPO_ROOT / "scripts" / "check_dead_code_coverage.py"

_EMPTY_DEAD_CODE_CONFIG = """
[tool.vulture]
ignore_decorators = []
ignore_names = []

[tool.milknado.dead-code]
allow = []
"""


def _write_fixture(tmp_path: Path, files: dict[str, str]) -> Path:
    for relpath, content in files.items():
        path = tmp_path / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return tmp_path


def _run_gate(fixture_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(GATE_SCRIPT)],
        cwd=fixture_dir,
        capture_output=True,
        text=True,
        check=False,
    )


def test_ac1_whole_dead_symbol_is_named_and_fails(tmp_path: Path) -> None:
    """AC-1: every measured line of dead_fn is zero-hit and unexempt."""
    coverage_xml = """<?xml version="1.0" ?>
<coverage>
  <packages>
    <package name=".">
      <classes>
        <class name="foo.py" filename="foo.py">
          <lines>
            <line number="1" hits="0"/>
            <line number="2" hits="0"/>
            <line number="3" hits="0"/>
          </lines>
        </class>
      </classes>
    </package>
  </packages>
</coverage>
"""
    fixture_dir = _write_fixture(
        tmp_path,
        {
            "coverage.xml": coverage_xml,
            "pyproject.toml": _EMPTY_DEAD_CODE_CONFIG,
            "foo.py": "def dead_fn():\n    x = 1\n    return x\n",
        },
    )

    result = _run_gate(fixture_dir)

    assert result.returncode == 1 and "dead_fn" in result.stdout, (
        "Fixture coverage.xml marks every measured line of foo:dead_fn as zero hits and "
        "it is unexempt; pre-implementation the whole-symbol detector is absent so "
        "dead_fn is never named and the test asserting exit 1 with dead_fn fails red"
    )


def test_ac2_exempt_zero_hit_symbols_are_not_flagged(tmp_path: Path) -> None:
    """AC-2: a project.scripts target and an allowlisted symbol are both zero-hit."""
    coverage_xml = """<?xml version="1.0" ?>
<coverage>
  <packages>
    <package name=".">
      <classes>
        <class name="entrypoints.py" filename="entrypoints.py">
          <lines>
            <line number="1" hits="0"/>
            <line number="2" hits="0"/>
            <line number="5" hits="0"/>
            <line number="6" hits="0"/>
          </lines>
        </class>
      </classes>
    </package>
  </packages>
</coverage>
"""
    pyproject = """
[project.scripts]
entry-tool = "entrypoints:cli_main"

[tool.vulture]
ignore_decorators = []
ignore_names = []

[tool.milknado.dead-code]
allow = ["entrypoints:reviewed_dead"]
"""
    fixture_dir = _write_fixture(
        tmp_path,
        {
            "coverage.xml": coverage_xml,
            "pyproject.toml": pyproject,
            "entrypoints.py": (
                "def cli_main():\n"
                "    x = 1\n"
                "    return x\n"
                "\n\n"
                "def reviewed_dead():\n"
                "    y = 2\n"
                "    return y\n"
            ),
        },
    )

    result = _run_gate(fixture_dir)

    assert (
        result.returncode == 0
        and "cli_main" not in result.stdout
        and "reviewed_dead" not in result.stdout
    ), (
        "Fixture has a zero-hit project-scripts entry point plus one allowlist symbol; "
        "pre-implementation no exemption resolver exists so the test asserting neither "
        "whole symbol is flagged fails red"
    )


def test_ac3_partial_hits_and_pragma_excluded_are_not_flagged(tmp_path: Path) -> None:
    """AC-3: a partially-covered function and a fully pragma-excluded function."""
    coverage_xml = """<?xml version="1.0" ?>
<coverage>
  <packages>
    <package name=".">
      <classes>
        <class name="partial.py" filename="partial.py">
          <lines>
            <line number="1" hits="1"/>
            <line number="2" hits="1"/>
            <line number="3" hits="1"/>
            <line number="4" hits="1"/>
            <line number="5" hits="0"/>
          </lines>
        </class>
      </classes>
    </package>
  </packages>
</coverage>
"""
    fixture_dir = _write_fixture(
        tmp_path,
        {
            "coverage.xml": coverage_xml,
            "pyproject.toml": _EMPTY_DEAD_CODE_CONFIG,
            "partial.py": (
                "def partial_fn():\n"
                "    x = 1\n"
                "    if x:\n"
                "        return x\n"
                "    return None\n"
                "\n\n"
                "def excluded_fn():  # pragma: no cover\n"
                "    return 1\n"
            ),
        },
    )

    result = _run_gate(fixture_dir)

    assert (
        result.returncode == 0
        and "partial_fn" not in result.stdout
        and "excluded_fn" not in result.stdout
    ), (
        "Fixture pairs a partially-covered function (one line with hits) with a fully "
        "pragma-excluded function; pre-implementation no hit filter exists so the test "
        "asserting neither symbol is flagged fails red"
    )


def test_ac4_no_dead_symbol_exits_zero_with_one_success_line(tmp_path: Path) -> None:
    """AC-4: no unexempt zero-hit symbol exists."""
    coverage_xml = """<?xml version="1.0" ?>
<coverage>
  <packages>
    <package name=".">
      <classes>
        <class name="clean.py" filename="clean.py">
          <lines>
            <line number="1" hits="1"/>
            <line number="2" hits="1"/>
          </lines>
        </class>
      </classes>
    </package>
  </packages>
</coverage>
"""
    fixture_dir = _write_fixture(
        tmp_path,
        {
            "coverage.xml": coverage_xml,
            "pyproject.toml": _EMPTY_DEAD_CODE_CONFIG,
            "clean.py": "def clean_fn():\n    return 1\n",
        },
    )

    result = _run_gate(fixture_dir)

    stdout_lines = result.stdout.strip().splitlines()
    assert result.returncode == 0 and len(stdout_lines) == 1 and stdout_lines[0] != "", (
        "Fixture coverage.xml has no unexempt zero-hit symbol; pre-implementation the "
        "step is absent so the test asserting exit zero and the success line fails red"
    )
