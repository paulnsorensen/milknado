"""US-211 FR-3: every source file stays under 300 lines; no circular imports."""

from __future__ import annotations

import importlib
from pathlib import Path

SRC_ROOT = Path(__file__).parent.parent / "src" / "milknado"
LINE_BUDGET = 300

_EXPLICIT = {
    "cli.py": SRC_ROOT / "cli.py",
    "run_loop/__init__.py": SRC_ROOT / "domains" / "execution" / "run_loop" / "__init__.py",
    "run_loop/input.py": SRC_ROOT / "domains" / "execution" / "run_loop" / "input.py",
    "run_loop/display.py": SRC_ROOT / "domains" / "execution" / "run_loop" / "display.py",
}


def _line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def test_cli_py_under_budget() -> None:
    path = _EXPLICIT["cli.py"]
    count = _line_count(path)
    assert count <= LINE_BUDGET, f"cli.py has {count} lines (budget {LINE_BUDGET})"


def test_run_loop_init_under_budget() -> None:
    path = _EXPLICIT["run_loop/__init__.py"]
    count = _line_count(path)
    assert count <= LINE_BUDGET, f"run_loop/__init__.py has {count} lines (budget {LINE_BUDGET})"


def test_run_loop_input_under_budget() -> None:
    path = _EXPLICIT["run_loop/input.py"]
    count = _line_count(path)
    assert count <= LINE_BUDGET, f"run_loop/input.py has {count} lines (budget {LINE_BUDGET})"


def test_run_loop_display_under_budget() -> None:
    path = _EXPLICIT["run_loop/display.py"]
    count = _line_count(path)
    assert count <= LINE_BUDGET, f"run_loop/display.py has {count} lines (budget {LINE_BUDGET})"


def test_all_source_files_under_budget() -> None:
    violations: list[str] = []
    for py_file in sorted(SRC_ROOT.rglob("*.py")):
        count = _line_count(py_file)
        if count > LINE_BUDGET:
            rel = py_file.relative_to(SRC_ROOT)
            violations.append(f"{rel}: {count} lines")
    assert not violations, "Files over {}-line budget:\n  {}".format(
        LINE_BUDGET, "\n  ".join(violations)
    )


def test_no_circular_import_cli_app() -> None:
    mod = importlib.import_module("milknado.cli")
    assert hasattr(mod, "app"), "milknado.cli must export 'app'"


def test_no_circular_import_run_loop() -> None:
    mod = importlib.import_module("milknado.domains.execution")
    assert hasattr(mod, "RunLoop"), "milknado.domains.execution must export 'RunLoop'"
