"""Exit-code contract for scripts/check_dead_code.py: exercises the classifier at its
real CLI seam (subprocess, exit code + stdout), no mocking the classifier itself.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GATE_SCRIPT = REPO_ROOT / "scripts" / "check_dead_code.py"


def _run_gate(*paths: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(GATE_SCRIPT), *paths],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_clean_tree_exits_zero() -> None:
    result = _run_gate("src")
    assert result.returncode == 0, result.stdout + result.stderr


def test_unclassified_dead_function_fails_the_gate(tmp_path: Path) -> None:
    probe = tmp_path / "probe.py"
    probe.write_text("def unused_probe():\n    return 1\n", encoding="utf-8")
    result = _run_gate(str(tmp_path))
    output = result.stdout + result.stderr
    assert result.returncode == 3
    assert "unused_probe" in output


def test_textual_lifecycle_override_is_classified_accepted(tmp_path: Path) -> None:
    probe = tmp_path / "probe.py"
    probe.write_text(
        textwrap.dedent(
            """\
            from textual.app import App, ComposeResult

            class ProbeApp(App):
                CSS = ""
                BINDINGS = []

                def compose(self) -> ComposeResult:
                    yield from ()

                def on_mount(self) -> None:
                    self.sub_title = "probe"

            ProbeApp()
            """
        ),
        encoding="utf-8",
    )
    result = _run_gate(str(tmp_path))
    output = result.stdout + result.stderr
    assert result.returncode == 0, output


def test_same_named_method_without_textual_base_is_not_classified(tmp_path: Path) -> None:
    probe = tmp_path / "probe.py"
    probe.write_text(
        textwrap.dedent(
            """\
            class NotTextual:
                def compose(self) -> None:
                    pass
            """
        ),
        encoding="utf-8",
    )
    result = _run_gate(str(tmp_path))
    output = result.stdout + result.stderr
    assert result.returncode == 3
    assert "compose" in output


def _defined_names(path: Path) -> set[str]:
    """Every function/method qualname (dotted through class nesting) and bare
    name definable at `path`, matching how allowlist entries and vendored-loop
    findings each name a symbol (dotted qualname vs. bare name).
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    stack: list[str] = []

    def _record(name: str) -> None:
        names.add(name)
        names.add(".".join([*stack, name]))

    class _Collector(ast.NodeVisitor):
        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            _record(node.name)
            stack.append(node.name)
            self.generic_visit(node)
            stack.pop()

        def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
            _record(node.name)
            stack.append(node.name)
            self.generic_visit(node)
            stack.pop()

        visit_FunctionDef = _visit_function
        visit_AsyncFunctionDef = _visit_function

        def visit_Assign(self, node: ast.Assign) -> None:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    _record(target.id)
            self.generic_visit(node)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            if isinstance(node.target, ast.Name):
                _record(node.target.id)
            self.generic_visit(node)

    _Collector().visit(tree)
    return names


def _module_to_path(module: str) -> Path:
    base = REPO_ROOT / "src" / Path(*module.split("."))
    file_path = base.with_suffix(".py")
    if file_path.exists():
        return file_path
    init_path = base / "__init__.py"
    assert init_path.exists(), f"module {module!r} has no source file"
    return init_path


def _vendored_loop_findings() -> set[tuple[str, str]]:
    """AST-extract _VENDORED_LOOP_FINDINGS without importing the gate script."""
    tree = ast.parse(GATE_SCRIPT.read_text(encoding="utf-8"), filename=str(GATE_SCRIPT))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_VENDORED_LOOP_FINDINGS"
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("_VENDORED_LOOP_FINDINGS not found in check_dead_code.py")


def test_dead_code_allow_entries_resolve_to_real_symbols() -> None:
    """A stale [tool.milknado.dead-code].allow entry (symbol renamed or deleted)
    would silently stop exempting anything -- catch it going stale instead.
    """
    with (REPO_ROOT / "pyproject.toml").open("rb") as stream:
        config = tomllib.load(stream)
    allow = config["tool"]["milknado"]["dead-code"]["allow"]
    assert allow, "allowlist extraction regression -- would pass vacuously"
    for entry in allow:
        module, qualname = entry.split(":", 1)
        path = _module_to_path(module)
        assert qualname in _defined_names(path), f"{entry}: {qualname!r} not found in {path}"


def test_vendored_loop_findings_resolve_to_real_symbols() -> None:
    """A stale _VENDORED_LOOP_FINDINGS entry would silently stop exempting the
    vendored engine's own unused surface -- catch it going stale instead.
    """
    findings = _vendored_loop_findings()
    assert findings, "_VENDORED_LOOP_FINDINGS extraction regression -- would pass vacuously"
    for relpath, name in findings:
        path = REPO_ROOT / relpath
        assert name in _defined_names(path), f"{relpath}:{name} not found"
