"""Adversarial tests-only hardening for scripts/check_dead_code_coverage.py.

Attacks the gate at its CLI seam (subprocess, exit code + stdout) with cases the
hand-written oracle fixtures in tests/dead_code_coverage_cut/ do not cover:
realistic coverage.py output (regression guard for the def-line-hits-at-import
no-op), call-syntax decorator exemptions (regression guard for @mcp.tool()),
AST edge cases, Cobertura <sources> resolution, and fail-loud behavior on
malformed input.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
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


def _coverage_xml(filename: str, lines_with_hits: list[tuple[str, int | None]]) -> str:
    """Build a minimal Cobertura fragment where hits align to source line by index."""
    line_elems = "\n".join(
        f'            <line number="{i}" hits="{hits}"/>'
        for i, (_text, hits) in enumerate(lines_with_hits, start=1)
        if hits is not None
    )
    return f"""<?xml version="1.0" ?>
<coverage>
  <packages>
    <package name=".">
      <classes>
        <class name="{filename}" filename="{filename}">
          <lines>
{line_elems}
          </lines>
        </class>
      </classes>
    </package>
  </packages>
</coverage>
"""


def _source(lines_with_hits: list[tuple[str, int | None]]) -> str:
    return "\n".join(text for text, _hits in lines_with_hits) + "\n"


def _write_lines_fixture(
    tmp_path: Path, filename: str, lines_with_hits: list[tuple[str, int | None]], pyproject: str
) -> Path:
    return _write_fixture(
        tmp_path,
        {
            "coverage.xml": _coverage_xml(filename, lines_with_hits),
            "pyproject.toml": pyproject,
            filename: _source(lines_with_hits),
        },
    )


# --- Regression guard: realistic coverage.py output (defect 1) --------------


def test_realistic_def_line_hit_body_zero_is_flagged_dead(tmp_path: Path) -> None:
    """coverage.py hits the ``def`` line at import time even when the body never
    runs; the gate must key off the body span, not the def line, or it no-ops.
    """
    lines: list[tuple[str, int | None]] = [
        ("def dead_fn():", 1),
        ("    x = 1", 0),
        ("    return x", 0),
        ("", None),
        ("def live_fn():", 1),
        ("    y = 2", 1),
        ("    return y", 1),
    ]
    fixture_dir = _write_lines_fixture(tmp_path, "realistic.py", lines, _EMPTY_DEAD_CODE_CONFIG)

    result = _run_gate(fixture_dir)

    assert result.returncode == 1
    assert "dead_fn" in result.stdout
    assert "live_fn" not in result.stdout


# --- Regression guard: call-syntax decorator exemptions (defect 2) ----------


def test_call_syntax_decorators_are_exempt_bare_decorators_too(tmp_path: Path) -> None:
    pyproject = """
[tool.vulture]
ignore_decorators = ["@*.command", "@mcp.tool", "@on", "@work"]
ignore_names = []

[tool.milknado.dead-code]
allow = []
"""
    lines: list[tuple[str, int | None]] = [
        ("@mcp.tool()", 1),
        ("def dead_tool():", 1),
        ("    x = 1", 0),
        ("", None),
        ('@app.command("run")', 1),
        ("def dead_command():", 1),
        ("    y = 1", 0),
        ("", None),
        ("@on", 1),
        ("def dead_on():", 1),
        ("    pass", 0),
        ("", None),
        ("@work", 1),
        ("def dead_work():", 1),
        ("    pass", 0),
        ("", None),
        ("def dead_plain():", 1),
        ("    pass", 0),
    ]
    fixture_dir = _write_lines_fixture(tmp_path, "decorated.py", lines, pyproject)

    result = _run_gate(fixture_dir)

    assert result.returncode == 1
    assert "dead_plain" in result.stdout
    for exempt_name in ("dead_tool", "dead_command", "dead_on", "dead_work"):
        assert exempt_name not in result.stdout


# --- [project.scripts] and [tool.milknado.dead-code].allow exemptions -------


def test_project_scripts_and_allowlist_exempt_zero_hit_symbols(tmp_path: Path) -> None:
    pyproject = """
[project.scripts]
entry-tool = "entrypoints:cli_main"

[tool.vulture]
ignore_decorators = []
ignore_names = []

[tool.milknado.dead-code]
allow = ["entrypoints:reviewed_dead"]
"""
    lines: list[tuple[str, int | None]] = [
        ("def cli_main():", 1),
        ("    x = 1", 0),
        ("", None),
        ("def reviewed_dead():", 1),
        ("    y = 1", 0),
        ("", None),
        ("def truly_dead():", 1),
        ("    z = 1", 0),
    ]
    fixture_dir = _write_lines_fixture(tmp_path, "entrypoints.py", lines, pyproject)

    result = _run_gate(fixture_dir)

    assert result.returncode == 1
    assert "truly_dead" in result.stdout
    assert "cli_main" not in result.stdout
    assert "reviewed_dead" not in result.stdout


# --- ignore_names globs ------------------------------------------------------


def test_ignore_names_globs_exempt_matching_bare_names(tmp_path: Path) -> None:
    pyproject = """
[tool.vulture]
ignore_decorators = []
ignore_names = ["on_*", "action_*"]

[tool.milknado.dead-code]
allow = []
"""
    lines: list[tuple[str, int | None]] = [
        ("def on_click():", 1),
        ("    x = 1", 0),
        ("", None),
        ("def action_quit():", 1),
        ("    y = 1", 0),
        ("", None),
        ("def not_exempt_dead():", 1),
        ("    z = 1", 0),
    ]
    fixture_dir = _write_lines_fixture(tmp_path, "widgets.py", lines, pyproject)

    result = _run_gate(fixture_dir)

    assert result.returncode == 1
    assert "not_exempt_dead" in result.stdout
    assert "on_click" not in result.stdout
    assert "action_quit" not in result.stdout


# --- AST edge cases: body-only span computation ------------------------------


def test_multiline_signature_body_span_is_exact(tmp_path: Path) -> None:
    lines: list[tuple[str, int | None]] = [
        ("def multiline_dead(", 1),
        ("    a,", None),
        ("    b,", None),
        ("):", None),
        ("    x = a + b", 0),
        ("    return x", 0),
        ("", None),
        ("def multiline_live(", 1),
        ("    a,", None),
        ("    b,", None),
        ("):", None),
        ("    x = a + b", 1),
        ("    return x", 1),
    ]
    fixture_dir = _write_lines_fixture(tmp_path, "multiline.py", lines, _EMPTY_DEAD_CODE_CONFIG)

    result = _run_gate(fixture_dir)

    assert result.returncode == 1
    assert "multiline_dead" in result.stdout
    assert "multiline_live" not in result.stdout


def test_decorated_function_not_in_exemption_list_is_still_flagged(tmp_path: Path) -> None:
    lines: list[tuple[str, int | None]] = [
        ("@some_decorator", 1),
        ("def decorated_dead():", 1),
        ("    return 1", 0),
        ("", None),
        ("@some_decorator", 1),
        ("def decorated_live():", 1),
        ("    return 2", 1),
    ]
    fixture_dir = _write_lines_fixture(tmp_path, "undecorated.py", lines, _EMPTY_DEAD_CODE_CONFIG)

    result = _run_gate(fixture_dir)

    assert result.returncode == 1
    assert "decorated_dead" in result.stdout
    assert "decorated_live" not in result.stdout


def test_nested_closure_dead_function_is_flagged_with_dotted_qualname(tmp_path: Path) -> None:
    lines: list[tuple[str, int | None]] = [
        ("def outer_live():", 1),
        ("    def inner_dead():", 1),
        ("        return 1", 0),
        ("    return inner_dead", 1),
    ]
    fixture_dir = _write_lines_fixture(tmp_path, "nested.py", lines, _EMPTY_DEAD_CODE_CONFIG)

    result = _run_gate(fixture_dir)

    assert result.returncode == 1
    assert "outer_live.inner_dead" in result.stdout
    assert "outer_live (line" not in result.stdout


def test_class_methods_async_staticmethod_classmethod_property(tmp_path: Path) -> None:
    lines: list[tuple[str, int | None]] = [
        ("class Foo:", None),
        ("    def method_dead(self):", 1),
        ("        return 1", 0),
        ("", None),
        ("    def method_live(self):", 1),
        ("        return 2", 1),
        ("", None),
        ("    async def amethod_dead(self):", 1),
        ("        return 1", 0),
        ("", None),
        ("    async def amethod_live(self):", 1),
        ("        return 2", 1),
        ("", None),
        ("    @staticmethod", 1),
        ("    def static_dead():", 1),
        ("        return 1", 0),
        ("", None),
        ("    @staticmethod", 1),
        ("    def static_live():", 1),
        ("        return 2", 1),
        ("", None),
        ("    @classmethod", 1),
        ("    def class_dead(cls):", 1),
        ("        return 1", 0),
        ("", None),
        ("    @classmethod", 1),
        ("    def class_live(cls):", 1),
        ("        return 2", 1),
        ("", None),
        ("    @property", 1),
        ("    def prop_dead(self):", 1),
        ("        return 1", 0),
        ("", None),
        ("    @property", 1),
        ("    def prop_live(self):", 1),
        ("        return 2", 1),
    ]
    fixture_dir = _write_lines_fixture(tmp_path, "methods.py", lines, _EMPTY_DEAD_CODE_CONFIG)

    result = _run_gate(fixture_dir)

    assert result.returncode == 1
    for dead_name in (
        "Foo.method_dead",
        "Foo.amethod_dead",
        "Foo.static_dead",
        "Foo.class_dead",
        "Foo.prop_dead",
    ):
        assert dead_name in result.stdout
    for live_name in (
        "Foo.method_live",
        "Foo.amethod_live",
        "Foo.static_live",
        "Foo.class_live",
        "Foo.prop_live",
    ):
        assert live_name not in result.stdout


# --- Cobertura <sources> resolution ------------------------------------------


def test_relative_filename_resolves_under_absolute_sources_dir(tmp_path: Path) -> None:
    pkg_dir = tmp_path / "pkg"
    pkg_dir.mkdir()
    lines: list[tuple[str, int | None]] = [
        ("def dead_fn():", 1),
        ("    return 1", 0),
    ]
    (pkg_dir / "mod.py").write_text(_source(lines))
    coverage_xml = f"""<?xml version="1.0" ?>
<coverage>
  <sources>
    <source>{pkg_dir}</source>
  </sources>
  <packages>
    <package name=".">
      <classes>
        <class name="mod.py" filename="mod.py">
          <lines>
            <line number="1" hits="1"/>
            <line number="2" hits="0"/>
          </lines>
        </class>
      </classes>
    </package>
  </packages>
</coverage>
"""
    fixture_dir = _write_fixture(
        tmp_path, {"coverage.xml": coverage_xml, "pyproject.toml": _EMPTY_DEAD_CODE_CONFIG}
    )

    result = _run_gate(fixture_dir)

    assert result.returncode == 1
    assert "dead_fn" in result.stdout


# --- Fail-loud on malformed input --------------------------------------------


def test_malformed_coverage_xml_fails_loud(tmp_path: Path) -> None:
    fixture_dir = _write_fixture(
        tmp_path,
        {"coverage.xml": "this is not xml", "pyproject.toml": _EMPTY_DEAD_CODE_CONFIG},
    )

    result = _run_gate(fixture_dir)

    assert result.returncode != 0
    assert "dead-code-coverage: no whole-symbol" not in result.stdout
    assert (
        "coverage.xml is malformed or empty — run the tests+coverage step first" in result.stderr
    )


def test_empty_coverage_xml_fails_loud(tmp_path: Path) -> None:
    fixture_dir = _write_fixture(
        tmp_path, {"coverage.xml": "", "pyproject.toml": _EMPTY_DEAD_CODE_CONFIG}
    )

    result = _run_gate(fixture_dir)

    assert result.returncode != 0


def test_coverage_xml_naming_missing_source_file_fails_loud(tmp_path: Path) -> None:
    coverage_xml = """<?xml version="1.0" ?>
<coverage>
  <packages>
    <package name=".">
      <classes>
        <class name="missing.py" filename="missing.py">
          <lines>
            <line number="1" hits="0"/>
          </lines>
        </class>
      </classes>
    </package>
  </packages>
</coverage>
"""
    fixture_dir = _write_fixture(
        tmp_path, {"coverage.xml": coverage_xml, "pyproject.toml": _EMPTY_DEAD_CODE_CONFIG}
    )

    result = _run_gate(fixture_dir)

    assert result.returncode != 0
