#!/usr/bin/env python3
"""Coverage-based whole-symbol dead-code gate.

Cross-checks coverage.xml against an AST pass: a function or method whose every
measured line has zero hits is whole-symbol dead code that vulture's name-based
matching misses (name-collision blindness, dead reference cycles). See
.hallouminate/wiki/history/dead-code-coverage-gate-decision.md.

Known limitations:
- A function whose entire body is on the ``def`` line (e.g.
  ``def f(): return x``) cannot be flagged. coverage.py records only the def
  line as hit (at import), and the body shares that line, so the measured span
  never shows zero hits even when the body never runs.
- A function with zero measured lines (a docstring-only body, or a Protocol
  stub body of just ``...``) is treated as live, not dead — coverage.py never
  instruments a line for it, so there is no hit/no-hit signal to check.
"""

from __future__ import annotations

import ast
import fnmatch
import sys
import tomllib
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

COVERAGE_XML = Path("coverage.xml")
PYPROJECT_TOML = Path("pyproject.toml")


@dataclass(frozen=True)
class DeadSymbol:
    module: str
    qualname: str
    lineno: int


@dataclass(frozen=True)
class Exemptions:
    ignore_decorators: tuple[str, ...]
    ignore_names: tuple[str, ...]
    script_targets: tuple[str, ...]
    allowlist: tuple[str, ...]

    def is_exempt(self, module_qualname: str, decorators: list[str]) -> bool:
        if module_qualname in self.allowlist or module_qualname in self.script_targets:
            return True
        return any(
            fnmatch.fnmatch(decorator, pattern)
            for decorator in decorators
            for pattern in self.ignore_decorators
        )


def _decorator_name(decorator: ast.expr) -> ast.expr:
    """Unwrap a call decorator to its callee, matching vulture's normalization."""
    return decorator.func if isinstance(decorator, ast.Call) else decorator


class _FunctionCollector(ast.NodeVisitor):
    """Walks a module tree, tracking class nesting to build dotted qualnames."""

    def __init__(self) -> None:
        self.functions: list[tuple[str, ast.AST, list[str]]] = []
        self._stack: list[str] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._stack.append(node.name)
        self.generic_visit(node)
        self._stack.pop()

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        qualname = ".".join([*self._stack, node.name])
        decorators = ["@" + ast.unparse(_decorator_name(d)) for d in node.decorator_list]
        self.functions.append((qualname, node, decorators))
        self._stack.append(node.name)
        self.generic_visit(node)
        self._stack.pop()

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function


def _load_exemptions(pyproject: Path) -> Exemptions:
    with pyproject.open("rb") as stream:
        config = tomllib.load(stream)
    vulture = config.get("tool", {}).get("vulture", {})
    scripts = config.get("project", {}).get("scripts", {})
    dead_code = config.get("tool", {}).get("milknado", {}).get("dead-code", {})
    return Exemptions(
        ignore_decorators=tuple(vulture.get("ignore_decorators", [])),
        ignore_names=tuple(vulture.get("ignore_names", [])),
        script_targets=tuple(scripts.values()),
        allowlist=tuple(dead_code.get("allow", [])),
    )


def _resolve_source_path(filename: str, sources: list[str], cwd: Path) -> Path:
    for source in sources:
        candidate = Path(source) / filename
        if not candidate.is_absolute():
            candidate = cwd / candidate
        if candidate.exists():
            return candidate.resolve()
    return (cwd / filename).resolve()


def _parse_coverage_xml(coverage_xml: Path, cwd: Path) -> dict[Path, dict[int, int]]:
    root = ET.parse(coverage_xml).getroot()
    sources = [elem.text.strip() for elem in root.findall("./sources/source") if elem.text]
    line_hits_by_file: dict[Path, dict[int, int]] = {}
    for class_elem in root.iter("class"):
        filename = class_elem.get("filename")
        if not filename:
            continue
        resolved = _resolve_source_path(filename, sources, cwd)
        line_hits = {
            int(line.get("number")): int(line.get("hits", "0"))
            for line in class_elem.findall("./lines/line")
        }
        line_hits_by_file.setdefault(resolved, {}).update(line_hits)
    return line_hits_by_file


def _module_name(path: Path, cwd: Path) -> str:
    relative = path.resolve().relative_to(cwd.resolve())
    parts = list(relative.with_suffix("").parts)
    if parts and parts[0] == "src":
        parts = parts[1:]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def find_whole_dead_symbols(coverage_xml: Path, cwd: Path, exempt: Exemptions) -> list[DeadSymbol]:
    dead_symbols: list[DeadSymbol] = []
    for resolved_path, line_hits in _parse_coverage_xml(coverage_xml, cwd).items():
        module = _module_name(resolved_path, cwd)
        tree = ast.parse(resolved_path.read_text(encoding="utf-8"), filename=str(resolved_path))
        collector = _FunctionCollector()
        collector.visit(tree)
        for qualname, node, decorators in collector.functions:
            body_start = node.body[0].lineno
            measured = {
                lineno: hits
                for lineno, hits in line_hits.items()
                if body_start <= lineno <= node.end_lineno
            }
            if not measured or any(hits > 0 for hits in measured.values()):
                continue
            module_qualname = f"{module}:{qualname}"
            if exempt.is_exempt(module_qualname, decorators):
                continue
            dead_symbols.append(DeadSymbol(module=module, qualname=qualname, lineno=node.lineno))
    return dead_symbols


def main() -> int:
    cwd = Path.cwd()
    exempt = _load_exemptions(cwd / PYPROJECT_TOML)
    try:
        dead_symbols = find_whole_dead_symbols(cwd / COVERAGE_XML, cwd, exempt)
    except (ET.ParseError, OSError):
        print(
            "coverage.xml is malformed, empty, or references a missing source file — "
            "run the tests+coverage step first",
            file=sys.stderr,
        )
        return 1
    if dead_symbols:
        for symbol in dead_symbols:
            print(
                f"{symbol.module}:{symbol.qualname} (line {symbol.lineno}) is whole-symbol "
                "dead code (zero coverage hits, unexempt). Test it, delete it, mark it "
                "'# pragma: no cover', or add it to [tool.milknado.dead-code].allow."
            )
        return 1
    print("dead-code-coverage: no whole-symbol zero-hit code found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
