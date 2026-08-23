"""Every LOCALLY-DEFINED barrel export must have a real use somewhere in the project.

Catches the wildcard-blindness class vulture misses: a name defined in a package's src,
listed in a barrel `__all__`, but never constructed, annotated, or otherwise referenced
anywhere in src or tests outside its own defining file. Combined-path vulture misses this
because `__all__` membership itself counts as a "use".

Scope choices:
- Use-set spans src AND tests: a test is a legitimate caller, so a name exercised only by
  tests is still live here. (Catching names dead in prod but alive only in tests is the
  separate src-only `vulture src` gate's job.)
- Only names DEFINED in the scanned src are candidates. A barrel export merely re-exported
  from elsewhere is shared vocabulary owned there; its liveness is that module's concern.
- Protocol (port) types are exempt: a port is public API by nature (implemented
  structurally, depended on by consumers) and need not be referenced internally.
- A use is only counted if it occurs outside the module that defines the name: a reference
  from within the defining file itself (e.g. one sibling helper calling another) is not
  evidence that the barrel export is consumed anywhere, so it does not count.
- Barrels in scope: src/milknado/domains/*/__init__.py (the 9 crusts), src/milknado/
  adapters/__init__.py, src/milknado/plugins/__init__.py. cli/__init__.py is out of scope
  as the CLI is milknado's entry-point surface, not an internal barrel; the mcp packages
  are out of scope because they declare no `__all__`. src/milknado/plugins/scaffold.py is
  excluded from all indices because it holds a plugin-scaffolding template string, not
  real code to trace liveness through.
- Assumption: milknado is a leaf app, not a library published for out-of-project
  consumers. A non-Protocol export that exists solely for external consumers (unused
  anywhere in this project's own src/tests) would be flagged dead here; that's accepted
  scope for a leaf app and is not an exemption this gate makes.
"""

from __future__ import annotations

import ast
import importlib
from collections.abc import Iterator
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parents[1] / "src" / "milknado"
_TESTS_ROOT = Path(__file__).resolve().parents[1] / "tests"
_SCAFFOLD_FILE = _PKG_ROOT / "plugins" / "scaffold.py"
_BARREL_INIT_FILES = sorted((_PKG_ROOT / "domains").glob("*/__init__.py")) + [
    _PKG_ROOT / "adapters" / "__init__.py",
    _PKG_ROOT / "plugins" / "__init__.py",
]


def _iter_py(root: Path, excluded: set[Path]) -> Iterator[Path]:
    for path in root.rglob("*.py"):
        if path.resolve() not in excluded:
            yield path


def _is_protocol_class(node: ast.ClassDef) -> bool:
    return any(
        (isinstance(base, ast.Name) and base.id == "Protocol")
        or (isinstance(base, ast.Attribute) and base.attr == "Protocol")
        for base in node.bases
    )


def _analyze_file(path: Path) -> tuple[set[str], set[str], set[str]]:
    """Single AST parse of `path`: (used names, locally-defined names, Protocol class names)."""
    tree = ast.parse(path.read_text(), filename=str(path))
    used: set[str] = set()
    defined: set[str] = set()
    protocols: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            used.add(node.id)
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            defined.add(node.name)
            if isinstance(node, ast.ClassDef) and _is_protocol_class(node):
                protocols.add(node.name)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    defined.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            defined.add(node.target.id)
    return used, defined, protocols


def _build_indices(
    pkg_root: Path, tests_root: Path, excluded: set[Path]
) -> tuple[dict[Path, set[str]], dict[str, set[Path]], set[str]]:
    """One pass per file over pkg_root + tests_root: use-by-file, name -> defining files,
    protocols.

    Definitions and Protocol membership are only tracked for pkg_root; tests_root only
    contributes to the use index.
    """
    use_by_file: dict[Path, set[str]] = {}
    defined_by_name: dict[str, set[Path]] = {}
    protocols: set[str] = set()
    for root in (pkg_root, tests_root):
        for path in _iter_py(root, excluded):
            used, defined, file_protocols = _analyze_file(path)
            use_by_file[path] = used
            if root is pkg_root:
                for name in defined:
                    defined_by_name.setdefault(name, set()).add(path)
                protocols |= file_protocols
    return use_by_file, defined_by_name, protocols


def _is_used_outside_definition(
    name: str, defining_files: set[Path], use_by_file: dict[Path, set[str]]
) -> bool:
    return any(name in uses for path, uses in use_by_file.items() if path not in defining_files)


def _dead_exports(
    all_names: list[str],
    use_by_file: dict[Path, set[str]],
    defined_by_name: dict[str, set[Path]],
    protocols: set[str],
) -> list[str]:
    return [
        name
        for name in all_names
        if name in defined_by_name
        and name not in protocols
        and not _is_used_outside_definition(name, defined_by_name[name], use_by_file)
    ]


def test_every_barrel_export_has_a_production_use() -> None:
    excluded = {path.resolve() for path in _BARREL_INIT_FILES} | {_SCAFFOLD_FILE.resolve()}
    use_by_file, defined_by_name, protocols = _build_indices(_PKG_ROOT, _TESTS_ROOT, excluded)
    assert defined_by_name, (
        "no locally-defined names extracted from src — extraction regression "
        "(the liveness gate would otherwise pass vacuously)"
    )

    for init_file in _BARREL_INIT_FILES:
        module_name = ".".join(init_file.parent.relative_to(_PKG_ROOT.parent).parts)
        module = importlib.import_module(module_name)
        dead = _dead_exports(list(module.__all__), use_by_file, defined_by_name, protocols)
        assert not dead, f"{module_name}: locally-defined export dead in src+tests: {dead}"


def test_liveness_checker_catches_a_fabricated_dead_name() -> None:
    # A locally-defined name absent outside its own defining file and not a Protocol must be
    # reported.
    defining_file = Path("fake_module.py")
    dead = _dead_exports(
        ["ZzzDefinitelyNotUsedSymbol"],
        use_by_file={},
        defined_by_name={"ZzzDefinitelyNotUsedSymbol": {defining_file}},
        protocols=set(),
    )
    assert dead == ["ZzzDefinitelyNotUsedSymbol"]
