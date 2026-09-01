from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import cache
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).parents[1] / "src" / "milknado"
_SCAFFOLD_TEMPLATE = _PACKAGE_ROOT / "plugins" / "scaffold.py"


@dataclass(frozen=True)
class _Export:
    path: Path
    name: str
    node: ast.AST


@cache
def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _literal_exports(path: Path) -> tuple[str, ...]:
    for statement in _parse(path).body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__" for target in statement.targets
        ):
            exports_raw: object = ast.literal_eval(statement.value)  # pyright: ignore[reportAny]
            if not isinstance(exports_raw, (list, tuple)):
                raise AssertionError(f"{path}: __all__ must contain only literal strings")
            exports: list[str] = []
            for item in exports_raw:  # pyright: ignore[reportUnknownVariableType]
                if not isinstance(item, str):
                    raise AssertionError(f"{path}: __all__ must contain only literal strings")
                exports.append(item)
            return tuple(exports)
    return ()


def _module_path(root: Path, importer: Path, imported: ast.ImportFrom) -> Path:
    if imported.level:
        base = importer.parent
        for _ in range(imported.level - 1):
            base = base.parent
        parts = imported.module.split(".") if imported.module else []
    else:
        parts = (imported.module or "").split(".")
        if parts and parts[0] == root.name:
            parts = parts[1:]
        base = root
    target = base.joinpath(*parts)
    module_file = target.with_suffix(".py")
    return module_file if module_file.is_file() else target / "__init__.py"


def _definitions(path: Path) -> dict[str, ast.AST]:
    definitions: dict[str, ast.AST] = {}
    for statement in _parse(path).body:
        if isinstance(statement, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            definitions[statement.name] = statement
        elif isinstance(statement, (ast.Assign, ast.AnnAssign)):
            targets = (
                statement.targets if isinstance(statement, ast.Assign) else [statement.target]
            )
            definitions.update(
                (target.id, statement) for target in targets if isinstance(target, ast.Name)
            )
    return definitions


def _imports(root: Path, path: Path) -> dict[str, tuple[Path, str]]:
    imports: dict[str, tuple[Path, str]] = {}
    for statement in ast.walk(_parse(path)):
        if not isinstance(statement, ast.ImportFrom):
            continue
        source = _module_path(root, path, statement)
        if source.is_file():
            for imported in statement.names:
                imports[imported.asname or imported.name] = (source, imported.name)
    return imports


def _resolve(
    root: Path, path: Path, name: str, seen: frozenset[tuple[Path, str]] | None = None
) -> _Export:
    seen = seen or frozenset()
    identity = (path, name)
    if identity in seen:
        raise AssertionError(f"cyclic export resolution for {path}:{name}")
    if node := _definitions(path).get(name):
        return _Export(path, name, node)
    if target := _imports(root, path).get(name):
        return _resolve(root, *target, seen | {identity})
    raise AssertionError(f"{path}: cannot resolve exported name {name!r}")


def _annotation_nodes(export: _Export) -> list[ast.AST]:
    node = export.node
    annotations: list[ast.AST] = []
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        for arg in (*node.args.posonlyargs, *node.args.args):
            if arg.annotation is not None:
                annotations.append(arg.annotation)
        for arg in node.args.kwonlyargs:
            if arg.annotation is not None:
                annotations.append(arg.annotation)
        if node.args.vararg and node.args.vararg.annotation is not None:
            annotations.append(node.args.vararg.annotation)
        if node.args.kwarg and node.args.kwarg.annotation is not None:
            annotations.append(node.args.kwarg.annotation)
        if node.returns is not None:
            annotations.append(node.returns)
    elif isinstance(node, ast.ClassDef):
        annotations.extend(
            statement.annotation for statement in node.body if isinstance(statement, ast.AnnAssign)
        )
    return annotations


def _loaded_names(node: ast.AST) -> set[str]:
    return {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
    }


def _annotation_names(node: ast.AST) -> set[str]:
    names = _loaded_names(node)
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            try:
                names.update(_loaded_names(ast.parse(child.value, mode="eval")))
            except SyntaxError:
                pass
    return names


def _reference_names(path: Path) -> set[str]:
    names: set[str] = set()
    for statement in _parse(path).body:
        if isinstance(statement, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__" for target in statement.targets
        ):
            continue
        names.update(_loaded_names(statement))
    return names


def _barrels(root: Path) -> tuple[Path, ...]:
    slices = root / "domains"
    search_root = slices if slices.is_dir() else root
    return tuple(path for path in search_root.rglob("__init__.py") if _literal_exports(path))


def _candidate_exports(root: Path, barrels: tuple[Path, ...]) -> dict[tuple[Path, str], _Export]:
    exports = (
        _resolve(root, barrel, name) for barrel in barrels for name in _literal_exports(barrel)
    )
    return {(export.path, export.name): export for export in exports}


def _directly_live(
    root: Path,
    candidates: dict[tuple[Path, str], _Export],
) -> set[tuple[Path, str]]:
    live: set[tuple[Path, str]] = set()
    for path in root.rglob("*.py"):
        if path == _SCAFFOLD_TEMPLATE:
            continue
        imports = _imports(root, path)
        for name in _reference_names(path):
            if not (target := imports.get(name)):
                continue
            try:
                resolved = _resolve(root, *target)
            except AssertionError:
                continue
            identity = (resolved.path, resolved.name)
            if identity in candidates and resolved.path != path:
                live.add(identity)
    return live


def _annotation_dependencies(
    root: Path, candidates: dict[tuple[Path, str], _Export]
) -> dict[tuple[Path, str], set[tuple[Path, str]]]:
    dependencies: dict[tuple[Path, str], set[tuple[Path, str]]] = {}
    for identity, export in candidates.items():
        imports = _imports(root, export.path)
        local = _definitions(export.path)
        edges: set[tuple[Path, str]] = set()
        for annotation in _annotation_nodes(export):
            for name in _annotation_names(annotation):
                if target := imports.get(name):
                    resolved = _resolve(root, *target)
                elif node := local.get(name):
                    resolved = _Export(export.path, name, node)
                else:
                    continue
                dependency = (resolved.path, resolved.name)
                if dependency in candidates:
                    edges.add(dependency)
        dependencies[identity] = edges
    return dependencies


def _dead_exports(root: Path) -> set[str]:
    barrels = _barrels(root)
    candidates = _candidate_exports(root, barrels)
    live = _directly_live(root, candidates)
    dependencies = _annotation_dependencies(root, candidates)
    pending = list(live)
    while pending:
        identity = pending.pop()
        for dependency in dependencies[identity] - live:
            live.add(dependency)
            pending.append(dependency)
    return {name for path, name in candidates if (path, name) not in live}


def _package(tmp_path: Path, files: dict[str, str]) -> Path:
    root = tmp_path / "pkg"
    for relative, source in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        _ = path.write_text(source, encoding="utf-8")
    return root


def test_production_barrel_exports_are_live() -> None:
    assert _dead_exports(_PACKAGE_ROOT) == set()


def test_test_only_consumer_does_not_keep_export_live(tmp_path: Path) -> None:
    root = _package(
        tmp_path,
        {
            "__init__.py": 'from pkg.models import TestOnly\n__all__ = ["TestOnly"]\n',
            "models.py": "class TestOnly: pass\n",
        },
    )
    tests = tmp_path / "tests"
    tests.mkdir()
    _ = (tests / "test_consumer.py").write_text(
        "from pkg import TestOnly\nassert TestOnly\n", encoding="utf-8"
    )
    assert _dead_exports(root) == {"TestOnly"}


def test_live_producer_keeps_return_contract_live(tmp_path: Path) -> None:
    root = _package(
        tmp_path,
        {
            "__init__.py": (
                'from pkg.api import Result, produce\n__all__ = ["Result", "produce"]\n'
            ),
            "api.py": "class Result: pass\ndef produce() -> Result: return Result()\n",
            "consumer.py": "from pkg import produce\nproduce()\n",
        },
    )
    assert _dead_exports(root) == set()


def test_nested_exported_model_field_contract_stays_live(tmp_path: Path) -> None:
    root = _package(
        tmp_path,
        {
            "__init__.py": 'from pkg.models import Inner, Outer\n__all__ = ["Inner", "Outer"]\n',
            "models.py": "class Inner: pass\nclass Outer:\n    inner: Inner\n",
            "consumer.py": "from pkg import Outer\nOuter()\n",
        },
    )
    assert _dead_exports(root) == set()


def test_same_module_body_reference_does_not_keep_helper_export_live(tmp_path: Path) -> None:
    root = _package(
        tmp_path,
        {
            "__init__.py": 'from pkg.api import Helper, run\n__all__ = ["Helper", "run"]\n',
            "api.py": "class Helper: pass\ndef run(): Helper()\n",
            "consumer.py": "from pkg import run\nrun()\n",
        },
    )
    assert _dead_exports(root) == {"Helper"}


def test_executable_use_inside_barrel_keeps_export_live(tmp_path: Path) -> None:
    root = _package(
        tmp_path,
        {
            "__init__.py": (
                "from pkg.api import Helper\n"
                '__all__ = ["Helper", "run"]\n'
                "def run(): return Helper()\n"
            ),
            "api.py": "class Helper: pass\n",
            "consumer.py": "from pkg import run\nrun()\n",
        },
    )
    assert _dead_exports(root) == set()


def test_string_literal_does_not_keep_export_live(tmp_path: Path) -> None:
    root = _package(
        tmp_path,
        {
            "__init__.py": 'from pkg.api import Ghost\n__all__ = ["Ghost"]\n',
            "api.py": "class Ghost: pass\n",
            "consumer.py": 'from pkg import Ghost\nname = "Ghost"\n',
        },
    )
    assert _dead_exports(root) == {"Ghost"}
