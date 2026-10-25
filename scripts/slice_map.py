"""Load `slices.toml` and build the module import graph for the slice guard and depth ratchet.

The schema follows sliced-bread-guard (paulnsorensen/sliced-bread-architecture#69) with a
Python backend: `[policy] language` is `"python"`, and `[policy] roots` names the import roots,
such as `src`. Each `.py` file under a root is a module; its dotted name is its path below the
root. An input error raises `MapError`; the callers exit 2 on it.
"""

from __future__ import annotations

import ast
import fnmatch
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import cast

SHAPES = frozenset({"thin-facade", "framework-bound", "service"})
SUPPRESSIBLE = frozenset({"SB-01", "SB-02", "SB-03", "SB-04", "SB-06"})
MAP_FILE = "slices.toml"


class MapError(Exception):
    """`slices.toml`, a baseline, or the source tree is not valid input."""


def _inside(file: str, paths: tuple[str, ...]) -> bool:
    return any(file == path or file.startswith(path + "/") for path in paths)


@dataclass(frozen=True)
class Node:
    name: str
    kind: str  # "slice" or "spine"
    paths: tuple[str, ...]
    crust: tuple[str, ...]
    may_depend_on: frozenset[str]
    requires: frozenset[str]

    def owns(self, file: str) -> bool:
        return _inside(file, self.paths)


@dataclass(frozen=True)
class Suppression:
    rule: str
    source: str
    target: str


@dataclass(frozen=True)
class SliceMap:
    roots: tuple[str, ...]
    ignore: tuple[str, ...]
    nodes: tuple[Node, ...]
    suppressions: tuple[Suppression, ...]

    def owner(self, file: str) -> Node | None:
        return next((node for node in self.nodes if node.owns(file)), None)

    def node(self, name: str) -> Node:
        return next(node for node in self.nodes if node.name == name)


def table(
    value: object, where: str, required: set[str], optional: set[str] | None
) -> dict[str, object]:
    """Return `value` as a table with every required field; `optional=None` admits any key."""
    if not isinstance(value, dict):
        raise MapError(f"{where} must be a table")
    found = {str(key): item for key, item in cast(dict[object, object], value).items()}
    unknown = [] if optional is None else sorted(found.keys() - required - optional)
    missing = sorted(required - found.keys())
    if unknown or missing:
        raise MapError(f"{where}: unknown fields {unknown}, missing fields {missing}")
    return found


def array(value: object, where: str) -> list[object]:
    if not isinstance(value, list):
        raise MapError(f"{where} must be an array")
    return cast(list[object], value)


def _text(value: object, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise MapError(f"{where} must be a non-empty string")
    return value


def _texts(value: object, where: str) -> tuple[str, ...]:
    items = tuple(_text(item, where) for item in array(value, where))
    if len(set(items)) != len(items):
        raise MapError(f"{where} holds a duplicate")
    return items


def _paths(value: object, where: str, root: Path) -> tuple[str, ...]:
    paths = _texts(value, where)
    for path in paths:
        globbed = any(mark in path for mark in "*?[")
        if globbed or path.endswith("/") or path.startswith("/") or ".." in path.split("/"):
            raise MapError(f"{where}: {path!r} must be a plain relative path without a glob")
        if not (root / path).exists():
            raise MapError(f"{where}: {path!r} does not exist")
    return paths


def _crust(
    fields: dict[str, object], where: str, paths: tuple[str, ...], root: Path
) -> tuple[str, ...]:
    crust = _paths(fields["crust"], f"{where} crust", root)
    if not crust:
        raise MapError(f"{where} crust must name at least one file")
    outside = [file for file in crust if not _inside(file, paths)]
    if outside or not all((root / file).is_file() for file in crust):
        raise MapError(f"{where} crust must be files inside its paths: {outside or list(crust)}")
    if fields["shape"] not in SHAPES:
        raise MapError(f"{where} shape must be one of {sorted(SHAPES)}")
    return crust


def _node(value: object, kind: str, root: Path) -> Node:
    is_slice = kind == "slice"
    required = {"name", "paths", "may_depend_on"} | (
        {"crust", "shape"} if is_slice else set[str]()
    )
    fields = table(value, f"[[{kind}]]", required, {"requires"} if is_slice else set())
    name = _text(fields["name"], f"[[{kind}]] name")
    where = f"[[{kind}]] {name}"
    paths = _paths(fields["paths"], f"{where} paths", root)
    crust = _crust(fields, where, paths, root) if is_slice else ()
    may = frozenset(_texts(fields["may_depend_on"], f"{where} may_depend_on"))
    requires = frozenset(_texts(fields.get("requires", []), f"{where} requires"))
    if not requires <= may:
        raise MapError(f"{where} requires {sorted(requires - may)}, not in may_depend_on")
    return Node(name, kind, paths, crust, may, requires)


def _check_nodes(nodes: list[Node]) -> None:
    names = [node.name for node in nodes]
    if len(set(names)) != len(names):
        raise MapError(f"{MAP_FILE}: duplicate node names")
    for node in nodes:
        unknown = sorted(node.may_depend_on - set(names))
        if unknown or node.name in node.may_depend_on:
            raise MapError(f"{node.name} may_depend_on names unknown nodes {unknown} or itself")
    claimed = [(path, node.name) for node in nodes for path in node.paths]
    for path, name in claimed:
        for other, other_name in claimed:
            if (path, name) != (other, other_name) and _inside(other, (path,)):
                raise MapError(
                    f"{MAP_FILE}: path {other!r} of {other_name} overlaps path {path!r} of {name}"
                )


def _suppression(entry: object) -> Suppression:
    fields = table(entry, "[[suppress]]", {"rule", "from", "to", "reason"}, set())
    if fields["rule"] not in SUPPRESSIBLE:
        raise MapError(f"[[suppress]] rule must be one of {sorted(SUPPRESSIBLE)}")
    _ = _text(fields["reason"], "[[suppress]] reason")
    return Suppression(
        str(fields["rule"]),
        _text(fields["from"], "[[suppress]] from"),
        _text(fields["to"], "[[suppress]] to"),
    )


def parse(text: str, root: Path) -> SliceMap:
    """Return the validated slice map that `text` declares for the tree at `root`."""
    try:
        document = cast(object, tomllib.loads(text))
    except tomllib.TOMLDecodeError as error:
        raise MapError(f"{MAP_FILE}: {error}") from error
    top = table(document, MAP_FILE, {"version", "policy"}, {"slice", "spine", "suppress"})
    if top["version"] != 1:
        raise MapError(f"{MAP_FILE}: version must be 1")
    policy = table(
        top["policy"],
        "[policy]",
        {"language", "unmapped_files", "suppressions", "roots"},
        {"ignore"},
    )
    expected = {"language": "python", "unmapped_files": "error", "suppressions": "human-only"}
    for field, value in expected.items():
        if policy[field] != value:
            raise MapError(f"[policy] {field} must be {value!r}")
    roots = _paths(policy["roots"], "[policy] roots", root)
    ignore = _texts(policy.get("ignore", []), "[policy] ignore")
    nodes = [
        _node(entry, kind, root)
        for kind in ("slice", "spine")
        for entry in array(top.get(kind, []), f"[[{kind}]]")
    ]
    _check_nodes(nodes)
    suppressions = tuple(map(_suppression, array(top.get("suppress", []), "[[suppress]]")))
    return SliceMap(roots, ignore, tuple(nodes), suppressions)


def load(root: Path) -> SliceMap:
    path = root / MAP_FILE
    if not path.is_file():
        raise MapError(f"{MAP_FILE} is missing at {root}")
    return parse(path.read_text(encoding="utf-8"), root)


@dataclass(frozen=True)
class Edge:
    source: str  # importing file, relative to the repository root
    target: str  # imported file
    line: int


@dataclass(frozen=True)
class Unresolved:
    source: str
    specifier: str
    line: int


@dataclass(frozen=True)
class Graph:
    files: dict[str, str]  # file -> dotted module name, for every inventoried file
    modules: dict[str, str]  # dotted module name -> file
    trees: dict[str, ast.Module]
    edges: tuple[Edge, ...]
    unresolved: tuple[Unresolved, ...]
    symlinks: tuple[str, ...]


def _symlinked_root(root: Path, base: str) -> bool:
    return any((root / part).is_symlink() for part in (Path(base), *Path(base).parents[:-1]))


def _inventory(root: Path, slice_map: SliceMap) -> tuple[dict[str, str], list[str]]:
    """Return the module name of each `.py` file under the roots, and each symlinked path.

    The walk does not follow a symlink, so a symlinked directory is reported, not skipped.
    """
    files: dict[str, str] = {}
    symlinks: list[str] = []
    for base in slice_map.roots:
        if _symlinked_root(root, base):
            symlinks.append(base)
            continue
        top = root / base
        for directory, dirnames, filenames in os.walk(top):
            dirnames.sort()
            for name in sorted(dirnames + filenames):
                path = Path(directory) / name
                relative = path.relative_to(root).as_posix()
                if any(fnmatch.fnmatchcase(relative, pattern) for pattern in slice_map.ignore):
                    continue
                if path.is_symlink():
                    if name in dirnames or name.endswith(".py"):
                        symlinks.append(relative)
                elif name in filenames and name.endswith(".py"):
                    parts = path.relative_to(top).with_suffix("").parts
                    files[relative] = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
    return files, sorted(symlinks)


def dotted(node: ast.expr) -> str | None:
    """Return `a.b.c` for a chain of attributes on a name, or None for any other expression."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute) and (base := dotted(node.value)):
        return f"{base}.{node.attr}"
    return None


def module_bindings(tree: ast.Module, module: str, is_package: bool) -> dict[str, str]:
    """Map each local name that an import binds to the absolute dotted module it names."""
    bound: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                bound[alias.asname or top] = alias.name if alias.asname else top
        elif isinstance(node, ast.ImportFrom) and (base := absolute(node, module, is_package)):
            for alias in node.names:
                if alias.name != "*":
                    bound[alias.asname or alias.name] = f"{base}.{alias.name}"
    return bound


def absolute(node: ast.ImportFrom, module: str, is_package: bool) -> str | None:
    """Return the absolute module of a `from ... import`, or None when it climbs above the top."""
    if not node.level:
        return node.module
    package = module.split(".") if is_package else module.split(".")[:-1]
    if node.level - 1 >= len(package):
        return None
    kept = package[: len(package) - (node.level - 1)]
    return ".".join([*kept, *([node.module] if node.module else [])])


@dataclass
class _Collector:
    """Collect the resolved edges and the unresolved first-party imports of each tree."""

    modules: dict[str, str]
    tops: set[str]
    edges: set[Edge]
    unresolved: set[Unresolved]

    def edge(self, source: str, target: str, line: int) -> None:
        if self.modules[target] != source:
            self.edges.add(Edge(source, self.modules[target], line))

    def imports(self, file: str, module: str, tree: ast.Module) -> None:
        is_package = file.endswith("/__init__.py")
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] not in self.tops:
                        continue
                    if alias.name in self.modules:
                        self.edge(file, alias.name, node.lineno)
                    else:
                        self.unresolved.add(Unresolved(file, alias.name, node.lineno))
            elif isinstance(node, ast.ImportFrom):
                self._import_from(file, absolute(node, module, is_package), node)

    def _import_from(self, file: str, base: str | None, node: ast.ImportFrom) -> None:
        if base is None:
            specifier = "." * node.level + (node.module or "")
            self.unresolved.add(Unresolved(file, specifier, node.lineno))
            return
        if node.level == 0 and base.split(".")[0] not in self.tops:
            return
        for alias in node.names:
            if f"{base}.{alias.name}" in self.modules:
                self.edge(file, f"{base}.{alias.name}", node.lineno)
            elif base in self.modules:
                self.edge(file, base, node.lineno)
            else:
                self.unresolved.add(Unresolved(file, base, node.lineno))

    def attributes(self, file: str, module: str, tree: ast.Module) -> None:
        bindings = module_bindings(tree, module, file.endswith("/__init__.py"))
        bound = {name: target for name, target in bindings.items() if target in self.modules}
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and (chain := dotted(node)):
                name, _, rest = chain.partition(".")
                if name in bound and (target := f"{bound[name]}.{rest}") in self.modules:
                    self.edge(file, target, node.lineno)


def build(root: Path, slice_map: SliceMap) -> Graph:
    """Parse every inventoried file and return its resolved and unresolved imports."""
    files, symlinks = _inventory(root, slice_map)
    modules = {module: file for file, module in files.items()}
    collector = _Collector(modules, {module.split(".")[0] for module in modules}, set(), set())
    trees: dict[str, ast.Module] = {}
    for file, module in files.items():
        try:
            tree = ast.parse((root / file).read_text(encoding="utf-8"), file)
        except SyntaxError as error:
            raise MapError(f"{file}: {error}") from error
        trees[file] = tree
        collector.imports(file, module, tree)
        collector.attributes(file, module, tree)
    return Graph(
        files,
        modules,
        trees,
        tuple(sorted(collector.edges, key=lambda e: (e.source, e.target, e.line))),
        tuple(sorted(collector.unresolved, key=lambda u: (u.source, u.specifier, u.line))),
        tuple(symlinks),
    )
