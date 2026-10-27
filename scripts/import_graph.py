"""Build the module import graph of the tree that `slices.toml` maps.

Each `.py` file under a root is a module; its dotted name is its path below the root.
Attribute chains resolve through the import bindings of their lexical scope.
"""

from __future__ import annotations

import ast
import fnmatch
import os
from dataclasses import dataclass
from pathlib import Path

from scripts.slice_map import MapError, SliceMap


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


Bindings = dict[str, frozenset[str]]
Comprehension = ast.ListComp | ast.SetComp | ast.GeneratorExp | ast.DictComp
_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)
_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef, *_COMPREHENSIONS)


def _function_header(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda) -> list[ast.AST]:
    """Return the defaults, decorators, and annotations that the enclosing scope evaluates."""
    args = node.args
    header: list[ast.AST] = [*args.defaults, *(d for d in args.kw_defaults if d)]
    if isinstance(node, ast.Lambda):
        return header
    every = (*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg)
    header += [arg.annotation for arg in every if arg and arg.annotation]
    return [*header, *node.decorator_list, *([node.returns] if node.returns else [])]


def _comprehension(node: Comprehension) -> tuple[list[ast.AST], list[ast.AST]]:
    """Split a comprehension: the enclosing scope evaluates only its first iterable."""
    first = node.generators[0]
    rest = [child for child in ast.iter_child_nodes(node) if child is not first]
    return [first.iter], [*rest, first.target, *first.ifs]


def _split(scope: ast.AST) -> tuple[list[ast.AST], list[ast.AST]]:
    """Return the nodes that the enclosing scope evaluates, and the nodes of `scope` itself."""
    match scope:
        case ast.FunctionDef() | ast.AsyncFunctionDef():
            return _function_header(scope), list(scope.body)
        case ast.Lambda():
            return _function_header(scope), [scope.body]
        case ast.ClassDef():
            keywords = [keyword.value for keyword in scope.keywords]
            return [*scope.decorator_list, *scope.bases, *keywords], list(scope.body)
        case ast.ListComp() | ast.SetComp() | ast.GeneratorExp() | ast.DictComp():
            return _comprehension(scope)
        case _:
            return [], list(ast.iter_child_nodes(scope))


def _own_nodes(scope: ast.AST) -> list[ast.AST]:
    """Return the nodes that `scope` evaluates; a nested scope is included but not entered."""
    found: list[ast.AST] = []
    stack = _split(scope)[1]
    while stack:
        node = stack.pop()
        found.append(node)
        stack.extend(_split(node)[0] if isinstance(node, _SCOPES) else ast.iter_child_nodes(node))
    return found


def _bind(nodes: list[ast.AST], module: str, is_package: bool) -> dict[str, set[str]]:
    """Map each local name that these imports bind to every absolute module it names."""
    bound: dict[str, set[str]] = {}
    for node in nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname or alias.name.split(".")[0]
                bound.setdefault(local, set()).add(alias.name if alias.asname else local)
        elif isinstance(node, ast.ImportFrom) and (base := absolute(node, module, is_package)):
            for alias in node.names:
                if alias.name != "*":
                    name = alias.asname or alias.name
                    bound.setdefault(name, set()).add(f"{base}.{alias.name}")
    return bound


@dataclass
class _Scope:
    node: ast.AST
    parent: _Scope | None
    own: list[ast.AST]
    bound: dict[str, set[str]]  # the import bindings that this scope owns


def _scopes(tree: ast.Module, module: str, is_package: bool) -> list[_Scope]:
    """Return every scope of `tree`; a parent always comes before its children."""
    own = _own_nodes(tree)
    scopes: list[_Scope] = []
    pending = [_Scope(tree, None, own, _bind(own, module, is_package))]
    while pending:
        scope = pending.pop()
        scopes.append(scope)
        for node in scope.own:
            if isinstance(node, _SCOPES):
                inner = _own_nodes(node)
                pending.append(_Scope(node, scope, inner, _bind(inner, module, is_package)))
    return scopes


def _declared_owner(scope: _Scope, declaration: ast.Global | ast.Nonlocal) -> _Scope:
    """Return the scope that owns a name that `declaration` declares in `scope`."""
    functions = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
    owner = scope
    while owner.parent is not None:
        owner = owner.parent
        if isinstance(declaration, ast.Nonlocal) and isinstance(owner.node, functions):
            return owner
    return owner


def _redirect(scopes: list[_Scope]) -> None:
    """Move each import binding of a `global` or `nonlocal` name into the scope that owns it."""
    for scope in reversed(scopes):  # A child moves its names before its parent does.
        for node in scope.own:
            if isinstance(node, ast.Global | ast.Nonlocal):
                owner = _declared_owner(scope, node)
                for name in node.names:
                    if name in scope.bound:
                        owner.bound.setdefault(name, set()).update(scope.bound.pop(name))


def scoped_bindings(
    tree: ast.Module, module: str, is_package: bool
) -> list[tuple[ast.AST, Bindings]]:
    """Pair each node with the import bindings of its lexical scope.

    A name that one scope binds more than once keeps every target. An import of a `global` or
    `nonlocal` name binds in the scope that owns the name.
    """
    scopes = _scopes(tree, module, is_package)
    _redirect(scopes)
    inner: dict[int, Bindings] = {}
    paired: list[tuple[ast.AST, Bindings]] = []
    for scope in scopes:
        outer = inner[id(scope.parent.node)] if scope.parent else {}
        visible = {**outer, **{name: frozenset(found) for name, found in scope.bound.items()}}
        inner[id(scope.node)] = outer if isinstance(scope.node, ast.ClassDef) else visible
        paired += [(node, visible) for node in scope.own]
    return paired


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
        for node, bindings in scoped_bindings(tree, module, file.endswith("/__init__.py")):
            if not isinstance(node, ast.Attribute) or not (chain := dotted(node)):
                continue
            name, _, rest = chain.partition(".")
            for bound in bindings.get(name, ()):
                if bound in self.modules and (target := f"{bound}.{rest}") in self.modules:
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
