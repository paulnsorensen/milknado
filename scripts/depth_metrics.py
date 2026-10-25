"""Depth metric definitions for the depth ratchet (`scripts/check_depth.py`), measurement v1.

A logical statement is an `ast.stmt` inside a function body, without nested `def` or `class`
lines and docstrings; a lambda counts 1. Complexity adds `if`, ternary, loops, `except`,
non-wildcard `case` and its guard, each extra boolean operand, each comprehension `for` and `if`,
and each parameter default. M09 counts the names of a literal `__all__`, else the public
top-level names. M12 counts crust exports that no file in another node imports or reads.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import cast

from scripts import slice_map
from scripts.slice_map import Graph, MapError, Node, SliceMap

FunctionLike = ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda
Value = int | tuple[int, int]  # a count, or a share as (numerator, denominator)
_DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _docstring(statement: ast.stmt, body: list[ast.stmt]) -> bool:
    return (
        statement is body[0]
        and isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, str)
    )


def logic_statements(tree: ast.AST) -> int:
    """Count statements inside function bodies; a lambda counts 1.

    Nested `def`/`class` lines and docstrings count 0.
    """
    total = 0

    def visit(node: ast.AST, inside: bool) -> None:
        nonlocal total
        body = cast(list[ast.stmt], getattr(node, "body", []))
        for child in ast.iter_child_nodes(node):
            logic = (
                inside
                and isinstance(child, ast.stmt)
                and not isinstance(child, _DEFINITIONS)
                and not (body and _docstring(child, body))
            )
            if logic or isinstance(child, ast.Lambda):
                total += 1
            visit(child, inside or isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)))

    visit(tree, False)
    return total


def _decisions(node: ast.AST) -> int:
    match node:
        case (
            ast.If() | ast.IfExp() | ast.For() | ast.AsyncFor() | ast.While() | ast.ExceptHandler()
        ):
            return 1
        case ast.BoolOp(values=values):
            return len(values) - 1
        case ast.comprehension(ifs=ifs):
            return 1 + len(ifs)
        case ast.match_case(pattern=pattern, guard=guard):
            wildcard = isinstance(pattern, ast.MatchAs) and pattern.pattern is None
            return (0 if wildcard else 1) + (guard is not None)
        case _:
            return 0


def complexity(tree: ast.AST) -> int:
    """Sum cyclomatic complexity over function-likes: 1 plus each decision and default."""
    total = 0
    for function in ast.walk(tree):
        if not isinstance(function, FunctionLike):
            continue
        arguments = function.args
        total += 1 + len(arguments.defaults) + sum(d is not None for d in arguments.kw_defaults)
        body = function.body
        pending: list[ast.AST] = [body] if isinstance(body, ast.expr) else list(body)
        while pending:
            node = pending.pop()
            if isinstance(node, FunctionLike):
                continue  # A nested function is its own unit.
            total += _decisions(node)
            pending.extend(ast.iter_child_nodes(node))
    return total


def _calls(node: ast.AST) -> bool:
    return any(isinstance(child, ast.Call) for child in ast.walk(node))


def module_init(tree: ast.Module) -> int:
    """Count top-level executable statements; logic at module scope leaves M08 and M10."""
    count = 0
    for statement in tree.body:
        if isinstance(statement, (ast.Import, ast.ImportFrom, *_DEFINITIONS)):
            continue
        type_guard = isinstance(statement, ast.If) and ast.unparse(statement.test) in {
            "TYPE_CHECKING",
            "typing.TYPE_CHECKING",
        }
        if _docstring(statement, tree.body) or type_guard:
            continue
        if isinstance(statement, (ast.Assign, ast.AnnAssign)) and not _calls(statement):
            continue
        count += 1
    return count


def _literal_names(value: ast.expr | None) -> list[str] | None:
    if not isinstance(value, (ast.List, ast.Tuple)):
        return None
    names = [
        item.value
        for item in value.elts
        if isinstance(item, ast.Constant) and isinstance(item.value, str)
    ]
    return sorted(set(names)) if len(names) == len(value.elts) else None


def _all(tree: ast.Module) -> list[str] | None:
    """Return the names of a literal `__all__`, or None when the module never names `__all__`.

    Raise MapError when `__all__` is computed, or when another statement reads or changes it.
    """
    references = [
        node for node in ast.walk(tree) if isinstance(node, ast.Name) and node.id == "__all__"
    ]
    if not references:
        return None
    for statement in tree.body:
        match statement:
            case ast.Assign(targets=targets, value=value):
                pass
            case ast.AnnAssign(target=target, value=value):
                targets = [target]
            case _:
                continue
        names = _literal_names(value)
        if len(references) == 1 and references[0] in targets and names is not None:
            return names
    raise MapError("a crust `__all__` must be one literal list or tuple that no statement reuses")


def exports(tree: ast.Module) -> list[str]:
    """Return the export slots of one crust module: a literal `__all__`, else its public names."""
    if (listed := _all(tree)) is not None:
        return listed
    names: set[str] = set()
    for statement in tree.body:
        match statement:
            case (
                ast.FunctionDef(name=name)
                | ast.AsyncFunctionDef(name=name)
                | ast.ClassDef(name=name)
            ):
                names.add(name)
            case ast.Import(names=aliases) | ast.ImportFrom(names=aliases):
                names.update(a.asname or a.name.split(".")[0] for a in aliases if a.name != "*")
            case ast.Assign(targets=targets):
                names.update(_bound_names(targets))
            case ast.AnnAssign(target=target):
                names.update(_bound_names([target]))
            case _:
                pass
    return sorted(name for name in names if not name.startswith("_"))


def _bound_names(targets: list[ast.expr]) -> set[str]:
    return {
        node.id for target in targets for node in ast.walk(target) if isinstance(node, ast.Name)
    }


def _imported_uses(tree: ast.Module, module: str, is_package: bool, crust: str) -> set[str] | None:
    """Return the names a file imports from `crust` and uses; None for a star import."""
    used: set[str] = set()
    loads = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    listed = set(_all(tree) or ())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and slice_map.absolute(node, module, is_package) == crust
        ):
            for alias in node.names:
                if alias.name == "*":
                    return None
                if (local := alias.asname or alias.name) in loads or local in listed:
                    used.add(alias.name)
    return used


def _uses(graph: Graph, file: str, crust: str, names: list[str]) -> set[str]:
    """Return the crust export names that one outside file uses through the crust module."""
    tree, module, is_package = graph.trees[file], graph.files[file], file.endswith("/__init__.py")
    imported = _imported_uses(tree, module, is_package, crust)
    if imported is None:
        return set(names)
    used = imported & set(names)
    bindings = slice_map.module_bindings(tree, module, is_package)
    bound = {local for local, target in bindings.items() if target == crust}
    chains = {
        local: target for local, target in bindings.items() if crust.startswith(target + ".")
    }
    values: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        values.add(id(node.value))
        local, _, rest = (slice_map.dotted(node) or "").partition(".")
        if local in bound:
            full = f"{crust}.{rest}"
        elif local in chains:
            full = f"{chains[local]}.{rest}"
        else:
            continue
        name = full.removeprefix(crust + ".").split(".")[0]
        if full.startswith(crust + ".") and name in names:
            used.add(name)
    bare = any(
        isinstance(node, ast.Name) and node.id in bound and id(node) not in values
        for node in ast.walk(tree)
    )
    return set(names) if bare else used


@dataclass(frozen=True)
class Measure:
    values: dict[str, Value]
    exports: dict[str, list[str]]
    unused: list[str]
    module_init: int


def _unused(slices: SliceMap, graph: Graph, node: Node, crust: str) -> list[str]:
    names = exports(graph.trees[crust])
    used: set[str] = set()
    for file in graph.files:
        owner = slices.owner(file)
        if owner is not None and owner.name != node.name:
            used |= _uses(graph, file, graph.files[crust], names)
    return [f"{graph.files[crust]}.{name}" for name in names if name not in used]


def _measure_slice(slices: SliceMap, graph: Graph, node: Node) -> Measure:
    owned = [file for file in graph.files if node.owns(file)]
    crust = [file for file in owned if file in node.crust]
    statements = {file: logic_statements(graph.trees[file]) for file in owned}
    branches = {file: complexity(graph.trees[file]) for file in owned}
    crust_statements = sum(statements[file] for file in crust)
    crust_complexity = sum(branches[file] for file in crust)
    slots = {file: exports(graph.trees[file]) for file in crust}
    unused = [name for file in crust for name in _unused(slices, graph, node, file)]
    values: dict[str, Value] = {
        "M08.crust_logic_statements": crust_statements,
        "M08.share": (crust_statements, sum(statements.values())),
        "M09.public_surface": sum(len(names) for names in slots.values()),
        "M10.crust_complexity": crust_complexity,
        "M10.share": (crust_complexity, sum(branches.values())),
        "M12.unused_exports": len(unused),
    }
    return Measure(values, slots, unused, sum(module_init(graph.trees[file]) for file in owned))


def measure(slices: SliceMap, graph: Graph) -> dict[str, Measure]:
    """Return the depth measure of each `[[slice]]`."""
    return {
        node.name: _measure_slice(slices, graph, node)
        for node in slices.nodes
        if node.kind == "slice"
    }


def computed_imports(graph: Graph) -> list[str]:
    """Name each computed import, which neither the guard nor M12 follows."""
    dynamic = {"importlib.import_module", "__import__", "import_module"}
    return [
        f"{file}:{node.lineno}"
        for file, tree in graph.trees.items()
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and slice_map.dotted(node.func) in dynamic
    ]
