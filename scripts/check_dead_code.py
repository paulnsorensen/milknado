"""Owner-qualified semantic classifier over Vulture's scanner API.

Wraps Vulture 2.16's scan and accepts findings only in narrow, owner-qualified
categories; everything else is reported and fails the gate. Textual exemptions
are structural. One exact loop payload exemption covers a string-key read;
see .hallouminate/wiki/history/dead-code-coverage-gate-decision.md.
"""

from __future__ import annotations

import ast
import contextlib
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypedDict, cast

from vulture import Vulture as _RawVulture  # pyright: ignore[reportMissingTypeStubs]
from vulture.config import InputError as _RawInputError  # pyright: ignore[reportMissingTypeStubs]
from vulture.config import (  # pyright: ignore[reportMissingTypeStubs]
    make_config as _raw_make_config,  # pyright: ignore[reportUnknownVariableType]
)
from vulture.core import ExitCode as _RawExitCode  # pyright: ignore[reportMissingTypeStubs]


class _VultureConfig(TypedDict):
    verbose: bool
    ignore_names: list[str]
    ignore_decorators: list[str]
    paths: list[str]
    exclude: list[str]
    min_confidence: int
    sort_by_size: bool


class _VultureItem(Protocol):
    filename: str | Path
    first_lineno: int
    name: str
    typ: str

    def get_report(self) -> str: ...


class _ExitCodeValue(Protocol):
    @property
    def value(self) -> int: ...


class _ExitCodes(Protocol):
    InvalidInput: _ExitCodeValue
    InvalidCmdlineArguments: _ExitCodeValue
    DeadCode: _ExitCodeValue
    NoDeadCode: _ExitCodeValue


class _Vulture(Protocol):
    exit_code: _ExitCodeValue

    def __init__(
        self,
        *,
        verbose: bool,
        ignore_names: list[str],
        ignore_decorators: list[str],
    ) -> None: ...

    def scavenge(self, paths: list[str], exclude: list[str] | None = None) -> None: ...

    def get_unused_code(
        self,
        *,
        min_confidence: int,
        sort_by_size: bool,
    ) -> list[_VultureItem]: ...


Vulture = cast(type[_Vulture], _RawVulture)
InputError = cast(type[Exception], _RawInputError)
make_config = cast(Callable[[list[str]], _VultureConfig], _raw_make_config)
ExitCode = cast(_ExitCodes, cast(object, _RawExitCode))

REPO_ROOT = Path(__file__).resolve().parent.parent

# ``echo_stdout`` is read through a string key in LoopAdapter, which Vulture
# cannot connect to this TypedDict field declaration.
_FALSE_POSITIVE_FINDINGS = {
    ("src/milknado/loop/_events.py", "echo_stdout"),
}

# Textual App/Widget/Screen lifecycle method names this classifier owner-qualifies:
# overridden on a class whose base is imported from a "textual." module.
_TEXTUAL_METHOD_EXACT = {"compose"}
_TEXTUAL_METHOD_PREFIXES = ("on_", "action_")
_TEXTUAL_CLASS_VARS = {"CSS", "DEFAULT_CSS", "BINDINGS"}
# self.<name> assignments Textual itself reads: App's window subtitle.
_TEXTUAL_APP_SELF_ATTRS = {"sub_title"}
# Reactive descriptors App declares. A subclass redeclares them as a bare class-level
# annotation (``title: Reactive[str]``) so a type checker can see the descriptor type;
# the annotation binds no value, so Vulture reads it as an unused variable.
_TEXTUAL_APP_REACTIVES = {"title", "sub_title"}
# Empty start value for the base-class walk below; a literal call in a parameter
# default is rejected by the type checker's recommended tier.
_NO_BASES: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class _Finding:
    path: Path
    first_line: int
    name: str
    typ: str
    report: str


def _findings(items: Sequence[_VultureItem]) -> tuple[_Finding, ...]:
    findings: list[_Finding] = []
    for i in items:
        path = Path(i.filename)
        if path.is_absolute():
            with contextlib.suppress(ValueError):
                path = path.relative_to(REPO_ROOT)
        findings.append(_Finding(path, i.first_lineno, i.name, i.typ, i.get_report()))
    return tuple(findings)


def _import_map(module: ast.Module) -> dict[str, str]:
    out: dict[str, str] = {}
    for node in module.body:
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                out[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return out


def _resolve_base(base: ast.expr, imports: Mapping[str, str]) -> str | None:
    if isinstance(base, ast.Subscript):
        base = base.value
    if isinstance(base, ast.Name):
        return imports.get(base.id, base.id)
    if isinstance(base, ast.Attribute):
        return ast.unparse(base)
    return None


def _module_ast(dotted: str) -> tuple[ast.Module, dict[str, str]] | None:
    """Parse the in-repo module named by *dotted*, or None when it is not ours."""
    if not dotted.startswith("milknado."):
        return None
    rel = Path(*dotted.split("."))
    candidates = (
        REPO_ROOT / "src" / rel.with_suffix(".py"),
        REPO_ROOT / "src" / rel / "__init__.py",
    )
    for candidate in candidates:
        try:
            tree = ast.parse(candidate.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        return tree, _import_map(tree)
    return None


def _has_textual_base(
    class_def: ast.ClassDef, imports: Mapping[str, str], _seen: frozenset[str] = _NO_BASES
) -> bool:
    """Whether *class_def* inherits from Textual, directly or through our own classes.

    A subclass of one of milknado's own App classes is still a Textual app, so the
    walk follows in-repo bases one module at a time. ``_seen`` stops an import loop.
    """
    for base in class_def.bases:
        resolved = _resolve_base(base, imports)
        if resolved is None:
            continue
        if resolved.startswith("textual."):
            return True
        if resolved in _seen:
            continue
        module_name, _, class_name = resolved.rpartition(".")
        parsed = _module_ast(module_name)
        if parsed is None:
            continue
        base_module, base_imports = parsed
        for node in base_module.body:
            if (
                isinstance(node, ast.ClassDef)
                and node.name == class_name
                and _has_textual_base(node, base_imports, _seen | {resolved})
            ):
                return True
    return False


def _is_textual_lifecycle_method(name: str) -> bool:
    return name in _TEXTUAL_METHOD_EXACT or name.startswith(_TEXTUAL_METHOD_PREFIXES)


def _method_start_line(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    return node.decorator_list[0].lineno if node.decorator_list else node.lineno


def _is_class_var_binding(item: ast.stmt, name: str, line: int) -> bool:
    """Whether *item* binds class variable *name* at *line*, annotated or not.

    Annotating a Textual class attribute (``CSS: ClassVar[str] = "..."``) turns the
    node from ``Assign`` into ``AnnAssign``; both are the same declaration.
    """
    if item.lineno != line:
        return False
    if isinstance(item, ast.AnnAssign):
        return (
            item.value is not None and isinstance(item.target, ast.Name) and item.target.id == name
        )
    return (
        isinstance(item, ast.Assign)
        and len(item.targets) == 1
        and isinstance(item.targets[0], ast.Name)
        and item.targets[0].id == name
    )


def _accepted_reason(finding: _Finding, module: ast.Module, imports: dict[str, str]) -> str | None:
    key = (finding.path.as_posix(), finding.name)
    if key in _FALSE_POSITIVE_FINDINGS:
        return "runtime payload field read by string key"

    for node in ast.walk(module):
        if not isinstance(node, ast.ClassDef):
            continue
        if not _has_textual_base(node, imports):
            continue
        if finding.typ == "method" and _is_textual_lifecycle_method(finding.name):
            for item in node.body:
                if (
                    isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and item.name == finding.name
                    and _method_start_line(item) == finding.first_line
                ):
                    return "Textual lifecycle method override"
        if finding.typ == "variable" and finding.name in _TEXTUAL_CLASS_VARS:
            for item in node.body:
                if _is_class_var_binding(item, finding.name, finding.first_line):
                    return "Textual lifecycle class attribute"
        if finding.typ == "variable" and finding.name in _TEXTUAL_APP_REACTIVES:
            for item in node.body:
                if (
                    isinstance(item, ast.AnnAssign)
                    and item.value is None
                    and isinstance(item.target, ast.Name)
                    and item.target.id == finding.name
                    and item.lineno == finding.first_line
                ):
                    return "Textual App reactive redeclaration"
        if finding.typ == "attribute" and finding.name in _TEXTUAL_APP_SELF_ATTRS:
            for inner in ast.walk(node):
                if (
                    isinstance(inner, ast.Attribute)
                    and isinstance(inner.ctx, ast.Store)
                    and inner.attr == finding.name
                    and inner.lineno == finding.first_line
                    and isinstance(inner.value, ast.Name)
                    and inner.value.id == "self"
                ):
                    return "Textual App self-attribute"
    return None


def main(argv: Sequence[str] | None = None) -> int:
    try:
        config = make_config(list(argv) if argv is not None else sys.argv[1:])
    except InputError as err:
        print(err, file=sys.stderr)
        return ExitCode.InvalidCmdlineArguments.value

    vult = Vulture(
        verbose=config["verbose"],
        ignore_names=config["ignore_names"],
        ignore_decorators=config["ignore_decorators"],
    )
    vult.scavenge(config["paths"], exclude=config["exclude"])
    if vult.exit_code == ExitCode.InvalidInput:
        return ExitCode.InvalidInput.value

    findings = _findings(
        vult.get_unused_code(
            min_confidence=config["min_confidence"], sort_by_size=config["sort_by_size"]
        )
    )
    unclassified: list[_Finding] = []
    module_cache: dict[Path, tuple[ast.Module | None, dict[str, str]]] = {}
    for finding in findings:
        if finding.path not in module_cache:
            source_path = finding.path if finding.path.is_absolute() else REPO_ROOT / finding.path
            try:
                tree = ast.parse(source_path.read_text(encoding="utf-8"))
            except (OSError, SyntaxError):
                module_cache[finding.path] = (None, {})
            else:
                module_cache[finding.path] = (tree, _import_map(tree))
        module, imports = module_cache[finding.path]
        if module is not None and _accepted_reason(finding, module, imports):
            continue
        unclassified.append(finding)

    for finding in unclassified:
        print(finding.report)
    return ExitCode.DeadCode.value if unclassified else ExitCode.NoDeadCode.value


if __name__ == "__main__":
    raise SystemExit(main())
