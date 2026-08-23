"""Owner-qualified semantic classifier over Vulture's scanner API.

Wraps Vulture 2.16's scan and accepts findings only in narrow, owner-qualified
categories; everything else is reported and fails the gate. Whitelist-free --
no checked-in symbol list (see .hallouminate/wiki/history/dead-code-coverage-gate-decision.md).
"""

from __future__ import annotations

import ast
import contextlib
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from vulture import Vulture
from vulture.config import InputError, make_config
from vulture.core import ExitCode, Item

REPO_ROOT = Path(__file__).resolve().parent.parent

# src/milknado/loop/ is vendored wholesale from ralphify (PR #137, 49ac978); see
# loop/__init__.py. These exact (relative path, name) pairs are the vendored engine's
# own unused-by-milknado surface (formerly vulture-whitelist-loop.py) -- only these
# enumerated findings are exempt, so genuinely new dead code in the engine still fails.
_VENDORED_LOOP_FINDINGS = {
    ("src/milknado/loop/_events.py", "ralph_name"),
    ("src/milknado/loop/_events.py", "duration_formatted"),
    ("src/milknado/loop/_events.py", "echo_stdout"),
    ("src/milknado/loop/_events.py", "echo_stderr"),
    ("src/milknado/loop/_events.py", "prompt_length"),
    ("src/milknado/loop/_events.py", "to_dict"),
    ("src/milknado/loop/_frontmatter.py", "FIELD_COMPLETION_SIGNAL"),
    ("src/milknado/loop/_frontmatter.py", "FIELD_STOP_ON_COMPLETION_SIGNAL"),
    ("src/milknado/loop/_frontmatter.py", "FIELD_MAX_TURNS"),
    ("src/milknado/loop/_frontmatter.py", "FIELD_MAX_TURNS_GRACE"),
    ("src/milknado/loop/_frontmatter.py", "FIELD_HOOKS"),
    ("src/milknado/loop/_frontmatter.py", "CMD_FIELD_NAME"),
    ("src/milknado/loop/_frontmatter.py", "CMD_FIELD_RUN"),
    ("src/milknado/loop/_frontmatter.py", "CMD_FIELD_TIMEOUT"),
    ("src/milknado/loop/_frontmatter.py", "HOOK_FIELD_EVENT"),
    ("src/milknado/loop/_frontmatter.py", "HOOK_FIELD_RUN"),
    ("src/milknado/loop/_frontmatter.py", "VALID_NAME_CHARS_MSG"),
    ("src/milknado/loop/_frontmatter.py", "serialize_frontmatter"),
    ("src/milknado/loop/_output.py", "format_count"),
    ("src/milknado/loop/adapters/_generic.py", "renders_structured_peek"),
    ("src/milknado/loop/adapters/_protocol.py", "renders_structured_peek"),
    ("src/milknado/loop/adapters/claude.py", "renders_structured_peek"),
    ("src/milknado/loop/adapters/codex.py", "renders_structured_peek"),
    ("src/milknado/loop/adapters/copilot.py", "renders_structured_peek"),
    ("src/milknado/loop/adapters/crush.py", "renders_structured_peek"),
    ("src/milknado/loop/adapters/opencode.py", "renders_structured_peek"),
    ("src/milknado/loop/manager.py", "add_listener"),
    ("src/milknado/loop/manager.py", "pause_run"),
    ("src/milknado/loop/manager.py", "resume_run"),
    ("src/milknado/loop/manager.py", "wait_for_any"),
    ("src/milknado/loop/manager.py", "wait_for_all"),
    ("src/milknado/loop/manager.py", "get_result"),
    ("src/milknado/loop/manager.py", "shutdown"),
}

# Textual App/Widget/Screen lifecycle method names this classifier owner-qualifies:
# overridden on a class whose base is imported from a "textual." module.
_TEXTUAL_METHOD_EXACT = {"compose"}
_TEXTUAL_METHOD_PREFIXES = ("on_", "action_")
_TEXTUAL_CLASS_VARS = {"CSS", "DEFAULT_CSS", "BINDINGS"}
# self.<name> assignments Textual itself reads: App's window subtitle.
_TEXTUAL_APP_SELF_ATTRS = {"sub_title"}


@dataclass(frozen=True, slots=True)
class _Finding:
    path: Path
    first_line: int
    name: str
    typ: str
    report: str


def _findings(items: Sequence[Item]) -> tuple[_Finding, ...]:
    findings = []
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


def _has_textual_base(class_def: ast.ClassDef, imports: Mapping[str, str]) -> bool:
    return any(
        (resolved := _resolve_base(base, imports)) is not None and resolved.startswith("textual.")
        for base in class_def.bases
    )


def _is_textual_lifecycle_method(name: str) -> bool:
    return name in _TEXTUAL_METHOD_EXACT or name.startswith(_TEXTUAL_METHOD_PREFIXES)


def _method_start_line(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    return node.decorator_list[0].lineno if node.decorator_list else node.lineno


def _accepted_reason(finding: _Finding, module: ast.Module, imports: dict[str, str]) -> str | None:
    key = (finding.path.as_posix(), finding.name)
    if key in _VENDORED_LOOP_FINDINGS:
        return "vendored ralph-loop engine"

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
                if (
                    isinstance(item, ast.Assign)
                    and len(item.targets) == 1
                    and isinstance(item.targets[0], ast.Name)
                    and item.targets[0].id == finding.name
                    and item.lineno == finding.first_line
                ):
                    return "Textual lifecycle class attribute"
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
    unclassified = []
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
