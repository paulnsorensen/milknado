from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from milknado.domains.common.agent_argv import ALLOWED_WORKER_EXECUTABLES


@dataclass(frozen=True)
class Gate:
    command: str
    fail_on_stdout: str | None = None


@dataclass(frozen=True)
class FlavorOverride:
    execution_agent: str | None = None
    tools: tuple[str, ...] | None = None
    brief_prepend: str | None = None
    quality_gates: tuple[Gate, ...] | None = None
    agent_type: str | None = None
    loop_mode: str | None = None
    max_iterations: int | None = None
    max_turns: int | None = None
    worktree: bool | None = None


def serialize_gates(gates: tuple[Gate, ...]) -> list[Any]:
    return [
        gate.command
        if gate.fail_on_stdout is None
        else {"command": gate.command, "fail_on_stdout": gate.fail_on_stdout}
        for gate in gates
    ]


def parse_gates(raw: Any, ctx: str) -> tuple[Gate, ...] | None:
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ValueError(f"{ctx} must be a list (use [] to explicitly skip gates)")
    return tuple(_parse_gate_entry(entry, ctx, i) for i, entry in enumerate(raw))


def _parse_gate_entry(entry: Any, ctx: str, index: int) -> Gate:
    if isinstance(entry, str):
        if not entry.strip():
            raise ValueError(f"{ctx}[{index}] must be a non-empty string")
        return Gate(command=entry)
    if not isinstance(entry, dict):
        raise ValueError(
            f"{ctx}[{index}] must be a string or a table with 'command', "
            f"got {type(entry).__name__}"
        )
    command = entry.get("command")
    if not isinstance(command, str) or not command.strip():
        raise ValueError(f"{ctx}[{index}].command must be a non-empty string")
    pattern = entry.get("fail_on_stdout")
    if pattern is not None:
        if not isinstance(pattern, str):
            raise ValueError(f"{ctx}[{index}].fail_on_stdout must be a string")
        pattern = pattern.strip() or None
        if pattern is not None:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(
                    f"{ctx}[{index}].fail_on_stdout is not a valid regex: {exc}"
                ) from exc
    return Gate(command=command, fail_on_stdout=pattern)


def coerce_tool_list(value: Any, ctx: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise ValueError(f"{ctx} must be a list of strings, got {type(value).__name__}")
    items: list[str] = []
    sentinels = 0
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item:
            raise ValueError(f"{ctx}[{index}] must be a non-empty string, got {item!r}")
        if item == "...":
            sentinels += 1
            if sentinels > 1:
                raise ValueError(f'{ctx} may contain at most one "..." sentinel, found multiple')
        items.append(item)
    return tuple(items)


def validate_loop_mode(value: Any, ctx: str) -> str:
    mode = str(value)
    if mode not in ("redispatch", "single"):
        raise ValueError(f"{ctx} loop_mode must be one of ['redispatch', 'single']; got {mode!r}")
    return mode


def validate_positive_int(value: Any, ctx: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{ctx} must be an integer")
    if value < 1:
        raise ValueError(f"{ctx} must be >= 1; got {value}")
    return value


def parse_flavor_tables(flavor_raw: Any, project_root: Path) -> dict[str, FlavorOverride]:
    if flavor_raw is None:
        return {}
    if not isinstance(flavor_raw, dict):
        raise ValueError("[milknado.flavor] must be a table")
    flavors: dict[str, FlavorOverride] = {}
    for name, entry in flavor_raw.items():
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"[milknado.flavor] flavor keys must be non-empty strings, got {name!r}"
            )
        if not isinstance(entry, dict):
            raise ValueError(f"[milknado.flavor.{name}] must be a table")
        flavors[name] = _parse_flavor_entry(entry, name, project_root)
    return flavors


def _parse_flavor_entry(entry: dict[str, Any], name: str, project_root: Path) -> FlavorOverride:
    ctx = f"[milknado.flavor.{name}]"
    execution_agent = entry.get("execution_agent")
    if execution_agent is not None:
        if not isinstance(execution_agent, str):
            raise ValueError(f"{ctx} execution_agent must be a string")
        argv = shlex.split(execution_agent)
        if argv and Path(argv[0]).name not in ALLOWED_WORKER_EXECUTABLES:
            raise ValueError(
                f"{ctx} execution_agent must start with one of "
                f"{sorted(ALLOWED_WORKER_EXECUTABLES)!r}; got {Path(argv[0]).name!r}"
            )
    tools_raw = entry.get("tools")
    tools = coerce_tool_list(tools_raw, f"{ctx} tools") if tools_raw is not None else None
    agent_type = entry.get("agent_type")
    if agent_type is not None and not isinstance(agent_type, str):
        raise ValueError(f"{ctx} agent_type must be a string")
    loop_raw = entry.get("loop_mode")
    loop_mode = validate_loop_mode(loop_raw, ctx) if loop_raw is not None else None
    worktree = entry.get("worktree")
    if worktree is not None and not isinstance(worktree, bool):
        raise ValueError(f"{ctx} worktree must be a boolean")
    return FlavorOverride(
        execution_agent=execution_agent,
        tools=tools,
        brief_prepend=_load_flavor_brief(entry, name, project_root),
        quality_gates=parse_gates(entry.get("quality_gates"), f"{ctx} quality_gates"),
        agent_type=agent_type,
        loop_mode=loop_mode,
        max_iterations=validate_positive_int(entry.get("max_iterations"), f"{ctx} max_iterations"),
        max_turns=validate_positive_int(entry.get("max_turns"), f"{ctx} max_turns"),
        worktree=worktree,
    )


def _load_flavor_brief(entry: dict[str, Any], name: str, project_root: Path) -> str | None:
    ctx = f"[milknado.flavor.{name}]"
    inline = entry.get("brief_prepend")
    paths_raw = entry.get("brief_prepend_path")
    if inline is not None and not isinstance(inline, str):
        raise ValueError(f"{ctx} brief_prepend must be a string")
    if inline is not None and paths_raw is not None:
        raise ValueError(f"{ctx} brief_prepend and brief_prepend_path are mutually exclusive")
    if inline is not None:
        return inline.strip() or None
    if paths_raw is None:
        return None
    paths = [paths_raw] if isinstance(paths_raw, str) else paths_raw
    if not isinstance(paths, list):
        raise ValueError(f"{ctx} brief_prepend_path must be a string or list of strings")
    parts: list[str] = []
    for value in paths:
        if not isinstance(value, str):
            raise ValueError(f"{ctx} brief_prepend_path entries must be strings")
        path = Path(value)
        if not path.is_absolute():
            path = project_root / path
        if not path.exists():
            raise FileNotFoundError(f"{ctx} brief_prepend_path does not exist: {path}")
        parts.append(path.read_text(encoding="utf-8").strip())
    return "\n\n".join(part for part in parts if part) or None


def serialize_flavor_tables(
    flavors: dict[str, FlavorOverride],
) -> dict[str, dict[str, Any]]:
    tables: dict[str, dict[str, Any]] = {}
    for name, flavor in flavors.items():
        entry = {
            key: value
            for key, value in (
                ("execution_agent", flavor.execution_agent),
                ("tools", list(flavor.tools) if flavor.tools is not None else None),
                ("brief_prepend", flavor.brief_prepend),
                (
                    "quality_gates",
                    serialize_gates(flavor.quality_gates)
                    if flavor.quality_gates is not None
                    else None,
                ),
                ("agent_type", flavor.agent_type),
                ("loop_mode", flavor.loop_mode),
                ("max_iterations", flavor.max_iterations),
                ("max_turns", flavor.max_turns),
                ("worktree", flavor.worktree),
            )
            if value is not None
        }
        if entry:
            tables[name] = entry
    return tables


def absolutize_global_flavor_paths(raw: dict[str, Any], base_dir: Path) -> None:
    flavors = raw.get("flavor")
    if not isinstance(flavors, dict):
        return
    for entry in flavors.values():
        if not isinstance(entry, dict):
            continue
        value = entry.get("brief_prepend_path")
        if isinstance(value, str) and not Path(value).is_absolute():
            entry["brief_prepend_path"] = str((base_dir / value).resolve())
        elif isinstance(value, list):
            entry["brief_prepend_path"] = [
                str((base_dir / item).resolve())
                if isinstance(item, str) and not Path(item).is_absolute()
                else item
                for item in value
            ]
