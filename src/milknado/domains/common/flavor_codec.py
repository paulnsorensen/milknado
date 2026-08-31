"""Flavor and quality-gate models with their TOML boundary schemas."""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import cast

import msgspec
from msgspec import structs

from milknado.domains.common.agent_argv import ALLOWED_WORKER_EXECUTABLES
from milknado.domains.common.paths import resolve_project_path, trust_global_path


class Gate(msgspec.Struct, frozen=True, kw_only=True):
    command: str
    fail_on_stdout: str | None = None

    def __post_init__(self) -> None:
        if not self.command.strip():
            raise ValueError("command must be a non-empty string")
        pattern = self.fail_on_stdout
        if pattern is None:
            return
        pattern = pattern.strip()
        if pattern:
            try:
                _ = re.compile(pattern)
            except re.error as exc:
                raise ValueError("fail_on_stdout is not a valid regex") from exc
        structs.force_setattr(self, "fail_on_stdout", pattern or None)


def normalize_gate(value: object) -> dict[str, object]:
    if isinstance(value, str):
        if not value.strip():
            raise ValueError("command must be a non-empty string")
        return {"command": value}
    if not isinstance(value, dict):
        raise ValueError(f"must be a string or a table with 'command', got {type(value).__name__}")
    return cast(dict[str, object], value)


def normalize_gates(value: object, ctx: str = "quality_gates") -> list[dict[str, object]] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"{ctx} must be a list (use [] to explicitly skip gates)")
    try:
        return [normalize_gate(gate) for gate in cast(list[object], value)]
    except ValueError as exc:
        raise ValueError(f"{ctx}: {exc}") from exc


def serialize_gates(gates: tuple[Gate, ...]) -> list[object]:
    return [
        gate.command
        if gate.fail_on_stdout is None
        else {"command": gate.command, "fail_on_stdout": gate.fail_on_stdout}
        for gate in gates
    ]


def coerce_tool_list(value: object, ctx: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise ValueError(f"{ctx} must be a list of strings, got {type(value).__name__}")
    values = cast(list[object] | tuple[object, ...], value)
    items: list[str] = []
    sentinels = 0
    for index, item in enumerate(values):
        if not isinstance(item, str) or not item:
            raise ValueError(f"{ctx}[{index}] must be a non-empty string")
        if item == "...":
            sentinels += 1
            if sentinels > 1:
                raise ValueError(f'{ctx} may contain at most one "..." sentinel, found multiple')
        items.append(item)
    return tuple(items)


def validate_loop_mode(value: object) -> str:
    mode = str(value)
    if mode not in ("redispatch", "single"):
        raise ValueError("loop_mode must be one of ['redispatch', 'single']")
    return mode


def validate_session_mode(value: object) -> str:
    mode = str(value)
    if mode not in ("fresh", "resume"):
        raise ValueError("session_mode must be one of ['fresh', 'resume']")
    return mode


def validate_on_reject(value: object) -> str:
    policy = str(value)
    if policy not in ("block", "warn"):
        raise ValueError("on_reject must be one of ['block', 'warn']")
    return policy


def validate_positive_int(value: object, ctx: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{ctx} must be an integer")
    if value < 1:
        raise ValueError(f"{ctx} must be >= 1")
    return value


class _FlavorFields(msgspec.Struct, frozen=True, kw_only=True):
    """Fields shared by ``FlavorOverride`` and ``FlavorTable``."""

    execution_agent: str | None = None
    tools: tuple[str, ...] | None = None
    brief_prepend: str | None = None
    quality_gates: tuple[Gate, ...] | None = None
    agent_type: str | None = None
    loop_mode: str | None = None
    max_iterations: int | None = None
    max_turns: int | None = None
    worktree: bool | None = None
    session_mode: str = "fresh"
    review: bool | None = None
    review_agent: str | None = None
    review_max_rounds: int = 2
    on_reject: str = "block"


class FlavorOverride(_FlavorFields, frozen=True, kw_only=True):
    """Validated per-flavor override, hydrated from a ``FlavorTable``."""


class FlavorTable(_FlavorFields, frozen=True, kw_only=True):
    """Schema for one ``[milknado.flavor.<name>]`` TOML table."""

    brief_prepend_path: str | list[str] | None = None

    def __post_init__(self) -> None:
        if self.execution_agent is not None:
            argv = shlex.split(self.execution_agent)
            if argv and Path(argv[0]).name not in ALLOWED_WORKER_EXECUTABLES:
                allowed = sorted(ALLOWED_WORKER_EXECUTABLES)
                raise ValueError(f"execution_agent must start with one of {allowed!r}")
        if self.brief_prepend is not None and self.brief_prepend_path is not None:
            raise ValueError("brief_prepend and brief_prepend_path are mutually exclusive")
        _ = validate_loop_mode(self.loop_mode) if self.loop_mode is not None else None
        _ = validate_session_mode(self.session_mode)
        _ = validate_on_reject(self.on_reject)
        _ = validate_positive_int(self.max_iterations, "max_iterations")
        _ = validate_positive_int(self.max_turns, "max_turns")
        _ = validate_positive_int(self.review_max_rounds, "review_max_rounds")
        if self.session_mode == "resume":
            self._reject_cursor_resume()

    def _reject_cursor_resume(self) -> None:
        for command in (self.execution_agent, self.review_agent):
            if command is None:
                continue
            argv = shlex.split(command)
            if argv and Path(argv[0]).name == "cursor-agent":
                raise ValueError(
                    "session_mode 'resume' is not supported for the cursor-agent family "
                    + "(headless resume mechanics are unverified — see "
                    + "adversarial-review-loops-F001); use session_mode = 'fresh' or a "
                    + "different agent"
                )

    def to_override(self, name: str, project_root: Path) -> FlavorOverride:
        return FlavorOverride(
            execution_agent=self.execution_agent,
            tools=self.tools,
            brief_prepend=self._load_brief(name, project_root),
            quality_gates=self.quality_gates,
            agent_type=self.agent_type,
            loop_mode=self.loop_mode,
            max_iterations=self.max_iterations,
            max_turns=self.max_turns,
            worktree=self.worktree,
            session_mode=self.session_mode,
            review=self.review,
            review_agent=self.review_agent,
            review_max_rounds=self.review_max_rounds,
            on_reject=self.on_reject,
        )

    def _load_brief(self, name: str, project_root: Path) -> str | None:
        if self.brief_prepend is not None:
            return self.brief_prepend.strip() or None
        if self.brief_prepend_path is None:
            return None
        ctx = f"[milknado.flavor.{name}]"
        values = (
            [self.brief_prepend_path]
            if isinstance(self.brief_prepend_path, str)
            else self.brief_prepend_path
        )
        parts: list[str] = []
        for value in values:
            path = resolve_project_path(value, project_root, label=f"{ctx} brief_prepend_path")
            if not path.exists():
                raise FileNotFoundError(f"{ctx} brief_prepend_path does not exist: {path}")
            parts.append(path.read_text(encoding="utf-8").strip())
        return "\n\n".join(part for part in parts if part) or None


def normalize_flavor_table(value: object) -> object:
    if not isinstance(value, dict):
        return value
    normalized = dict(cast(dict[str, object], value))
    for key in ("execution_agent", "brief_prepend", "agent_type", "review_agent"):
        field = normalized.get(key)
        if field is not None and not isinstance(field, str):
            raise ValueError(f"{key} must be a string")
    for key, validator in (
        ("loop_mode", validate_loop_mode),
        ("session_mode", validate_session_mode),
        ("on_reject", validate_on_reject),
    ):
        if key in normalized:
            if not isinstance(normalized[key], str):
                raise ValueError(f"{key} must be a string")
            _ = validator(normalized[key])
    for key in ("review", "worktree"):
        field = normalized.get(key)
        if field is not None and not isinstance(field, bool):
            raise ValueError(f"{key} must be a boolean")
    for key in ("max_iterations", "max_turns", "review_max_rounds"):
        if key in normalized:
            _ = validate_positive_int(normalized[key], key)
    path = normalized.get("brief_prepend_path")
    if path is not None and not (
        isinstance(path, str)
        or isinstance(path, list)
        and all(isinstance(item, str) for item in cast(list[object], path))
    ):
        raise ValueError("brief_prepend_path must be a string or list of strings")
    if "tools" in normalized and normalized["tools"] is not None:
        normalized["tools"] = coerce_tool_list(normalized["tools"], "tools")
    if "quality_gates" in normalized:
        normalized["quality_gates"] = normalize_gates(normalized["quality_gates"])
    return normalized


def serialize_flavor_tables(
    flavors: dict[str, FlavorOverride],
) -> dict[str, dict[str, object]]:
    tables: dict[str, dict[str, object]] = {}
    for name, flavor in flavors.items():
        entry: dict[str, object] = {
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
                ("session_mode", flavor.session_mode),
                ("review", flavor.review),
                ("review_agent", flavor.review_agent),
                ("review_max_rounds", flavor.review_max_rounds),
                ("on_reject", flavor.on_reject),
            )
            if value is not None
        }
        if entry:
            tables[name] = entry
    return tables


def absolutize_global_flavor_paths(raw: dict[str, object], base_dir: Path) -> None:
    flavors = raw.get("flavor")
    if not isinstance(flavors, dict):
        return
    flavors = cast(dict[str, object], flavors)
    for entry in flavors.values():
        if not isinstance(entry, dict):
            continue
        entry = cast(dict[str, object], entry)
        value = entry.get("brief_prepend_path")
        if isinstance(value, str):
            entry["brief_prepend_path"] = trust_global_path(value, base_dir)
        elif isinstance(value, list):
            items = cast(list[object], value)
            entry["brief_prepend_path"] = [
                trust_global_path(item, base_dir) if isinstance(item, str) else item
                for item in items
            ]
