"""Flavor and quality-gate models with their TOML boundary schemas.

``Gate`` and ``FlavorOverride`` are the validated records the rest of the
system consumes. ``FlavorTable`` is the Pydantic schema for one
``[milknado.flavor.<name>]`` TOML table; it owns the flavor grammar (type
checks, literal vocabularies, mutual exclusion) and ``to_override`` hydrates
file-backed fields into a ``FlavorOverride``.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from milknado.domains.common.agent_argv import ALLOWED_WORKER_EXECUTABLES
from milknado.domains.common.paths import resolve_project_path, trust_global_path


class Gate(BaseModel):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)

    command: str
    fail_on_stdout: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _shorthand(cls, value: Any) -> Any:
        if isinstance(value, str):
            if not value.strip():
                raise ValueError("command must be a non-empty string")
            return {"command": value}
        if not isinstance(value, dict):
            raise ValueError(
                f"must be a string or a table with 'command', got {type(value).__name__}"
            )
        return value

    @field_validator("command", mode="before")
    @classmethod
    def _command_present(cls, value: Any) -> Any:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("command must be a non-empty string")
        return value

    @field_validator("fail_on_stdout", mode="before")
    @classmethod
    def _pattern_valid(cls, value: Any) -> Any:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("fail_on_stdout must be a string")
        pattern = value.strip() or None
        if pattern is not None:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError("fail_on_stdout is not a valid regex") from exc
        return pattern


def serialize_gates(gates: tuple[Gate, ...]) -> list[Any]:
    return [
        gate.command
        if gate.fail_on_stdout is None
        else {"command": gate.command, "fail_on_stdout": gate.fail_on_stdout}
        for gate in gates
    ]


def coerce_tool_list(value: Any, ctx: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise ValueError(f"{ctx} must be a list of strings, got {type(value).__name__}")
    items: list[str] = []
    sentinels = 0
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item:
            raise ValueError(f"{ctx}[{index}] must be a non-empty string")
        if item == "...":
            sentinels += 1
            if sentinels > 1:
                raise ValueError(f'{ctx} may contain at most one "..." sentinel, found multiple')
        items.append(item)
    return tuple(items)


def validate_loop_mode(value: Any) -> str:
    mode = str(value)
    if mode not in ("redispatch", "single"):
        raise ValueError("loop_mode must be one of ['redispatch', 'single']")
    return mode


def validate_session_mode(value: Any) -> str:
    mode = str(value)
    if mode not in ("fresh", "resume"):
        raise ValueError("session_mode must be one of ['fresh', 'resume']")
    return mode


def validate_on_reject(value: Any) -> str:
    policy = str(value)
    if policy not in ("block", "warn"):
        raise ValueError("on_reject must be one of ['block', 'warn']")
    return policy


def validate_positive_int(value: Any, ctx: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{ctx} must be an integer")
    if value < 1:
        raise ValueError(f"{ctx} must be >= 1")
    return value


class FlavorOverride(BaseModel):
    """Validated per-flavor override, hydrated from a ``FlavorTable``."""

    model_config = ConfigDict(frozen=True)

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


class FlavorTable(BaseModel):
    """Schema for one ``[milknado.flavor.<name>]`` TOML table."""

    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)

    execution_agent: str | None = None
    tools: tuple[str, ...] | None = None
    brief_prepend: str | None = None
    brief_prepend_path: str | list[str] | None = None
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

    @field_validator("execution_agent", mode="before")
    @classmethod
    def _execution_agent_allowed(cls, value: Any) -> Any:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("execution_agent must be a string")
        argv = shlex.split(value)
        if argv and Path(argv[0]).name not in ALLOWED_WORKER_EXECUTABLES:
            raise ValueError(
                f"execution_agent must start with one of {sorted(ALLOWED_WORKER_EXECUTABLES)!r}"
            )
        return value

    @field_validator("tools", mode="before")
    @classmethod
    def _tools_valid(cls, value: Any) -> Any:
        if value is None:
            return None
        return coerce_tool_list(value, "tools")

    @field_validator("brief_prepend", mode="before")
    @classmethod
    def _brief_str(cls, value: Any) -> Any:
        if value is not None and not isinstance(value, str):
            raise ValueError("brief_prepend must be a string")
        return value

    @field_validator("brief_prepend_path", mode="before")
    @classmethod
    def _brief_path_valid(cls, value: Any) -> Any:
        if value is None or isinstance(value, str):
            return value
        if not isinstance(value, list):
            raise ValueError("brief_prepend_path must be a string or list of strings")
        for item in value:
            if not isinstance(item, str):
                raise ValueError("brief_prepend_path entries must be strings")
        return value

    @field_validator("quality_gates", mode="before")
    @classmethod
    def _gates_list(cls, value: Any) -> Any:
        if value is not None and not isinstance(value, list):
            raise ValueError("quality_gates must be a list (use [] to explicitly skip gates)")
        return value

    @field_validator("agent_type", mode="before")
    @classmethod
    def _agent_type_str(cls, value: Any) -> Any:
        if value is not None and not isinstance(value, str):
            raise ValueError("agent_type must be a string")
        return value

    @field_validator("review_agent", mode="before")
    @classmethod
    def _review_agent_str(cls, value: Any) -> Any:
        if value is not None and not isinstance(value, str):
            raise ValueError("review_agent must be a string")
        return value

    @field_validator("loop_mode", mode="before")
    @classmethod
    def _loop_mode_valid(cls, value: Any) -> Any:
        if value is None:
            return None
        return validate_loop_mode(value)

    @field_validator("session_mode", mode="before")
    @classmethod
    def _session_mode_valid(cls, value: Any) -> str:
        return validate_session_mode(value)

    @field_validator("on_reject", mode="before")
    @classmethod
    def _on_reject_valid(cls, value: Any) -> str:
        return validate_on_reject(value)

    @field_validator("review", mode="before")
    @classmethod
    def _review_bool(cls, value: Any) -> Any:
        if value is not None and not isinstance(value, bool):
            raise ValueError("review must be a boolean")
        return value

    @field_validator("worktree", mode="before")
    @classmethod
    def _worktree_bool(cls, value: Any) -> Any:
        if value is not None and not isinstance(value, bool):
            raise ValueError("worktree must be a boolean")
        return value

    @field_validator("max_iterations", mode="before")
    @classmethod
    def _max_iterations_positive(cls, value: Any) -> Any:
        return validate_positive_int(value, "max_iterations")

    @field_validator("max_turns", mode="before")
    @classmethod
    def _max_turns_positive(cls, value: Any) -> Any:
        return validate_positive_int(value, "max_turns")

    @field_validator("review_max_rounds", mode="before")
    @classmethod
    def _review_max_rounds_positive(cls, value: Any) -> Any:
        return validate_positive_int(value, "review_max_rounds")

    @model_validator(mode="after")
    def _check_cross_field(self) -> FlavorTable:
        if self.brief_prepend is not None and self.brief_prepend_path is not None:
            raise ValueError("brief_prepend and brief_prepend_path are mutually exclusive")
        if self.session_mode == "resume":
            for command in (self.execution_agent, self.review_agent):
                if command is None:
                    continue
                argv = shlex.split(command)
                if argv and Path(argv[0]).name == "cursor-agent":
                    raise ValueError(
                        "session_mode 'resume' is not supported for the cursor-agent "
                        "family (headless resume mechanics are unverified — see "
                        "adversarial-review-loops-F001); use session_mode = 'fresh' or a "
                        "different agent"
                    )
        return self

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


def absolutize_global_flavor_paths(raw: dict[str, Any], base_dir: Path) -> None:
    flavors = raw.get("flavor")
    if not isinstance(flavors, dict):
        return
    for entry in flavors.values():
        if not isinstance(entry, dict):
            continue
        value = entry.get("brief_prepend_path")
        if isinstance(value, str):
            entry["brief_prepend_path"] = trust_global_path(value, base_dir)
        elif isinstance(value, list):
            entry["brief_prepend_path"] = [
                trust_global_path(item, base_dir) if isinstance(item, str) else item
                for item in value
            ]
