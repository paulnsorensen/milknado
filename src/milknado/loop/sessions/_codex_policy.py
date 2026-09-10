from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast


@dataclass(frozen=True, slots=True)
class CodexPolicy:
    command: tuple[str, ...]
    cwd: Path
    resume_id: str | None
    thread: dict[str, object]
    turn: dict[str, object]
    images: tuple[str, ...]


@dataclass(slots=True)
class _Options:
    root: Path
    values: dict[str, object]
    passthrough: list[str]
    images: list[str]
    extra_roots: list[str]
    output_schema: dict[str, object] | None = None


def _value(args: list[str], index: int, option: str) -> tuple[str, int]:
    if index + 1 >= len(args) or args[index + 1].startswith("-"):
        raise ValueError(f"{option} requires a value")
    return args[index + 1], index + 2


def _set(values: dict[str, object], key: str, value: object, option: str) -> None:
    previous = values.get(key)
    if previous is not None and previous != value:
        raise ValueError(f"conflicting Codex options for {option}")
    values[key] = value


def _split(argument: str, names: tuple[str, ...]) -> tuple[str, str] | None:
    for name in names:
        prefix = f"{name}="
        if argument.startswith(prefix):
            return name, argument[len(prefix) :]
    return None


def _path(root: Path, value: str) -> str:
    candidate = Path(value).expanduser()
    return str(candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve())


def _load_schema(value: str) -> dict[str, object]:
    try:
        loaded = cast(object, json.loads(Path(value).expanduser().read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid Codex output schema: {value!r}") from exc
    if not isinstance(loaded, dict):
        raise ValueError("Codex output schema must be a JSON object")
    return cast(dict[str, object], loaded)


def _apply_value(option: str, value: str, state: _Options) -> None:
    if option in {"--model", "-m"}:
        _set(state.values, "model", value, option)
    elif option in {"--sandbox", "-s"}:
        if value not in {"read-only", "workspace-write", "danger-full-access"}:
            raise ValueError(f"unsupported Codex sandbox mode: {value!r}")
        _set(state.values, "sandbox", value, option)
    elif option in {"--ask-for-approval", "--approval-policy"}:
        if value == "on-failure":
            raise ValueError(
                "Codex app-server has no equivalent for --ask-for-approval on-failure"
            )
        if value not in {"untrusted", "on-request", "never"}:
            raise ValueError(f"unsupported Codex approval policy: {value!r}")
        _set(state.values, "approvalPolicy", value, option)
    elif option in {"--cd", "-C"}:
        state.root = Path(_path(state.root, value))
        state.values["cwd"] = str(state.root)
    elif option == "--add-dir":
        resolved = _path(state.root, value)
        if resolved not in state.extra_roots:
            state.extra_roots.append(resolved)
    elif option == "--thread-source":
        _set(state.values, "threadSource", value, option)
    elif option == "--output-schema":
        state.output_schema = _load_schema(value)
    elif option in {"--image", "-i"}:
        state.images.append(_path(state.root, value))
    elif option == "--listen":
        if value != "stdio://":
            raise ValueError("CodexSession requires the app-server stdio transport")
    else:
        raise ValueError(f"unsupported Codex app-server flag: {option}")


def _consume_value(args: list[str], index: int, state: _Options) -> int | None:
    names = (
        "--model",
        "-m",
        "--sandbox",
        "-s",
        "--ask-for-approval",
        "--approval-policy",
        "--cd",
        "-C",
        "--add-dir",
        "--thread-source",
        "--output-schema",
        "--image",
        "-i",
        "--listen",
    )
    inline = _split(args[index], names)
    if inline is not None:
        option, value = inline
        if not value:
            raise ValueError(f"{option} requires a value")
        _apply_value(option, value, state)
        return index + 1
    if args[index] not in names:
        return None
    value, next_index = _value(args, index, args[index])
    _apply_value(args[index], value, state)
    return next_index


def _consume_simple(argument: str, state: _Options) -> bool:
    if argument == "--full-auto":
        _set(state.values, "sandbox", "workspace-write", argument)
        _set(state.values, "approvalPolicy", "on-request", argument)
    elif argument == "--approve-for-me":
        _set(state.values, "sandbox", "workspace-write", argument)
        _set(state.values, "approvalPolicy", "on-request", argument)
        _set(state.values, "approvalsReviewer", "auto_review", argument)
    elif argument == "--dangerously-bypass-approvals-and-sandbox":
        _set(state.values, "sandbox", "danger-full-access", argument)
        _set(state.values, "approvalPolicy", "never", argument)
    elif argument in {"--json", "--stdio"}:
        pass
    elif argument == "--strict-config" or argument == "--analytics-default-enabled":
        state.passthrough.append(argument)
    else:
        return False
    return True


def _consume_passthrough(args: list[str], index: int, state: _Options) -> int | None:
    argument = args[index]
    if argument in {"-c", "--config", "--enable", "--disable"}:
        value, next_index = _value(args, index, argument)
        state.passthrough.extend((argument, value))
        return next_index
    if argument.startswith(("--config=", "--enable=", "--disable=")):
        state.passthrough.append(argument)
        return index + 1
    return None


def _validate_policy(state: _Options) -> dict[str, object]:
    if state.extra_roots:
        sandbox = state.values.get("sandbox")
        if sandbox is None:
            state.values["sandbox"] = "workspace-write"
            sandbox = "workspace-write"
        if sandbox != "workspace-write":
            raise ValueError("Codex --add-dir requires --sandbox workspace-write")
    turn: dict[str, object] = {}
    if state.output_schema is not None:
        turn["outputSchema"] = state.output_schema
    if state.extra_roots:
        turn["sandboxPolicy"] = {
            "type": "workspaceWrite",
            "writableRoots": [str(state.root), *state.extra_roots],
        }
    return turn


def _unsupported(argument: str) -> None:
    unsupported = {
        "--profile",
        "-p",
        "--oss",
        "--local-provider",
        "--skip-git-repo-check",
        "--ignore-user-config",
        "--ignore-rules",
        "--color",
        "-o",
        "--output-last-message",
    }
    if argument in unsupported or argument.startswith("--profile="):
        raise ValueError(f"unsupported Codex app-server flag: {argument}")


def translate_argv(argv: tuple[str, ...], cwd: Path) -> CodexPolicy:
    if not argv or Path(argv[0]).stem != "codex":
        raise ValueError("CodexSession requires a codex executable")
    args = list(argv[1:])
    if args and args[0] in {"exec", "app-server"}:
        _ = args.pop(0)
    resume_id: str | None = None
    if args and args[0] == "resume":
        _ = args.pop(0)
        if not args or args[0].startswith("-"):
            raise ValueError("Codex resume requires an explicit thread id")
        resume_id, args = args[0], args[1:]
    elif args and args[0] in {"fork", "review"}:
        raise ValueError(f"Codex app-server does not support exec {args[0]} here")

    root = Path(cwd).expanduser().resolve()
    state = _Options(root, {"cwd": str(root)}, [], [], [])
    index = 0
    while index < len(args):
        next_index = _consume_value(args, index, state)
        if next_index is not None:
            index = next_index
            continue
        if _consume_simple(args[index], state):
            index += 1
            continue
        next_index = _consume_passthrough(args, index, state)
        if next_index is not None:
            index = next_index
            continue
        _unsupported(args[index])
        if args[index].startswith("-"):
            raise ValueError(f"unsupported Codex app-server flag: {args[index]}")
        raise ValueError("CodexSession receives the prompt through start(), not argv")

    turn = _validate_policy(state)
    command = (argv[0], "app-server", *state.passthrough)
    return CodexPolicy(
        command, state.root, resume_id, dict(state.values), turn, tuple(state.images)
    )
