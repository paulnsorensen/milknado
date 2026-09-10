from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import msgspec
import pytest

from milknado.domains.common import SessionEvent
from milknado.loop.sessions._codex import CodexSession
from milknado.loop.sessions._codex_policy import translate_argv


def wire(command: bytes) -> dict[str, object]:
    return msgspec.json.decode(command, type=dict[str, object])


def mapping(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def test_translate_argv_resolves_relative_workspaces_from_session_cwd(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    shared = tmp_path / "shared"
    root.mkdir()
    shared.mkdir()

    policy = translate_argv(
        ("codex", "--cd", "repo", "--add-dir", "../shared"),
        tmp_path,
    )

    assert policy.cwd == root.resolve()
    assert policy.thread == {"cwd": str(root.resolve()), "sandbox": "workspace-write"}
    assert policy.turn == {
        "sandboxPolicy": {
            "type": "workspaceWrite",
            "writableRoots": [str(root.resolve()), str(shared.resolve())],
        }
    }


@pytest.mark.parametrize(
    ("flag", "expected"),
    (
        (
            "--full-auto",
            {"sandbox": "workspace-write", "approvalPolicy": "on-request"},
        ),
        (
            "--approve-for-me",
            {
                "sandbox": "workspace-write",
                "approvalPolicy": "on-request",
                "approvalsReviewer": "auto_review",
            },
        ),
        (
            "--dangerously-bypass-approvals-and-sandbox",
            {"sandbox": "danger-full-access", "approvalPolicy": "never"},
        ),
    ),
)
def test_explicit_codex_automation_flags_preserve_their_policy(
    tmp_path: Path, flag: str, expected: dict[str, object]
) -> None:
    policy = translate_argv(("codex", flag), tmp_path)

    assert policy.thread == {"cwd": str(tmp_path.resolve()), **expected}
    assert policy.turn == {}


def test_resume_requires_a_non_option_thread_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires an explicit thread id"):
        _ = translate_argv(("codex", "resume", "--model", "gpt-5.6"), tmp_path)


def test_resume_preserves_cwd_workspace_and_schema_in_wire_commands(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    extra = tmp_path / "extra"
    schema_path = tmp_path / "schema.json"
    root.mkdir()
    extra.mkdir()
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    _ = schema_path.write_text(json.dumps(schema), encoding="utf-8")

    session = CodexSession(
        (
            "codex",
            "resume",
            "thread-old",
            "--cd",
            str(root),
            "--add-dir",
            str(extra),
            "--output-schema",
            str(schema_path),
        ),
        tmp_path,
    )

    initialize = wire(session.start("continue").commands[0])
    thread_step = session.receive(
        msgspec.json.encode({"id": initialize["id"], "result": {}}) + b"\n"
    )
    assert wire(thread_step.commands[0]) == {"method": "initialized", "params": {}}
    thread = wire(thread_step.commands[1])
    assert thread == {
        "method": "thread/resume",
        "id": thread["id"],
        "params": {
            "threadId": "thread-old",
            "cwd": str(root.resolve()),
            "sandbox": "workspace-write",
        },
    }

    started = session.receive(
        msgspec.json.encode(
            {
                "id": thread["id"],
                "result": {"thread": {"id": "thread-old", "sessionId": "session-old"}},
            }
        )
        + b"\n"
    )
    turn = wire(started.commands[0])
    assert turn["method"] == "turn/start"
    assert turn["params"] == {
        "threadId": "thread-old",
        "input": [{"type": "text", "text": "continue"}],
        "clientUserMessageId": str(turn["id"]),
        "outputSchema": schema,
        "sandboxPolicy": {
            "type": "workspaceWrite",
            "writableRoots": [str(root.resolve()), str(extra.resolve())],
        },
    }
    assert started.events == (
        SessionEvent(kind="user", text="continue", event_id=str(turn["id"]), state="submitted"),
    )


def test_output_schema_is_loaded_from_an_object_and_sent_only_on_turn(tmp_path: Path) -> None:
    schema_path = tmp_path / "schema.json"
    schema = {"type": "string", "enum": ["ok", "done"]}
    _ = schema_path.write_text(json.dumps(schema), encoding="utf-8")
    session = CodexSession(("codex", "--output-schema", str(schema_path)), tmp_path)

    initialize = wire(session.start("prompt").commands[0])
    thread_step = session.receive(
        msgspec.json.encode({"id": initialize["id"], "result": {}}) + b"\n"
    )
    thread = wire(thread_step.commands[1])
    thread_params = mapping(thread["params"])
    assert "outputSchema" not in thread_params

    started = session.receive(
        msgspec.json.encode(
            {
                "id": thread["id"],
                "result": {"thread": {"id": "thread-1", "sessionId": "session-1"}},
            }
        )
        + b"\n"
    )
    turn = wire(started.commands[0])
    turn_params = mapping(turn["params"])
    assert turn_params["outputSchema"] == schema


@pytest.mark.parametrize(
    ("filename", "contents", "message"),
    (
        ("missing.json", None, "invalid Codex output schema"),
        ("broken.json", "{", "invalid Codex output schema"),
        ("array.json", "[]", "must be a JSON object"),
    ),
)
def test_malformed_output_schema_fails_closed(
    tmp_path: Path, filename: str, contents: str | None, message: str
) -> None:
    schema_path = tmp_path / filename
    if contents is not None:
        _ = schema_path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        _ = CodexSession(("codex", "--output-schema", str(schema_path)), tmp_path)


@pytest.mark.parametrize("sandbox", ("read-only", "danger-full-access"))
def test_additional_workspace_cannot_broaden_non_write_sandbox(
    tmp_path: Path, sandbox: str
) -> None:
    with pytest.raises(ValueError, match="requires --sandbox workspace-write"):
        _ = translate_argv(("codex", "--sandbox", sandbox, "--add-dir", "extra"), tmp_path)


@pytest.mark.parametrize(
    ("argv", "message"),
    (
        (("codex", "--sandbox", "network"), "unsupported Codex sandbox mode"),
        (("codex", "--approval-policy", "sometimes"), "unsupported Codex approval policy"),
        (("codex", "--profile", "fast"), "unsupported Codex app-server flag"),
        (("codex", "--profile=fast"), "unsupported Codex app-server flag"),
        (("codex", "--listen", "tcp://"), "requires the app-server stdio transport"),
        (("codex", "--add-dir", "--sandbox"), "--add-dir requires a value"),
    ),
)
def test_unsupported_or_malformed_options_fail_closed(
    tmp_path: Path, argv: tuple[str, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _ = translate_argv(argv, tmp_path)


def test_conflicting_permission_options_fail_instead_of_broadening_access(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="conflicting Codex options"):
        _ = translate_argv(("codex", "--sandbox", "read-only", "--full-auto"), tmp_path)
