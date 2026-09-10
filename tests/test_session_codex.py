from __future__ import annotations

from pathlib import Path
from typing import cast

import msgspec
import pytest

from milknado.domains.common import SessionAction, SessionEvent, SessionInput
from milknado.domains.common.session import SessionKind
from milknado.loop.sessions._codex import CodexSession
from milknado.loop.sessions._protocol import ProtocolStep


def frame(payload: dict[str, object]) -> bytes:
    return msgspec.json.encode(payload) + b"\n"


def wire(command: bytes) -> dict[str, object]:
    return msgspec.json.decode(command, type=dict[str, object])


def mapping(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def last_event(step: ProtocolStep, kind: str, state: str | None = None) -> SessionEvent:
    events = [event for event in step.events if event.kind == kind]
    if state is not None:
        events = [event for event in events if event.state == state]
    assert events
    return events[-1]


def bootstrap(session: CodexSession) -> tuple[dict[str, object], str, dict[str, object]]:
    initialized = session.start("implement the change")
    initialize = wire(initialized.commands[0])
    assert initialize["method"] == "initialize"
    thread_step = session.receive(
        frame({"id": initialize["id"], "result": {"userAgent": "codex"}})
    )
    assert wire(thread_step.commands[0]) == {"method": "initialized", "params": {}}
    thread = wire(thread_step.commands[1])
    thread_step = session.receive(
        frame(
            {
                "id": thread["id"],
                "result": {"thread": {"id": "thread-1", "sessionId": "session-1"}},
            }
        )
    )
    turn = wire(thread_step.commands[0])
    assert turn["method"] == "turn/start"
    _ = session.receive(
        frame(
            {
                "id": turn["id"],
                "result": {"turn": {"id": "turn-1", "status": "inProgress"}},
            }
        )
    )
    return turn, "turn-1", thread


def test_initialize_starts_thread_and_turn_with_schema_fields(tmp_path: Path) -> None:
    session = CodexSession(("codex", "--model", "gpt-5.6"), tmp_path)

    turn, _, _ = bootstrap(session)

    params = mapping(turn["params"])
    assert params["threadId"] == "thread-1"
    assert params["input"] == [{"type": "text", "text": "implement the change"}]
    assert params["clientUserMessageId"]
    assert session.command == ("codex", "app-server")


def test_resume_uses_thread_resume_and_preserves_identity(tmp_path: Path) -> None:
    session = CodexSession(("codex", "resume", "thread-old"), tmp_path)
    initialized = session.start("continue")
    initialize = wire(initialized.commands[0])

    thread_step = session.receive(frame({"id": initialize["id"], "result": {}}))
    thread = wire(thread_step.commands[1])
    assert thread["method"] == "thread/resume"
    assert thread["params"] == {"threadId": "thread-old", "cwd": str(tmp_path)}

    started = session.receive(
        frame(
            {
                "id": thread["id"],
                "result": {"thread": {"id": "thread-old", "sessionId": "session-old"}},
            }
        )
    )
    assert started.session_id == "session-old"
    assert mapping(wire(started.commands[0])["params"])["threadId"] == "thread-old"


def test_policy_translation_keeps_sandbox_and_approval_constraints(tmp_path: Path) -> None:
    session = CodexSession(
        (
            "codex",
            "--model",
            "gpt-5.6",
            "--sandbox",
            "workspace-write",
            "--approval-policy",
            "on-request",
            "--add-dir",
            "../shared",
        ),
        tmp_path,
    )
    turn, _, thread = bootstrap(session)
    params = mapping(turn["params"])
    thread_params = mapping(thread["params"])
    assert thread_params["model"] == "gpt-5.6"
    assert thread_params["sandbox"] == "workspace-write"
    assert thread_params["approvalPolicy"] == "on-request"
    assert params["sandboxPolicy"] == {
        "type": "workspaceWrite",
        "writableRoots": [str(tmp_path.resolve()), str((tmp_path / "../shared").resolve())],
    }


def test_unsupported_policy_is_rejected_instead_of_dropped(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no equivalent"):
        _ = CodexSession(("codex", "--ask-for-approval", "on-failure"), tmp_path)


def test_active_steer_acknowledges_only_the_current_turn(tmp_path: Path) -> None:
    session = CodexSession(("codex",), tmp_path)
    _, turn_id, _ = bootstrap(session)

    submitted = session.submit(
        SessionInput(action="steer", text="also update tests", request_id="input-7")
    )
    command = wire(submitted.commands[0])
    assert command["method"] == "turn/steer"
    assert mapping(command["params"])["expectedTurnId"] == turn_id
    assert submitted.events == ()

    acknowledged = session.receive(frame({"id": command["id"], "result": {}}))
    assert last_event(acknowledged, "user", "delivered").text == "also update tests"

    stale = session.submit(SessionInput(action="steer", text="stale input", request_id="input-8"))
    stale_command = wire(stale.commands[0])
    _ = session.receive(
        frame(
            {
                "method": "turn/completed",
                "params": {"turn": {"id": turn_id, "status": "completed", "items": []}},
            }
        )
    )
    rejected = session.receive(frame({"id": stale_command["id"], "result": {}}))
    assert last_event(rejected, "user", "rejected").text == "stale input"
    assert all(event.state != "delivered" for event in rejected.events)


def test_steer_error_is_visible_and_does_not_end_session(tmp_path: Path) -> None:
    session = CodexSession(("codex",), tmp_path)
    _ = bootstrap(session)
    command = wire(
        session.submit(
            SessionInput(action="steer", text="reroute", request_id="input-9")
        ).commands[0]
    )

    failed = session.receive(
        frame(
            {
                "id": command["id"],
                "error": {"code": -32000, "message": "turn is no longer active"},
            }
        )
    )
    assert last_event(failed, "error").text == "turn is no longer active"
    assert last_event(failed, "user").state == "rejected"
    assert failed.done is False
    assert failed.failed is False


@pytest.mark.parametrize(
    ("will_retry", "done", "failed"),
    ((True, False, False), (False, True, True)),
)
def test_turn_error_is_visible_and_terminal_only_without_retry(
    tmp_path: Path, will_retry: bool, done: bool, failed: bool
) -> None:
    session = CodexSession(("codex",), tmp_path)
    _ = bootstrap(session)

    step = session.receive(
        frame(
            {
                "method": "error",
                "params": {
                    "turnId": "turn-1",
                    "error": {"message": "provider turn failed"},
                    "willRetry": will_retry,
                },
            }
        )
    )

    error = last_event(step, "error", "rejected")
    assert error.text == "provider turn failed"
    assert error.event_id == "turn-1"
    assert step.done is done
    assert step.failed is failed
    assert step.session_id == "session-1"
    if not will_retry:
        with pytest.raises(ValueError, match="not accepting input"):
            _ = session.submit(SessionInput(action="steer", text="late input"))


def test_interrupt_uses_turn_identity_and_reports_interrupted_completion(tmp_path: Path) -> None:
    session = CodexSession(("codex",), tmp_path)
    _, turn_id, _ = bootstrap(session)

    submitted = session.submit(SessionInput(action="interrupt", request_id="interrupt-1"))
    command = wire(submitted.commands[0])
    assert command == {
        "method": "turn/interrupt",
        "id": command["id"],
        "params": {"threadId": "thread-1", "turnId": turn_id},
    }
    completed = session.receive(
        frame(
            {
                "method": "turn/completed",
                "params": {"turn": {"id": turn_id, "status": "interrupted", "items": []}},
            }
        )
    )
    assert completed.done is True
    assert completed.interrupted is True
    assert completed.failed is False


@pytest.mark.parametrize(("action", "choice"), (("approve", "accept"), ("deny", "decline")))
def test_approval_reply_uses_exact_request_id_and_documented_choice(
    tmp_path: Path, action: SessionAction, choice: str
) -> None:
    session = CodexSession(("codex",), tmp_path)
    _ = bootstrap(session)
    requested = session.receive(
        frame(
            {
                "id": 41,
                "method": "item/commandExecution/requestApproval",
                "params": {"threadId": "thread-1", "turnId": "turn-1", "command": "ls"},
            }
        )
    )
    assert last_event(requested, "permission", "requested").event_id == "41"

    reply = session.submit(SessionInput(action=action, request_id="41"))
    assert wire(reply.commands[0]) == {"id": 41, "result": {"decision": choice}}


def test_terminal_tool_and_message_notifications_decode_to_observable_events(
    tmp_path: Path,
) -> None:
    session = CodexSession(("codex",), tmp_path)

    tool = session.receive(
        frame(
            {
                "method": "item/started",
                "params": {
                    "threadId": "thread-1",
                    "turnId": "turn-1",
                    "item": {"id": "item-1", "type": "commandExecution", "command": "pwd"},
                },
            }
        )
    )
    assert last_event(tool, "tool", "streaming").text == "pwd"

    terminal = session.receive(
        frame(
            {
                "method": "item/commandExecution/terminalInteraction",
                "params": {"itemId": "item-1", "processId": "proc-1", "stdin": "y\n"},
            }
        )
    )
    terminal_event = last_event(terminal, "tool", "streaming")
    assert terminal_event.event_id == "item-1"
    assert terminal_event.text == "y\n"
    assert terminal_event.delta is True

    message = session.receive(
        frame(
            {
                "method": "item/agentMessage/delta",
                "params": {"itemId": "message-1", "delta": "done"},
            }
        )
    )
    assert last_event(message, "assistant", "streaming").text == "done"
    complete = session.receive(
        frame(
            {
                "method": "item/completed",
                "params": {"item": {"id": "message-1", "type": "agentMessage", "text": "done"}},
            }
        )
    )
    assert last_event(complete, "assistant", "complete").text == "done"


def test_approval_reply_requires_exact_request_id_without_consuming_request(
    tmp_path: Path,
) -> None:
    session = CodexSession(("codex",), tmp_path)
    _ = bootstrap(session)
    _ = session.receive(
        frame(
            {
                "id": "approval-1",
                "method": "tool/requestUserInput",
                "params": {"message": "Choose a value"},
            }
        )
    )

    with pytest.raises(ValueError, match="unknown or already resolved"):
        _ = session.submit(SessionInput(action="approve", request_id="approval-2"))

    reply = session.submit(SessionInput(action="approve", request_id="approval-1"))
    assert wire(reply.commands[0]) == {
        "id": "approval-1",
        "result": {"answers": {}},
    }


def test_approval_choice_is_rejected_before_request_is_resolved(tmp_path: Path) -> None:
    session = CodexSession(("codex",), tmp_path)
    _ = bootstrap(session)
    _ = session.receive(
        frame(
            {
                "id": 42,
                "method": "item/commandExecution/requestApproval",
                "params": {"command": "rm -i file"},
            }
        )
    )

    with pytest.raises(ValueError, match="unsupported Codex approval choice"):
        _ = session.submit(SessionInput(action="approve", request_id="42", text="decline"))

    reply = session.submit(
        SessionInput(action="approve", request_id="42", text="acceptForSession")
    )
    assert wire(reply.commands[0]) == {
        "id": 42,
        "result": {"decision": "acceptForSession"},
    }


@pytest.mark.parametrize(
    "case",
    (
        (
            "item/fileChange/requestApproval",
            {"reason": "Apply the patch?"},
            "approve",
            "Apply the patch?",
            {"decision": "accept"},
        ),
        (
            "item/permissions/requestApproval",
            {
                "reason": "Need read access",
                "permissions": {"fileSystem": {"read": ["/repo"]}},
            },
            "approve",
            "Need read access",
            {"permissions": {"fileSystem": {"read": ["/repo"]}}},
        ),
        (
            "item/permissions/requestApproval",
            {
                "reason": "Need network access",
                "permissions": {"network": {"domains": ["example.com"]}},
            },
            "deny",
            "Need network access",
            {"permissions": {"fileSystem": None, "network": None}},
        ),
        (
            "tool/requestUserInput",
            {"message": "Select an option"},
            "approve",
            "Select an option",
            {"answers": {}},
        ),
        (
            "mcpServer/elicitation/request",
            {"message": "Authorize the tool?"},
            "deny",
            "Authorize the tool?",
            {"action": "decline", "content": None},
        ),
        (
            "execCommandApproval",
            {"command": "git status"},
            "approve",
            "git status",
            {"decision": "approved"},
        ),
        (
            "applyPatchApproval",
            {"reason": "Update source"},
            "deny",
            "Update source",
            {"decision": "denied"},
        ),
    ),
)
def test_supported_codex_approval_methods_preserve_request_and_reply(
    tmp_path: Path,
    case: tuple[str, dict[str, object], SessionAction, str, dict[str, object]],
) -> None:
    method, params, action, text, result = case
    session = CodexSession(("codex",), tmp_path)
    _ = bootstrap(session)
    requested = session.receive(frame({"id": "approval-1", "method": method, "params": params}))

    assert requested.events == (
        SessionEvent(
            kind="permission",
            text=text,
            event_id="approval-1",
            state="requested",
        ),
    )

    reply = session.submit(SessionInput(action=action, request_id="approval-1"))
    assert wire(reply.commands[0]) == {"id": "approval-1", "result": result}


@pytest.mark.parametrize(
    ("action", "state", "decision"),
    (("approve", "approved", "accept"), ("deny", "denied", "decline")),
)
def test_approval_resolution_publishes_approval_or_cancellation(
    tmp_path: Path, action: SessionAction, state: str, decision: str
) -> None:
    session = CodexSession(("codex",), tmp_path)
    _ = bootstrap(session)
    _ = session.receive(
        frame(
            {
                "id": "approval-1",
                "method": "item/commandExecution/requestApproval",
                "params": {"command": "git status"},
            }
        )
    )

    reply = session.submit(SessionInput(action=action, request_id="approval-1"))
    assert wire(reply.commands[0]) == {
        "id": "approval-1",
        "result": {"decision": decision},
    }

    resolved = session.receive(
        frame(
            {
                "method": "serverRequest/resolved",
                "params": {"requestId": "approval-1"},
            }
        )
    )
    assert resolved.events == (
        SessionEvent(
            kind="permission",
            text="git status",
            event_id="approval-1",
            state=state,
        ),
    )
    assert resolved.session_id == "session-1"
    cleared = session.receive(frame({"method": "serverRequest/resolved", "params": {}}))
    assert cleared.events == ()


def test_unsupported_codex_server_request_is_observable_and_terminal(
    tmp_path: Path,
) -> None:
    session = CodexSession(("codex",), tmp_path)
    _ = bootstrap(session)
    method = "item/input/requestApproval"

    rejected = session.receive(
        frame({"id": "unsupported-1", "method": method, "params": {"message": "input"}})
    )

    assert rejected.done is True
    assert rejected.failed is True
    error = last_event(rejected, "error")
    assert error.state == "rejected"
    assert method in error.text


def test_duplicate_codex_approval_ids_fail_closed(tmp_path: Path) -> None:
    session = CodexSession(("codex",), tmp_path)
    _ = bootstrap(session)
    _ = session.receive(
        frame(
            {
                "id": 7,
                "method": "item/commandExecution/requestApproval",
                "params": {"command": "pwd"},
            }
        )
    )

    rejected = session.receive(
        frame(
            {
                "id": "7",
                "method": "item/fileChange/requestApproval",
                "params": {"reason": "duplicate"},
            }
        )
    )

    assert rejected.done is True
    assert rejected.failed is True
    assert last_event(rejected, "error").text == "duplicate Codex approval request id '7'"


def test_wrong_turn_completion_does_not_misreport_current_turn_success(
    tmp_path: Path,
) -> None:
    session = CodexSession(("codex",), tmp_path)
    _, turn_id, _ = bootstrap(session)

    stale = session.receive(
        frame(
            {
                "method": "turn/completed",
                "params": {
                    "turn": {
                        "id": "turn-stale",
                        "status": "completed",
                        "items": [
                            {"id": "stale-message", "type": "agentMessage", "text": "stale"}
                        ],
                    }
                },
            }
        )
    )
    assert stale.events == ()
    assert stale.done is False
    assert stale.failed is False

    completed = session.receive(
        frame(
            {
                "method": "turn/completed",
                "params": {
                    "turn": {
                        "id": turn_id,
                        "status": "completed",
                        "items": [
                            {"id": "current-message", "type": "agentMessage", "text": "current"}
                        ],
                    }
                },
            }
        )
    )
    assert completed.done is True
    assert completed.failed is False
    assert completed.result_text == "current"


def test_non_object_codex_response_result_is_rejected_instead_of_accepted(
    tmp_path: Path,
) -> None:
    session = CodexSession(("codex",), tmp_path)
    initialize = wire(session.start("prompt").commands[0])

    rejected = session.receive(frame({"id": initialize["id"], "result": []}))

    assert rejected.done is True
    assert rejected.failed is True
    assert rejected.commands == ()
    assert last_event(rejected, "error").state == "rejected"
    assert "object" in last_event(rejected, "error").text


@pytest.mark.parametrize("line", (b"not-json\n", b"[]\n", b"{}\n"))
def test_malformed_codex_frames_become_terminal_errors(tmp_path: Path, line: bytes) -> None:
    session = CodexSession(("codex",), tmp_path)

    rejected = session.receive(line)

    assert rejected.done is True
    assert rejected.failed is True
    assert rejected.commands == ()
    assert last_event(rejected, "error").state == "rejected"


def test_missing_delta_item_id_is_not_silently_rendered(tmp_path: Path) -> None:
    session = CodexSession(("codex",), tmp_path)

    rejected = session.receive(
        frame({"method": "item/agentMessage/delta", "params": {"delta": "lost"}})
    )

    assert rejected.done is True
    assert rejected.failed is True
    assert last_event(rejected, "error").text == "Codex delta omitted itemId"


@pytest.mark.parametrize(
    ("method", "kind"),
    (
        ("item/plan/delta", "assistant"),
        ("item/reasoning/summaryTextDelta", "assistant"),
        ("item/reasoning/textDelta", "assistant"),
        ("item/commandExecution/outputDelta", "tool"),
        ("item/commandExecution/output_delta", "tool"),
        ("item/fileChange/outputDelta", "tool"),
    ),
)
def test_real_codex_delta_frames_preserve_reasoning_and_tool_output(
    tmp_path: Path, method: str, kind: SessionKind
) -> None:
    session = CodexSession(("codex",), tmp_path)

    streamed = session.receive(
        frame({"method": method, "params": {"itemId": "item-1", "delta": "visible"}})
    )

    assert streamed.events == (
        SessionEvent(
            kind=kind,
            text="visible",
            event_id="item-1",
            state="streaming",
            delta=True,
        ),
    )


def test_real_mcp_progress_frame_preserves_tool_output(tmp_path: Path) -> None:
    session = CodexSession(("codex",), tmp_path)

    streamed = session.receive(
        frame(
            {
                "method": "item/mcpToolCall/progress",
                "params": {"itemId": "mcp-1", "message": "still running"},
            }
        )
    )

    assert streamed.events == (
        SessionEvent(
            kind="tool",
            text="still running",
            event_id="mcp-1",
            state="streaming",
            delta=True,
        ),
    )


@pytest.mark.parametrize(
    ("item", "expected"),
    (
        (
            {"id": "command-1", "type": "commandExecution", "command": "git status"},
            "git status",
        ),
        (
            {
                "id": "command-2",
                "type": "commandExecution",
                "command": "git status",
                "aggregatedOutput": " M file.py",
            },
            " M file.py",
        ),
        ({"id": "mcp-1", "type": "mcpToolCall", "name": "search"}, "search"),
        (
            {
                "id": "file-1",
                "type": "fileChange",
                "changes": {"src/a.py": {}, "src/b.py": {}},
            },
            "src/a.py, src/b.py",
        ),
        (
            {
                "id": "function-1",
                "type": "functionCallOutput",
                "name": "lookup",
                "output": "42",
            },
            "42",
        ),
        ({"id": "dynamic-1", "type": "dynamicToolCall", "name": "lookup"}, "lookup"),
        (
            {"id": "collab-1", "type": "collabAgentToolCall", "tool": "delegate"},
            "delegate",
        ),
        (
            {"id": "subagent-1", "type": "subAgentActivity"},
            '{"id": "subagent-1", "type": "subAgentActivity"}',
        ),
    ),
)
def test_real_codex_tool_items_emit_complete_observable_events(
    tmp_path: Path, item: dict[str, object], expected: str
) -> None:
    session = CodexSession(("codex",), tmp_path)

    completed = session.receive(frame({"method": "item/completed", "params": {"item": item}}))

    assert completed.events == (
        SessionEvent(
            kind="tool",
            text=expected,
            event_id=str(item["id"]),
            state="complete",
        ),
    )


def test_real_user_message_frame_preserves_text_parts_and_client_identity(
    tmp_path: Path,
) -> None:
    session = CodexSession(("codex",), tmp_path)
    item = {
        "id": "user-item-1",
        "type": "userMessage",
        "clientId": "client-message-1",
        "content": [
            {"type": "text", "text": "first"},
            {"type": "image", "path": "/tmp/image.png"},
            {"type": "text", "text": "second"},
        ],
    }

    delivered = session.receive(frame({"method": "item/completed", "params": {"item": item}}))

    assert delivered.events == (
        SessionEvent(
            kind="user",
            text="first\nsecond",
            event_id="client-message-1",
            state="delivered",
        ),
    )


def test_unmatched_codex_response_is_visible_without_consuming_pending_request(
    tmp_path: Path,
) -> None:
    session = CodexSession(("codex",), tmp_path)
    initialize = wire(session.start("prompt").commands[0])

    stale = session.receive(frame({"id": 999, "result": {}}))
    assert stale.events == (
        SessionEvent(
            kind="error",
            text="unmatched Codex response id 999",
            event_id="999",
            state="rejected",
        ),
    )
    assert stale.done is False
    assert stale.failed is False

    thread_step = session.receive(frame({"id": initialize["id"], "result": {}}))
    assert wire(thread_step.commands[1])["method"] == "thread/start"


def test_item_without_type_is_rejected_instead_of_dropping_message_text(
    tmp_path: Path,
) -> None:
    session = CodexSession(("codex",), tmp_path)

    rejected = session.receive(
        frame(
            {
                "method": "item/completed",
                "params": {"item": {"id": "message-1", "text": "answer"}},
            }
        )
    )

    assert rejected.done is True
    assert rejected.failed is True
    assert last_event(rejected, "error").state == "rejected"


def test_completed_plan_item_preserves_observable_plan_text(tmp_path: Path) -> None:
    session = CodexSession(("codex",), tmp_path)

    completed = session.receive(
        frame(
            {
                "method": "item/completed",
                "params": {"item": {"id": "plan-1", "type": "plan", "text": "inspect then edit"}},
            }
        )
    )

    assert completed.events == (
        SessionEvent(
            kind="assistant",
            text="inspect then edit",
            event_id="plan-1",
            state="complete",
        ),
    )


def test_agent_message_deltas_accumulate_in_completion_result(tmp_path: Path) -> None:
    session = CodexSession(("codex",), tmp_path)

    first = session.receive(
        frame(
            {
                "method": "item/agentMessage/delta",
                "params": {"itemId": "message-1", "delta": "first"},
            }
        )
    )
    second = session.receive(
        frame(
            {
                "method": "item/agentMessage/delta",
                "params": {"itemId": "message-1", "delta": " second"},
            }
        )
    )
    completed = session.receive(
        frame(
            {
                "method": "turn/completed",
                "params": {"turn": {"id": "turn-1", "status": "completed", "items": []}},
            }
        )
    )

    assert first.events[0].text == "first"
    assert second.events[0].text == " second"
    assert completed.result_text == "first second"
