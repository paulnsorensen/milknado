from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import msgspec
import pytest

from milknado.domains.common.session import SessionEvent, SessionInput
from milknado.loop.sessions._claude import ClaudeSession


def _frame(payload: dict[str, object]) -> bytes:
    return (json.dumps(payload) + "\n").encode()


def _result(text: str, **extra: object) -> bytes:
    return _frame(
        {"type": "result", "subtype": "success", "is_error": False, "result": text, **extra}
    )


def _wire(command: bytes) -> dict[str, object]:
    return msgspec.json.decode(command, type=dict[str, object])


def _mapping(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def _string(value: object) -> str:
    assert isinstance(value, str)
    return value


def _session(tmp_path: Path, *argv: str) -> ClaudeSession:
    return ClaudeSession(("claude", *argv), tmp_path)


def test_command_enables_streaming_without_dropping_cli_flags(tmp_path: Path) -> None:
    session = _session(
        tmp_path,
        "-p",
        "--model",
        "opus",
        "--allowedTools",
        "Read",
        "--mcp-config",
        "mcp.json",
        "--permission-mode",
        "plan",
        "--resume",
        "session-name",
        "--permission-prompt-tool",
        "ask",
    )

    assert session.command[:3] == ("claude", "-p", "--model")
    assert "--model" in session.command and "opus" in session.command
    assert "--allowedTools" in session.command and "Read" in session.command
    assert "--mcp-config" in session.command and "mcp.json" in session.command
    assert "--permission-mode" in session.command and "plan" in session.command
    assert "--resume" in session.command and "session-name" in session.command
    assert "--permission-prompt-tool" in session.command and "ask" in session.command
    assert session.command[session.command.index("--output-format") + 1] == "stream-json"
    assert session.command[session.command.index("--input-format") + 1] == "stream-json"
    assert "--verbose" in session.command
    assert "--include-partial-messages" in session.command
    assert "--replay-user-messages" in session.command
    assert session.actions == ("follow_up", "interrupt", "approve", "deny")


def test_start_and_echo_require_complete_json_frames(tmp_path: Path) -> None:
    session = _session(tmp_path, "-p")
    started = session.start("hello")
    assert _mapping(_wire(started.commands[1])["message"])["content"] == "hello"
    assert started.events[0].state == "submitted"

    partial = b'{"type":"user","uuid":"echo-1","message":{"content":"hel'
    assert session.receive(partial).events == ()
    echoed = session.receive(b'lo"}}\n')
    assert echoed.events == (
        SessionEvent(
            kind="user", text="hello", event_id=started.events[0].event_id, state="delivered"
        ),
    )


def test_known_envelope_fields_reject_wrong_wire_types(tmp_path: Path) -> None:
    session = _session(tmp_path)
    _ = session.start("go")
    invalid = session.receive(_frame({"type": "result", "is_error": "false"}))
    assert invalid.done is True
    assert invalid.failed is True
    assert invalid.result_text == invalid.events[0].text


def test_stream_event_deltas_and_tool_readable_events(tmp_path: Path) -> None:
    session = _session(tmp_path)
    _ = session.start("go")
    assert (
        session.receive(
            _frame({"type": "system", "subtype": "init", "session_id": "sid-1", "model": "opus"})
        ).session_id
        == "sid-1"
    )
    _ = session.receive(
        _frame(
            {
                "type": "stream_event",
                "session_id": "sid-1",
                "event": {"type": "message_start", "message": {"id": "msg-1"}},
            }
        )
    )
    _ = session.receive(
        _frame(
            {
                "type": "stream_event",
                "session_id": "sid-1",
                "event": {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            }
        )
    )
    delta = session.receive(
        _frame(
            {
                "type": "stream_event",
                "session_id": "sid-1",
                "event": {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "Hi"},
                },
            }
        )
    )
    assert delta.events[0].kind == "assistant"
    assert delta.events[0].delta is True
    assert delta.events[0].text == "Hi"
    tool = session.receive(
        _frame(
            {
                "type": "assistant",
                "message": {
                    "id": "msg-2",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tool-1",
                            "name": "Bash",
                            "input": {"command": "pwd"},
                        }
                    ],
                },
            }
        )
    )
    assert tool.events[0].kind == "tool"
    assert tool.events[0].event_id == "tool-1"
    assert tool.events[0].text == "Bash: pwd"
    assert tool.events[0].state == "streaming"


def test_permission_approval_and_denial_use_exact_request_id(tmp_path: Path) -> None:
    session = _session(tmp_path)
    _ = session.start("go")
    request = _frame(
        {
            "type": "control_request",
            "request_id": "perm-42",
            "request": {
                "subtype": "can_use_tool",
                "tool_name": "Bash",
                "input": {"command": "rm x"},
            },
        }
    )
    pending = session.receive(request)
    assert pending.events[0].state == "requested"
    approved = session.submit(SessionInput(action="approve", request_id="perm-42"))
    wire = _wire(approved.commands[0])
    response = _mapping(wire["response"])
    assert response["request_id"] == "perm-42"
    assert response["response"] == {
        "behavior": "allow",
        "updatedInput": {"command": "rm x"},
    }
    assert approved.events[0].state == "submitted"
    assert approved.after_write_events[0].state == "approved"
    assert approved.after_write_events[0].event_id == "perm-42"

    denied_request = request.replace(b"perm-42", b"perm-43")
    _ = session.receive(denied_request)
    denied = session.submit(SessionInput(action="deny", request_id="perm-43", text="not safe"))
    denied_wire = _wire(denied.commands[0])
    denied_response = _mapping(denied_wire["response"])
    assert denied_response["request_id"] == "perm-43"
    assert denied_response["response"] == {"behavior": "deny", "message": "not safe"}
    assert denied.events[0].state == "submitted"
    assert denied.after_write_events[0].state == "denied"
    assert denied.after_write_events[0].event_id == "perm-43"


def test_queued_follow_up_delays_done_until_its_result(tmp_path: Path) -> None:
    session = _session(tmp_path)
    _ = session.start("first")
    queued = session.submit(SessionInput(action="follow_up", text="second", request_id="input-2"))
    assert queued.events == ()
    first = session.receive(_result("first result", session_id="sid-2"))
    assert first.done is False
    assert first.result_text is None
    second = session.receive(_result("second result", session_id="sid-2"))
    assert second.done is True
    assert second.result_text == "second result"
    assert second.session_id == "sid-2"


def test_interrupt_is_controlled_and_terminal_only_when_vendor_aborts(tmp_path: Path) -> None:
    session = _session(tmp_path)
    _ = session.start("stop")
    submitted = session.submit(SessionInput(action="interrupt"))
    interrupt_id = _string(_wire(submitted.commands[0])["request_id"])
    ack = session.receive(
        _frame(
            {
                "type": "control_response",
                "response": {"subtype": "success", "request_id": interrupt_id, "response": {}},
            }
        )
    )
    assert ack.events[0].text == "Interrupt acknowledged"
    stopped = session.receive(
        _frame(
            {
                "type": "result",
                "subtype": "error_during_execution",
                "is_error": True,
                "result": "Interrupted by user",
                "session_id": "sid-3",
            }
        )
    )
    assert stopped.done is True
    assert stopped.failed is False
    assert stopped.interrupted is True


def test_result_and_control_errors_preserve_real_text(tmp_path: Path) -> None:
    session = _session(tmp_path)
    started = session.start("go")
    init_id = _string(_wire(started.commands[0])["request_id"])
    control = session.receive(
        _frame(
            {
                "type": "control_response",
                "response": {"subtype": "error", "request_id": init_id, "error": "bad control"},
            }
        )
    )
    assert control.events[0].kind == "error"
    assert control.events[0].text == "bad control"
    failed = session.receive(
        _frame(
            {
                "type": "result",
                "subtype": "error_max_turns",
                "is_error": True,
                "errors": ["turn cap reached"],
                "session_id": "sid-4",
            }
        )
    )
    assert failed.done is True
    assert failed.failed is True
    assert failed.result_text == "turn cap reached"
    assert failed.events[0].text == "turn cap reached"


def test_native_steer_is_not_claimed(tmp_path: Path) -> None:
    session = _session(tmp_path)
    _ = session.start("go")
    with pytest.raises(ValueError, match="native steer"):
        _ = session.submit(SessionInput(action="steer", text="change"))


def test_receive_requires_start_and_start_is_single_use(tmp_path: Path) -> None:
    session = _session(tmp_path)
    with pytest.raises(ValueError, match="ClaudeSession has not started"):
        _ = session.receive(b"")

    started = session.start("go")
    assert started.events == (
        SessionEvent(kind="user", text="go", event_id="user-1", state="submitted"),
    )
    with pytest.raises(ValueError, match="already started"):
        _ = session.start("again")


def test_streamed_blocks_assemble_text_and_thinking_before_result(tmp_path: Path) -> None:
    session = _session(tmp_path)
    _ = session.start("go")

    started = session.receive(
        _frame(
            {
                "type": "stream_event",
                "session_id": "sid-stream",
                "event": {"type": "message_start", "message": {"id": "msg-stream"}},
            }
        )
    )
    assert started.events == ()
    assert started.session_id == "sid-stream"

    text_block = session.receive(
        _frame(
            {
                "type": "stream_event",
                "session_id": "sid-stream",
                "event": {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text"},
                },
            }
        )
    )
    assert text_block.events == (
        SessionEvent(kind="assistant", text="", event_id="msg-stream", state="streaming"),
    )

    first_delta = session.receive(
        _frame(
            {
                "type": "stream_event",
                "session_id": "sid-stream",
                "event": {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "Hello"},
                },
            }
        )
    )
    assert first_delta.events == (
        SessionEvent(
            kind="assistant",
            text="Hello",
            event_id="msg-stream",
            state="streaming",
            delta=True,
        ),
    )

    second_delta = session.receive(
        _frame(
            {
                "type": "stream_event",
                "session_id": "sid-stream",
                "event": {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": " world"},
                },
            }
        )
    )
    assert second_delta.events[0].text == " world"
    assert second_delta.events[0].event_id == "msg-stream"
    assert second_delta.events[0].delta is True

    thinking_block = session.receive(
        _frame(
            {
                "type": "stream_event",
                "session_id": "sid-stream",
                "event": {
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": {"type": "thinking"},
                },
            }
        )
    )
    assert thinking_block.events == (
        SessionEvent(kind="assistant", text="", event_id="msg-stream", state="streaming"),
    )
    thinking_delta = session.receive(
        _frame(
            {
                "type": "stream_event",
                "session_id": "sid-stream",
                "event": {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "thinking_delta", "thinking": " plan"},
                },
            }
        )
    )
    assert thinking_delta.events == (
        SessionEvent(
            kind="assistant",
            text=" plan",
            event_id="msg-stream",
            state="streaming",
            delta=True,
        ),
    )

    stopped = session.receive(
        _frame(
            {
                "type": "stream_event",
                "session_id": "sid-stream",
                "event": {"type": "message_stop"},
            }
        )
    )
    assert stopped.events == (
        SessionEvent(
            kind="assistant",
            text="Hello world plan",
            event_id="msg-stream",
            state="complete",
        ),
    )

    result = session.receive(_result("Hello world plan", session_id="sid-stream"))
    assert result.done is True
    assert result.failed is False
    assert result.interrupted is False
    assert result.result_text == "Hello world plan"
    assert result.events == ()


def test_stream_delta_without_started_block_emits_no_phantom_content(tmp_path: Path) -> None:
    session = _session(tmp_path)
    _ = session.start("go")
    unknown = session.receive(
        _frame(
            {
                "type": "stream_event",
                "session_id": "sid-stream",
                "event": {
                    "type": "content_block_delta",
                    "index": 99,
                    "delta": {"type": "text_delta", "text": "ghost"},
                },
            }
        )
    )
    assert unknown.events == ()
    assert unknown.done is False
    assert unknown.result_text is None
    assert unknown.session_id == "sid-stream"


def test_malformed_assistant_frame_is_terminal_and_stops_submission(tmp_path: Path) -> None:
    session = _session(tmp_path)
    _ = session.start("go")
    invalid = session.receive(_frame({"type": "assistant"}))

    assert invalid.done is True
    assert invalid.failed is True
    assert invalid.result_text == "Claude assistant frame is missing message"
    assert invalid.events == (
        SessionEvent(
            kind="error",
            text="Claude assistant frame is missing message",
            event_id="error-3",
            state="stopped",
        ),
    )
    with pytest.raises(ValueError, match="ClaudeSession is not active"):
        _ = session.submit(SessionInput(action="follow_up", text="after failure"))


def test_assistant_string_content_is_not_duplicated_by_terminal_result(tmp_path: Path) -> None:
    session = _session(tmp_path)
    _ = session.start("go")
    assistant = session.receive(
        _frame(
            {
                "type": "assistant",
                "session_id": "sid-result",
                "message": {"id": "msg-result", "content": "All done."},
            }
        )
    )
    assert assistant.events == (
        SessionEvent(
            kind="assistant",
            text="All done.",
            event_id="msg-result",
            state="complete",
        ),
    )

    result = session.receive(_result("All done.", session_id="sid-result"))
    assert result.done is True
    assert result.failed is False
    assert result.result_text == "All done."
    assert result.events == ()


def test_tool_results_preserve_content_and_pending_prompt_identity(tmp_path: Path) -> None:
    session = _session(tmp_path)
    started = session.start("go")
    echoed = session.receive(
        _frame(
            {
                "type": "user",
                "uuid": "echo-tools",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool-7",
                            "content": [
                                {"type": "text", "text": "first"},
                                {"type": "text", "text": "\nsecond"},
                                "malformed block",
                                {"type": "text", "text": 42},
                            ],
                        },
                        {"type": "text", "text": "go"},
                    ]
                },
            }
        )
    )
    assert echoed.events == (
        SessionEvent(kind="tool", text="first\nsecond", event_id="tool-7", state="complete"),
        SessionEvent(
            kind="user",
            text="go",
            event_id=started.events[0].event_id,
            state="delivered",
        ),
    )

    assistant_tool = session.receive(
        _frame(
            {
                "type": "assistant",
                "message": {
                    "id": "msg-tool",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool-8",
                            "content": "permission denied",
                        }
                    ],
                },
            }
        )
    )
    assert assistant_tool.events == (
        SessionEvent(
            kind="tool",
            text="permission denied",
            event_id="tool-8",
            state="complete",
        ),
    )


def test_unmatched_user_echo_does_not_consume_pending_prompt(tmp_path: Path) -> None:
    session = _session(tmp_path)
    started = session.start("go")
    unrelated = session.receive(
        _frame({"type": "user", "uuid": "echo-other", "message": "other input"})
    )
    assert unrelated.events == (
        SessionEvent(kind="user", text="other input", event_id="echo-other", state="delivered"),
    )

    matched = session.receive(_frame({"type": "user", "uuid": "echo-prompt", "message": "go"}))
    assert matched.events == (
        SessionEvent(
            kind="user",
            text="go",
            event_id=started.events[0].event_id,
            state="delivered",
        ),
    )


def test_unsupported_control_request_returns_wire_error_and_visible_event(
    tmp_path: Path,
) -> None:
    session = _session(tmp_path)
    _ = session.start("go")
    unsupported = session.receive(
        _frame(
            {
                "type": "control_request",
                "request_id": "control-1",
                "request": {"subtype": "ask_user"},
            }
        )
    )
    assert _wire(unsupported.commands[0]) == {
        "type": "control_response",
        "response": {
            "subtype": "error",
            "request_id": "control-1",
            "error": "Unsupported Claude control request: ask_user",
        },
    }
    assert unsupported.events == (
        SessionEvent(
            kind="error",
            text="Unsupported Claude control request: ask_user",
            event_id="control-1",
            state="running",
        ),
    )
    assert unsupported.done is False
    assert unsupported.failed is False


def test_duplicate_permission_request_keeps_original_input_pending(tmp_path: Path) -> None:
    session = _session(tmp_path)
    _ = session.start("go")
    request = _frame(
        {
            "type": "control_request",
            "request_id": "perm-duplicate",
            "request": {
                "subtype": "can_use_tool",
                "tool_name": "Bash",
                "input": {"command": "printf '%s' one"},
            },
        }
    )
    first = session.receive(request)
    assert first.events[0].state == "requested"
    duplicate = session.receive(request)
    assert duplicate.events == (
        SessionEvent(
            kind="error",
            text="Duplicate Claude permission request_id: perm-duplicate",
            event_id="error-3",
            state="running",
        ),
    )
    assert duplicate.done is False
    assert duplicate.failed is False
    assert duplicate.result_text is None

    approved = session.submit(SessionInput(action="approve", request_id="perm-duplicate"))
    assert _mapping(_wire(approved.commands[0])["response"])["response"] == {
        "behavior": "allow",
        "updatedInput": {"command": "printf '%s' one"},
    }


def test_permission_approval_preserves_updated_input_and_human_text(tmp_path: Path) -> None:
    session = _session(tmp_path)
    _ = session.start("go")
    pending = session.receive(
        _frame(
            {
                "type": "control_request",
                "request_id": "perm-read",
                "request": {
                    "subtype": "can_use_tool",
                    "tool_name": "Read",
                    "title": "Read configuration",
                    "input": {"file_path": "/tmp/config", "limit": 20, "offset": 4},
                },
            }
        )
    )
    assert pending.events == (
        SessionEvent(
            kind="permission",
            text="Read configuration",
            event_id="perm-read",
            state="requested",
        ),
    )

    approved = session.submit(SessionInput(action="approve", request_id="perm-read"))
    assert _wire(approved.commands[0]) == {
        "type": "control_response",
        "response": {
            "subtype": "success",
            "request_id": "perm-read",
            "response": {
                "behavior": "allow",
                "updatedInput": {"file_path": "/tmp/config", "limit": 20, "offset": 4},
            },
        },
    }
    assert approved.events == (
        SessionEvent(
            kind="permission",
            text="Permission approved: Read configuration",
            event_id="perm-read",
            state="submitted",
        ),
    )
    assert approved.after_write_events == (
        SessionEvent(
            kind="permission",
            text="Permission approved: Read configuration",
            event_id="perm-read",
            state="approved",
        ),
    )


def test_permission_denial_uses_default_message_and_clears_request(tmp_path: Path) -> None:
    session = _session(tmp_path)
    _ = session.start("go")
    _ = session.receive(
        _frame(
            {
                "type": "control_request",
                "request_id": "perm-deny",
                "request": {
                    "subtype": "can_use_tool",
                    "tool_name": "Bash",
                    "input": {"command": "rm -rf /tmp/cache"},
                },
            }
        )
    )
    denied = session.submit(SessionInput(action="deny", request_id="perm-deny"))
    assert _mapping(_wire(denied.commands[0])["response"])["response"] == {
        "behavior": "deny",
        "message": "Denied by operator.",
    }
    assert denied.events == (
        SessionEvent(
            kind="permission",
            text="Permission denied: Bash: rm -rf /tmp/cache",
            event_id="perm-deny",
            state="submitted",
        ),
    )
    assert denied.after_write_events == (
        SessionEvent(
            kind="permission",
            text="Permission denied: Bash: rm -rf /tmp/cache",
            event_id="perm-deny",
            state="denied",
        ),
    )
    with pytest.raises(ValueError, match=r"Unknown Claude permission request_id: 'perm-deny'"):
        _ = session.submit(SessionInput(action="deny", request_id="perm-deny"))


def test_permission_cancel_is_visible_and_removes_pending_request(tmp_path: Path) -> None:
    session = _session(tmp_path)
    _ = session.start("go")
    _ = session.receive(
        _frame(
            {
                "type": "control_request",
                "request_id": "perm-cancel",
                "request": {
                    "subtype": "can_use_tool",
                    "tool_name": "Read",
                    "display_name": "Read file",
                    "input": {"file_path": "/tmp/file"},
                },
            }
        )
    )
    cancelled = session.receive(
        _frame({"type": "control_cancel_request", "request_id": "perm-cancel"})
    )
    assert cancelled.events == (
        SessionEvent(
            kind="permission",
            text="Permission cancelled: Read file",
            event_id="perm-cancel",
            state="denied",
        ),
    )
    with pytest.raises(ValueError, match=r"Unknown Claude permission request_id: 'perm-cancel'"):
        _ = session.submit(SessionInput(action="approve", request_id="perm-cancel"))


def test_outer_control_response_identity_is_retained_for_later_frames(tmp_path: Path) -> None:
    session = _session(tmp_path)
    started = session.start("go")
    initialize_id = _string(_wire(started.commands[0])["request_id"])
    initialized = session.receive(
        _frame(
            {
                "type": "control_response",
                "session_id": "sid-control",
                "response": {
                    "subtype": "success",
                    "request_id": initialize_id,
                    "response": {},
                },
            }
        )
    )
    assert initialized.session_id == "sid-control"
    assert initialized.events == (
        SessionEvent(
            kind="status",
            text="Claude session initialized",
            event_id=initialize_id,
            state="running",
        ),
    )

    later = session.receive(_frame({"type": "notice"}))
    assert later.session_id == "sid-control"
    assert later.events == (
        SessionEvent(
            kind="status",
            text="Claude event: notice",
            event_id="notice-3",
            state="running",
        ),
    )


def test_unknown_control_response_is_nonterminal_and_does_not_consume_request(
    tmp_path: Path,
) -> None:
    session = _session(tmp_path)
    started = session.start("go")
    initialize_id = _string(_wire(started.commands[0])["request_id"])
    unknown = session.receive(
        _frame(
            {
                "type": "control_response",
                "response": {
                    "subtype": "success",
                    "request_id": "not-pending",
                    "response": {},
                },
            }
        )
    )
    assert unknown.events == (
        SessionEvent(
            kind="error",
            text="Unexpected Claude control response",
            event_id="error-3",
            state="running",
        ),
    )
    assert unknown.done is False
    assert unknown.failed is False
    assert unknown.result_text is None

    initialized = session.receive(
        _frame(
            {
                "type": "control_response",
                "response": {
                    "subtype": "success",
                    "request_id": initialize_id,
                    "response": {},
                },
            }
        )
    )
    assert initialized.events[0].text == "Claude session initialized"


def test_control_error_without_message_preserves_fallback_text(tmp_path: Path) -> None:
    session = _session(tmp_path)
    _ = session.start("go")
    interrupt = session.submit(SessionInput(action="interrupt"))
    interrupt_id = _string(_wire(interrupt.commands[0])["request_id"])
    failed = session.receive(
        _frame(
            {
                "type": "control_response",
                "response": {
                    "subtype": "error",
                    "request_id": interrupt_id,
                },
            }
        )
    )
    assert failed.events == (
        SessionEvent(
            kind="error",
            text="Claude control request failed",
            event_id=interrupt_id,
            state="running",
        ),
    )
    assert failed.done is False
    assert failed.failed is False

    retry = session.submit(SessionInput(action="interrupt"))
    assert _string(_wire(retry.commands[0])["request_id"]) != interrupt_id


def test_system_error_object_preserves_vendor_message(tmp_path: Path) -> None:
    session = _session(tmp_path)
    _ = session.start("go")
    failure = session.receive(
        _frame(
            {
                "type": "system",
                "subtype": "failure",
                "error": {"message": "Claude backend offline"},
            }
        )
    )
    assert failure.events == (
        SessionEvent(
            kind="error",
            text="Claude backend offline",
            event_id="error-3",
            state="stopped",
        ),
    )
    assert failure.done is True
    assert failure.failed is True
    assert failure.result_text == "Claude backend offline"


def test_error_frame_without_detail_uses_observable_fallback(tmp_path: Path) -> None:
    session = _session(tmp_path)
    _ = session.start("go")
    failure = session.receive(_frame({"type": "error"}))
    assert failure.events == (
        SessionEvent(
            kind="error",
            text="Claude returned an error",
            event_id="error-3",
            state="stopped",
        ),
    )
    assert failure.done is True
    assert failure.failed is True
    assert failure.result_text == "Claude returned an error"


def test_follow_up_rejects_empty_text_and_duplicate_request_id(tmp_path: Path) -> None:
    session = _session(tmp_path)
    _ = session.start("first")
    with pytest.raises(ValueError, match="Claude follow_up requires text"):
        _ = session.submit(SessionInput(action="follow_up"))

    queued = session.submit(SessionInput(action="follow_up", text="second", request_id="follow-1"))
    assert queued.events == ()
    with pytest.raises(ValueError, match="Duplicate Claude user request_id: follow-1"):
        _ = session.submit(
            SessionInput(action="follow_up", text="replacement", request_id="follow-1")
        )
    first = session.receive(_result("first result", session_id="sid-follow"))
    assert first.done is False
    assert first.result_text is None
    assert first.events == (
        SessionEvent(
            kind="assistant",
            text="first result",
            event_id="sid-follow",
            state="complete",
        ),
    )
    second = session.receive(_result("second result", session_id="sid-follow"))
    assert second.done is True
    assert second.result_text == "second result"


def test_interrupt_acknowledgement_does_not_abort_normal_result(tmp_path: Path) -> None:
    session = _session(tmp_path)
    _ = session.start("stop")
    requested = session.submit(SessionInput(action="interrupt"))
    interrupt_id = _string(_wire(requested.commands[0])["request_id"])
    acknowledged = session.receive(
        _frame(
            {
                "type": "control_response",
                "response": {
                    "subtype": "success",
                    "request_id": interrupt_id,
                    "response": {},
                },
            }
        )
    )
    assert acknowledged.events == (
        SessionEvent(
            kind="status",
            text="Interrupt acknowledged",
            event_id=interrupt_id,
            state="running",
        ),
    )
    assert acknowledged.done is False
    assert acknowledged.failed is False
    assert acknowledged.interrupted is False

    completed = session.receive(_result("finished", session_id="sid-normal"))
    assert completed.done is True
    assert completed.failed is False
    assert completed.interrupted is False
    assert completed.result_text == "finished"
    assert completed.events == (
        SessionEvent(
            kind="assistant",
            text="finished",
            event_id="sid-normal",
            state="complete",
        ),
    )


def test_control_response_with_unknown_subtype_is_not_acknowledged(tmp_path: Path) -> None:
    session = _session(tmp_path)
    started = session.start("go")
    initialize_id = _string(_wire(started.commands[0])["request_id"])
    invalid = session.receive(
        _frame(
            {
                "type": "control_response",
                "response": {
                    "subtype": "unexpected",
                    "request_id": initialize_id,
                    "response": {},
                },
            }
        )
    )
    assert invalid.events == (
        SessionEvent(
            kind="error",
            text="Claude control request failed",
            event_id=initialize_id,
            state="running",
        ),
    )
    assert invalid.done is False
    assert invalid.failed is False
    assert invalid.result_text is None
