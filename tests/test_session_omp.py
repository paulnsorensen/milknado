from __future__ import annotations

import base64
from pathlib import Path

import msgspec
import pytest

from milknado.domains.common import SessionEvent, SessionInput
from milknado.loop.sessions._omp import OmpSession
from milknado.loop.sessions._protocol import ProtocolStep


def frame(payload: dict[str, object]) -> bytes:
    return msgspec.json.encode(payload) + b"\n"


def event(step: ProtocolStep, kind: str, state: str | None = None) -> SessionEvent:
    matches = [item for item in step.events if item.kind == kind]
    assert matches
    if state is not None:
        matches = [item for item in matches if item.state == state]
        assert matches
    return matches[-1]


def test_ready_negotiates_v2_and_recovers_session_identity() -> None:
    session = OmpSession(("omp",), Path("/repo"))

    ready = session.receive(
        frame(
            {
                "type": "ready",
                "protocolVersion": 1,
                "supportedProtocolVersions": [1, 2],
                "maxFrameBytes": 1048576,
            }
        )
    )
    commands = [msgspec.json.decode(command, type=dict[str, object]) for command in ready.commands]
    assert commands == [
        {"id": "protocol-1", "type": "negotiate_protocol", "protocolVersion": 2},
        {"id": "state-1", "type": "get_state"},
    ]

    identity = session.receive(
        frame(
            {
                "id": "state-1",
                "type": "response",
                "command": "get_state",
                "success": True,
                "data": {"sessionId": "omp-session-42"},
            }
        )
    )
    assert identity.session_id == "omp-session-42"


def test_rpc_chunk_reassembles_a_complete_json_record() -> None:
    session = OmpSession(("omp",), Path("/repo"))
    _ = session.receive(frame({"type": "ready", "supportedProtocolVersions": [1, 2]}))
    payload = msgspec.json.encode(
        {
            "id": "state-1",
            "type": "response",
            "command": "get_state",
            "success": True,
            "data": {"sessionId": "chunked-session"},
        }
    )
    result = session.receive(
        frame(
            {
                "type": "rpc_chunk",
                "chunkId": "rpc-1",
                "index": 0,
                "count": 1,
                "byteLength": len(payload),
                "data": base64.b64encode(payload).decode(),
            }
        )
    )
    assert result.session_id == "chunked-session"


def test_command_replaces_only_mode_and_preserves_safety_flags() -> None:
    session = OmpSession(
        (
            "/usr/local/bin/omp",
            "--model",
            "openai/gpt-5.6",
            "--mode=json",
            "--config",
            "secure.yml",
            "--profile",
            "isolated",
            "--add-dir",
            "/repo/extra",
            "--no-session",
        ),
        Path("/repo"),
    )

    assert session.command == (
        "/usr/local/bin/omp",
        "--model",
        "openai/gpt-5.6",
        "--mode",
        "rpc",
        "--config",
        "secure.yml",
        "--profile",
        "isolated",
        "--add-dir",
        "/repo/extra",
        "--no-session",
    )


def test_streamed_messages_tools_and_result_preserve_wire_ids() -> None:
    session = OmpSession(("omp", "--mode", "text", "--model", "gpt-5.6"), Path("/repo"))
    started = session.start("inspect the project")
    request_id = msgspec.json.decode(started.commands[0], type=dict[str, object])["id"]
    assert event(started, "user", "submitted").event_id == request_id

    accepted = session.receive(
        frame({"id": request_id, "type": "response", "command": "prompt", "success": True})
    )
    assert accepted.events == ()
    delivered = session.receive(
        frame(
            {
                "type": "message_start",
                "message": {"role": "user", "content": "inspect the project"},
            }
        )
    )
    assert event(delivered, "user", "delivered").event_id == request_id

    delta = session.receive(
        frame(
            {
                "type": "message_update",
                "assistantMessageEvent": {
                    "type": "text_delta",
                    "contentIndex": 0,
                    "delta": "I found ",
                },
            }
        )
    )
    assistant_delta = event(delta, "assistant", "streaming")
    assert assistant_delta.delta is True
    assert assistant_delta.text == "I found "

    tool_start = session.receive(
        frame({"type": "tool_execution_start", "toolCallId": "call-7", "toolName": "read"})
    )
    assert event(tool_start, "tool", "streaming").event_id == "call-7"
    tool_update = session.receive(
        frame(
            {
                "type": "tool_execution_update",
                "toolCallId": "call-7",
                "partialResult": {"content": [{"type": "text", "text": "README.md\n"}]},
            }
        )
    )
    assert event(tool_update, "tool", "streaming").text == "README.md\n"
    tool_end = session.receive(
        frame(
            {
                "type": "tool_execution_end",
                "toolCallId": "call-7",
                "result": {"content": [{"type": "text", "text": "done"}]},
                "isError": False,
            }
        )
    )
    assert event(tool_end, "tool", "complete").event_id == "call-7"

    _ = session.receive(
        frame(
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "<promise>RALPH_PROMISE_COMPLETE</promise>"}
                    ],
                    "stopReason": "stop",
                },
            }
        )
    )
    finished = session.receive(
        frame(
            {
                "type": "agent_end",
                "isTerminal": True,
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "<promise>RALPH_PROMISE_COMPLETE</promise>"}
                        ],
                    }
                ],
            }
        )
    )
    assert finished.done is True
    assert finished.failed is False
    assert finished.result_text == "<promise>RALPH_PROMISE_COMPLETE</promise>"


def test_follow_up_is_not_dropped_before_follow_up_echo() -> None:
    session = OmpSession(("omp",), Path("/repo"))
    started = session.start("first")
    prompt_id = msgspec.json.decode(started.commands[0], type=dict[str, object])["id"]
    _ = session.receive(
        frame({"id": prompt_id, "type": "response", "command": "prompt", "success": True})
    )
    delivered = session.receive(
        frame({"type": "message_start", "message": {"role": "user", "content": "first"}})
    )
    assert event(delivered, "user", "delivered").event_id == prompt_id
    follow_up = session.submit(
        SessionInput(action="follow_up", request_id="follow-1", text="also check tests")
    )
    assert follow_up.events == ()
    queued = session.receive(
        frame({"id": "follow-1", "type": "response", "command": "follow_up", "success": True})
    )
    assert queued.events == ()

    waiting = session.receive(frame({"type": "agent_end", "isTerminal": True, "messages": []}))
    assert waiting.done is False
    assert event(waiting, "status", "running").text == "OMP agent turn complete"
    echoed = session.receive(
        frame(
            {"type": "message_start", "message": {"role": "user", "content": "also check tests"}}
        )
    )
    assert event(echoed, "user", "delivered").event_id == "follow-1"
    complete = session.receive(frame({"type": "agent_end", "isTerminal": True, "messages": []}))
    assert complete.done is True


def test_permission_confirm_reply_is_supported_without_fake_ack() -> None:
    session = OmpSession(("omp",), Path("/repo"))
    _ = session.start("approve the safe edit")
    request = session.receive(
        frame(
            {
                "type": "extension_ui_request",
                "id": "ui-1",
                "method": "confirm",
                "title": "Apply edit?",
            }
        )
    )
    assert event(request, "permission", "requested").event_id == "ui-1"
    assert "approve" in session.actions and "deny" in session.actions

    reply = session.submit(SessionInput(action="approve", request_id="ui-1"))
    payload = msgspec.json.decode(reply.commands[0], type=dict[str, object])
    assert payload == {"type": "extension_ui_response", "id": "ui-1", "confirmed": True}
    assert event(reply, "permission", "submitted").event_id == "ui-1"
    assert "approve" not in session.actions


def test_stale_rejected_and_unsupported_records_are_visible() -> None:
    session = OmpSession(("omp",), Path("/repo"))
    _ = session.start("run")
    stale = session.receive(
        frame({"id": "unknown", "type": "response", "command": "prompt", "success": True})
    )
    assert event(stale, "error", "stale_response").text == (
        "OMP response for unknown request unknown"
    )

    unsupported = session.receive(
        frame({"type": "extension_ui_request", "id": "ui-2", "method": "custom_dialog"})
    )
    assert (
        event(unsupported, "error", "unsupported_ui").text
        == "Unsupported OMP extension UI request: custom_dialog"
    )
    with pytest.raises(ValueError, match="Unknown or stale OMP UI request"):
        _ = session.submit(SessionInput(action="deny", request_id="ui-2"))

    rejected = session.receive(
        frame(
            {
                "id": "prompt-1",
                "type": "response",
                "command": "prompt",
                "success": False,
                "error": "permission denied",
            }
        )
    )
    assert event(rejected, "user", "rejected").text == "permission denied"
    assert event(rejected, "error", "rejected").text == "permission denied"


def test_extension_error_rejects_pending_unsupported_input() -> None:
    session = OmpSession(("omp",), Path("/repo"))
    started = session.start("run")
    request_id = msgspec.json.decode(started.commands[0], type=dict[str, object])["id"]

    rejected = session.receive(
        frame(
            {
                "type": "extension_error",
                "id": request_id,
                "message": "unsupported OMP input",
            }
        )
    )

    assert rejected.done is False
    assert rejected.failed is True
    assert event(rejected, "user", "rejected").event_id == request_id
    assert event(rejected, "user", "rejected").text == "unsupported OMP input"
    assert event(rejected, "error", "rejected").text == "unsupported OMP input"

    resolved = session.receive(
        frame({"id": request_id, "type": "response", "command": "prompt", "success": True})
    )
    assert event(resolved, "error", "resolved_response").text == (
        f"OMP response for unknown request {request_id}"
    )


def test_extension_error_without_pending_input_is_visible() -> None:
    session = OmpSession(("omp",), Path("/repo"))

    rejected = session.receive(
        frame(
            {
                "type": "extension_error",
                "id": "missing-1",
                "error": "unsupported OMP input",
            }
        )
    )

    assert rejected.done is False
    assert rejected.failed is True
    error = event(rejected, "error", "extension_error")
    assert error.event_id == "missing-1"
    assert error.text == "unsupported OMP input"


def test_interrupt_reports_only_actual_aborted_terminal_turn() -> None:
    session = OmpSession(("omp",), Path("/repo"))
    started = session.start("long task")
    prompt_id = msgspec.json.decode(started.commands[0], type=dict[str, object])["id"]
    _ = session.receive(
        frame({"id": prompt_id, "type": "response", "command": "prompt", "success": True})
    )
    submitted = session.submit(SessionInput(action="interrupt", request_id="interrupt-1"))
    assert event(submitted, "status", "submitted").event_id == "interrupt-1"
    accepted = session.receive(
        frame({"id": "interrupt-1", "type": "response", "command": "abort", "success": True})
    )
    assert event(accepted, "status", "queued").event_id == "interrupt-1"
    _ = session.receive(
        frame(
            {
                "type": "message_end",
                "message": {"role": "assistant", "content": [], "stopReason": "aborted"},
            }
        )
    )
    finished = session.receive(frame({"type": "agent_end", "isTerminal": True, "messages": []}))
    assert finished.done is True
    assert finished.interrupted is True
    assert finished.failed is False


def test_duplicate_start_and_inactive_submit_are_rejected() -> None:
    session = OmpSession(("omp",), Path("/repo"))
    _ = session.start("first")

    with pytest.raises(ValueError, match="OMP session already started"):
        _ = session.start("second")

    finished = session.receive(frame({"type": "agent_end", "isTerminal": True, "messages": []}))
    assert finished.done is True
    assert finished.failed is False
    with pytest.raises(ValueError, match="OMP session is inactive"):
        _ = session.submit(SessionInput(action="follow_up", text="late input"))


def test_malformed_rpc_frames_are_reported_without_false_events() -> None:
    session = OmpSession(("omp",), Path("/repo"))

    syntax = session.receive(b"not valid json\n")
    syntax_error = event(syntax, "error", "invalid_frame")
    assert syntax_error.text.startswith("Invalid OMP RPC frame:")
    assert syntax.failed is False

    typed = session.receive(frame({"type": 42}))
    typed_error = event(typed, "error", "invalid_frame")
    assert typed_error.text.startswith("Invalid OMP RPC frame:")
    assert typed.failed is False


def test_rpc_chunks_reject_missing_metadata_and_invalid_base64() -> None:
    session = OmpSession(("omp",), Path("/repo"))

    missing = session.receive(
        frame({"type": "rpc_chunk", "chunkId": "chunk-1", "count": 1, "byteLength": 1})
    )
    assert event(missing, "error", "invalid_chunk").text == "Invalid OMP rpc_chunk metadata"

    encoded = session.receive(
        frame(
            {
                "type": "rpc_chunk",
                "chunkId": "chunk-2",
                "index": 0,
                "count": 1,
                "byteLength": 1,
                "data": "***",
            }
        )
    )
    assert event(encoded, "error", "invalid_chunk").text.startswith("Invalid OMP rpc_chunk data:")


def test_rpc_chunks_enforce_advertised_limit_and_declared_length() -> None:
    session = OmpSession(("omp",), Path("/repo"))
    _ = session.receive(frame({"type": "ready", "maxReassembledFrameBytes": 2}))

    oversized = session.receive(
        frame(
            {
                "type": "rpc_chunk",
                "chunkId": "chunk-1",
                "index": 0,
                "count": 1,
                "byteLength": 3,
                "data": base64.b64encode(b"abc").decode(),
            }
        )
    )
    assert event(oversized, "error", "invalid_chunk").text == (
        "OMP rpc_chunk exceeds advertised limits"
    )

    wrong_length = session.receive(
        frame(
            {
                "type": "rpc_chunk",
                "chunkId": "chunk-2",
                "index": 0,
                "count": 1,
                "byteLength": 2,
                "data": base64.b64encode(b"x").decode(),
            }
        )
    )
    assert event(wrong_length, "error", "invalid_chunk").text == (
        "OMP rpc_chunk ended with wrong byte length"
    )


def test_rpc_chunks_reject_interleaving_and_recover_for_next_record() -> None:
    session = OmpSession(("omp",), Path("/repo"))
    payload = msgspec.json.encode({"type": "agent_settled"})
    midpoint = len(payload) // 2

    partial = session.receive(
        frame(
            {
                "type": "rpc_chunk",
                "chunkId": "chunk-1",
                "index": 0,
                "count": 2,
                "byteLength": len(payload),
                "data": base64.b64encode(payload[:midpoint]).decode(),
            }
        )
    )
    assert partial.commands == ()
    assert partial.events == ()

    interleaved = session.receive(
        frame(
            {
                "type": "rpc_chunk",
                "chunkId": "chunk-2",
                "index": 1,
                "count": 2,
                "byteLength": len(payload),
                "data": base64.b64encode(payload[midpoint:]).decode(),
            }
        )
    )
    assert event(interleaved, "error", "invalid_chunk").text == (
        "OMP rpc_chunk sequence was interleaved or interrupted"
    )

    recovered = session.receive(
        frame(
            {
                "type": "rpc_chunk",
                "chunkId": "chunk-3",
                "index": 0,
                "count": 1,
                "byteLength": len(payload),
                "data": base64.b64encode(payload).decode(),
            }
        )
    )
    assert event(recovered, "status", "settled").text == "OMP session settled"


def test_rpc_chunks_reject_non_utf8_payloads() -> None:
    session = OmpSession(("omp",), Path("/repo"))

    invalid = session.receive(
        frame(
            {
                "type": "rpc_chunk",
                "chunkId": "chunk-1",
                "index": 0,
                "count": 1,
                "byteLength": 1,
                "data": base64.b64encode(b"\xff").decode(),
            }
        )
    )
    assert event(invalid, "error", "invalid_chunk").text.startswith("OMP rpc_chunk is not UTF-8:")


def test_rpc_response_without_correlation_is_visible() -> None:
    session = OmpSession(("omp",), Path("/repo"))

    missing = session.receive(frame({"type": "response", "success": True}))
    assert event(missing, "error", "stale_response").text == (
        "OMP response is missing correlation id or command"
    )
    assert missing.failed is False


def test_mismatched_rpc_response_is_not_an_acknowledgement() -> None:
    session = OmpSession(("omp",), Path("/repo"))
    started = session.start("run")
    request_id = msgspec.json.decode(started.commands[0], type=dict[str, object])["id"]

    mismatch = session.receive(
        frame({"id": request_id, "type": "response", "command": "steer", "success": True})
    )
    assert event(mismatch, "error", "response_mismatch").text == (
        "OMP response command mismatch: expected prompt, got steer"
    )
    assert all(item.kind != "user" for item in mismatch.events)

    resolved = session.receive(
        frame({"id": request_id, "type": "response", "command": "prompt", "success": True})
    )
    assert event(resolved, "error", "resolved_response").text == (
        f"OMP response for unknown request {request_id}"
    )


def test_pending_identical_inputs_are_delivered_to_distinct_request_ids() -> None:
    session = OmpSession(("omp",), Path("/repo"))
    started = session.start("initial")
    prompt_id = msgspec.json.decode(started.commands[0], type=dict[str, object])["id"]
    accepted = session.receive(
        frame({"id": prompt_id, "type": "response", "command": "prompt", "success": True})
    )
    assert accepted.events == ()

    first = session.submit(SessionInput(action="follow_up", request_id="follow-a", text="repeat"))
    second = session.submit(SessionInput(action="follow_up", request_id="follow-b", text="repeat"))
    assert first.events == ()
    assert second.events == ()

    first_echo = session.receive(
        frame({"type": "message_start", "message": {"role": "user", "content": "repeat"}})
    )
    second_echo = session.receive(
        frame({"type": "message_start", "message": {"role": "user", "content": "repeat"}})
    )
    assert event(first_echo, "user", "delivered").event_id == "follow-a"
    assert event(second_echo, "user", "delivered").event_id == "follow-b"


def test_permission_select_reply_validates_and_preserves_selected_value() -> None:
    session = OmpSession(("omp",), Path("/repo"))
    _ = session.start("choose a mode")
    requested = session.receive(
        frame(
            {
                "type": "extension_ui_request",
                "id": "ui-select",
                "method": "select",
                "title": "Pick a mode",
                "options": ["safe", "fast"],
            }
        )
    )
    assert event(requested, "permission", "requested").text == "Pick a mode: safe, fast"

    with pytest.raises(ValueError, match=r"OMP select reply must match one of \('safe', 'fast'\)"):
        _ = session.submit(SessionInput(action="approve", request_id="ui-select", text="unknown"))

    reply = session.submit(SessionInput(action="approve", request_id="ui-select", text="fast"))
    payload = msgspec.json.decode(reply.commands[0], type=dict[str, object])
    assert payload == {
        "type": "extension_ui_response",
        "id": "ui-select",
        "value": "fast",
    }
    assert event(reply, "permission", "submitted").text == "Pick a mode"


def test_permission_denial_and_text_replies_use_exact_wire_shapes() -> None:
    session = OmpSession(("omp",), Path("/repo"))
    _ = session.start("handle permissions")

    _ = session.receive(
        frame(
            {
                "type": "extension_ui_request",
                "id": "ui-confirm",
                "method": "confirm",
                "title": "Apply change?",
            }
        )
    )
    denied = session.submit(SessionInput(action="deny", request_id="ui-confirm"))
    denied_payload = msgspec.json.decode(denied.commands[0], type=dict[str, object])
    assert denied_payload == {
        "type": "extension_ui_response",
        "id": "ui-confirm",
        "confirmed": False,
    }

    _ = session.receive(
        frame(
            {
                "type": "extension_ui_request",
                "id": "ui-input",
                "method": "input",
                "title": "Environment name",
            }
        )
    )
    entered = session.submit(SessionInput(action="approve", request_id="ui-input", text="staging"))
    entered_payload = msgspec.json.decode(entered.commands[0], type=dict[str, object])
    assert entered_payload == {
        "type": "extension_ui_response",
        "id": "ui-input",
        "value": "staging",
    }

    _ = session.receive(
        frame(
            {
                "type": "extension_ui_request",
                "id": "ui-editor",
                "method": "editor",
                "title": "Edit instructions",
            }
        )
    )
    cancelled = session.submit(SessionInput(action="deny", request_id="ui-editor"))
    cancelled_payload = msgspec.json.decode(cancelled.commands[0], type=dict[str, object])
    assert cancelled_payload == {
        "type": "extension_ui_response",
        "id": "ui-editor",
        "cancelled": True,
    }
    assert session.actions == ("steer", "follow_up", "interrupt")


def test_passive_ui_requests_are_status_only_and_do_not_add_actions() -> None:
    session = OmpSession(("omp",), Path("/repo"))
    _ = session.start("show status")

    notification = session.receive(
        frame(
            {
                "type": "extension_ui_request",
                "id": "ui-notify",
                "method": "notify",
                "message": "Edit saved",
            }
        )
    )
    assert notification.commands == ()
    assert event(notification, "status", "notify").text == "Edit saved"

    status = session.receive(
        frame(
            {
                "type": "extension_ui_request",
                "id": "ui-status",
                "method": "setStatus",
                "title": "Waiting for agent",
            }
        )
    )
    assert status.commands == ()
    assert event(status, "status", "setStatus").text == "Waiting for agent"
    assert session.actions == ("steer", "follow_up", "interrupt")


def test_malformed_ui_request_is_observable_without_permission_actions() -> None:
    session = OmpSession(("omp",), Path("/repo"))
    _ = session.start("inspect")

    invalid = session.receive(frame({"type": "extension_ui_request", "id": "ui-missing-method"}))
    assert event(invalid, "error", "invalid_ui_request").text == (
        "OMP UI request is missing id or method"
    )
    assert session.actions == ("steer", "follow_up", "interrupt")


def test_assistant_stream_error_is_visible_and_fails_the_terminal_turn() -> None:
    session = OmpSession(("omp",), Path("/repo"))
    _ = session.start("answer")

    stream_error = session.receive(
        frame(
            {
                "type": "message_update",
                "assistantMessageEvent": {
                    "type": "error",
                    "error": {"message": "model stream failed"},
                },
            }
        )
    )
    error = event(stream_error, "error", "assistant_error")
    assert error.text == "model stream failed"
    assert error.event_id == "assistant-2"
    assert stream_error.failed is True

    finished = session.receive(frame({"type": "agent_end", "isTerminal": True, "messages": []}))
    assert finished.done is True
    assert finished.failed is True


def test_assistant_message_end_error_preserves_text_and_failure_reason() -> None:
    session = OmpSession(("omp",), Path("/repo"))
    _ = session.start("answer")
    _ = session.receive(
        frame(
            {
                "type": "message_start",
                "message": {
                    "role": "assistant",
                    "id": "assistant-msg",
                    "content": "partial answer",
                },
            }
        )
    )

    ended = session.receive(
        frame(
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "id": "assistant-msg",
                    "content": "partial answer",
                    "stopReason": "error",
                    "errorMessage": "provider disconnected",
                },
            }
        )
    )
    assert event(ended, "assistant", "complete").text == "partial answer"
    assert event(ended, "error", "assistant_error").text == "provider disconnected"

    finished = session.receive(frame({"type": "agent_end", "isTerminal": True, "messages": []}))
    assert finished.done is True
    assert finished.failed is True
    assert finished.result_text == "partial answer"


def test_tool_streaming_error_keeps_call_identity_and_error_state() -> None:
    session = OmpSession(("omp",), Path("/repo"))
    _ = session.start("run the tool")

    update = session.receive(
        frame(
            {
                "type": "tool_execution_update",
                "toolCallId": "call-error",
                "partialResult": {"error": "permission denied"},
            }
        )
    )
    update_event = event(update, "tool", "streaming")
    assert update_event.event_id == "call-error"
    assert update_event.text == "permission denied"
    assert update_event.delta is True

    ended = session.receive(
        frame(
            {
                "type": "tool_execution_end",
                "toolCallId": "call-error",
                "result": {"error": "permission denied"},
                "isError": True,
            }
        )
    )
    tool_error = event(ended, "tool", "error")
    assert tool_error.event_id == "call-error"
    assert tool_error.text == "permission denied"


def test_non_agent_prompt_response_is_terminal_without_duplicate_receipt() -> None:
    session = OmpSession(("omp",), Path("/repo"))
    started = session.start("no-op")
    request_id = msgspec.json.decode(started.commands[0], type=dict[str, object])["id"]

    result = session.receive(
        frame(
            {
                "id": request_id,
                "type": "response",
                "command": "prompt",
                "success": True,
                "data": {"agentInvoked": False},
            }
        )
    )
    assert result.events == ()
    assert result.done is True
    assert result.failed is False
    assert result.interrupted is False


def test_prompt_result_without_agent_invocation_reports_complete() -> None:
    session = OmpSession(("omp",), Path("/repo"))
    _ = session.start("no-op")

    result = session.receive(
        frame(
            {
                "id": "prompt-result-1",
                "type": "prompt_result",
                "agentInvoked": False,
            }
        )
    )
    assert event(result, "status", "complete").text == (
        "OMP prompt completed without agent invocation"
    )
    assert result.done is True
    assert result.failed is False


def test_non_terminal_turn_and_settled_event_are_read_only() -> None:
    session = OmpSession(("omp",), Path("/repo"))
    _ = session.start("continue")

    running = session.receive(frame({"type": "agent_end", "isTerminal": False, "messages": []}))
    assert running.commands == ()
    assert running.done is False
    assert event(running, "status", "running").text == "OMP agent turn complete"

    settled = session.receive(frame({"type": "agent_settled"}))
    assert settled.commands == ()
    assert settled.done is False
    assert event(settled, "status", "settled").text == "OMP session settled"


def test_status_failure_and_unknown_events_remain_observable() -> None:
    session = OmpSession(("omp",), Path("/repo"))
    _ = session.start("retry")

    retry = session.receive(
        frame(
            {
                "type": "auto_retry_end",
                "success": False,
                "errorMessage": "retry budget exhausted",
            }
        )
    )
    assert event(retry, "status", "error").text == "retry budget exhausted"
    assert retry.failed is True

    unsupported = session.receive(frame({"type": "unexpected_event"}))
    assert event(unsupported, "error", "unsupported_event").text == (
        "Unsupported OMP RPC event: unexpected_event"
    )


def test_unmatched_user_echo_is_not_attributed_to_pending_input() -> None:
    session = OmpSession(("omp",), Path("/repo"))
    started = session.start("expected")
    request_id = msgspec.json.decode(started.commands[0], type=dict[str, object])["id"]

    echoed = session.receive(
        frame({"type": "message_start", "message": {"role": "user", "content": "unexpected"}})
    )
    receipt = event(echoed, "user", "delivered")
    assert receipt.text == "unexpected"
    assert receipt.event_id == "user-2"
    assert receipt.event_id != request_id


def test_rpc_chunks_require_zero_based_order() -> None:
    session = OmpSession(("omp",), Path("/repo"))
    payload = msgspec.json.encode({"type": "agent_settled"})

    out_of_order = session.receive(
        frame(
            {
                "type": "rpc_chunk",
                "chunkId": "chunk-1",
                "index": 1,
                "count": 2,
                "byteLength": len(payload),
                "data": base64.b64encode(payload).decode(),
            }
        )
    )
    assert event(out_of_order, "error", "invalid_chunk").text == (
        "OMP rpc_chunk sequence must start at index 0"
    )

    recovered = session.receive(
        frame(
            {
                "type": "rpc_chunk",
                "chunkId": "chunk-2",
                "index": 0,
                "count": 1,
                "byteLength": len(payload),
                "data": base64.b64encode(payload).decode(),
            }
        )
    )
    assert event(recovered, "status", "settled").text == "OMP session settled"


def test_rpc_chunks_reject_piece_longer_than_declared_length() -> None:
    session = OmpSession(("omp",), Path("/repo"))

    oversized_piece = session.receive(
        frame(
            {
                "type": "rpc_chunk",
                "chunkId": "chunk-1",
                "index": 0,
                "count": 1,
                "byteLength": 1,
                "data": base64.b64encode(b"too long").decode(),
            }
        )
    )
    assert event(oversized_piece, "error", "invalid_chunk").text == (
        "OMP rpc_chunk byte length mismatch"
    )


def test_prompt_result_with_agent_invocation_is_only_queued() -> None:
    session = OmpSession(("omp",), Path("/repo"))
    _ = session.start("work")

    scheduled = session.receive(
        frame(
            {
                "id": "prompt-result-1",
                "type": "prompt_result",
                "agentInvoked": True,
            }
        )
    )
    status = event(scheduled, "status", "queued")
    assert status.event_id == "prompt-result-1"
    assert status.text == "OMP prompt scheduled"
    assert scheduled.done is False


def test_message_frames_require_messages_and_supported_roles() -> None:
    session = OmpSession(("omp",), Path("/repo"))
    _ = session.start("inspect")

    missing = session.receive(frame({"type": "message_start"}))
    assert event(missing, "error", "invalid_message").text == ("OMP message_start has no message")

    missing_event = session.receive(frame({"type": "message_update"}))
    assert event(missing_event, "error", "invalid_message").text == (
        "OMP message_update has no assistant event"
    )

    unsupported = session.receive(
        frame(
            {
                "type": "message_start",
                "message": {"role": "system", "content": "not a user message"},
            }
        )
    )
    assert event(unsupported, "error", "unsupported_message").text == (
        "Unsupported OMP message role: system"
    )
