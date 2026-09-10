from __future__ import annotations

import textwrap
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import pytest

from milknado.domains.common import SessionContext, SessionEvent, SessionInput
from milknado.loop._agent import AgentRunSpec
from milknado.loop.sessions import SessionChannel, run_session

_CONTEXT = SessionContext(family="claude", cwd="/repo")
_WORKER = """
import json
import sys

for raw in sys.stdin:
    payload = json.loads(raw)
    if payload.get("type") != "user":
        continue
    text = payload["message"]["content"]
    print(
        json.dumps(
            {"type": "user", "uuid": f"echo-{text}", "message": {"content": text}}
        ),
        flush=True,
    )
    print(
        json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "result": text,
                "session_id": "receipt-test",
            }
        ),
        flush=True,
    )
"""


def test_channel_keeps_persisted_receipts_monotonic_and_unique() -> None:
    persisted: list[SessionEvent] = []
    channel = SessionChannel(sink=persisted.append)
    channel.start(_CONTEXT, ("follow_up",))
    command = SessionInput(action="follow_up", text="second", request_id="follow-1")

    assert channel.submit(command)
    (submitted,) = channel.drain()
    channel.publish(
        SessionEvent(
            kind="user",
            text=submitted.text,
            event_id=submitted.request_id,
            state="queued",
        )
    )
    channel.publish(
        SessionEvent(
            kind="user",
            text=submitted.text,
            event_id=submitted.request_id,
            state="delivered",
        )
    )

    assert [(event.state, event.text) for event in persisted] == [
        ("queued", "second"),
        ("submitted", "second"),
        ("delivered", "second"),
    ]


def test_runtime_persists_one_monotonic_receipt_per_follow_up(tmp_path: Path) -> None:
    worker = tmp_path / "claude"
    _ = worker.write_text("#!/usr/bin/env python3\n" + textwrap.dedent(_WORKER), encoding="utf-8")
    _ = worker.chmod(0o755)
    persisted: list[SessionEvent] = []
    channel = SessionChannel()

    def sink(event: SessionEvent) -> None:
        persisted.append(event)
        if event.kind == "user" and event.state == "submitted" and event.text == "first":
            assert channel.submit(
                SessionInput(action="follow_up", text="second", request_id="follow-1")
            )

    channel.set_sink(sink)
    result = run_session(
        AgentRunSpec(
            cmd=[str(worker)],
            prompt="first",
            timeout=5.0,
            log_dir=None,
            iteration=1,
            capture_result_text=True,
            cwd=tmp_path,
        ),
        channel,
    )

    receipts = [event for event in persisted if event.kind == "user" and event.text == "second"]
    assert [(event.state, event.text) for event in receipts] == [
        ("queued", "second"),
        ("submitted", "second"),
        ("delivered", "second"),
    ]
    assert result.result_text == "second"


_PERMISSION_WORKER = """
import json
import os
import sys
import time
from pathlib import Path

for raw in sys.stdin:
    payload = json.loads(raw)
    if payload.get("type") == "user":
        if sys.argv[1] == "closed":
            os.close(0)
        print(json.dumps({
            "type": "control_request",
            "request_id": "permission-write",
            "request": {
                "subtype": "can_use_tool",
                "tool_name": "Write",
                "input": {"file_path": "allowed.txt", "content": "approved"},
            },
        }), flush=True)
        if sys.argv[1] == "closed":
            time.sleep(30)
    elif payload.get("type") == "control_response":
        response = payload["response"]
        assert response["request_id"] == "permission-write"
        assert response["response"]["behavior"] == "allow"
        Path("allowed.txt").write_text("approved", encoding="utf-8")
        print(json.dumps({
            "type": "result", "subtype": "success", "result": "permission received"
        }), flush=True)
        break
"""


@pytest.mark.parametrize("pipe_state", ("open", "closed"))
def test_permission_decision_requires_successful_pipe_write(
    tmp_path: Path, pipe_state: str
) -> None:
    worker = tmp_path / "claude"
    _ = worker.write_text("#!/usr/bin/env python3\n" + _PERMISSION_WORKER, encoding="utf-8")
    _ = worker.chmod(0o755)
    persisted: list[SessionEvent] = []
    channel = SessionChannel()
    approvals: list[Future[bool]] = []

    with ThreadPoolExecutor(max_workers=1) as operator:

        def sink(event: SessionEvent) -> None:
            persisted.append(event)
            if event.kind == "permission" and event.state == "requested":
                approvals.append(
                    operator.submit(
                        channel.submit, SessionInput(action="approve", request_id=event.event_id)
                    )
                )

        channel.set_sink(sink)
        result = run_session(
            AgentRunSpec(
                cmd=[str(worker), pipe_state],
                prompt="write with permission",
                timeout=2.0,
                log_dir=None,
                iteration=1,
                capture_result_text=True,
                cwd=tmp_path,
            ),
            channel,
        )
    assert [approval.result() for approval in approvals] == [True]

    states = [event.state for event in persisted if event.kind == "permission"]
    if pipe_state == "open":
        assert states == ["requested", "submitted", "approved"]
        assert result.returncode == 0
        assert result.result_text == "permission received"
        assert (tmp_path / "allowed.txt").read_text(encoding="utf-8") == "approved"
    else:
        assert states == ["requested", "submitted", "cancelled"]
        assert result.returncode != 0
        assert not (tmp_path / "allowed.txt").exists()
