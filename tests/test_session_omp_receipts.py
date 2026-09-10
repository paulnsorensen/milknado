from __future__ import annotations

import json
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import pytest

from milknado.domains.common import SessionEvent, SessionInput
from milknado.loop._agent import AgentRunSpec
from milknado.loop.sessions import SessionChannel, run_session

_SCRIPT = """#!/usr/bin/env python3
import json
import os
import sys
import time
from pathlib import Path

def emit(frame):
    print(json.dumps(frame), flush=True)

for raw in sys.stdin:
    frame = json.loads(raw)
    if frame['type'] == 'prompt':
        emit({'type': 'response', 'id': frame['id'], 'command': 'prompt', 'success': True})
        emit({'type': 'message_start', 'message': {'role': 'user', 'content': frame['message']}})
        if sys.argv[1] == 'closed':
            os.close(0)
        emit({'type': 'extension_ui_request', 'id': 'permission-1',
              'method': 'confirm', 'title': 'Apply edit?'})
        if sys.argv[1] == 'closed':
            time.sleep(30)
    elif frame['type'] == 'extension_ui_response':
        Path('decision.json').write_text(json.dumps(frame['confirmed']))
    elif frame['type'] == 'steer':
        emit({'type': 'response', 'id': frame['id'], 'command': 'steer', 'success': True})
        emit({'type': 'message_start', 'message': {'role': 'user', 'content': frame['message']}})
        emit({'type': 'message_end', 'message': {'role': 'assistant', 'content': 'done'}})
        emit({'type': 'agent_end', 'messages': [{'role': 'assistant',
              'content': [{'type': 'text', 'text': 'done'}]}]})
        break
"""


def _spec(tmp_path: Path, pipe_state: str) -> AgentRunSpec:
    worker = tmp_path / "omp"
    _ = worker.write_text(_SCRIPT, encoding="utf-8")
    worker.chmod(0o755)
    return AgentRunSpec(
        cmd=[str(worker), pipe_state],
        prompt="request permission",
        timeout=5.0,
        log_dir=None,
        iteration=1,
        capture_result_text=True,
        cwd=tmp_path,
    )


def _submit_when_available(channel: SessionChannel, command: SessionInput) -> bool:
    deadline = time.monotonic() + 3.0
    while command.action not in channel.view().actions:
        if time.monotonic() >= deadline:
            raise AssertionError(f"session action never became available: {command.action}")
        time.sleep(0.01)
    return channel.submit(command)


@pytest.mark.parametrize("decision", ("approve", "deny"))
@pytest.mark.parametrize("pipe_state", ("open", "closed"))
def test_omp_permission_settles_only_after_real_write(
    tmp_path: Path, decision: Literal["approve", "deny"], pipe_state: str
) -> None:
    events: list[SessionEvent] = []
    admissions: list[Future[bool]] = []
    channel = SessionChannel(max_inputs=1)
    with ThreadPoolExecutor(max_workers=1) as operator:

        def sink(event: SessionEvent) -> None:
            events.append(event)
            if event.kind != "permission":
                return
            if event.state == "requested":
                command = SessionInput(action=decision, request_id=event.event_id)
            elif event.state in {"approved", "denied"}:
                command = SessionInput(action="steer", text="continue")
            else:
                return
            admissions.append(operator.submit(_submit_when_available, channel, command))

        channel.set_sink(sink)
        result = run_session(_spec(tmp_path, pipe_state), channel)
    assert [item.result() for item in admissions] == (
        [True, True] if pipe_state == "open" else [True]
    ), ([(event.kind, event.state, event.event_id) for event in events], result.timed_out)
    terminal = {"approve": "approved", "deny": "denied"}[decision]
    terminal = terminal if pipe_state == "open" else "cancelled"
    assert [event.state for event in events if event.kind == "permission"] == [
        "requested",
        "submitted",
        terminal,
    ]
    if pipe_state == "open":
        assert result.returncode == 0
        assert result.result_text == "done"
        assert json.loads((tmp_path / "decision.json").read_text()) is (decision == "approve")
    else:
        assert result.returncode != 0
        assert not (tmp_path / "decision.json").exists()
