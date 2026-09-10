from __future__ import annotations

import os
import shlex
import textwrap
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from milknado.domains.common import SessionContext, SessionEvent, SessionInput
from milknado.loop._agent import AgentResult, AgentRunSpec
from milknado.loop._events import OutputStream
from milknado.loop._run_types import RunConfig, RunState, RunStatus
from milknado.loop.engine import run_loop
from milknado.loop.sessions import SessionChannel, run_session

_SCRIPT_HEADER = """#!/usr/bin/env python3
import json
import os
import sys
import time
from pathlib import Path


def emit(payload):
    print(json.dumps(payload), flush=True)
"""

_SCRIPTS = {
    "drain": """
for raw in sys.stdin:
    if json.loads(raw).get("type") == "user":
        emit({"type": "system", "subtype": "ready", "session_id": "drain"})
        print("stderr message", file=sys.stderr, flush=True)
        emit({
            "type": "result",
            "subtype": "success",
            "result": "drained",
            "session_id": "drain",
        })
        break
""",
    "followup": """
count = 0
for raw in sys.stdin:
    if json.loads(raw).get("type") == "user":
        count += 1
        emit({
            "type": "result",
            "subtype": "success",
            "result": f"result-{count}",
            "session_id": "followup",
        })
        if count == 2:
            break
""",
    "hang": """
pid_path = Path(sys.argv[2])
pid_path.write_text(str(os.getpid()), encoding="utf-8")
while True:
    time.sleep(0.05)
""",
    "sink": """
pid_path = Path(sys.argv[2])
pid_path.write_text(str(os.getpid()), encoding="utf-8")
for raw in sys.stdin:
    if json.loads(raw).get("type") == "user":
        emit({"type": "system", "subtype": "boom", "session_id": "sink"})
        while True:
            time.sleep(0.05)
""",
    "engine": """
counter = Path(sys.argv[2])
count = int(counter.read_text(encoding="utf-8") or "0") + 1
counter.write_text(str(count), encoding="utf-8")
for raw in sys.stdin:
    payload = json.loads(raw)
    if payload.get("type") == "control_request":
        request = payload.get("request", {})
        if count == 1 and request.get("subtype") == "interrupt":
            emit({
                "type": "result",
                "subtype": "interrupted",
                "result": "interrupted",
                "session_id": "engine",
            })
            break
    if count > 1 and payload.get("type") == "user":
        emit({
            "type": "result",
            "subtype": "success",
            "result": "later success",
            "session_id": "engine",
        })
        break
""",
}


def _worker(tmp_path: Path, mode: str) -> Path:
    worker = tmp_path / "claude"
    _ = worker.write_text(_SCRIPT_HEADER + textwrap.dedent(_SCRIPTS[mode]), encoding="utf-8")
    _ = worker.chmod(0o755)
    return worker


def _spec(worker: Path, tmp_path: Path, args: tuple[str, ...]) -> AgentRunSpec:
    return AgentRunSpec(
        cmd=[str(worker), *args],
        prompt="initial prompt",
        timeout=5.0,
        log_dir=None,
        iteration=1,
        capture_result_text=True,
        cwd=tmp_path,
    )


def _wait_for_pid(pid_path: Path) -> int:
    deadline = time.monotonic() + 3.0
    while not pid_path.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert pid_path.exists()
    return int(pid_path.read_text(encoding="utf-8"))


def _assert_dead(pid: int) -> None:
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.02)
    pytest.fail(f"worker process {pid} survived cleanup")


def test_run_session_drains_stdout_and_stderr(tmp_path: Path) -> None:
    worker = _worker(tmp_path, "drain")
    events: list[SessionEvent] = []
    output_lines: list[tuple[str, OutputStream]] = []

    def output_line(text: str, stream: OutputStream) -> None:
        output_lines.append((text, stream))

    result = run_session(
        replace(_spec(worker, tmp_path, ("drain",)), on_output_line=output_line),
        SessionChannel(events.append),
    )

    stdout_lines = [text for text, stream in output_lines if stream == "stdout"]
    stderr_lines = [text for text, stream in output_lines if stream == "stderr"]
    assert result.returncode == 0
    assert result.result_text == "drained"
    assert '"subtype": "ready"' in (result.captured_stdout or "")
    assert "stderr message" in (result.captured_stderr or "")
    assert len(stdout_lines) == 2
    assert all(line.startswith('{"type":') for line in stdout_lines)
    assert all('"session_id": "drain"' in line for line in stdout_lines)
    assert stderr_lines == ["stderr message\n"]
    assert any(event.text == "Claude ready" for event in events)
    assert any(event.state == "stopped" for event in events)


def test_run_session_sends_follow_up_before_first_terminal_result(tmp_path: Path) -> None:
    worker = _worker(tmp_path, "followup")
    channel = SessionChannel()
    events: list[SessionEvent] = []
    follow_up_admitted: list[bool] = []

    def sink(event: SessionEvent) -> None:
        events.append(event)
        if event.kind == "user" and event.state == "submitted" and not follow_up_admitted:
            follow_up_admitted.append(
                channel.submit(
                    SessionInput(action="follow_up", text="second prompt", request_id="follow-1")
                )
            )

    channel.set_sink(sink)
    result = run_session(_spec(worker, tmp_path, ("followup",)), channel)

    assert result.result_text == "result-2"
    assert follow_up_admitted == [True]
    assert {event.text for event in events if event.kind == "assistant"} == {
        "result-1",
        "result-2",
    }
    assert any(
        event.kind == "user" and event.text == "second prompt" and event.state == "submitted"
        for event in events
    )


def test_run_session_timeout_cleans_up_process_group(tmp_path: Path) -> None:
    worker = _worker(tmp_path, "hang")
    pid_path = tmp_path / "timeout.pid"
    spec = replace(_spec(worker, tmp_path, ("hang", str(pid_path))), timeout=1.0)

    result = run_session(spec, SessionChannel())

    assert result.timed_out
    _assert_dead(_wait_for_pid(pid_path))


def test_run_session_force_stop_cleans_up_process_group(tmp_path: Path) -> None:
    worker = _worker(tmp_path, "hang")
    pid_path = tmp_path / "force.pid"
    force_stop = threading.Event()
    spec = replace(
        _spec(worker, tmp_path, ("hang", str(pid_path))),
        force_stop_event=force_stop,
    )
    results: list[AgentResult] = []
    thread = threading.Thread(target=lambda: results.append(run_session(spec, SessionChannel())))
    thread.start()
    pid = _wait_for_pid(pid_path)
    force_stop.set()
    thread.join(timeout=3.0)

    assert not thread.is_alive()
    assert results and results[0].force_stopped
    _assert_dead(pid)


def test_run_session_propagates_sink_failure_and_cleans_up(tmp_path: Path) -> None:
    worker = _worker(tmp_path, "sink")
    pid_path = tmp_path / "sink.pid"

    def sink(event: SessionEvent) -> None:
        if event.text == "Claude boom":
            raise RuntimeError("sink broke")

    channel = SessionChannel(sink=sink)
    with pytest.raises(RuntimeError, match="sink broke"):
        _ = run_session(_spec(worker, tmp_path, ("sink", str(pid_path))), channel)

    assert not channel.view().active
    _assert_dead(_wait_for_pid(pid_path))


def test_interrupted_iteration_does_not_stop_on_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = _worker(tmp_path, "engine")
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ.get("PATH", ""))
    counter = tmp_path / "iterations.txt"
    _ = counter.write_text("0", encoding="utf-8")
    channel = SessionChannel()
    interrupt_sent = False

    def sink(event: SessionEvent) -> None:
        nonlocal interrupt_sent
        if event.kind == "user" and event.state == "submitted" and not interrupt_sent:
            interrupt_sent = True
            assert channel.submit(SessionInput(action="interrupt"))

    channel.set_sink(sink)
    config = RunConfig(
        agent=shlex.join(("claude", "engine", str(counter))),
        ralph_dir=tmp_path,
        prompt="initial prompt",
        max_iterations=2,
        stop_on_error=True,
        project_root=tmp_path,
        session_context=SessionContext(family="claude", cwd=str(tmp_path)),
        session_sink=sink,
    )
    state = RunState(run_id="engine-test", session=channel)

    run_loop(config, state)

    assert state.status is RunStatus.COMPLETED
    assert state.iteration == 2
    assert state.interrupted == 1
    assert state.completed == 1
    assert state.failed == 0
    assert counter.read_text(encoding="utf-8") == "2"
