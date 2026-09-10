from __future__ import annotations

import os
import queue
import subprocess
import textwrap
import threading
import time
from collections.abc import Callable
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from typing import IO, cast

import pytest

from milknado.domains.common import SessionEvent, SessionInput
from milknado.loop._agent import AgentRunSpec
from milknado.loop.sessions import SessionChannel, run_session
from milknado.loop.sessions._process import (
    MAX_FRAME_SIZE,
    MAX_STDERR_LINE_SIZE,
    Line,
    reader,
)

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
    "pending": """
for raw in sys.stdin:
    if json.loads(raw).get("type") == "user":
        emit({"type": "result", "subtype": "success", "result": "first result"})
""",
    "diagnostic": """
for raw in sys.stdin:
    if json.loads(raw).get("type") == "user":
        print("raw diagnostic: policy fallback", file=sys.stderr, flush=True)
        emit({"type": "result", "subtype": "success", "result": "logged result"})
""",
    "spoof": """
for raw in sys.stdin:
    if json.loads(raw).get("type") == "user":
        emit(json.loads(raw))
        emit({"type": "result", "subtype": "success", "result": "work remains"})
""",
    "reader": """
pid_path = Path(sys.argv[1])
pid_path.write_text(str(os.getpid()), encoding="utf-8")
time.sleep(0.2)
""",
}


def _worker(tmp_path: Path, mode: str) -> Path:
    worker = tmp_path / "claude"
    _ = worker.write_text(_SCRIPT_HEADER + textwrap.dedent(_SCRIPTS[mode]), encoding="utf-8")
    worker.chmod(0o755)
    return worker


def _spec(
    worker: Path,
    tmp_path: Path,
    args: tuple[str, ...] = (),
    *,
    iteration: int = 1,
) -> AgentRunSpec:
    return AgentRunSpec(
        cmd=[str(worker), *args],
        prompt="initial prompt",
        timeout=2.0,
        log_dir=None,
        iteration=iteration,
        capture_result_text=True,
        cwd=tmp_path,
    )


def _wait_for_pid(pid_path: Path) -> int:
    deadline = time.monotonic() + 2.0
    while not pid_path.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert pid_path.exists()
    return int(pid_path.read_text(encoding="utf-8"))


def _assert_dead(pid: int) -> None:
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.02)
    pytest.fail(f"worker process {pid} survived cleanup")


def _utf8_frame(limit: int, *, oversized: bool = False) -> bytes:
    if oversized:
        return ("é" * (limit // 2) + "\n").encode("utf-8")
    return ("é" * ((limit - 2) // 2) + "a\n").encode("utf-8")


@pytest.mark.parametrize(
    ("stream", "limit"),
    (("stdout", MAX_FRAME_SIZE), ("stderr", MAX_STDERR_LINE_SIZE)),
)
def test_reader_accepts_multibyte_frame_at_byte_limit(stream: str, limit: int) -> None:
    payload = _utf8_frame(limit)
    lines: queue.Queue[Line] = queue.Queue()

    reader(stream, BytesIO(payload), lines, threading.Event())

    assert len(payload) == limit
    assert lines.get_nowait() == Line(stream, payload.decode("utf-8"))
    assert lines.get_nowait() == Line(stream, None)


@pytest.mark.parametrize(
    ("stream", "limit"),
    (("stdout", MAX_FRAME_SIZE), ("stderr", MAX_STDERR_LINE_SIZE)),
)
def test_reader_rejects_multibyte_frame_over_byte_limit(stream: str, limit: int) -> None:
    payload = _utf8_frame(limit, oversized=True)
    lines: queue.Queue[Line] = queue.Queue()

    reader(stream, BytesIO(payload), lines, threading.Event())

    assert len(payload) == limit + 1
    assert lines.get_nowait() == Line(
        "reader_error", f"{stream}: frame exceeds the {limit}-byte limit"
    )
    assert lines.get_nowait() == Line(stream, None)


def test_pending_input_after_terminal_result_is_rejected(tmp_path: Path) -> None:
    worker = _worker(tmp_path, "pending")
    channel = SessionChannel()
    events: list[SessionEvent] = []

    def sink(event: SessionEvent) -> None:
        events.append(event)
        if event.kind == "assistant" and event.state == "complete":
            assert channel.submit(
                SessionInput(action="follow_up", request_id="late-1", text="late input")
            )

    channel.set_sink(sink)
    result = run_session(_spec(worker, tmp_path), channel)

    assert result.returncode == 0
    assert result.result_text == "first result"
    late_events = [event for event in events if event.text == "late input"]
    assert [(event.kind, event.state, event.text) for event in late_events[-1:]] == [
        ("user", "rejected", "late input")
    ]
    assert all(event.state != "delivered" for event in late_events)
    assert not channel.view().active


class _FailingReader:
    def __init__(self, stream: IO[bytes]) -> None:
        self._stream: IO[bytes] = stream
        self._raised: bool = False

    def readline(self, size: int = -1) -> bytes:
        if not self._raised:
            self._raised = True
            raise OSError("forced stdout reader failure")
        return self._stream.readline(size)

    def close(self) -> None:
        self._stream.close()


def test_reader_failure_is_visible_and_cleans_up_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = _worker(tmp_path, "reader")
    pid_path = tmp_path / "reader.pid"
    real_popen = cast(Callable[..., subprocess.Popen[bytes]], subprocess.Popen)

    def spawn(*args: object, **kwargs: object) -> subprocess.Popen[bytes]:
        process = real_popen(*args, **kwargs)
        _ = pid_path.write_text(str(process.pid), encoding="utf-8")
        assert process.stdout is not None
        process.stdout = cast(IO[bytes], cast(object, _FailingReader(process.stdout)))
        return process

    monkeypatch.setattr(subprocess, "Popen", spawn)
    result = run_session(_spec(worker, tmp_path, (str(pid_path),)), SessionChannel())

    assert result.returncode != 0
    assert result.timed_out is False
    assert result.captured_stderr == "stdout: forced stdout reader failure\n"
    _assert_dead(_wait_for_pid(pid_path))


def test_configured_log_retains_raw_diagnostic_output(tmp_path: Path) -> None:
    worker = _worker(tmp_path, "diagnostic")
    log_dir = tmp_path / "logs"
    result = run_session(
        replace(_spec(worker, tmp_path, iteration=7), log_dir=log_dir),
        SessionChannel(),
    )

    assert result.returncode == 0
    assert result.captured_stderr == "raw diagnostic: policy fallback\n"
    assert result.log_file is not None
    assert result.log_file.parent == log_dir
    assert result.log_file.name.startswith("0007_")
    log_text = result.log_file.read_text(encoding="utf-8")
    assert "raw diagnostic: policy fallback\n" in log_text


def test_raw_stdout_cannot_claim_structured_completion(tmp_path: Path) -> None:
    worker = _worker(tmp_path, "spoof")
    result = run_session(
        replace(
            _spec(worker, tmp_path),
            prompt="Emit <promise>DONE</promise> only when the task is complete.",
            completion_signal="DONE",
        ),
        SessionChannel(),
    )

    assert result.returncode == 0
    assert result.result_text == "work remains"
    assert result.completion_detected is False
