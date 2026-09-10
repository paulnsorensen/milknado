from __future__ import annotations

import os
import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from milknado.loop._agent import AgentRunSpec, OutputLineCallback
from milknado.loop._events import OutputStream
from milknado.loop.sessions import SessionChannel, run_session
from milknado.loop.sessions._process import CAPTURE_LIMIT, MAX_FRAME_SIZE

_SCRIPT = """\
#!/usr/bin/env python3
import json
import os
import subprocess
import sys
import time
from pathlib import Path
def emit(payload):
    print(json.dumps(payload), flush=True)
def assistant(content):
    emit(
        {
            "type": "assistant",
            "session_id": "sid",
            "message": {"role": "assistant", "content": content},
        }
    )
mode = sys.argv[1]
marker = Path(sys.argv[2])
for _ in range(2):
    if not sys.stdin.readline():
        raise SystemExit(1)
if mode.endswith("-child"):
    child_pid = os.fork()
    if child_pid == 0:
        time.sleep(30)
        os._exit(0)
    marker.write_text(str(child_pid), encoding="utf-8")
if mode == "success-child":
    emit({"type": "result", "subtype": "success", "result": "done", "session_id": "sid"})
elif mode == "timeout-child":
    time.sleep(30)
elif mode == "force-child":
    assistant([{"type": "text", "text": "working"}])
    time.sleep(30)
elif mode == "stderr-cancel":
    marker.write_text(str(os.getpid()), encoding="utf-8")
    print("stderr cancellation", file=sys.stderr, flush=True)
    time.sleep(30)
elif mode.startswith("oversize-"):
    stream = sys.stdout if mode == "oversize-stdout" else sys.stderr
    stream.write("x" * (FRAME_LIMIT + 1))
    stream.flush()
    time.sleep(30)
elif mode == "soft-hook":
    config_dir = Path(os.environ.get("CLAUDE_CONFIG_DIR", ""))
    settings_path = config_dir / "settings.json"
    if not settings_path.is_file():
        marker.write_text("missing settings", encoding="utf-8")
        raise SystemExit(2)
    assistant([{"type": "tool_use", "id": "tool-1", "name": "Bash"}])
    counter_path = config_dir / "turncount"
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and counter_path.read_text().strip() != "1":
        time.sleep(0.01)
    if counter_path.read_text().strip() != "1":
        marker.write_text("counter not updated", encoding="utf-8")
        raise SystemExit(3)
    settings = json.loads(settings_path.read_text())
    command = settings["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
    hook_output = subprocess.run(command, shell=True, capture_output=True, text=True).stdout
    marker.write_text(hook_output, encoding="utf-8")
    emit({"type": "result", "subtype": "success", "result": "done", "session_id": "sid"})
elif mode == "tool-cap":
    assistant([{"type": "tool_use", "id": "tool-1", "name": "Bash"}])
    time.sleep(30)
else:
    raise SystemExit(f"unknown mode: {mode}")
"""


@dataclass(frozen=True, slots=True)
class _Scenario:
    mode: str
    marker: Path
    timeout: float | None = 2.0
    max_turns: int | None = None
    max_turns_grace: int = 0
    force_stop_event: threading.Event | None = None
    on_output_line: OutputLineCallback | None = None


def _worker(tmp_path: Path) -> Path:
    worker = tmp_path / "claude"
    script = _SCRIPT.replace("FRAME_LIMIT", str(MAX_FRAME_SIZE))
    _ = worker.write_text(script, encoding="utf-8")
    _ = worker.chmod(0o755)
    return worker


def _spec(worker: Path, tmp_path: Path, scenario: _Scenario) -> AgentRunSpec:
    return AgentRunSpec(
        cmd=[str(worker), scenario.mode, str(scenario.marker)],
        prompt="initial prompt",
        timeout=scenario.timeout,
        log_dir=None,
        iteration=1,
        capture_result_text=True,
        max_turns=scenario.max_turns,
        max_turns_grace=scenario.max_turns_grace,
        force_stop_event=scenario.force_stop_event,
        on_output_line=scenario.on_output_line,
        cwd=tmp_path,
    )


def _wait_for_pid(path: Path) -> int:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if path.exists():
            return int(path.read_text(encoding="utf-8"))
        time.sleep(0.01)
    raise AssertionError("child pid was not recorded")


def _assert_dead(pid: int) -> None:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            _ = os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.01)
    raise AssertionError(f"child process {pid} survived session cleanup")


@pytest.mark.parametrize("mode", ("success-child", "timeout-child", "force-child"))
def test_process_group_cleanup_kills_descendants(tmp_path: Path, mode: str) -> None:
    worker = _worker(tmp_path)
    child_pid_path = tmp_path / "child.pid"
    force_stop = threading.Event() if mode == "force-child" else None
    scenario = _Scenario(
        mode=mode,
        marker=child_pid_path,
        force_stop_event=force_stop,
        on_output_line=(
            lambda line, _stream: (
                force_stop.set() if force_stop is not None and "working" in line else None
            )
        ),
    )
    result = run_session(_spec(worker, tmp_path, scenario), SessionChannel())

    _assert_dead(_wait_for_pid(child_pid_path))
    assert result.timed_out is (mode == "timeout-child")
    assert result.force_stopped is (mode == "force-child")
    if mode == "success-child":
        assert result.result_text == "done"


def test_process_group_cleanup_consumes_identity_after_first_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if os.name == "nt":
        pytest.skip("process-group cleanup is POSIX-specific")
    signals: list[int] = []
    real_killpg = os.killpg

    def record_killpg(pgid: int, sig: int) -> None:
        if sig in (signal.SIGTERM, signal.SIGKILL):
            signals.append(sig)
        real_killpg(pgid, sig)

    monkeypatch.setattr(os, "killpg", record_killpg)
    worker = _worker(tmp_path)
    child_pid_path = tmp_path / "child.pid"
    result = run_session(
        _spec(worker, tmp_path, _Scenario(mode="success-child", marker=child_pid_path)),
        SessionChannel(),
    )

    _assert_dead(_wait_for_pid(child_pid_path))
    assert result.result_text == "done"
    assert signals.count(signal.SIGTERM) == 1
    assert signals.count(signal.SIGKILL) <= 1


def test_stderr_output_callback_can_stop_runtime(tmp_path: Path) -> None:
    worker = _worker(tmp_path)
    marker = tmp_path / "stderr-cancel.pid"
    force_stop = threading.Event()
    output_lines: list[tuple[str, OutputStream]] = []

    def cancel_on_stderr(text: str, stream: OutputStream) -> None:
        output_lines.append((text, stream))
        if stream == "stderr":
            force_stop.set()

    result = run_session(
        _spec(
            worker,
            tmp_path,
            _Scenario(
                mode="stderr-cancel",
                marker=marker,
                force_stop_event=force_stop,
                on_output_line=cancel_on_stderr,
            ),
        ),
        SessionChannel(),
    )

    _assert_dead(_wait_for_pid(marker))
    assert result.force_stopped is True
    assert output_lines == [("stderr cancellation\n", "stderr")]
    assert result.captured_stderr == "stderr cancellation\n"


@pytest.mark.parametrize("mode", ("oversize-stdout", "oversize-stderr"))
def test_oversize_output_is_bounded_visible_and_terminal(tmp_path: Path, mode: str) -> None:
    worker = _worker(tmp_path)
    channel = SessionChannel()
    result = run_session(
        _spec(
            worker,
            tmp_path,
            _Scenario(mode=mode, marker=tmp_path / "unused"),
        ),
        channel,
    )

    assert result.returncode != 0
    assert result.timed_out is False
    assert any(
        event.kind == "error" and "frame exceeds" in event.text for event in channel.view().events
    )
    captured = result.captured_stdout if mode == "oversize-stdout" else result.captured_stderr
    assert len(captured or "") <= CAPTURE_LIMIT


def test_max_turns_grace_uses_the_existing_soft_hook_and_hard_cap(tmp_path: Path) -> None:
    worker = _worker(tmp_path)
    marker = tmp_path / "hook.json"
    result = run_session(
        _spec(
            worker,
            tmp_path,
            _Scenario(
                mode="soft-hook",
                marker=marker,
                max_turns=2,
                max_turns_grace=1,
            ),
        ),
        SessionChannel(),
    )

    hook_output = marker.read_text(encoding="utf-8")
    assert result.returncode == 0
    assert result.turn_capped is False
    assert result.tool_use_count == 1
    assert "additionalContext" in hook_output
    assert "1 of 2" in hook_output


def test_structured_session_hard_cap_stops_at_the_tool_boundary(tmp_path: Path) -> None:
    worker = _worker(tmp_path)
    result = run_session(
        _spec(
            worker,
            tmp_path,
            _Scenario(mode="tool-cap", marker=tmp_path / "unused", max_turns=1),
        ),
        SessionChannel(),
    )

    assert result.returncode == 0
    assert result.turn_capped is True
    assert result.tool_use_count == 1
