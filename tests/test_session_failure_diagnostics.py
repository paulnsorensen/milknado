from __future__ import annotations

import json
import sys
from pathlib import Path

from milknado.loop._agent import AgentResult, AgentRunSpec
from milknado.loop.sessions import SessionChannel, run_session


def _worker(tmp_path: Path, *, failed: bool) -> Path:
    executable = tmp_path / "claude"
    executable.symlink_to(sys.executable)
    payload = json.dumps(
        {
            "type": "result",
            "subtype": "error_during_execution" if failed else "success",
            "is_error": failed,
            "result": (
                "Cannot open sandbox directory; token=PRIVATE_SAMPLE"
                if failed
                else "Task completed"
            ),
            "raw_tool_payload": "RAW_TOOL_SENTINEL",
        }
    )
    script = tmp_path / "claude.py"
    status_frame = '{"type":"system","subtype":"status"}'
    noise = ""
    if failed:
        noise = f"        for _ in range(3500): print({status_frame!r}, flush=True)\n"
    _ = script.write_text(
        "import json, sys, time\n"
        + "for line in sys.stdin:\n"
        + "    if json.loads(line).get('type') == 'user':\n"
        + noise
        + f"        print({payload!r}, flush=True)\n"
        + "        time.sleep(0.2)\n",
        encoding="utf-8",
    )
    return executable


def _run(tmp_path: Path, *, failed: bool) -> AgentResult:
    executable = _worker(tmp_path, failed=failed)
    spec = AgentRunSpec(
        cmd=[str(executable), str(executable.with_suffix(".py"))],
        prompt="Work",
        timeout=2.0,
        log_dir=tmp_path / "logs",
        iteration=1,
        capture_result_text=True,
        cwd=tmp_path,
    )
    return run_session(spec, SessionChannel())


def test_native_failure_retains_redacted_parsed_diagnostic(tmp_path: Path) -> None:
    result = _run(tmp_path, failed=True)

    assert result.returncode != 0
    assert not result.terminal_confirmed
    assert result.captured_stdout is not None
    assert result.captured_stderr is not None
    assert "Cannot open sandbox directory" in result.captured_stdout
    assert "token=[REDACTED]" in result.captured_stdout
    assert result.log_file is not None
    log_text = result.log_file.read_text(encoding="utf-8")
    assert len(log_text) > 64 * 1024 - 1024
    assert "Cannot open sandbox directory" in log_text
    assert "token=[REDACTED]" in log_text
    for text in (result.captured_stdout, result.captured_stderr, log_text):
        assert "PRIVATE_SAMPLE" not in text
        assert "RAW_TOOL_SENTINEL" not in text
        assert len(text) <= 64 * 1024


def test_native_success_keeps_result_without_failure_diagnostic(tmp_path: Path) -> None:
    result = _run(tmp_path, failed=False)

    assert result.returncode == 0
    assert result.terminal_confirmed
    assert result.result_text == "Task completed"
    assert result.captured_stdout is not None
    assert "Task completed" not in result.captured_stdout
