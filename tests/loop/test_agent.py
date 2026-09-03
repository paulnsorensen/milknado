"""Tests for the _agent module — subprocess execution, log writing, and stream parsing."""

import io
import itertools
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest.mock import ANY, MagicMock, call, patch

import pytest
from _pytest.capture import CaptureFixture
from _pytest.logging import LogCaptureFixture
from _pytest.monkeypatch import MonkeyPatch
from typing_extensions import override

import milknado.loop._agent as agent_module
from milknado.loop._agent import (
    _OUTPUT_TAIL_CHARS,  # pyright: ignore[reportPrivateUsage]
    _STREAM_QUEUE_MAX_LINES,  # pyright: ignore[reportPrivateUsage]
    AgentResult,
    AgentRunSpec,
    OutputLineCallback,
    _BoundedOutput,  # pyright: ignore[reportPrivateUsage]
    _extract_result_text_from_line,  # pyright: ignore[reportPrivateUsage]
    _kill_process_group,  # pyright: ignore[reportPrivateUsage]
    _pump_stream,  # pyright: ignore[reportPrivateUsage]
    _read_agent_stream,  # pyright: ignore[reportPrivateUsage]
    _ResolvedAgentRun,  # pyright: ignore[reportPrivateUsage]
    _run_agent_blocking,  # pyright: ignore[reportPrivateUsage]
    _run_agent_streaming,  # pyright: ignore[reportPrivateUsage]
    _terminate_lingering_group,  # pyright: ignore[reportPrivateUsage]
    execute_agent,
)
from milknado.loop._events import OutputStream
from milknado.loop.adapters import select_adapter
from milknado.loop.adapters.claude import ClaudeAdapter
from tests.loop.helpers import MOCK_SUBPROCESS, fail_proc, make_mock_popen, ok_proc, timeout_proc


class TestReadAgentStream:
    def test_collects_all_lines(self):
        stream = io.StringIO("line1\nline2\nline3\n")
        result = _read_agent_stream(stream, deadline=None, on_activity=None)

        assert result.stdout_lines is not None
        assert len(result.stdout_lines) == 3
        assert result.timed_out is False

    def test_force_stop_returns_captured_lines(self):
        force_stop = threading.Event()
        force_stop.set()

        result = _read_agent_stream(
            io.StringIO("line\n"),
            deadline=None,
            on_activity=None,
            force_stop_event=force_stop,
        )

        assert result.force_stopped is True
        assert result.stdout_lines == ()

    def test_parses_json_lines(self):
        activities = []
        stream = io.StringIO('{"type": "status", "msg": "ok"}\n')
        _ = _read_agent_stream(stream, deadline=None, on_activity=activities.append)  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]

        assert len(activities) == 1  # pyright: ignore[reportUnknownArgumentType]
        assert activities[0]["type"] == "status"

    def test_captures_result_text(self):
        stream = io.StringIO('{"type": "result", "result": "All done"}\n')
        result = _read_agent_stream(stream, deadline=None, on_activity=None)

        assert result.result_text == "All done"

    @pytest.mark.parametrize("line", ["42", '["result"]'], ids=["scalar", "list"])
    def test_result_text_extraction_ignores_non_object_json(self, line: str) -> None:
        assert _extract_result_text_from_line(line) is None

    def test_ignores_non_json_lines(self):
        activities = []
        stream = io.StringIO("not json\n")
        result = _read_agent_stream(stream, deadline=None, on_activity=activities.append)  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]

        assert result.stdout_lines is not None
        assert len(result.stdout_lines) == 1
        assert len(activities) == 0  # pyright: ignore[reportUnknownArgumentType]

    @pytest.mark.parametrize("line", ["42\n", '"hello"\n', "[1,2,3]\n", "true\n", "null\n"])
    def test_ignores_valid_json_non_object_lines(self, line: str):
        """Valid JSON that is not a JSON object (e.g. number, string, array)
        should be silently skipped, not crash with AttributeError."""
        activities = []
        stream = io.StringIO(line)
        result = _read_agent_stream(stream, deadline=None, on_activity=activities.append)  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]

        assert result.stdout_lines is not None
        assert len(result.stdout_lines) == 1
        assert len(activities) == 0  # pyright: ignore[reportUnknownArgumentType]
        assert result.result_text is None

    def test_skips_empty_lines(self):
        activities = []
        stream = io.StringIO("\n\n")
        result = _read_agent_stream(stream, deadline=None, on_activity=activities.append)  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]

        assert result.stdout_lines is not None
        assert len(result.stdout_lines) == 2
        assert len(activities) == 0  # pyright: ignore[reportUnknownArgumentType]

    def test_timeout_returns_early(self):
        stream = io.StringIO("line1\nline2\n")
        # Deadline already in the past — the reader thread may or may not
        # have queued lines yet, but timed_out must be True.
        result = _read_agent_stream(stream, deadline=0.001, on_activity=None)

        assert result.timed_out is True

    def test_timeout_preserves_already_read_lines(self):
        """When timeout fires after the reader thread has queued lines,
        those lines must still appear in stdout_lines ��� the non-blocking
        drain on an expired deadline must not silently discard them."""
        r_fd, w_fd = os.pipe()
        reader = os.fdopen(r_fd, "r")
        writer = os.fdopen(w_fd, "w")
        try:
            _ = writer.write("line1\nline2\n")
            writer.flush()
            # Don't close writer — stream stays open so reader blocks on
            # readline after the two lines, and the deadline fires.
            deadline = time.monotonic() + 0.5
            result = _read_agent_stream(reader, deadline=deadline, on_activity=None)

            assert result.timed_out is True
            assert result.stdout_lines is not None
            assert len(result.stdout_lines) >= 1
            assert result.stdout_lines[0] == "line1\n"
        finally:
            writer.close()
            reader.close()

    def test_timeout_still_processes_last_line(self):
        """When timeout fires on a line containing a result event, the result
        text must still be extracted — the deadline check should not skip
        processing of the already-read line."""
        r_fd, w_fd = os.pipe()
        reader = os.fdopen(r_fd, "r")
        writer = os.fdopen(w_fd, "w")
        try:
            _ = writer.write('{"type": "result", "result": "Done"}\n')
            writer.flush()
            deadline = time.monotonic() + 0.5
            result = _read_agent_stream(reader, deadline=deadline, on_activity=None)

            assert result.timed_out is True
            assert result.result_text == "Done"
        finally:
            writer.close()
            reader.close()

    def test_timeout_still_calls_on_activity_for_last_line(self):
        """When timeout fires on a line, on_activity must still be called
        for that line so the last event is not silently swallowed."""
        r_fd, w_fd = os.pipe()
        reader = os.fdopen(r_fd, "r")
        writer = os.fdopen(w_fd, "w")
        try:
            activities = []
            _ = writer.write('{"type": "status", "msg": "working"}\n')
            writer.flush()
            deadline = time.monotonic() + 0.5
            result = _read_agent_stream(reader, deadline=deadline, on_activity=activities.append)  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]

            assert result.timed_out is True
            assert len(activities) == 1  # pyright: ignore[reportUnknownArgumentType]
            assert activities[0]["type"] == "status"
        finally:
            writer.close()
            reader.close()

    def test_result_type_without_result_field_ignored(self):
        stream = io.StringIO('{"type": "result"}\n')
        result = _read_agent_stream(stream, deadline=None, on_activity=None)

        assert result.result_text is None

    def test_non_string_result_field_ignored(self):
        stream = io.StringIO('{"type": "result", "result": 42}\n')
        result = _read_agent_stream(stream, deadline=None, on_activity=None)

        assert result.result_text is None

    def test_last_result_wins(self):
        stream = io.StringIO(
            '{"type": "result", "result": "first"}\n{"type": "result", "result": "second"}\n'
        )
        result = _read_agent_stream(stream, deadline=None, on_activity=None)

        assert result.result_text == "second"

    def test_continues_when_on_output_line_raises(self):
        """A raising on_output_line callback must not crash the stream reader.

        This mirrors _pump_stream's behavior where callback exceptions are
        caught per-line so that draining continues.  Without this guard, a
        transient rendering error in the console emitter would kill the
        entire streaming run."""
        call_count = 0

        def raising_callback(line: str, stream: OutputStream):  # pyright: ignore[reportUnusedParameter]
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("boom")

        stream = io.StringIO("line1\nline2\nline3\n")
        result = _read_agent_stream(
            stream, deadline=None, on_activity=None, on_output_line=raising_callback
        )

        assert result.stdout_lines is not None
        assert len(result.stdout_lines) == 3
        assert call_count == 3
        assert result.timed_out is False

    def test_continues_when_on_activity_raises(self):
        """A raising on_activity callback must not crash the stream reader."""
        call_count = 0

        def raising_activity(data: dict[str, object]):  # pyright: ignore[reportUnusedParameter]
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("boom")

        stream = io.StringIO('{"type": "status", "msg": "a"}\n{"type": "status", "msg": "b"}\n')
        result = _read_agent_stream(stream, deadline=None, on_activity=raising_activity)

        assert result.stdout_lines is not None
        assert len(result.stdout_lines) == 2
        assert call_count == 2
        assert result.timed_out is False


class TestExecuteAgentBlocking:
    @patch(MOCK_SUBPROCESS, side_effect=ok_proc)
    def test_success(self, mock_popen: MagicMock):  # pyright: ignore[reportUnusedParameter]
        result = execute_agent(
            AgentRunSpec(["echo"], "prompt", timeout=None, log_dir=None, iteration=1)
        )

        assert result.returncode == 0
        assert result.timed_out is False
        assert result.log_file is None
        assert result.elapsed >= 0

    @patch(MOCK_SUBPROCESS, side_effect=fail_proc)
    def test_failure(self, mock_popen: MagicMock):  # pyright: ignore[reportUnusedParameter]
        result = execute_agent(
            AgentRunSpec(["echo"], "prompt", timeout=None, log_dir=None, iteration=1)
        )

        assert result.returncode == 1
        assert result.timed_out is False

    @patch(MOCK_SUBPROCESS, side_effect=timeout_proc)
    def test_timeout(self, mock_popen: MagicMock):  # pyright: ignore[reportUnusedParameter]
        result = execute_agent(
            AgentRunSpec(["echo"], "prompt", timeout=5, log_dir=None, iteration=1)
        )

        assert result.returncode is None
        assert result.timed_out is True

    @patch(MOCK_SUBPROCESS, side_effect=timeout_proc)
    def test_returncode_is_none_on_blocking_timeout(self, mock_popen: MagicMock):  # pyright: ignore[reportUnusedParameter]
        """Blocking path returns returncode=None on timeout, matching the
        ProcessResult contract and the streaming path's behavior."""
        result = _run_agent_blocking(
            _ResolvedAgentRun(["echo"], "prompt", timeout=5, log_dir=None, iteration=1)
        )

        assert result.returncode is None
        assert result.timed_out is True

    @patch(MOCK_SUBPROCESS)
    def test_writes_log_on_success(self, mock_popen: MagicMock, tmp_path: Path):
        mock_popen.return_value = ok_proc(stdout_text="agent output\n")
        result = execute_agent(
            AgentRunSpec(
                ["echo"],
                "prompt",
                timeout=None,
                log_dir=tmp_path,
                iteration=3,
            )
        )

        assert result.log_file is not None
        assert result.log_file.exists()
        assert result.log_file.name.startswith("003_")
        assert "agent output" in result.log_file.read_text()

    @patch(MOCK_SUBPROCESS)
    def test_writes_log_on_timeout(self, mock_popen: MagicMock, tmp_path: Path):
        mock_popen.return_value = timeout_proc(stdout_text="partial", stderr_text="err")
        result = execute_agent(
            AgentRunSpec(
                ["echo"],
                "prompt",
                timeout=5,
                log_dir=tmp_path,
                iteration=1,
            )
        )

        assert result.log_file is not None
        assert result.log_file.exists()

    @patch(MOCK_SUBPROCESS)
    def test_timeout_captures_partial_output(self, mock_popen: MagicMock, tmp_path: Path):
        """When logging is enabled and the agent times out, partial output
        is captured on the AgentResult for the engine to echo."""
        mock_popen.return_value = timeout_proc(
            stdout_text="partial stdout", stderr_text="partial stderr"
        )
        result = execute_agent(
            AgentRunSpec(
                ["echo"],
                "prompt",
                timeout=5,
                log_dir=tmp_path,
                iteration=1,
            )
        )

        assert result.captured_stdout == "partial stdout"
        assert result.captured_stderr == "partial stderr"

    @patch(MOCK_SUBPROCESS)
    def test_capture_result_text_does_not_buffer_blocking_output_without_log_dir(
        self, mock_popen: MagicMock
    ):
        mock_popen.return_value = ok_proc(
            stdout_text="<promise>done</promise>\n",
            stderr_text="some stderr\n",
        )
        result = _run_agent_blocking(
            _ResolvedAgentRun(
                ["echo"],
                "prompt",
                timeout=None,
                log_dir=None,
                iteration=1,
                capture_result_text=True,
            )
        )

        assert result.captured_stdout is None
        assert result.captured_stderr is None
        assert result.result_text is None

        call_kwargs = mock_popen.call_args[1]  # pyright: ignore[reportAny]
        assert call_kwargs.get("stdin") == subprocess.PIPE  # pyright: ignore[reportAny]
        assert call_kwargs.get("stdout") == subprocess.PIPE  # pyright: ignore[reportAny]
        assert call_kwargs.get("stderr") is None  # pyright: ignore[reportAny]

    @patch(MOCK_SUBPROCESS, side_effect=ok_proc)
    def test_no_log_when_dir_not_set(self, mock_popen: MagicMock):  # pyright: ignore[reportUnusedParameter]
        result = execute_agent(
            AgentRunSpec(["echo"], "prompt", timeout=None, log_dir=None, iteration=1)
        )

        assert result.log_file is None

    @patch(MOCK_SUBPROCESS)
    def test_bounded_capture_keeps_complete_log_and_detects_early_marker(
        self,
        mock_popen: MagicMock,
        tmp_path: Path,
    ) -> None:
        lines = ["<promise>done</promise>\n"] + [
            f"line-{index}-{'x' * 80}\n" for index in range(_STREAM_QUEUE_MAX_LINES * 3)
        ]
        callbacks: list[str] = []
        mock_popen.return_value = ok_proc(stdout_text="".join(lines))

        def on_output_line(line: str, _stream: OutputStream) -> None:
            time.sleep(0.0001)
            callbacks.append(line)

        result = execute_agent(
            AgentRunSpec(
                ["omp"],
                "prompt",
                timeout=None,
                log_dir=tmp_path,
                iteration=1,
                capture_stdout=True,
                completion_signal="done",
                on_output_line=on_output_line,
            )
        )

        assert result.completion_detected is True
        assert result.captured_stdout is not None
        assert len(result.captured_stdout) <= _OUTPUT_TAIL_CHARS
        assert len(callbacks) == len(lines)
        assert result.log_file is not None
        complete_log = result.log_file.read_text(encoding="utf-8")
        assert complete_log.startswith("<promise>done</promise>\n")
        assert f"line-{len(lines) - 2}-" in complete_log
        assert len(complete_log) > _OUTPUT_TAIL_CHARS

    @pytest.mark.parametrize(
        ("missing", "expected"),
        [
            ("stdin", "PIPE stdin"),
            ("stdout", "PIPE stdout"),
            ("stderr", "PIPE stderr"),
        ],
    )
    @patch(MOCK_SUBPROCESS)
    def test_missing_subprocess_pipes_raise_runtime_error(
        self,
        mock_popen: MagicMock,
        tmp_path: Path,
        missing: str,
        expected: str,
    ) -> None:
        proc = ok_proc()
        setattr(proc, missing, None)
        mock_popen.return_value = proc
        run = _ResolvedAgentRun(
            ["omp"],
            "prompt",
            timeout=None,
            log_dir=tmp_path if missing == "stderr" else None,
            iteration=1,
            capture_stdout=missing == "stdout",
        )

        with pytest.raises(RuntimeError, match=expected):
            _ = _run_agent_blocking(run)

    @patch(MOCK_SUBPROCESS)
    def test_file_not_found_propagates(self, mock_popen: MagicMock):
        mock_popen.side_effect = FileNotFoundError("not found")

        with pytest.raises(FileNotFoundError):
            _ = execute_agent(
                AgentRunSpec(
                    ["nonexistent"],
                    "prompt",
                    timeout=None,
                    log_dir=None,
                    iteration=1,
                )
            )


class TestAgentResult:
    def test_defaults(self):
        result = AgentResult(returncode=0, elapsed=1.0, log_file=None)
        assert result.result_text is None
        assert result.timed_out is False

    def test_timed_out(self):
        result = AgentResult(returncode=None, elapsed=5.0, log_file=None, timed_out=True)
        assert result.timed_out is True
        assert result.returncode is None

    @pytest.mark.parametrize(
        ("result", "expected"),
        [
            (AgentResult(returncode=0, elapsed=1.0, log_file=None), True),
            (AgentResult(returncode=1, elapsed=1.0, log_file=None), False),
            (
                AgentResult(returncode=None, elapsed=5.0, log_file=None, timed_out=True),
                False,
            ),
        ],
        ids=["zero-exit", "nonzero-exit", "timed-out"],
    )
    def test_success(self, result: AgentResult, expected: bool):
        assert result.success is expected


class TestExecuteAgentDispatch:
    """Tests for execute_agent routing to streaming vs blocking mode."""

    @patch(MOCK_SUBPROCESS)
    def test_dispatches_to_streaming_for_claude(
        self, mock_popen: MagicMock, monkeypatch: MonkeyPatch
    ):
        """execute_agent uses the streaming path when the adapter renders
        structured output."""
        claude_adapter = select_adapter(["claude"])
        assert isinstance(claude_adapter, ClaudeAdapter)
        monkeypatch.setattr(claude_adapter, "supports_streaming", True)
        mock_popen.return_value = make_mock_popen(
            stdout_lines='{"type": "result", "result": "done"}\n',
            returncode=0,
        )
        result = execute_agent(
            AgentRunSpec(
                ["claude", "-p"],
                "prompt",
                timeout=None,
                log_dir=None,
                iteration=1,
            )
        )

        assert result.returncode == 0
        assert result.result_text == "done"
        mock_popen.assert_called_once()

    def test_execute_agent_passes_capture_result_text_to_streaming_helper(
        self, monkeypatch: MonkeyPatch
    ):
        on_activity = MagicMock()
        on_output_line = MagicMock()
        fake_streaming = MagicMock(return_value=AgentResult(returncode=0, elapsed=0.01))

        claude_adapter = select_adapter(["claude"])
        assert isinstance(claude_adapter, ClaudeAdapter)
        monkeypatch.setattr(claude_adapter, "supports_streaming", True)
        monkeypatch.setattr("milknado.loop._agent._run_agent_streaming", fake_streaming)

        _ = execute_agent(
            AgentRunSpec(
                ["claude", "-p"],
                "prompt",
                timeout=None,
                log_dir=None,
                iteration=1,
                on_activity=on_activity,
                on_output_line=on_output_line,
                capture_result_text=True,
            )
        )

        fake_streaming.assert_called_once_with(
            _ResolvedAgentRun(
                cmd=claude_adapter.build_command(["claude", "-p"]),
                stdin_text="prompt",
                timeout=None,
                log_dir=None,
                iteration=1,
                on_activity=on_activity,
                on_output_line=on_output_line,
                capture_result_text=True,
                capture_stdout=False,
                adapter=claude_adapter,
                max_turns=None,
                on_tool_use=None,
                env=None,
                cwd=None,
            )
        )

    def test_execute_agent_passes_capture_result_text_to_blocking_helper(
        self, monkeypatch: MonkeyPatch
    ):
        on_output_line = MagicMock()
        fake_blocking = MagicMock(return_value=AgentResult(returncode=0, elapsed=0.01))

        # The autouse conftest fixture already forces every adapter's
        # supports_streaming flag to False, so ``echo`` falls into the
        # blocking path through the GenericAdapter fallback.  The fallback
        # is constructed fresh per ``select_adapter`` call, so the dispatch
        # assertion matches the adapter arg with ``ANY``.
        monkeypatch.setattr("milknado.loop._agent._run_agent_blocking", fake_blocking)

        _ = execute_agent(
            AgentRunSpec(
                ["echo"],
                "prompt",
                timeout=None,
                log_dir=None,
                iteration=1,
                on_output_line=on_output_line,
                capture_result_text=True,
            )
        )

        fake_blocking.assert_called_once_with(
            _ResolvedAgentRun(
                cmd=["echo"],
                stdin_text="prompt",
                timeout=None,
                log_dir=None,
                iteration=1,
                on_output_line=on_output_line,
                capture_result_text=True,
                capture_stdout=False,
                adapter=ANY,
                max_turns=None,
                on_tool_use=None,
                env=None,
                cwd=None,
            )
        )


class TestExecuteAgentStreaming:
    """Tests for the streaming execution path (_run_agent_streaming)."""

    @patch(MOCK_SUBPROCESS)
    def test_streaming_result_event_populates_result_text(self, mock_popen: MagicMock):
        mock_popen.return_value = make_mock_popen(
            stdout_lines='{"type": "result", "result": "early done"}\n',
            returncode=0,
        )
        result = _run_agent_streaming(
            _ResolvedAgentRun(
                ["claude", "-p"],
                "prompt",
                timeout=10,
                log_dir=None,
                iteration=1,
            )
        )
        assert result.result_text == "early done"
        assert result.returncode == 0
        assert result.timed_out is False

    @patch(MOCK_SUBPROCESS)
    def test_blocking_result_event_populates_result_text_when_captured(
        self, mock_popen: MagicMock, tmp_path: Path
    ):
        mock_popen.return_value = make_mock_popen(
            stdout_lines='{"type": "result", "result": "early done"}\n',
            returncode=0,
        )
        result = _run_agent_blocking(
            _ResolvedAgentRun(
                ["claude", "-p"],
                "prompt",
                timeout=10,
                log_dir=tmp_path,
                iteration=1,
            )
        )

        assert result.result_text == "early done"
        assert result.returncode == 0
        assert result.timed_out is False

    @patch(MOCK_SUBPROCESS)
    def test_result_text_absent_when_no_result_event(self, mock_popen: MagicMock):
        mock_popen.return_value = make_mock_popen(
            stdout_lines="status: working\n",
            returncode=0,
        )
        result = _run_agent_streaming(
            _ResolvedAgentRun(
                ["claude", "-p"],
                "prompt",
                timeout=10,
                log_dir=None,
                iteration=1,
            )
        )
        assert result.result_text is None
        assert result.returncode == 0
        assert result.timed_out is False

    @patch(MOCK_SUBPROCESS)
    def test_last_result_event_wins(self, mock_popen: MagicMock):
        mock_popen.return_value = make_mock_popen(
            stdout_lines=(
                '{"type": "result", "result": "first"}\n{"type": "result", "result": "second"}\n'
            ),
            returncode=0,
        )
        result = _run_agent_streaming(
            _ResolvedAgentRun(
                ["claude", "-p"],
                "prompt",
                timeout=10,
                log_dir=None,
                iteration=1,
            )
        )
        assert result.result_text == "second"
        assert result.returncode == 0
        assert result.timed_out is False

    @patch(MOCK_SUBPROCESS)
    def test_success(self, mock_popen: MagicMock):
        mock_popen.return_value = make_mock_popen(
            stdout_lines='{"type": "status", "msg": "working"}\n',
            returncode=0,
        )
        result = _run_agent_streaming(
            _ResolvedAgentRun(
                ["claude", "-p"],
                "prompt",
                timeout=None,
                log_dir=None,
                iteration=1,
            )
        )

        assert result.returncode == 0
        assert result.timed_out is False
        assert result.elapsed >= 0

    @patch(MOCK_SUBPROCESS)
    def test_failure(self, mock_popen: MagicMock):
        mock_popen.return_value = make_mock_popen(returncode=1)
        result = _run_agent_streaming(
            _ResolvedAgentRun(
                ["claude", "-p"],
                "prompt",
                timeout=None,
                log_dir=None,
                iteration=1,
            )
        )

        assert result.returncode == 1
        assert result.timed_out is False
        assert result.success is False

    @patch(MOCK_SUBPROCESS)
    def test_captures_result_text(self, mock_popen: MagicMock):
        mock_popen.return_value = make_mock_popen(
            stdout_lines='{"type": "result", "result": "All tests passed"}\n',
            returncode=0,
        )
        result = _run_agent_streaming(
            _ResolvedAgentRun(
                ["claude", "-p"],
                "prompt",
                timeout=None,
                log_dir=None,
                iteration=1,
            )
        )

        assert result.result_text == "All tests passed"

    @patch(MOCK_SUBPROCESS)
    def test_sends_prompt_to_stdin(self, mock_popen: MagicMock):
        proc = make_mock_popen(returncode=0)
        mock_popen.return_value = proc
        _ = _run_agent_streaming(
            _ResolvedAgentRun(
                ["claude", "-p"],
                "my prompt text",
                timeout=None,
                log_dir=None,
                iteration=1,
            )
        )

        proc.stdin.write.assert_called_once_with("my prompt text")  # pyright: ignore[reportAny]
        proc.stdin.close.assert_called_once()  # pyright: ignore[reportAny]

    @patch(MOCK_SUBPROCESS)
    def test_passes_cmd_verbatim_to_popen(self, mock_popen: MagicMock):
        """_run_agent_streaming no longer appends its own flags — the caller
        (via ``adapter.build_command``) owns CLI flags now."""
        mock_popen.return_value = make_mock_popen(returncode=0)
        _ = _run_agent_streaming(
            _ResolvedAgentRun(
                ["claude", "-p", "--output-format", "stream-json", "--verbose"],
                "prompt",
                timeout=None,
                log_dir=None,
                iteration=1,
            )
        )

        call_args = mock_popen.call_args
        cmd = call_args[0][0]  # pyright: ignore[reportAny]
        assert cmd == [
            "claude",
            "-p",
            "--output-format",
            "stream-json",
            "--verbose",
        ]

    @patch(MOCK_SUBPROCESS)
    def test_writes_log_on_success(self, mock_popen: MagicMock, tmp_path: Path):
        mock_popen.return_value = make_mock_popen(
            stdout_lines="agent output\n",
            stderr_text="some stderr\n",
            returncode=0,
        )
        result = _run_agent_streaming(
            _ResolvedAgentRun(
                ["claude", "-p"],
                "prompt",
                timeout=None,
                log_dir=tmp_path,
                iteration=3,
            )
        )

        assert result.log_file is not None
        assert result.log_file.exists()
        assert result.log_file.name.startswith("003_")
        content = result.log_file.read_text()
        assert "agent output" in content
        assert "some stderr" in content

    @patch(MOCK_SUBPROCESS)
    def test_captured_output_set_when_logging(self, mock_popen: MagicMock, tmp_path: Path):
        """When log_dir is set, captured_stdout and captured_stderr
        must be populated so the engine can echo them via the event system
        — matching the blocking path's behavior."""
        mock_popen.return_value = make_mock_popen(
            stdout_lines="agent output\n",
            stderr_text="some stderr\n",
            returncode=0,
        )
        result = _run_agent_streaming(
            _ResolvedAgentRun(
                ["claude", "-p"],
                "prompt",
                timeout=None,
                log_dir=tmp_path,
                iteration=1,
            )
        )

        assert result.captured_stdout == "agent output\n"
        assert result.captured_stderr == "some stderr\n"

    @patch(MOCK_SUBPROCESS)
    def test_capture_result_text_does_not_buffer_stream_output_without_log_dir(
        self, mock_popen: MagicMock
    ):
        mock_popen.return_value = make_mock_popen(
            stdout_lines="<promise>done</promise>\n",
            stderr_text="some stderr\n",
            returncode=0,
        )
        result = _run_agent_streaming(
            _ResolvedAgentRun(
                ["claude", "-p"],
                "prompt",
                timeout=None,
                log_dir=None,
                iteration=1,
                capture_result_text=True,
            )
        )

        assert result.captured_stdout is None
        assert result.captured_stderr is None
        assert result.result_text is None

        call_args = mock_popen.call_args
        assert call_args.args[0] == ["claude", "-p"]
        call_kwargs = call_args[1]  # pyright: ignore[reportAny]
        assert call_kwargs.get("stdin") == subprocess.PIPE  # pyright: ignore[reportAny]
        assert call_kwargs.get("stdout") == subprocess.PIPE  # pyright: ignore[reportAny]
        assert call_kwargs.get("stderr") is None  # pyright: ignore[reportAny]

    @patch(MOCK_SUBPROCESS)
    def test_no_log_when_dir_not_set(self, mock_popen: MagicMock):
        mock_popen.return_value = make_mock_popen(returncode=0)
        result = _run_agent_streaming(
            _ResolvedAgentRun(
                ["claude", "-p"],
                "prompt",
                timeout=None,
                log_dir=None,
                iteration=1,
            )
        )

        assert result.log_file is None

    @patch(MOCK_SUBPROCESS)
    def test_on_activity_callback_invoked(self, mock_popen: MagicMock):
        mock_popen.return_value = make_mock_popen(
            stdout_lines='{"type": "status", "msg": "working"}\n{"type": "progress"}\n',
            returncode=0,
        )
        activities = []
        _ = _run_agent_streaming(
            _ResolvedAgentRun(
                ["claude", "-p"],
                "prompt",
                timeout=None,
                log_dir=None,
                iteration=1,
                on_activity=activities.append,  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
            )
        )

        assert len(activities) == 2  # pyright: ignore[reportUnknownArgumentType]
        assert activities[0]["type"] == "status"
        assert activities[1]["type"] == "progress"

    @patch("milknado.loop._agent.time.monotonic")
    @patch(MOCK_SUBPROCESS)
    def test_timeout_kills_process(self, mock_popen: MagicMock, mock_time: MagicMock):
        # start=0 → deadline=5.  First loop check: remaining=5 (positive,
        # so queue.get waits up to 5 real seconds — succeeds immediately
        # because the StringIO reader fills the queue).  Post-line check
        # returns 100 → past deadline → timed_out.
        mock_time.side_effect = itertools.chain(
            [0.0, 0.0, 100.0],
            itertools.repeat(100.0),
        )
        proc = make_mock_popen(
            stdout_lines="line1\nline2\n",
            returncode=0,
        )
        proc.poll.return_value = (  # pyright: ignore[reportAny]
            None  # process still running when timeout fires
        )
        mock_popen.return_value = proc

        result = _run_agent_streaming(
            _ResolvedAgentRun(
                ["claude", "-p"],
                "prompt",
                timeout=5,
                log_dir=None,
                iteration=1,
            )
        )

        assert result.timed_out is True
        assert result.returncode is None


class TestInterruptibleReap:
    def test_force_stop_after_stdout_eof_uses_bounded_cleanup(
        self, monkeypatch: MonkeyPatch, caplog: LogCaptureFixture
    ):
        force_stop = threading.Event()
        proc = make_mock_popen(stdout_lines="", returncode=0)
        proc.pid = 4242
        proc.returncode = None
        proc.poll.return_value = None  # pyright: ignore[reportAny]

        def wait_until_force_observed(timeout: float | None = None):
            assert timeout is not None
            force_stop.set()
            raise subprocess.TimeoutExpired(proc.args, timeout)  # pyright: ignore[reportAny]

        proc.wait.side_effect = wait_until_force_observed  # pyright: ignore[reportAny]
        claude_adapter = select_adapter(["claude"])
        monkeypatch.setattr(claude_adapter, "supports_streaming", True)

        with (
            patch(MOCK_SUBPROCESS, return_value=proc),
            patch("milknado.loop._agent._ensure_process_dead") as cleanup,
            caplog.at_level(logging.WARNING, logger="milknado.loop._agent"),
        ):
            result = execute_agent(
                AgentRunSpec(
                    ["claude", "-p"],
                    "prompt",
                    timeout=None,
                    log_dir=None,
                    iteration=7,
                    force_stop_event=force_stop,
                )
            )

        assert result.force_stopped is True
        assert result.returncode is None
        cleanup.assert_called()
        assert "pid=4242 correlation=iteration=7" in caplog.text


class TestWindowsJobContainment:
    def test_force_stop_terminates_assigned_job_instead_of_parent_only(
        self, monkeypatch: MonkeyPatch
    ):
        kernel32 = MagicMock()
        kernel32.CreateJobObjectW.return_value = 123  # pyright: ignore[reportAny]
        kernel32.SetInformationJobObject.return_value = 1  # pyright: ignore[reportAny]
        kernel32.AssignProcessToJobObject.return_value = 1  # pyright: ignore[reportAny]
        kernel32.TerminateJobObject.return_value = 1  # pyright: ignore[reportAny]
        kernel32.CloseHandle.return_value = 1  # pyright: ignore[reportAny]
        proc = MagicMock(pid=42)
        proc._handle = 456
        resume_process = MagicMock()
        monkeypatch.setattr(agent_module, "_resume_windows_process", resume_process)

        monkeypatch.setattr(agent_module, "IS_WINDOWS", True)
        monkeypatch.setattr(agent_module, "_load_kernel32", lambda: kernel32)

        job = agent_module._WindowsJob.assign(proc)  # pyright: ignore[reportPrivateUsage]
        assert job is not None
        agent_module._kill_process_group(proc, job)  # pyright: ignore[reportPrivateUsage]
        resume_process.assert_called_once_with(proc)

        kernel32.AssignProcessToJobObject.assert_called_once_with(123, 456)  # pyright: ignore[reportAny]
        kernel32.TerminateJobObject.assert_called_once_with(123, 1)  # pyright: ignore[reportAny]
        kernel32.CloseHandle.assert_called_once_with(123)  # pyright: ignore[reportAny]
        proc.kill.assert_not_called()  # pyright: ignore[reportAny]


class TestProcessGroupCleanup:
    """Process group cleanup, isolation, and _kill_process_group tests."""

    pytestmark: object = pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only behavior")

    @patch("milknado.loop._agent.os.killpg")
    @patch("milknado.loop._agent.os.getpgid")
    def test_session_leader_gets_sigterm(self, mock_getpgid: MagicMock, mock_killpg: MagicMock):
        proc = MagicMock(pid=42, poll=MagicMock(return_value=None))
        mock_getpgid.return_value = 42

        _kill_process_group(proc)

        mock_killpg.assert_any_call(42, signal.SIGTERM)
        proc.wait.assert_called_once_with(timeout=3)  # pyright: ignore[reportAny]

    @patch("milknado.loop._agent.os.killpg")
    @patch("milknado.loop._agent.os.getpgid")
    def test_escalates_to_sigkill_on_timeout(
        self, mock_getpgid: MagicMock, mock_killpg: MagicMock
    ):
        proc = MagicMock(pid=42, poll=MagicMock(return_value=None))
        proc.wait.side_effect = subprocess.TimeoutExpired(cmd="agent", timeout=3)  # pyright: ignore[reportAny]
        mock_getpgid.return_value = 42

        _kill_process_group(proc)

        mock_killpg.assert_any_call(42, signal.SIGTERM)
        mock_killpg.assert_any_call(42, signal.SIGKILL)

    @patch("milknado.loop._agent.os.killpg")
    @patch("milknado.loop._agent.os.getpgid")
    def test_sigkill_failure_falls_back_to_proc_kill(
        self, mock_getpgid: MagicMock, mock_killpg: MagicMock
    ):
        """When SIGTERM times out and SIGKILL also fails, fall back to proc.kill()."""
        proc = MagicMock(pid=42, poll=MagicMock(return_value=None))
        proc.wait.side_effect = subprocess.TimeoutExpired(cmd="agent", timeout=3)  # pyright: ignore[reportAny]
        mock_getpgid.return_value = 42
        # SIGTERM succeeds but SIGKILL fails (process vanished between attempts)
        mock_killpg.side_effect = [None, ProcessLookupError("No such process")]

        _kill_process_group(proc)

        mock_killpg.assert_any_call(42, signal.SIGTERM)
        mock_killpg.assert_any_call(42, signal.SIGKILL)
        proc.kill.assert_called_once()  # pyright: ignore[reportAny]

    @patch("milknado.loop._agent.os.killpg")
    @patch("milknado.loop._agent.os.getpgid")
    def test_not_session_leader_falls_back_to_kill(
        self, mock_getpgid: MagicMock, mock_killpg: MagicMock
    ):
        proc = MagicMock(pid=42, poll=MagicMock(return_value=None))
        mock_getpgid.return_value = 1

        _kill_process_group(proc)

        mock_killpg.assert_not_called()
        proc.kill.assert_called_once()  # pyright: ignore[reportAny]

    def test_already_exited_falls_back_to_kill(self):
        proc = MagicMock(pid=42, poll=MagicMock(return_value=0))

        _kill_process_group(proc)

        proc.kill.assert_called_once()  # pyright: ignore[reportAny]

    @patch(MOCK_SUBPROCESS, side_effect=ok_proc)
    def test_blocking_uses_start_new_session(self, mock_popen: MagicMock):
        _ = execute_agent(
            AgentRunSpec(["echo"], "prompt", timeout=None, log_dir=None, iteration=1)
        )
        assert mock_popen.call_args[1].get("start_new_session") is True  # pyright: ignore[reportAny]

    @patch(MOCK_SUBPROCESS)
    def test_streaming_uses_start_new_session(self, mock_popen: MagicMock):
        mock_popen.return_value = make_mock_popen(returncode=0)
        _ = _run_agent_streaming(
            _ResolvedAgentRun(
                ["claude", "-p"],
                "prompt",
                timeout=None,
                log_dir=None,
                iteration=1,
            )
        )
        assert mock_popen.call_args[1].get("start_new_session") is True  # pyright: ignore[reportAny]

    @patch("milknado.loop._agent.os.killpg")
    @patch("milknado.loop._agent.os.getpgid")
    def test_killpg_oserror_falls_back_to_proc_kill(
        self, mock_getpgid: MagicMock, mock_killpg: MagicMock
    ):
        """When os.killpg raises OSError (e.g. process already gone), fall back to proc.kill()."""
        proc = MagicMock(pid=42, poll=MagicMock(return_value=None))
        mock_getpgid.return_value = 42  # session leader
        mock_killpg.side_effect = OSError("No such process")

        _kill_process_group(proc)

        mock_killpg.assert_called_once_with(42, signal.SIGTERM)
        proc.kill.assert_called_once()  # pyright: ignore[reportAny]

    @patch("milknado.loop._agent.os.killpg")
    @patch("milknado.loop._agent.os.getpgid")
    def test_killpg_process_lookup_error_falls_back_to_proc_kill(
        self, mock_getpgid: MagicMock, mock_killpg: MagicMock
    ):
        """When os.killpg raises ProcessLookupError, fall back to proc.kill()."""
        proc = MagicMock(pid=42, poll=MagicMock(return_value=None))
        mock_getpgid.return_value = 42
        mock_killpg.side_effect = ProcessLookupError("No such process")

        _kill_process_group(proc)

        proc.kill.assert_called_once()  # pyright: ignore[reportAny]

    @pytest.mark.parametrize("pid", [0, -1, None])
    @patch("milknado.loop._agent.os.killpg")
    @patch("milknado.loop._agent.os.getpgid")
    def test_kill_process_group_short_circuits_on_sentinel_pid(
        self, mock_getpgid: MagicMock, mock_killpg: MagicMock, pid: int
    ):
        """Non-positive or None pids must never reach os.getpgid/os.killpg."""
        proc = MagicMock(pid=pid, poll=MagicMock(return_value=None))

        _kill_process_group(proc)

        mock_getpgid.assert_not_called()
        mock_killpg.assert_not_called()
        proc.kill.assert_not_called()  # pyright: ignore[reportAny]


class TestRunAgentStreamingPipeGuard:
    """Test the defensive RuntimeError when Popen fails to create PIPE streams."""

    @patch(MOCK_SUBPROCESS)
    def test_raises_when_stdin_is_none(self, mock_popen: MagicMock):
        proc = make_mock_popen(returncode=0)
        proc.stdin = None
        proc.poll.return_value = (  # pyright: ignore[reportAny]
            None  # process still running for finally cleanup
        )
        mock_popen.return_value = proc

        with pytest.raises(RuntimeError, match="PIPE stdin"):
            _ = _run_agent_streaming(
                _ResolvedAgentRun(
                    ["claude", "-p"],
                    "prompt",
                    timeout=None,
                    log_dir=None,
                    iteration=1,
                )
            )


class TestRunAgentBlockingKeyboardInterrupt:
    """Test that KeyboardInterrupt during blocking execution kills the process and re-raises."""

    @patch(MOCK_SUBPROCESS)
    def test_keyboard_interrupt_kills_and_reraises(self, mock_popen: MagicMock):
        proc = MagicMock()
        proc.pid = 0  # sentinel: skip real process-group manipulation
        proc.stdin = MagicMock()
        proc.stdout = io.StringIO("")
        proc.stderr = io.StringIO("")
        # First wait() raises KeyboardInterrupt; all subsequent calls
        # return -2.  Uses itertools.chain so the exact number of wait()
        # calls made by cleanup paths can drift without breaking the test.
        proc.wait.side_effect = itertools.chain([KeyboardInterrupt], itertools.repeat(-2))  # pyright: ignore[reportAny]
        # First poll() (inside _kill_process_group) sees a live process;
        # every later poll() sees it as reaped so cleanup paths don't
        # re-enter kill/wait.
        proc.poll.side_effect = itertools.chain([None], itertools.repeat(0))  # pyright: ignore[reportAny]
        mock_popen.return_value = proc

        with pytest.raises(KeyboardInterrupt):
            _ = execute_agent(
                AgentRunSpec(
                    ["echo"],
                    "prompt",
                    timeout=None,
                    log_dir=None,
                    iteration=1,
                )
            )

        proc.wait.assert_called()  # pyright: ignore[reportAny]


class TestRunAgentBlockingLineStreaming:
    """Real-subprocess tests for live line streaming via _run_agent_blocking.

    Uses tiny ``python -c`` subprocesses rather than mocks so we exercise
    the actual reader-thread path and observe lines as the process runs.
    """

    def test_on_output_line_receives_lines_in_order(self, tmp_path: Path):
        script = "import sys; print('first'); print('second'); print('third')"
        received: list[tuple[str, str]] = []

        result = _run_agent_blocking(
            _ResolvedAgentRun(
                [sys.executable, "-c", script],
                stdin_text="",
                timeout=10,
                log_dir=tmp_path,
                iteration=1,
                on_output_line=lambda line, stream: received.append((line, stream)),
            )
        )

        assert result.returncode == 0
        assert [line for line, _ in received] == ["first", "second", "third"]
        assert all(stream == "stdout" for _, stream in received)

        # Log file still receives the full output (behavior preserved).
        assert result.log_file is not None
        log_text = result.log_file.read_text()
        assert "first" in log_text
        assert "second" in log_text
        assert "third" in log_text

    def test_on_output_line_captures_stderr_separately(self, tmp_path: Path):
        script = (
            "import sys; "
            "print('out-line', file=sys.stdout); "
            "print('err-line', file=sys.stderr); "
            "sys.stdout.flush(); sys.stderr.flush()"
        )
        received: list[tuple[str, str]] = []

        _ = _run_agent_blocking(
            _ResolvedAgentRun(
                [sys.executable, "-u", "-c", script],
                stdin_text="",
                timeout=10,
                log_dir=tmp_path,
                iteration=1,
                on_output_line=lambda line, stream: received.append((line, stream)),
            )
        )

        streams = {stream for _, stream in received}
        assert streams == {"stdout", "stderr"}
        lines_by_stream = {stream: line for line, stream in received}
        assert lines_by_stream["stdout"] == "out-line"
        assert lines_by_stream["stderr"] == "err-line"

    def test_stdin_prompt_delivered_to_subprocess(self, tmp_path: Path):
        """The prompt must reach the child via stdin so real agents get it."""
        script = "import sys; sys.stdout.write(sys.stdin.read())"
        received: list[str] = []

        _ = _run_agent_blocking(
            _ResolvedAgentRun(
                [sys.executable, "-c", script],
                stdin_text="hello-from-prompt\n",
                timeout=10,
                log_dir=tmp_path,
                iteration=1,
                on_output_line=lambda line, stream: received.append(line),
            )
        )

        assert received == ["hello-from-prompt"]

    def test_large_prompt_with_concurrent_stderr_does_not_deadlock(self, tmp_path: Path):
        """Regression guard for the pipe-buffer deadlock.

        If the child writes a burst of stderr larger than the OS pipe
        buffer before reading its stdin, and the parent is simultaneously
        writing a stdin prompt larger than the buffer, the only way to
        avoid deadlock is for the parent to drain stderr concurrently on
        a reader thread that was started **before** the stdin write began.

        This test fails (hangs) if the reader threads are started after
        the stdin write.
        """
        script = (
            "import sys\n"
            "sys.stderr.write('y' * 200000)\n"
            "sys.stderr.flush()\n"
            "sys.stdin.read()\n"
            "sys.stdout.write('done\\n')\n"
        )
        large_prompt = "x" * 200000

        result = _run_agent_blocking(
            _ResolvedAgentRun(
                [sys.executable, "-c", script],
                stdin_text=large_prompt,
                timeout=15,
                log_dir=tmp_path,
                iteration=1,
            )
        )

        assert result.returncode == 0
        assert result.timed_out is False

    def test_early_exit_with_large_prompt_does_not_crash(self, tmp_path: Path):
        """If the agent exits without consuming its stdin, the parent's
        write will raise ``BrokenPipeError``; the blocking path must
        swallow it and still return the child's real exit code.
        """
        script = "import sys; sys.exit(0)"
        large_prompt = "x" * 200000

        result = _run_agent_blocking(
            _ResolvedAgentRun(
                [sys.executable, "-c", script],
                stdin_text=large_prompt,
                timeout=10,
                log_dir=tmp_path,
                iteration=1,
            )
        )

        assert result.returncode == 0
        assert result.timed_out is False

    def test_streaming_large_stderr_drained_concurrently(self, tmp_path: Path):
        """Regression guard for the streaming path.

        An agent that writes substantial stderr before emitting its result
        event would previously deadlock the streaming path because stderr
        was only read after ``proc.wait()``.  With a concurrent stderr
        pump thread this finishes normally.
        """
        script = (
            "import sys\n"
            "sys.stderr.write('y' * 200000)\n"
            "sys.stderr.flush()\n"
            "sys.stdin.read()\n"
            'sys.stdout.write(\'{"type": "result", "result": "ok"}\\n\')\n'
        )

        result = _run_agent_streaming(
            _ResolvedAgentRun(
                [sys.executable, "-c", script],
                stdin_text="hi",
                timeout=15,
                log_dir=tmp_path,
                iteration=1,
            )
        )

        assert result.returncode == 0
        assert result.timed_out is False
        assert result.result_text == "ok"

    def test_timeout_enforced_when_agent_does_not_read_stdin(self, tmp_path: Path):
        """If the agent never reads stdin, --timeout must still fire.

        Before the writer-thread fix, proc.stdin.write(prompt) blocked on
        the main thread when the OS pipe buffer was full, and
        proc.wait(timeout=...) was never reached — so --timeout silently
        did nothing.

        Uses a prompt large enough to fill the pipe buffer (64 KB on
        Linux, ~8 KB on macOS) so the write would block if it were on
        the main thread.
        """
        # Agent that never reads stdin and sleeps for 30 seconds.
        script = "import time; time.sleep(30)"
        # Prompt larger than OS pipe buffer on any platform.
        large_prompt = "x" * 200_000

        start = time.monotonic()
        result = _run_agent_blocking(
            _ResolvedAgentRun(
                [sys.executable, "-c", script],
                stdin_text=large_prompt,
                timeout=2.0,
                log_dir=tmp_path,
                iteration=1,
            )
        )
        elapsed = time.monotonic() - start

        assert result.timed_out is True
        assert result.returncode is None
        # Must complete well before the child's 30-second sleep finishes.
        assert elapsed < 15.0


class TestExecuteAgentCwd:
    """Real-subprocess tests proving ``cwd`` lands the spawned agent process
    in the requested working directory.

    The forensic defect this guards: worker agents inherited the
    orchestrator's cwd and edited the wrong repo.  ``RunConfig.project_root``
    must be honoured at the spawn layer so a worker edits its own worktree.
    """

    @staticmethod
    def _pwd_probe_cmd(out_file: Path):
        # A fake agent that records its own working directory and exits.
        script = "import os, sys; open(sys.argv[1], 'w').write(os.getcwd())"
        return [sys.executable, "-c", script, str(out_file)]

    def test_execute_agent_spawns_in_requested_cwd(self, tmp_path: Path):
        """execute_agent(..., cwd=dir) runs the agent process inside *dir*."""
        work_dir = tmp_path / "worker-worktree"
        work_dir.mkdir()
        out_file = tmp_path / "pwd.txt"

        result = execute_agent(
            AgentRunSpec(
                self._pwd_probe_cmd(out_file),
                "prompt",
                timeout=10,
                log_dir=None,
                iteration=1,
                cwd=work_dir,
            )
        )

        assert result.returncode == 0
        # The child reported its real cwd; resolve both sides so macOS
        # /var → /private/var symlinking can't produce a spurious mismatch.
        recorded = Path(out_file.read_text())
        assert recorded.resolve() == work_dir.resolve()

    def test_execute_agent_cwd_none_inherits_parent_cwd(self, tmp_path: Path):
        """cwd=None preserves the inherit-cwd behaviour for direct engine
        callers — the child must NOT run in some unrelated directory."""
        out_file = tmp_path / "pwd.txt"

        result = execute_agent(
            AgentRunSpec(
                self._pwd_probe_cmd(out_file),
                "prompt",
                timeout=10,
                log_dir=None,
                iteration=1,
            )
        )

        assert result.returncode == 0
        recorded = Path(out_file.read_text())
        assert recorded.resolve() == Path(os.getcwd()).resolve()

    def test_blocking_helper_honours_cwd(self, tmp_path: Path):
        """The blocking helper passes cwd straight through to Popen."""
        work_dir = tmp_path / "blocking-cwd"
        work_dir.mkdir()
        out_file = tmp_path / "pwd.txt"

        _ = _run_agent_blocking(
            _ResolvedAgentRun(
                self._pwd_probe_cmd(out_file),
                stdin_text="",
                timeout=10,
                log_dir=None,
                iteration=1,
                cwd=work_dir,
            )
        )

        assert Path(out_file.read_text()).resolve() == work_dir.resolve()

    def test_streaming_helper_honours_cwd(self, tmp_path: Path):
        """The streaming helper passes cwd straight through to Popen."""
        work_dir = tmp_path / "streaming-cwd"
        work_dir.mkdir()
        out_file = tmp_path / "pwd.txt"

        _ = _run_agent_streaming(
            _ResolvedAgentRun(
                self._pwd_probe_cmd(out_file),
                stdin_text="",
                timeout=10,
                log_dir=None,
                iteration=1,
                cwd=work_dir,
            )
        )

        assert Path(out_file.read_text()).resolve() == work_dir.resolve()


class TestStreamingDeadlineAndBuffering:
    """Tests for deadline enforcement and line-at-a-time delivery in the
    streaming path (_read_agent_stream / _run_agent_streaming).

    Uses real subprocesses to exercise the actual I/O layer — the bugs
    these tests guard against live at the pipe-buffering and thread-scheduling
    level, where mocks would be misleading.
    """

    def test_streaming_timeout_enforced_on_silent_agent(self, tmp_path: Path):
        """A hung agent that produces no output must still be killed by --timeout.

        Before the fix, ``for line in stdout`` blocked in readline()
        indefinitely, and the deadline check only ran between lines.  The
        queue-based approach unblocks via ``queue.get(timeout=remaining)``
        so the deadline fires even with zero output.
        """
        # Agent reads stdin then sleeps 30s, writing nothing to stdout.
        script = "import sys, time; sys.stdin.read(); time.sleep(30)"

        start = time.monotonic()
        result = _run_agent_streaming(
            _ResolvedAgentRun(
                [sys.executable, "-u", "-c", script],
                stdin_text="go",
                timeout=1.0,
                log_dir=tmp_path,
                iteration=1,
            )
        )
        elapsed = time.monotonic() - start

        assert result.timed_out is True
        assert result.returncode is None
        # Must return well before the child's 30-second sleep.
        assert elapsed < 10.0

    def test_streaming_peek_flows_line_at_a_time(self, tmp_path: Path):
        """Peek callbacks must fire promptly as lines arrive, not in 8KB bursts.

        A fake agent emits timestamped lines with 200ms sleeps.  Each
        ``on_output_line`` callback must fire within 500ms of the line's
        emission — if they arrived in readahead bursts the later lines
        would be delayed.
        """
        # Agent emits 5 lines at ~200ms intervals with wall-clock timestamps.
        script = (
            "import sys, time\n"
            "sys.stdin.read()\n"
            "for i in range(5):\n"
            "    print(f'{time.monotonic():.6f} line{i}', flush=True)\n"
            "    if i < 4:\n"
            "        time.sleep(0.2)\n"
        )
        receive_times: list[float] = []

        def on_line(line: str, stream: str) -> None:  # pyright: ignore[reportUnusedParameter]
            receive_times.append(time.monotonic())

        result = _run_agent_streaming(
            _ResolvedAgentRun(
                [sys.executable, "-u", "-c", script],
                stdin_text="go",
                timeout=15,
                log_dir=tmp_path,
                iteration=1,
                on_output_line=on_line,
            )
        )

        assert result.returncode == 0
        assert result.timed_out is False
        # Should have received all 5 lines (stdout) plus any stderr lines
        # forwarded by the stderr pump — at least 5.
        stdout_callbacks = len(receive_times)
        assert stdout_callbacks >= 5

        # Check that lines were delivered promptly.  The gap between
        # consecutive callbacks should be roughly 200ms (±300ms for
        # scheduling jitter).  If buffered in 8KB bursts, the last 4
        # lines would all arrive at once with ~0ms gaps.
        for i in range(1, min(5, stdout_callbacks)):
            gap = receive_times[i] - receive_times[i - 1]
            assert gap > 0.05, (
                f"Gap between line {i - 1} and {i} was {gap:.3f}s — "
                "lines likely arrived in a buffered burst"
            )


class TestBlockingInheritPath:
    """Tests for the fd-inheritance path in _run_agent_blocking.

    When both ``log_dir`` and ``on_output_line`` are ``None``, the
    blocking path should inherit stdout/stderr from the parent (no PIPE,
    no reader threads) so that ``ralph run | cat`` shows output.
    """

    @patch(MOCK_SUBPROCESS, side_effect=ok_proc)
    def test_no_pipe_when_no_log_no_callback(self, mock_popen: MagicMock):
        """Popen must NOT receive stdout/stderr=PIPE when no one needs capture."""
        result = _run_agent_blocking(
            _ResolvedAgentRun(
                ["echo"],
                "prompt",
                timeout=None,
                log_dir=None,
                iteration=1,
                on_output_line=None,
            )
        )

        call_kwargs = mock_popen.call_args[1]  # pyright: ignore[reportAny]
        assert call_kwargs.get("stdout") is None  # pyright: ignore[reportAny]
        assert call_kwargs.get("stderr") is None  # pyright: ignore[reportAny]
        assert result.returncode == 0
        assert result.log_file is None

    @patch(MOCK_SUBPROCESS, side_effect=ok_proc)
    def test_still_pipes_when_callback_set(self, mock_popen: MagicMock):
        """When on_output_line is provided, stdout/stderr must be PIPE'd."""
        _ = _run_agent_blocking(
            _ResolvedAgentRun(
                ["echo"],
                "prompt",
                timeout=None,
                log_dir=None,
                iteration=1,
                on_output_line=lambda line, stream: None,
            )
        )

        call_kwargs = mock_popen.call_args[1]  # pyright: ignore[reportAny]
        assert call_kwargs.get("stdout") == subprocess.PIPE  # pyright: ignore[reportAny]
        assert call_kwargs.get("stderr") == subprocess.PIPE  # pyright: ignore[reportAny]

    @patch(MOCK_SUBPROCESS, side_effect=ok_proc)
    def test_still_pipes_when_log_dir_set(self, mock_popen: MagicMock, tmp_path: Path):
        """When log_dir is provided, stdout/stderr must be PIPE'd."""
        _ = _run_agent_blocking(
            _ResolvedAgentRun(
                ["echo"],
                "prompt",
                timeout=None,
                log_dir=tmp_path,
                iteration=1,
                on_output_line=None,
            )
        )

        call_kwargs = mock_popen.call_args[1]  # pyright: ignore[reportAny]
        assert call_kwargs.get("stdout") == subprocess.PIPE  # pyright: ignore[reportAny]
        assert call_kwargs.get("stderr") == subprocess.PIPE  # pyright: ignore[reportAny]

    def test_inherit_path_shows_output(self, capfd: CaptureFixture[str]):
        """Real subprocess in inherit mode: child output reaches the parent's
        stdout, verifying the ``ralph run | cat`` scenario works.

        Uses ``capfd`` (fd-level capture) rather than ``capsys`` because
        the inherit path writes to the raw file descriptor, bypassing
        Python's ``sys.stdout``.
        """
        script = "import sys; print('visible-output')"

        result = _run_agent_blocking(
            _ResolvedAgentRun(
                [sys.executable, "-c", script],
                stdin_text="",
                timeout=10,
                log_dir=None,
                iteration=1,
                on_output_line=None,
            )
        )

        assert result.returncode == 0
        captured = capfd.readouterr()
        assert "visible-output" in captured.out

    def test_callback_only_does_not_buffer(self, tmp_path: Path):  # pyright: ignore[reportUnusedParameter]
        """When on_output_line is set but log_dir is None, lines should
        be forwarded to the callback but NOT accumulated (no unbounded
        buffering).  Verified indirectly: log_file is None (no buffer to
        write) but the callback still receives lines."""
        script = "import sys; print('line1'); print('line2')"
        received: list[str] = []

        result = _run_agent_blocking(
            _ResolvedAgentRun(
                [sys.executable, "-c", script],
                stdin_text="",
                timeout=10,
                log_dir=None,
                iteration=1,
                on_output_line=lambda line, stream: received.append(line),
            )
        )

        assert result.returncode == 0
        assert result.log_file is None
        assert received == ["line1", "line2"]


class TestPumpStreamExceptionHandling:
    """Tests for _pump_stream resilience against callback and I/O errors."""

    def test_continues_when_callback_raises(self):
        """A callback that raises on the first line must not prevent
        subsequent lines from being buffered."""
        script = "import sys; print('line1'); print('line2'); print('line3'); sys.stdout.flush()"
        call_count = 0

        def raising_callback(line: str, stream: OutputStream):  # pyright: ignore[reportUnusedParameter]
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("boom")

        result = _run_agent_blocking(
            _ResolvedAgentRun(
                [sys.executable, "-u", "-c", script],
                stdin_text="",
                timeout=10,
                log_dir=None,
                iteration=1,
                on_output_line=raising_callback,
            )
        )

        assert result.returncode == 0
        # The callback was called for all three lines, even though
        # the first invocation raised.
        assert call_count == 3

    def test_buffers_all_lines_when_callback_raises(self, tmp_path: Path):
        """When logging is enabled, all lines must be captured in the log
        even if the callback raises on every single line."""
        script = "import sys; print('a'); print('b'); print('c'); sys.stdout.flush()"

        def always_raises(line: str, stream: OutputStream):  # pyright: ignore[reportUnusedParameter]
            raise ValueError("always fails")

        result = _run_agent_blocking(
            _ResolvedAgentRun(
                [sys.executable, "-u", "-c", script],
                stdin_text="",
                timeout=10,
                log_dir=tmp_path,
                iteration=1,
                on_output_line=always_raises,
            )
        )

        assert result.returncode == 0
        assert result.log_file is not None
        log_text = result.log_file.read_text()
        assert "a" in log_text
        assert "b" in log_text
        assert "c" in log_text

    def test_exits_cleanly_on_closed_stream(self):
        """When the read end of a pipe is closed, the pump thread must
        exit within a short bounded join — not hang forever."""
        import os
        import threading

        read_fd, write_fd = os.pipe()
        read_file = os.fdopen(read_fd, "r")
        write_file = os.fdopen(write_fd, "w")

        buffer: list[str] = []
        thread = threading.Thread(
            target=_pump_stream,
            args=(read_file, buffer, "stdout", None),
            daemon=True,
        )
        thread.start()

        # Write a line so the thread is actively reading, then close.
        _ = write_file.write("hello\n")
        write_file.flush()
        write_file.close()

        # The thread must exit promptly — EOF from the closed write end.
        thread.join(timeout=5)
        assert not thread.is_alive(), "_pump_stream thread did not exit after pipe closed"
        assert buffer == ["hello\n"]

        read_file.close()

    def test_exits_cleanly_on_valueerror(self):
        """A stream whose readline raises ValueError (e.g. closed file)
        must not crash the thread — it should exit cleanly."""
        import threading

        class ClosedStream:
            """Fake stream that raises ValueError on readline, simulating
            a concurrent close of the underlying file descriptor."""

            def readline(self):
                raise ValueError("I/O operation on closed file")

        buffer: list[str] = []
        thread = threading.Thread(
            target=_pump_stream,
            args=(ClosedStream(), buffer, "stdout", None),
            daemon=True,
        )
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive(), "_pump_stream thread did not exit after ValueError"
        assert buffer == []

    def test_exits_cleanly_on_oserror(self):
        """A stream whose readline raises OSError must not crash the thread."""
        import threading

        class BrokenStream:
            def readline(self):
                raise OSError("stream error")

        buffer: list[str] = []
        thread = threading.Thread(
            target=_pump_stream,
            args=(BrokenStream(), buffer, "stdout", None),
            daemon=True,
        )
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive(), "_pump_stream thread did not exit after OSError"
        assert buffer == []


class TestBoundedReaderThreadJoins:
    """Tests for bounded reader-thread joins and pipe-closing in finally blocks.

    Validates that grandchild processes holding stdout/stderr pipes cannot
    hang the CLI, and that joins always happen in the finally block.
    """

    def test_grandchild_inheriting_stdout_does_not_hang(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        """Close pipes when a detached grandchild keeps inherited descriptors open."""
        script = (
            "import subprocess, sys, time\n"
            "subprocess.Popen(\n"
            "    [sys.executable, '-c', 'import time; time.sleep(2)'],\n"
            "    start_new_session=True,\n"
            ")\n"
            "print('parent-output')\n"
            "sys.stdout.flush()\n"
            "time.sleep(0.1)\n"
        )
        close_pipes = MagicMock(
            wraps=agent_module._close_pipes  # pyright: ignore[reportPrivateUsage]
        )
        monkeypatch.setattr(agent_module, "_THREAD_JOIN_TIMEOUT", 0.05)
        monkeypatch.setattr(agent_module, "_close_pipes", close_pipes)

        start = time.monotonic()
        result = _run_agent_blocking(
            _ResolvedAgentRun(
                [sys.executable, "-c", script],
                stdin_text="",
                timeout=15,
                log_dir=tmp_path,
                iteration=1,
            )
        )
        elapsed = time.monotonic() - start

        assert result.returncode == 0
        assert result.timed_out is False
        assert elapsed < 1.0, (
            f"_run_agent_blocking took {elapsed:.1f}s — likely hung on"
            " grandchild holding stdout pipe"
        )
        close_pipes.assert_called_once()

        assert result.log_file is not None
        log_text = result.log_file.read_text()
        assert "parent-output" in log_text

    def test_joins_happen_in_finally(self):
        """Inject an exception after threads start but before proc.wait.

        Both reader threads must have exited by the time the exception
        propagates — proving that the finally block joined them.
        """
        import threading

        stdout_thread_ref: list[threading.Thread] = []
        stderr_thread_ref: list[threading.Thread] = []

        # Script that produces output on both streams then exits.
        script = (
            "import sys; "
            "print('out'); sys.stdout.flush(); "
            "print('err', file=sys.stderr); sys.stderr.flush()"
        )

        class InjectErrorPopen(subprocess.Popen[str]):
            """Popen subclass whose wait() raises RuntimeError on first call.

            This simulates an unexpected exception after the process has
            been started and reader threads are running.
            """

            _first_wait: bool = True

            @override
            def wait(self, timeout: float | None = None) -> int:
                if InjectErrorPopen._first_wait:
                    InjectErrorPopen._first_wait = False
                    raise RuntimeError("injected error")
                return super().wait(timeout=timeout)

        with pytest.raises(RuntimeError, match="injected error"):
            with patch(MOCK_SUBPROCESS, side_effect=lambda *a, **kw: InjectErrorPopen(*a, **kw)):  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
                # Capture thread references by patching _start_pump_thread
                original_start_pump = __import__(  # pyright: ignore[reportAny]
                    "milknado.loop._agent", fromlist=["_start_pump_thread"]
                )._start_pump_thread

                def tracking_start_pump(  # pyright: ignore[reportAny]
                    stream: io.StringIO,
                    buffer: _BoundedOutput | None,
                    stream_name: OutputStream,
                    on_output_line: OutputLineCallback | None,
                ):
                    t = original_start_pump(stream, buffer, stream_name, on_output_line)  # pyright: ignore[reportAny]
                    if stream_name == "stdout":
                        stdout_thread_ref.append(t)  # pyright: ignore[reportAny]
                    else:
                        stderr_thread_ref.append(t)  # pyright: ignore[reportAny]
                    return t  # pyright: ignore[reportAny]

                with patch(
                    "milknado.loop._agent._start_pump_thread",
                    side_effect=tracking_start_pump,
                ):
                    _ = _run_agent_blocking(
                        _ResolvedAgentRun(
                            [sys.executable, "-c", script],
                            stdin_text="",
                            timeout=10,
                            log_dir=None,
                            iteration=1,
                            on_output_line=lambda line, stream: None,
                        )
                    )

        # After the exception propagated, the finally block must have
        # joined (and the pipe close must have unblocked) both threads.
        for ref, name in [(stdout_thread_ref, "stdout"), (stderr_thread_ref, "stderr")]:
            assert len(ref) == 1, f"expected 1 {name} thread, got {len(ref)}"
            thread = ref[0]
            # Give a small grace period — the finally block's join should
            # have already completed, but allow for thread scheduling.
            thread.join(timeout=2.0)
            assert not thread.is_alive(), (
                f"{name} reader thread still alive after exception — "
                "joins are not in the finally block"
            )


class TestArgDeliveryStdin:
    """Tests for the arg-delivery stdin decision threaded through _agent.

    When an adapter's ``deliver_prompt`` returns ``stdin_text=None`` the
    spawn must use ``stdin=DEVNULL`` and start no writer thread; when it
    returns a string the prior stdin=PIPE + writer-thread path is preserved
    byte-for-byte.
    """

    @patch("milknado.loop._agent._start_writer_thread")
    @patch(MOCK_SUBPROCESS, side_effect=ok_proc)
    def test_streaming_arg_delivery_uses_devnull_and_no_writer(
        self, mock_popen: MagicMock, mock_writer: MagicMock
    ):
        _ = _run_agent_streaming(
            _ResolvedAgentRun(
                ["opencode", "run", "--format", "json", "hi"],
                None,
                timeout=None,
                log_dir=None,
                iteration=1,
            )
        )

        assert mock_popen.call_args.kwargs["stdin"] == subprocess.DEVNULL
        mock_writer.assert_not_called()

    @patch("milknado.loop._agent._start_writer_thread")
    @patch(MOCK_SUBPROCESS, side_effect=ok_proc)
    def test_streaming_stdin_delivery_pipes_and_writes(
        self, mock_popen: MagicMock, mock_writer: MagicMock
    ):
        _ = _run_agent_streaming(
            _ResolvedAgentRun(
                ["claude", "-p"],
                "the prompt",
                timeout=None,
                log_dir=None,
                iteration=1,
            )
        )

        assert mock_popen.call_args.kwargs["stdin"] == subprocess.PIPE
        mock_writer.assert_called_once()

    @patch("milknado.loop._agent._start_writer_thread")
    @patch(MOCK_SUBPROCESS, side_effect=ok_proc)
    def test_blocking_arg_delivery_uses_devnull_and_no_writer(
        self, mock_popen: MagicMock, mock_writer: MagicMock
    ):
        _ = _run_agent_blocking(
            _ResolvedAgentRun(
                ["aider", "the prompt"],
                None,
                timeout=None,
                log_dir=None,
                iteration=1,
            )
        )

        assert mock_popen.call_args.kwargs["stdin"] == subprocess.DEVNULL
        mock_writer.assert_not_called()

    @patch("milknado.loop._agent._start_writer_thread")
    @patch(MOCK_SUBPROCESS, side_effect=ok_proc)
    def test_blocking_stdin_delivery_pipes_and_writes(
        self, mock_popen: MagicMock, mock_writer: MagicMock
    ):
        _ = _run_agent_blocking(
            _ResolvedAgentRun(
                ["aider"],
                "the prompt",
                timeout=None,
                log_dir=None,
                iteration=1,
            )
        )

        assert mock_popen.call_args.kwargs["stdin"] == subprocess.PIPE
        mock_writer.assert_called_once()

    def test_execute_agent_threads_arg_delivery_through_opencode(self, tmp_path: Path):  # pyright: ignore[reportUnusedParameter]
        """execute_agent must route the opencode adapter to DEVNULL stdin.

        The opencode adapter appends the prompt to argv and reports
        ``stdin_text=None``; execute_agent must spawn with stdin=DEVNULL.
        """
        with patch(MOCK_SUBPROCESS, side_effect=ok_proc) as mock_popen:
            _ = execute_agent(
                AgentRunSpec(
                    ["opencode", "run"],
                    "do the work",
                    timeout=None,
                    log_dir=None,
                    iteration=1,
                )
            )

        spawn_cmd = mock_popen.call_args.args[0]  # pyright: ignore[reportAny]
        assert spawn_cmd == ["opencode", "run", "--format", "json", "do the work"]
        assert mock_popen.call_args.kwargs["stdin"] == subprocess.DEVNULL

    def test_execute_agent_threads_arg_delivery_through_omp(self):
        with patch(MOCK_SUBPROCESS, side_effect=ok_proc) as mock_popen:
            _ = execute_agent(
                AgentRunSpec(
                    ["omp", "-p", "--auto-approve"],
                    "do the work",
                    timeout=None,
                    log_dir=None,
                    iteration=1,
                )
            )

        assert mock_popen.call_args.args[0] == [
            "omp",
            "-p",
            "--auto-approve",
            "--mode",
            "json",
            "do the work",
        ]
        assert mock_popen.call_args.kwargs["stdin"] == subprocess.DEVNULL

    def test_arg_delivery_does_not_hang_when_child_ignores_stdin(self, tmp_path: Path):
        """Real subprocess: an arg-delivery agent that never reads stdin must
        still complete and have its stdout parsed.

        With ``stdin=DEVNULL`` the child gets immediate EOF; nothing writes
        to a pipe, so there is no deadlock and the result event is parsed.
        """
        # Child writes a JSON result line and exits, never touching stdin.
        script = 'import sys; sys.stdout.write(\'{"type": "result", "result": "ok"}\\n\')'

        start = time.monotonic()
        result = _run_agent_streaming(
            _ResolvedAgentRun(
                [sys.executable, "-u", "-c", script],
                None,
                timeout=10,
                log_dir=tmp_path,
                iteration=1,
            )
        )
        elapsed = time.monotonic() - start

        assert result.returncode == 0
        assert result.timed_out is False
        assert result.result_text == "ok"
        assert elapsed < 9.0

    def test_arg_delivery_devnull_gives_child_eof_when_it_reads_stdin(self, tmp_path: Path):
        """Real subprocess: a child that *blocks reading stdin* must still
        finish under arg delivery.

        This is the hardening guard for the DEVNULL choice. The child does
        ``sys.stdin.read()`` (blocks until EOF) before emitting its result.
        ``stdin=DEVNULL`` delivers immediate EOF so ``read()`` returns ``""``
        and the child proceeds. If the spawn ever regressed to leaving stdin
        as an unwritten open pipe, ``read()`` would block forever and the run
        would hit the timeout — so a deadline-driven completion here would be
        a failure, not a pass.
        """
        # Child blocks on stdin.read(), then reports what it received.
        script = (
            "import sys, json\n"
            "data = sys.stdin.read()\n"
            'sys.stdout.write(json.dumps({"type": "result", "result": data}) + "\\n")\n'
        )

        start = time.monotonic()
        result = _run_agent_streaming(
            _ResolvedAgentRun(
                [sys.executable, "-u", "-c", script],
                None,
                timeout=10,
                log_dir=tmp_path,
                iteration=1,
            )
        )
        elapsed = time.monotonic() - start

        assert result.timed_out is False, "child blocked on stdin — DEVNULL gave no EOF"
        assert result.returncode == 0
        # EOF on DEVNULL means read() returned the empty string, not a hang.
        assert result.result_text == ""
        assert elapsed < 9.0


class TestBoundedOutput:
    def test_drops_oldest_complete_lines_when_tail_exceeds_limit(self) -> None:
        output = _BoundedOutput(limit=6)
        output.append("abc")
        output.append("def")
        output.append("ghi")

        assert list(output) == ["def", "ghi"]

    def test_single_oversized_line_keeps_only_tail(self) -> None:
        output = _BoundedOutput(limit=4)
        output.append("abcdefgh")

        assert list(output) == ["efgh"]


def test_lingering_group_that_exits_during_grace_is_not_sigkilled() -> None:
    proc = MagicMock(pid=123)
    with (
        patch("milknado.loop._agent.IS_WINDOWS", False),
        patch("milknado.loop._agent.time.monotonic", return_value=0),
        patch(
            "milknado.loop._agent.os.killpg",
            side_effect=[None, ProcessLookupError()],
        ) as killpg,
    ):
        _terminate_lingering_group(proc)

    assert killpg.call_args_list == [
        call(123, signal.SIGTERM),
        call(123, 0),
    ]


def test_lingering_group_escalates_and_ignores_sigkill_race() -> None:
    proc = MagicMock(pid=123)
    with (
        patch("milknado.loop._agent.IS_WINDOWS", False),
        patch("milknado.loop._agent.time.monotonic", side_effect=[0, 0, 3]),
        patch("milknado.loop._agent.time.sleep") as sleep,
        patch(
            "milknado.loop._agent.os.killpg",
            side_effect=[None, None, ProcessLookupError()],
        ) as killpg,
    ):
        _terminate_lingering_group(proc)

    assert killpg.call_args_list == [
        call(123, signal.SIGTERM),
        call(123, 0),
        call(123, signal.SIGKILL),
    ]
    sleep.assert_called_once_with(0.1)
