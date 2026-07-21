"""Run agent subprocesses with output capture and timeout enforcement.

This module handles the mechanics of executing agent commands — spawning
processes, enforcing timeouts, capturing output, and writing log files.
The engine module uses these functions but owns the orchestration: state
updates, event emission, and loop control.

Two execution modes are supported:

- **Streaming** (``_run_agent_streaming``) — line-by-line stdout reading
  via ``Popen``, used for agents that emit JSON streams (e.g. Claude Code).
- **Blocking** (``_run_agent_blocking``) — ``subprocess.run`` with optional
  output capture, used for all other agents.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any

from milknado.loop._events import OutputStream
from milknado.loop._output import (
    IS_WINDOWS,
    SESSION_KWARGS,
    SUBPROCESS_TEXT_KWARGS,
    ProcessResult,
    collect_output,
    warn,
)
from milknado.loop._promise import has_promise_completion
from milknado.loop.adapters import CLIAdapter, select_adapter

_log = logging.getLogger(__name__)

_counter_write_failure_logged = False
"""Module-level latch so the wind-down counter write failure logs only once."""

# ── Callback type aliases ──────────────────────────────────────────────
# Used across the streaming and blocking execution paths for callbacks
# that observe live agent output.

ActivityCallback = Callable[[dict[str, Any]], None]
"""Receives parsed JSON activity dicts from the agent's stream."""

OutputLineCallback = Callable[[str, OutputStream], None]
"""Receives raw output lines with their stream name ("stdout"/"stderr")."""

ToolUseCallback = Callable[[str, int], None]
"""Invoked by both the streaming path (as lines arrive) and the blocking
path (after the subprocess exits, during post-hoc counting).  Best-effort:
exceptions are swallowed so a buggy subscriber cannot kill the agent loop.
"""


def _call_safely(
    callback: Callable[..., Any] | None,
    *args: Any,
    correlation: str = "unknown",
) -> None:
    if callback is None:
        return
    try:
        callback(*args)
    except Exception:
        _log.exception(
            "agent observer callback failed callback=%s correlation=%s",
            getattr(callback, "__name__", type(callback).__name__),
            correlation,
        )


# Typed constants for the OutputStream literal so the type checker enforces
# that only "stdout" / "stderr" ever reach ``on_output_line``.
_STDOUT: OutputStream = "stdout"
_STDERR: OutputStream = "stderr"

# JSON stream event types and fields for result extraction.
_RESULT_EVENT_TYPE = "result"
_RESULT_FIELD = "result"

# Log file naming — timestamp format and iteration zero-padding width.
_LOG_TIMESTAMP_FORMAT = "%Y%m%d-%H%M%S"
_LOG_ITERATION_PAD_WIDTH = 3

# Seconds to wait for graceful shutdown after SIGTERM before escalating to SIGKILL.
_SIGTERM_GRACE_PERIOD = 3

# Seconds to wait for reader threads to drain during cleanup.
# Generous bound: after parent-side pipe close, EOF propagates within
# milliseconds — 5s is headroom for slow kernels / loaded CI boxes.
_THREAD_JOIN_TIMEOUT = 5.0

# Seconds to wait for the agent process to exit after a kill signal.
_PROCESS_WAIT_TIMEOUT = 5.0
_OUTPUT_TAIL_CHARS = 64 * 1024
_STREAM_QUEUE_MAX_LINES = 256


@dataclass(slots=True)
class _BoundedOutput:
    limit: int = _OUTPUT_TAIL_CHARS
    _lines: list[str] = field(default_factory=list)
    _chars: int = 0

    def append(self, line: str) -> None:
        if len(line) > self.limit:
            line = line[-self.limit :]
        self._lines.append(line)
        self._chars += len(line)
        while self._chars > self.limit and len(self._lines) > 1:
            self._chars -= len(self._lines.pop(0))

    def __iter__(self):
        return iter(self._lines)

    @property
    def text(self) -> str:
        return "".join(self._lines)


@dataclass(slots=True)
class _FileSink:
    """Serialize complete output to a file without retaining it in memory."""

    handle: IO[str]
    path: Path | None = None

    def write(self, line: str) -> None:
        self.handle.write(line)
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


@dataclass(slots=True)
class _OutputCapture:
    """Bounded tail plus an optional file-backed complete-output sink."""

    tail: _BoundedOutput | None = None
    sink: _FileSink | None = None
    mirror: _FileSink | None = None

    def append(self, line: str) -> None:
        if self.tail is not None:
            self.tail.append(line)
        if self.sink is not None:
            self.sink.write(line)
        if self.mirror is not None:
            self.mirror.write(line)

    @property
    def text(self) -> str | None:
        return self.tail.text if self.tail is not None else None

    def iter_complete_lines(self):
        if self.sink is None:
            return ()
        self.sink.handle.flush()
        self.sink.handle.seek(0)
        return iter(self.sink.handle.readline, "")

    def close(self) -> None:
        if self.sink is not None:
            self.sink.close()


def _new_output_sink(log_dir: Path | None, iteration: int) -> _FileSink | None:
    if log_dir is None:
        return _FileSink(tempfile.TemporaryFile(mode="w+", encoding="utf-8"))
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime(_LOG_TIMESTAMP_FORMAT)
    path = log_dir / f"{iteration:0{_LOG_ITERATION_PAD_WIDTH}d}_{timestamp}.log"
    return _FileSink(path.open("w", encoding="utf-8"), path)


# Tempdir prefix for per-iteration wind-down hook config.  Surfaces in
# ``$TMPDIR`` listings so users can spot stale dirs after a crash.
_HOOK_TEMPDIR_PREFIX = "milknado-loop-"

# Counter filename when ``log_dir`` is unset and the file lives inside
# the per-iteration tempdir.  When ``log_dir`` is set, the counter lives
# at ``<log_dir>/<iter>.turncount``.
_COUNTER_FILENAME = "turncount"
_COUNTER_LOG_SUFFIX = ".turncount"
_COUNTER_PAD_WIDTH = _LOG_ITERATION_PAD_WIDTH


def _extract_result_text_from_lines(lines: list[str] | None) -> str | None:
    """Return the last string payload from any JSON ``result`` event in *lines*."""
    if lines is None:
        return None

    result_text = None
    for line in lines:
        extracted = _extract_result_text_from_line(line)
        if extracted is not None:
            result_text = extracted
    return result_text


def _extract_result_text_from_line(line: str) -> str | None:
    """Return the string payload from a single JSON ``result`` event line."""
    try:
        parsed = json.loads(line.strip())
    except json.JSONDecodeError:
        return None
    if (
        isinstance(parsed, dict)
        and parsed.get("type") == _RESULT_EVENT_TYPE
        and isinstance(parsed.get(_RESULT_FIELD), str)
    ):
        return parsed[_RESULT_FIELD]
    return None


def _try_graceful_group_kill(proc: subprocess.Popen[Any]) -> bool:
    """Attempt to kill the process via its POSIX process group.

    Sends SIGTERM, waits briefly, then escalates to SIGKILL if needed.
    Only acts when the process is a session leader (pgid == pid) to avoid
    accidentally killing the caller's group in tests.

    Returns ``True`` if the group kill succeeded, ``False`` if the caller
    should fall back to ``proc.kill()``.
    """
    try:
        pgid = os.getpgid(proc.pid)
    except (OSError, ProcessLookupError):
        return False

    if pgid != proc.pid:
        return False

    try:
        os.killpg(pgid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        return False

    try:
        proc.wait(timeout=_SIGTERM_GRACE_PERIOD)
        return True
    except subprocess.TimeoutExpired:
        pass

    try:
        os.killpg(pgid, signal.SIGKILL)
        return True
    except (OSError, ProcessLookupError):
        return False


def _kill_process_group(proc: subprocess.Popen[Any]) -> None:
    """Kill the agent process and its entire process group.

    On POSIX, attempts a graceful group kill (SIGTERM → SIGKILL) via
    :func:`_try_graceful_group_kill`.  Falls back to ``proc.kill()``
    on Windows, when the process already exited, or if the group kill fails.

    Short-circuits when ``proc.pid`` is ``None`` or non-positive — these
    are sentinel values used by test mocks.  A non-positive pid would also
    be dangerous to pass to ``os.getpgid`` / ``os.killpg`` (pid 0 means
    the caller's own process group).
    """
    if proc.pid is None or proc.pid <= 0:
        return
    if not IS_WINDOWS and proc.poll() is None:
        if _try_graceful_group_kill(proc):
            return
    proc.kill()


def _ensure_process_dead(proc: subprocess.Popen[Any]) -> None:
    """Kill the agent process if still running, then wait for exit.

    Safe to call multiple times — no-ops when the process has already
    exited.  Used in ``finally`` and exception-handler blocks to
    guarantee the child is reaped before we move on.

    The wait is bounded so that a stuck process (e.g. grandchild holding
    a session) cannot hang the CLI forever.
    """
    if proc.poll() is None:
        _kill_process_group(proc)
    try:
        proc.wait(timeout=_PROCESS_WAIT_TIMEOUT)
    except subprocess.TimeoutExpired:
        warn(f"agent process did not exit within {_PROCESS_WAIT_TIMEOUT}s after kill")


def _close_pipes(proc: subprocess.Popen[Any]) -> None:
    """Close parent-side stdout/stderr pipe file descriptors.

    Forces EOF to propagate to reader threads even when grandchild
    processes have inherited the write end of the pipe.  The pump
    thread's ``readline()`` wakes with ``OSError`` (EBADF), which
    ``_pump_stream`` catches, so the thread exits and ``join()``
    returns promptly.

    Uses ``os.close()`` on the raw fd rather than ``pipe.close()``
    because Python's ``TextIOWrapper`` / ``BufferedReader`` hold an
    internal lock during ``readline()``.  Calling ``pipe.close()``
    from the main thread would block waiting for that lock — exactly
    the hang we're trying to break.  ``os.close()`` bypasses the
    Python lock and directly invalidates the fd at the OS level.

    Safe to call multiple times or on already-closed pipes.
    """
    for pipe in (proc.stdout, proc.stderr):
        if pipe is not None:
            try:
                os.close(pipe.fileno())
            except Exception:
                pass


def _finalize_pipes(proc: subprocess.Popen[Any]) -> None:
    """Mark parent-side pipe file objects as closed at the Python level.

    Called AFTER :func:`_close_pipes` and :func:`_drain_readers` to set
    the ``closed`` flag on the Python file objects.  This prevents
    "Bad file descriptor" warnings when the garbage collector finalizes
    the objects whose underlying fd was already closed by
    :func:`_close_pipes`.

    Must be called after reader threads have exited — the pipe's
    internal lock is held during ``readline()``, so ``close()`` would
    block on a live thread.
    """
    for pipe in (proc.stdout, proc.stderr):
        if pipe is not None:
            try:
                pipe.close()
            except Exception:
                pass


def _deliver_prompt(proc: subprocess.Popen[Any], prompt: str) -> None:
    """Write *prompt* to the agent's stdin and close the pipe.

    Silently handles ``BrokenPipeError`` — the agent may exit before
    consuming the full prompt, which is a normal (if uncommon) lifecycle
    event.
    """
    if proc.stdin is None:
        raise RuntimeError("subprocess.Popen did not provide stdin for prompt delivery")
    try:
        proc.stdin.write(prompt)
    except BrokenPipeError:
        pass
    finally:
        try:
            proc.stdin.close()
        except BrokenPipeError:
            pass


@dataclass(slots=True)
class AgentResult(ProcessResult):
    """Result of running the agent subprocess."""

    elapsed: float = 0.0
    log_file: Path | None = None
    result_text: str | None = None
    captured_stdout: str | None = None
    completion_detected: bool = False
    captured_stderr: str | None = None
    # Tool-use events the adapter reported for this iteration; ``0`` when
    # no adapter was active or the adapter's ``counts_what != "tool_use"``.
    tool_use_count: int = 0
    # True when the cap was reached and the agent process was terminated
    # (streaming) or the cap was exceeded post-hoc (blocking path).
    turn_capped: bool = False


@dataclass(slots=True)
class _WindDownContext:
    """Per-iteration tempdir and counter state for soft wind-down hooks.

    Built by :func:`_setup_wind_down` for adapters whose
    ``supports_soft_wind_down`` is True and a ``max_turns`` cap is
    configured.  Carries the env-var overrides the spawned agent needs,
    plus the cleanup hook called from the streaming path's ``finally``.
    """

    tempdir: Path
    counter_path: Path
    env_overrides: dict[str, str]

    def cleanup(self) -> None:
        try:
            self.counter_path.unlink()
        except (FileNotFoundError, OSError):
            pass
        shutil.rmtree(self.tempdir, ignore_errors=True)


@dataclass(frozen=True, slots=True)
class _StreamResult:
    """Accumulated output from reading the agent's JSON stream."""

    stdout_lines: tuple[str, ...] | None
    result_text: str | None
    timed_out: bool
    tool_use_count: int = 0
    turn_capped: bool = False
    completion_detected: bool = False


@dataclass(frozen=True, slots=True)
class AgentRunSpec:
    """Everything execute_agent needs for one subprocess run."""

    cmd: list[str]
    prompt: str
    timeout: float | None
    log_dir: Path | None
    iteration: int
    adapter: CLIAdapter | None = None
    on_activity: ActivityCallback | None = None
    on_output_line: OutputLineCallback | None = None
    capture_result_text: bool = False
    capture_stdout: bool | None = None
    max_turns: int | None = None
    max_turns_grace: int = 0
    completion_signal: str | None = None
    on_tool_use: ToolUseCallback | None = None
    cwd: Path | None = None


@dataclass(frozen=True, slots=True)
class _ResolvedAgentRun:
    """Post-adapter-resolution run parameters shared by the streaming and
    blocking execution paths."""

    cmd: list[str]
    stdin_text: str | None
    timeout: float | None
    log_dir: Path | None
    iteration: int
    on_activity: ActivityCallback | None = None
    on_output_line: OutputLineCallback | None = None
    capture_result_text: bool = False
    capture_stdout: bool = False
    adapter: CLIAdapter | None = None
    max_turns: int | None = None
    on_tool_use: ToolUseCallback | None = None
    env: dict[str, str] | None = None
    completion_signal: str | None = None
    cwd: Path | None = None


def _write_log(
    log_dir: Path | None,
    iteration: int,
    stdout: str | bytes | None,
    stderr: str | bytes | None,
) -> Path | None:
    """Write iteration output to a timestamped log file if logging is configured.

    Returns the log file path, or ``None`` when *log_dir* is not set.
    """
    if log_dir is None:
        return None
    timestamp = datetime.now(UTC).strftime(_LOG_TIMESTAMP_FORMAT)
    log_file = log_dir / f"{iteration:0{_LOG_ITERATION_PAD_WIDTH}d}_{timestamp}.log"
    log_file.write_text(collect_output(stdout, stderr), encoding="utf-8")
    return log_file


def _readline_pump(
    stdout: IO[str],
    line_queue: queue.Queue[str | None],
    stop_event: threading.Event | None = None,
) -> None:
    """Read stdout into a bounded queue without losing complete log output."""
    try:
        for line in iter(stdout.readline, ""):
            while True:
                if stop_event is not None and stop_event.is_set():
                    return
                try:
                    line_queue.put(line, timeout=0.1)
                    break
                except queue.Full:
                    continue
    except (ValueError, OSError):
        pass
    finally:
        while stop_event is None or not stop_event.is_set():
            try:
                line_queue.put(None, timeout=0.1)
                break
            except queue.Full:
                continue


@dataclass(slots=True)
class _StreamState:
    stdout_lines: _BoundedOutput | None
    output_capture: _OutputCapture | None = None
    completion_signal: str | None = None
    result_text: str | None = None
    tool_use_count: int = 0
    turn_capped: bool = False
    completion_detected: bool = False

    def to_result(self, *, timed_out: bool) -> _StreamResult:
        return _StreamResult(
            stdout_lines=tuple(self.stdout_lines) if self.stdout_lines is not None else None,
            result_text=self.result_text,
            timed_out=timed_out,
            tool_use_count=self.tool_use_count,
            turn_capped=self.turn_capped,
            completion_detected=self.completion_detected,
        )


def _record_stream_line(
    state: _StreamState,
    line: str,
    on_output_line: OutputLineCallback | None,
    correlation: str,
) -> None:
    if state.output_capture is not None:
        state.output_capture.append(line)
    elif state.stdout_lines is not None:
        state.stdout_lines.append(line)
    if state.completion_signal and has_promise_completion(line, state.completion_signal):
        state.completion_detected = True
    _call_safely(on_output_line, line.rstrip("\r\n"), _STDOUT, correlation=correlation)


def _update_tool_use(
    state: _StreamState,
    adapter: CLIAdapter,
    on_tool_use: ToolUseCallback | None,
    max_turns: int | None,
    stripped_line: str,
    correlation: str,
) -> bool:
    event = adapter.parse_event(stripped_line)
    if event is None or event.kind != "tool_use":
        return False
    state.tool_use_count += 1
    _call_safely(on_tool_use, event.name or "", state.tool_use_count, correlation=correlation)
    if max_turns is not None and state.tool_use_count >= max_turns:
        state.turn_capped = True
        return True
    return False


def _read_agent_stream(
    stdout: IO[str],
    deadline: float | None,
    on_activity: ActivityCallback | None,
    on_output_line: OutputLineCallback | None = None,
    *,
    capture_stdout: bool = True,
    output_capture: _OutputCapture | None = None,
    completion_signal: str | None = None,
    adapter: CLIAdapter | None = None,
    max_turns: int | None = None,
    on_tool_use: ToolUseCallback | None = None,
    correlation: str = "unknown",
    pump_threads: list[threading.Thread] | None = None,
) -> _StreamResult:
    state = _StreamState(
        stdout_lines=(
            output_capture.tail
            if output_capture is not None
            else (_BoundedOutput() if capture_stdout else None)
        ),
        output_capture=output_capture,
        completion_signal=completion_signal,
    )
    count_tool_use = adapter is not None and adapter.counts_what == "tool_use"
    line_q: queue.Queue[str | None] = queue.Queue(maxsize=_STREAM_QUEUE_MAX_LINES)
    stop_event = threading.Event()
    pump_thread = threading.Thread(
        target=_readline_pump,
        args=(stdout, line_q, stop_event),
        daemon=True,
    )
    pump_thread.start()
    if pump_threads is not None:
        pump_threads.append(pump_thread)

    while True:
        get_timeout = max(deadline - time.monotonic(), 0) if deadline is not None else None
        try:
            line = line_q.get(timeout=get_timeout)
        except queue.Empty:
            stop_event.set()
            return state.to_result(timed_out=True)
        if line is None:
            return state.to_result(timed_out=False)

        _record_stream_line(state, line, on_output_line, correlation)
        stripped = line.strip()
        if stripped:
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                pass
            else:
                if isinstance(parsed, dict):
                    if parsed.get("type") == _RESULT_EVENT_TYPE and isinstance(
                        parsed.get(_RESULT_FIELD), str
                    ):
                        state.result_text = parsed[_RESULT_FIELD]
                    _call_safely(on_activity, parsed, correlation=correlation)
            if count_tool_use and adapter is not None:
                if _update_tool_use(
                    state,
                    adapter,
                    on_tool_use,
                    max_turns,
                    stripped,
                    correlation,
                ):
                    stop_event.set()
                    return state.to_result(timed_out=False)
        if deadline is not None and time.monotonic() > deadline:
            stop_event.set()
            return state.to_result(timed_out=True)


def _run_agent_streaming(run: _ResolvedAgentRun) -> AgentResult:
    start = time.monotonic()
    deadline = (start + run.timeout) if run.timeout is not None else None
    capture_stdout_text = run.log_dir is not None or run.capture_stdout
    pipe_stderr = run.log_dir is not None or run.on_output_line is not None
    capture_stderr_text = run.log_dir is not None
    pipe_stdin = run.stdin_text is not None
    writer_thread: threading.Thread | None = None
    stderr_thread: threading.Thread | None = None
    pump_threads: list[threading.Thread] = []
    correlation = f"iteration={run.iteration}"
    log_sink = _new_output_sink(run.log_dir, run.iteration) if run.log_dir is not None else None
    stdout_capture = _OutputCapture(
        tail=_BoundedOutput() if capture_stdout_text else None,
        sink=_new_output_sink(None, run.iteration) if capture_stdout_text else None,
        mirror=log_sink,
    )
    stderr_capture = _OutputCapture(
        tail=_BoundedOutput() if capture_stderr_text else None,
        mirror=log_sink,
    )

    try:
        proc = subprocess.Popen(
            run.cmd,
            stdin=subprocess.PIPE if pipe_stdin else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE if pipe_stderr else None,
            env=_build_spawn_env(run.env),
            cwd=run.cwd,
            **SUBPROCESS_TEXT_KWARGS,
            **SESSION_KWARGS,
        )
    except Exception:
        stdout_capture.close()
        if log_sink is not None:
            log_sink.close()
        raise

    try:
        if proc.stdout is None:
            raise RuntimeError("subprocess.Popen failed to create PIPE stdout")
        if pipe_stdin and proc.stdin is None:
            raise RuntimeError("subprocess.Popen failed to create PIPE stdin")
        if pipe_stderr and proc.stderr is None:
            raise RuntimeError("subprocess.Popen failed to create PIPE stderr")
        if proc.stderr is not None:
            stderr_thread = _start_pump_thread(
                proc.stderr,
                stderr_capture,
                _STDERR,
                run.on_output_line,
                correlation,
            )
            pump_threads.append(stderr_thread)
        if run.stdin_text is not None:
            writer_thread = _start_writer_thread(proc, run.stdin_text)
        stream = _read_agent_stream(
            proc.stdout,
            deadline,
            run.on_activity,
            run.on_output_line,
            capture_stdout=capture_stdout_text,
            output_capture=stdout_capture,
            completion_signal=run.completion_signal,
            adapter=run.adapter,
            max_turns=run.max_turns,
            on_tool_use=run.on_tool_use,
            correlation=correlation,
            pump_threads=pump_threads,
        )
        if stream.timed_out or stream.turn_capped:
            _kill_process_group(proc)
        proc.wait()
    finally:
        _cleanup_agent(proc, *pump_threads, writer_thread)

    stdout = stdout_capture.text
    stderr = stderr_capture.text
    stdout_capture.close()
    if log_sink is not None:
        log_sink.close()
    return AgentResult(
        returncode=None if stream.timed_out else proc.returncode,
        elapsed=time.monotonic() - start,
        log_file=log_sink.path if log_sink is not None else None,
        result_text=stream.result_text,
        timed_out=stream.timed_out,
        captured_stdout=stdout if capture_stdout_text else None,
        captured_stderr=stderr if capture_stderr_text else None,
        tool_use_count=stream.tool_use_count,
        turn_capped=stream.turn_capped,
        completion_detected=stream.completion_detected,
    )


def _pump_stream(
    stream: IO[str],
    buffer: Any | None,
    stream_name: OutputStream,
    on_output_line: OutputLineCallback | None,
    correlation: str = "unknown",
) -> None:
    try:
        for line in iter(stream.readline, ""):
            if buffer is not None:
                buffer.append(line)
            _call_safely(
                on_output_line,
                line.rstrip("\r\n"),
                stream_name,
                correlation=correlation,
            )
    except (ValueError, OSError):
        pass


def _start_writer_thread(
    proc: subprocess.Popen[Any],
    prompt: str,
) -> threading.Thread:
    thread = threading.Thread(target=_deliver_prompt, args=(proc, prompt), daemon=True)
    thread.start()
    return thread


def _start_pump_thread(
    stream: IO[str],
    buffer: Any | None,
    stream_name: OutputStream,
    on_output_line: OutputLineCallback | None,
    correlation: str = "unknown",
) -> threading.Thread:
    thread = threading.Thread(
        target=_pump_stream,
        args=(stream, buffer, stream_name, on_output_line, correlation),
        daemon=True,
    )
    thread.start()
    return thread


def _drain_readers(
    *threads: threading.Thread | None,
    timeout: float = _THREAD_JOIN_TIMEOUT,
) -> None:
    """Join reader threads, skipping any that are ``None``.

    Used in ``finally`` blocks to ensure background pump threads finish
    draining before the caller continues.  Logs a warning to stderr for
    any thread that fails to exit within the timeout — this is visible
    feedback that a grandchild may be holding a pipe open and the log
    may be incomplete.
    """
    for thread in threads:
        if thread is not None:
            thread.join(timeout=timeout)
            if thread.is_alive():
                warn(
                    f"reader thread {thread.name!r} did not exit within"
                    f" {timeout}s — log output may be incomplete"
                )


def _terminate_lingering_group(proc: subprocess.Popen[Any]) -> None:
    if IS_WINDOWS or proc.pid is None or proc.pid <= 0:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        pass


def _cleanup_agent(
    proc: subprocess.Popen[Any],
    *threads: threading.Thread | None,
) -> None:
    """Perform the full four-step shutdown for a piped agent subprocess.

    1. Kill the process if still running and wait for exit.
    2. Close parent-side pipe fds to unblock reader threads.
    3. Join reader/writer threads to drain remaining output.
    4. Finalize Python pipe objects to suppress GC warnings.

    Used in ``finally`` blocks of the streaming and blocking capture
    paths, which previously duplicated this exact sequence inline.
    """
    _ensure_process_dead(proc)
    if proc.poll() is not None:
        _terminate_lingering_group(proc)
    _close_pipes(proc)
    _drain_readers(*threads)
    _finalize_pipes(proc)


def _run_agent_blocking(run: _ResolvedAgentRun) -> AgentResult:
    """Run a non-streaming agent with bounded tails and file-backed output."""
    start = time.monotonic()
    needs_post_hoc_count = (
        run.max_turns is not None
        and run.adapter is not None
        and run.adapter.counts_what == "tool_use"
    )
    capture_stdout_text = run.log_dir is not None or run.capture_stdout or needs_post_hoc_count
    capture_stderr_text = run.log_dir is not None
    pipe_stdout = capture_stdout_text or run.on_output_line is not None or run.capture_result_text
    pipe_stderr = capture_stderr_text or run.on_output_line is not None
    pipe_stdin = run.stdin_text is not None

    returncode: int | None = None
    timed_out = False
    writer_thread: threading.Thread | None = None
    stdout_thread: threading.Thread | None = None
    stderr_thread: threading.Thread | None = None
    log_sink = _new_output_sink(run.log_dir, run.iteration) if run.log_dir is not None else None
    stdout_capture = _OutputCapture(
        tail=_BoundedOutput() if capture_stdout_text else None,
        sink=_new_output_sink(None, run.iteration) if capture_stdout_text else None,
        mirror=log_sink,
    )
    stderr_capture = _OutputCapture(
        tail=_BoundedOutput() if capture_stderr_text else None,
        mirror=log_sink,
    )
    result_text: str | None = None
    completion_detected = False

    def _on_output_line(line: str, stream_name: OutputStream) -> None:
        nonlocal result_text, completion_detected
        if stream_name == _STDOUT:
            if run.capture_result_text:
                extracted = _extract_result_text_from_line(line)
                if extracted is not None:
                    result_text = extracted
            if run.completion_signal and has_promise_completion(line, run.completion_signal):
                completion_detected = True
        _call_safely(
            run.on_output_line,
            line,
            stream_name,
            correlation=f"iteration={run.iteration}",
        )

    try:
        proc = subprocess.Popen(
            run.cmd,
            stdin=subprocess.PIPE if pipe_stdin else subprocess.DEVNULL,
            stdout=subprocess.PIPE if pipe_stdout else None,
            stderr=subprocess.PIPE if pipe_stderr else None,
            env=_build_spawn_env(run.env),
            cwd=run.cwd,
            **SUBPROCESS_TEXT_KWARGS,
            **SESSION_KWARGS,
        )
    except Exception:
        stdout_capture.close()
        if log_sink is not None:
            log_sink.close()
        raise

    try:
        if pipe_stdin and proc.stdin is None:
            raise RuntimeError("subprocess.Popen failed to create PIPE stdin")
        if pipe_stdout:
            if proc.stdout is None:
                raise RuntimeError("subprocess.Popen failed to create PIPE stdout")
            stdout_thread = _start_pump_thread(
                proc.stdout, stdout_capture, _STDOUT, _on_output_line
            )
        if pipe_stderr:
            if proc.stderr is None:
                raise RuntimeError("subprocess.Popen failed to create PIPE stderr")
            stderr_thread = _start_pump_thread(
                proc.stderr, stderr_capture, _STDERR, _on_output_line
            )
        if run.stdin_text is not None:
            writer_thread = _start_writer_thread(proc, run.stdin_text)

        try:
            returncode = proc.wait(timeout=run.timeout)
        except subprocess.TimeoutExpired:
            _ensure_process_dead(proc)
            timed_out = True
    finally:
        _cleanup_agent(proc, stdout_thread, stderr_thread, writer_thread)

    stdout = stdout_capture.text
    stderr = stderr_capture.text
    tool_use_count, turn_capped = _count_tool_uses_post_hoc(
        adapter=run.adapter,
        stdout_lines=stdout_capture.tail,
        stdout_capture=stdout_capture,
        max_turns=run.max_turns,
        on_tool_use=run.on_tool_use,
    )
    stdout_capture.close()
    if log_sink is not None:
        log_sink.close()

    return AgentResult(
        returncode=None if timed_out else returncode,
        elapsed=time.monotonic() - start,
        log_file=log_sink.path if log_sink is not None else None,
        result_text=result_text or _extract_result_text_from_lines(stdout_capture.tail),
        timed_out=timed_out,
        captured_stdout=stdout if capture_stdout_text else None,
        captured_stderr=stderr if capture_stderr_text else None,
        tool_use_count=tool_use_count,
        turn_capped=turn_capped,
        completion_detected=completion_detected,
    )


def _prepare_agent_run(
    spec: AgentRunSpec,
    adapter: CLIAdapter,
    cmd: list[str],
    stdin_text: str | None,
    on_tool_use: ToolUseCallback | None,
    env: dict[str, str] | None,
) -> _ResolvedAgentRun:
    """Resolve the ``capture_stdout`` default and assemble the run object
    shared by the streaming and blocking execution paths."""
    capture_stdout = spec.capture_stdout
    if capture_stdout is None:
        capture_stdout = spec.log_dir is not None or (
            not adapter.supports_streaming
            and spec.on_output_line is None
            and spec.capture_result_text
        )
    return _ResolvedAgentRun(
        cmd=cmd,
        stdin_text=stdin_text,
        timeout=spec.timeout,
        log_dir=spec.log_dir,
        iteration=spec.iteration,
        on_activity=spec.on_activity,
        on_output_line=spec.on_output_line,
        capture_result_text=spec.capture_result_text,
        capture_stdout=capture_stdout,
        adapter=adapter,
        max_turns=spec.max_turns,
        on_tool_use=on_tool_use,
        env=env,
        cwd=spec.cwd,
        completion_signal=spec.completion_signal,
    )


def execute_agent(spec: AgentRunSpec) -> AgentResult:
    """Run the agent subprocess, auto-selecting streaming or blocking mode.

    ``spec.adapter`` (or :func:`select_adapter` when omitted) decides which
    execution path runs: adapters whose ``supports_streaming`` flag is True
    take the line-streaming path that drives ``on_activity`` callbacks; all
    others take the blocking path with concurrent stdout/stderr drain.
    ``adapter.build_command(spec.cmd)`` is applied before spawning, so the
    CLI receives any flags the adapter requires (e.g. Claude's
    ``--output-format stream-json --verbose`` or Codex's ``--json``).

    When ``spec.max_turns`` is set, the streaming path counts adapter-reported
    tool-use events and terminates the subprocess once the cap is reached.
    The blocking path cannot preempt but records the post-hoc count.
    ``spec.max_turns_grace`` enables a soft wind-down: if the adapter supports
    it, a per-iteration tempdir is set up with a counter file and environment
    variables pointing the agent at ``_wind_down_shim`` so it can warn the
    agent when the cap is ``grace`` tool-uses away.

    This is the single entry point the engine should use — callers don't need
    to know which execution mode is selected.

    ``spec.cwd`` is the working directory the spawned agent process runs in.
    When ``None`` (the default), the child inherits the parent's cwd —
    preserving behaviour for direct engine callers. Pass the worktree path so
    a worker agent edits its own node's repo rather than the orchestrator's.
    """
    adapter = spec.adapter if spec.adapter is not None else select_adapter(spec.cmd)
    cmd = adapter.build_command(spec.cmd)
    # Let the adapter decide where the prompt goes: stdin adapters return
    # the command unchanged with ``stdin_text=prompt``; arg-delivery adapters
    # (e.g. opencode) append the prompt to argv and return ``stdin_text=None``
    # so the child is spawned with ``stdin=DEVNULL`` and no writer thread.
    inv = adapter.deliver_prompt(cmd, spec.prompt)

    wind_down = _setup_wind_down(
        adapter=adapter,
        max_turns=spec.max_turns,
        max_turns_grace=spec.max_turns_grace,
        log_dir=spec.log_dir,
        iteration=spec.iteration,
    )
    wrapped_on_tool_use = _wrap_tool_use_with_counter(
        on_tool_use=spec.on_tool_use,
        counter_path=wind_down.counter_path if wind_down is not None else None,
    )
    env = wind_down.env_overrides if wind_down is not None else None

    run = _prepare_agent_run(spec, adapter, inv.argv, inv.stdin_text, wrapped_on_tool_use, env)

    try:
        if adapter.supports_streaming:
            return _run_agent_streaming(run)
        return _run_agent_blocking(run)
    finally:
        if wind_down is not None:
            wind_down.cleanup()


def _build_spawn_env(overrides: dict[str, str] | None) -> dict[str, str] | None:
    """Compose a spawn environment merging *overrides* onto ``os.environ``.

    Returns ``None`` when no overrides are requested so ``Popen`` inherits
    the parent's environment directly (the common case).
    """
    if not overrides:
        return None
    merged = os.environ.copy()
    merged.update(overrides)
    return merged


def _setup_wind_down(
    *,
    adapter: CLIAdapter,
    max_turns: int | None,
    max_turns_grace: int,
    log_dir: Path | None,
    iteration: int,
) -> _WindDownContext | None:
    """Prepare the per-iteration tempdir + counter for soft wind-down.

    Returns ``None`` — skipping the hook wiring entirely — when any of the
    preconditions are not met:

    - *max_turns* is unset (no cap = nothing to wind down toward).
    - *max_turns_grace* is ``0`` (user opted out of the warning window).
    - The adapter's ``supports_soft_wind_down`` flag is ``False``.
    - The adapter's ``install_wind_down_hook`` raises
      ``NotImplementedError`` (e.g. Copilot has no hook system today).

    The tempdir is created with a ``milknado-loop-`` prefix so stale dirs are
    easy to spot in ``$TMPDIR`` after a crash.  The counter file lives in
    *log_dir* when logging is enabled (so it's co-located with the
    iteration log) and inside the tempdir otherwise.  Either way the
    caller must invoke :meth:`_WindDownContext.cleanup` in its ``finally``.
    """
    if max_turns is None or max_turns_grace <= 0:
        return None
    if not adapter.supports_soft_wind_down:
        return None
    # Clamp the grace below the cap.  A grace >= max_turns is reachable via
    # the library API (RunConfig does not validate it the way the CLI does);
    # left unclamped the shim's threshold (max(cap - grace, 0)) collapses to
    # 0 and injects the wind-down nudge on the very first tool use.
    effective_grace = min(max_turns_grace, max(max_turns - 1, 0))
    tempdir = Path(tempfile.mkdtemp(prefix=_HOOK_TEMPDIR_PREFIX))
    counter_path = _resolve_counter_path(log_dir, iteration, tempdir)
    _atomic_write_counter(counter_path, 0)
    try:
        env_overrides = adapter.install_wind_down_hook(
            tempdir=tempdir,
            counter_path=counter_path,
            cap=max_turns,
            grace=effective_grace,
        )
    except NotImplementedError:
        # counter_path may live in log_dir rather than the tempdir, so
        # removing the tempdir alone would orphan the "0" counter file in
        # the log directory.  Remove it explicitly, best-effort.
        with suppress(OSError):
            counter_path.unlink(missing_ok=True)
        shutil.rmtree(tempdir, ignore_errors=True)
        return None
    return _WindDownContext(
        tempdir=tempdir,
        counter_path=counter_path,
        env_overrides=env_overrides,
    )


def _resolve_counter_path(
    log_dir: Path | None,
    iteration: int,
    tempdir: Path,
) -> Path:
    """Pick the counter file location: co-located with the log or inside tempdir."""
    if log_dir is not None:
        filename = f"{iteration:0{_COUNTER_PAD_WIDTH}d}{_COUNTER_LOG_SUFFIX}"
        return log_dir / filename
    return tempdir / _COUNTER_FILENAME


def _atomic_write_counter(counter_path: Path, value: int) -> None:
    """Write *value* to *counter_path* via rename so readers never see a partial int.

    Best-effort: on any I/O failure the caller proceeds without wind-down
    state — the warning subscription is advisory, not required.  The first
    failure is logged once at WARNING so a silently-disabled wind-down is
    greppable; subsequent failures stay quiet to avoid per-iteration spam.
    """
    global _counter_write_failure_logged
    try:
        tmp_path = counter_path.with_name(counter_path.name + ".tmp")
        tmp_path.write_text(str(value), encoding="utf-8")
        os.replace(tmp_path, counter_path)
    except OSError as exc:
        if not _counter_write_failure_logged:
            _counter_write_failure_logged = True
            _log.warning(
                "soft wind-down counter write to %s failed (%s); wind-down will not fire this run",
                counter_path,
                exc,
            )


def _wrap_tool_use_with_counter(
    on_tool_use: ToolUseCallback | None,
    counter_path: Path | None,
) -> ToolUseCallback | None:
    """Return a callback that atomically updates *counter_path* on each tool-use.

    Returns *on_tool_use* unchanged when no counter file is in play, so the
    streaming path only pays for the disk write when wind-down is active.
    """
    if counter_path is None:
        return on_tool_use

    def _wrapped(name: str, count: int) -> None:
        _atomic_write_counter(counter_path, count)
        _call_safely(on_tool_use, name, count)

    return _wrapped


def _count_tool_uses_post_hoc(
    *,
    adapter: CLIAdapter | None,
    stdout_lines: _BoundedOutput | None,
    stdout_capture: _OutputCapture | None = None,
    max_turns: int | None,
    on_tool_use: ToolUseCallback | None,
) -> tuple[int, bool]:
    """Scan complete file-backed stdout while retaining only a bounded tail."""
    if adapter is None or adapter.counts_what != "tool_use":
        return 0, False
    lines = (
        stdout_capture.iter_complete_lines()
        if stdout_capture is not None and stdout_capture.sink is not None
        else iter(stdout_lines or ())
    )
    count = 0
    for line in lines:
        event = adapter.parse_event(line)
        if event is None or event.kind != "tool_use":
            continue
        count += 1
        _call_safely(on_tool_use, event.name or "", count)
    turn_capped = max_turns is not None and count >= max_turns
    return count, turn_capped
