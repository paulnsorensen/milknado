from __future__ import annotations

import os
import queue
import signal
import subprocess
import threading
import time
import uuid
from collections import deque
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import IO
from weakref import WeakKeyDictionary

from milknado.loop._agent import (
    AgentRunSpec,
    _atomic_write_counter,  # pyright: ignore[reportPrivateUsage]
    _setup_wind_down,  # pyright: ignore[reportPrivateUsage]
    _WindDownContext,  # pyright: ignore[reportPrivateUsage]
)
from milknado.loop.adapters import select_adapter
from milknado.loop.sessions._protocol import SessionProtocol

CAPTURE_LIMIT = 64 * 1024
READER_QUEUE_LIMIT = 256
POLL_INTERVAL = 0.05
TERMINATE_GRACE = 0.5
READ_CHUNK_SIZE = 4096
MAX_FRAME_SIZE = 1024 * 1024
MAX_STDERR_LINE_SIZE = CAPTURE_LIMIT

_PROCESS_GROUP_IDS: WeakKeyDictionary[subprocess.Popen[bytes], int] = WeakKeyDictionary()


class BoundedTail:
    def __init__(self) -> None:
        self._lines: deque[str] = deque()
        self._chars: int = 0

    def append(self, line: str) -> None:
        if len(line) > CAPTURE_LIMIT:
            line = line[-CAPTURE_LIMIT:]
        self._lines.append(line)
        self._chars += len(line)
        while self._chars > CAPTURE_LIMIT and len(self._lines) > 1:
            self._chars -= len(self._lines.popleft())

    @property
    def text(self) -> str:
        return "".join(self._lines)


@dataclass(frozen=True, slots=True)
class Line:
    stream: str
    text: str | None


def _put_line(lines: queue.Queue[Line], item: Line, stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            lines.put(item, timeout=POLL_INTERVAL)
            return
        except queue.Full:
            continue


def _read_frame(pipe: IO[bytes], limit: int) -> str | None:
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = pipe.readline(min(READ_CHUNK_SIZE, limit - size + 1))
        if chunk == b"":
            if not chunks:
                return None
            return b"".join(chunks).decode("utf-8", errors="replace")
        size += len(chunk)
        if size > limit:
            raise ValueError(f"frame exceeds the {limit}-byte limit")
        chunks.append(chunk)
        if chunk.endswith(b"\n"):
            return b"".join(chunks).decode("utf-8", errors="replace")


def reader(
    stream: str,
    pipe: IO[bytes],
    lines: queue.Queue[Line],
    stop: threading.Event,
) -> None:
    limit = MAX_FRAME_SIZE if stream == "stdout" else MAX_STDERR_LINE_SIZE
    try:
        while (text := _read_frame(pipe, limit)) is not None:
            _put_line(lines, Line(stream, text), stop)
    except (OSError, ValueError) as exc:
        if not stop.is_set():
            _put_line(lines, Line("reader_error", f"{stream}: {exc}"), stop)
    finally:
        _put_line(lines, Line(stream, None), stop)


def start_readers(
    proc: subprocess.Popen[bytes],
    lines: queue.Queue[Line],
    stop: threading.Event,
    iteration: int,
) -> list[threading.Thread]:
    threads: list[threading.Thread] = []
    for stream, pipe in (("stdout", proc.stdout), ("stderr", proc.stderr)):
        if pipe is None:
            continue
        thread = threading.Thread(
            target=reader,
            args=(stream, pipe, lines, stop),
            daemon=True,
            name=f"session-{stream}-{iteration}",
        )
        thread.start()
        threads.append(thread)
    return threads


def write_commands(proc: subprocess.Popen[bytes], commands: tuple[bytes, ...]) -> None:
    stdin = proc.stdin
    if stdin is None and commands:
        raise BrokenPipeError("session worker has no stdin")
    if stdin is None:
        return
    for command in commands:
        payload = command if command.endswith(b"\n") else command + b"\n"
        _ = stdin.write(payload)
        _ = stdin.flush()


def _group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _wait_for_group_exit(pgid: int) -> bool:
    deadline = time.monotonic() + TERMINATE_GRACE
    while time.monotonic() < deadline:
        if not _group_exists(pgid):
            return True
        time.sleep(POLL_INTERVAL)
    return not _group_exists(pgid)


def _process_group_id(proc: subprocess.Popen[bytes]) -> int | None:
    return _PROCESS_GROUP_IDS.get(proc)


def _terminate_process_group(pgid: int) -> bool:
    with suppress(OSError, ProcessLookupError):
        os.killpg(pgid, signal.SIGTERM)
    if _wait_for_group_exit(pgid):
        return True
    with suppress(OSError, ProcessLookupError):
        os.killpg(pgid, signal.SIGKILL)
    return _wait_for_group_exit(pgid)


def terminate(proc: subprocess.Popen[bytes]) -> None:
    """Stop the retained worker process group, including surviving descendants."""
    pgid = _process_group_id(proc)
    group_pgid = pgid if os.name != "nt" and pgid == proc.pid else None
    group_terminated = False
    if group_pgid is not None:
        group_terminated = _terminate_process_group(group_pgid)
        _ = _PROCESS_GROUP_IDS.pop(proc, None)
    elif proc.poll() is None:
        proc.terminate()
    if proc.poll() is not None:
        return
    try:
        _ = proc.wait(timeout=TERMINATE_GRACE)
    except subprocess.TimeoutExpired:
        if group_pgid is not None and not group_terminated:
            with suppress(OSError, ProcessLookupError):
                os.killpg(group_pgid, signal.SIGKILL)
        else:
            proc.kill()
        with suppress(subprocess.TimeoutExpired):
            _ = proc.wait(timeout=TERMINATE_GRACE)


def finish_process(proc: subprocess.Popen[bytes], *, graceful: bool) -> None:
    if graceful:
        if proc.stdin is not None:
            with suppress(OSError, ValueError):
                _ = proc.stdin.close()
        with suppress(subprocess.TimeoutExpired):
            _ = proc.wait(timeout=TERMINATE_GRACE)
    terminate(proc)


def close_pipes(proc: subprocess.Popen[bytes], threads: tuple[threading.Thread, ...]) -> None:
    for thread in threads:
        thread.join(timeout=1.0)
    for pipe in (proc.stdin, proc.stdout, proc.stderr):
        if pipe is not None:
            with suppress(OSError, ValueError):
                pipe.close()


def cleanup_process(
    proc: subprocess.Popen[bytes],
    stop: threading.Event,
    threads: tuple[threading.Thread, ...],
) -> None:
    stop.set()
    terminate(proc)
    close_pipes(proc, threads)


def prepare_wind_down(spec: AgentRunSpec) -> _WindDownContext | None:
    adapter = spec.adapter if spec.adapter is not None else select_adapter(spec.cmd)
    return _setup_wind_down(
        adapter=adapter,
        max_turns=spec.max_turns,
        max_turns_grace=spec.max_turns_grace,
        log_dir=spec.log_dir,
        iteration=spec.iteration,
    )


def record_tool_count(wind_down: _WindDownContext | None, count: int) -> None:
    if wind_down is not None:
        _atomic_write_counter(wind_down.counter_path, count)


def log_path(log_dir: Path, iteration: int) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{iteration:04d}_{uuid.uuid4().hex}.log"


def start_process(
    protocol: SessionProtocol,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> subprocess.Popen[bytes]:
    spawn_env = {**os.environ, **env} if env else None
    if os.name == "nt":
        proc = subprocess.Popen(
            protocol.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=spawn_env,
        )
    else:
        proc = subprocess.Popen(
            protocol.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=spawn_env,
            start_new_session=True,
        )
        _PROCESS_GROUP_IDS[proc] = proc.pid
    return proc
