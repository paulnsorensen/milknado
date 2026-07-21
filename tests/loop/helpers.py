"""Shared constants and helpers for the vendored loop engine tests.

Import these via ``from tests.loop.helpers import MOCK_SUBPROCESS, ok_result``.
Fixtures stay in conftest.py — pytest discovers them automatically.
"""

from __future__ import annotations

import io
import itertools
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from milknado.loop._events import Event, EventType, QueueEmitter
from milknado.loop._frontmatter import RALPH_MARKER, serialize_frontmatter
from milknado.loop._run_types import RunConfig, RunState
from milknado.loop._runner import RunResult

# ── Patch targets ─────────────────────────────────────────────────────

MOCK_SUBPROCESS = "milknado.loop._agent.subprocess.Popen"
"""Patch target for subprocess.Popen inside the agent module."""

MOCK_RUNNER_SUBPROCESS = "milknado.loop._runner.subprocess.run"
"""Patch target for subprocess.run inside the runner module."""

MOCK_WHICH = "milknado.loop.cli.shutil.which"
"""Patch target for shutil.which inside the CLI module."""

MOCK_RUN_COMMAND = "milknado.loop.engine.run_command"
"""Patch target for run_command inside the engine module."""

MOCK_WAIT_FOR_STOP = "milknado.loop._run_types.RunState.wait_for_stop"
"""Patch target for RunState.wait_for_stop used by the engine delay logic."""


# ── Factory helpers ───────────────────────────────────────────────────


def make_ralph(
    tmp_path: Path,
    prompt: str = "go",
    agent: str = "claude -p --dangerously-skip-permissions",
    commands: list[dict[str, Any]] | None = None,
    args: list[str] | None = None,
) -> Path:
    """Create a ralph directory with a proper RALPH.md for CLI-level tests.

    Returns the ralph directory path.  Uses :func:`serialize_frontmatter`
    to build the file, so the YAML is always well-formed.
    """
    ralph_dir = tmp_path / "my-ralph"
    ralph_dir.mkdir(exist_ok=True)
    frontmatter: dict[str, Any] = {"agent": agent}
    if commands:
        frontmatter["commands"] = commands
    if args:
        frontmatter["args"] = args
    content = serialize_frontmatter(frontmatter, prompt)
    (ralph_dir / RALPH_MARKER).write_text(content)
    return ralph_dir


def make_config(tmp_path: Path, ralph_content: str = "test prompt", **overrides) -> RunConfig:
    """Create a RunConfig pointing at a temp ralph directory.

    *ralph_content* is written to the ``RALPH.md`` file every time, so
    tests can supply custom frontmatter + body without manually creating
    the ralph directory first.
    """
    ralph_dir = tmp_path / "my-ralph"
    ralph_dir.mkdir(exist_ok=True)
    ralph_file = ralph_dir / RALPH_MARKER
    ralph_file.write_text(ralph_content)

    defaults = dict(
        agent="omp",
        ralph_dir=ralph_dir,
        ralph_file=ralph_file,
        max_iterations=1,
        project_root=tmp_path,
    )
    defaults.update(overrides)
    return RunConfig(**defaults)


def make_state() -> RunState:
    """Create a RunState with a fixed test run ID."""
    return RunState(run_id="test-run-001")


def _make_completed_process(
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    """Build a CompletedProcess with the given values."""
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr
    )


def ok_result(
    *_args: Any,
    stdout: str = "",
    stderr: str = "",
    **_kwargs: Any,
) -> subprocess.CompletedProcess[str]:
    """Subprocess result with exit code 0.

    Works as a direct factory — ``ok_result(stdout="out\\n")`` — and as a
    ``side_effect`` callable where mock call args are silently absorbed.
    Used by runner tests that mock ``subprocess.run``.
    """
    return _make_completed_process(returncode=0, stdout=stdout, stderr=stderr)


def fail_result(
    *_args: Any,
    stdout: str = "",
    stderr: str = "",
    **_kwargs: Any,
) -> subprocess.CompletedProcess[str]:
    """Subprocess result with exit code 1.

    Works as a direct factory and as a ``side_effect`` callable (see
    :func:`ok_result`).
    """
    return _make_completed_process(returncode=1, stdout=stdout, stderr=stderr)


def _make_mock_proc(
    returncode: int = 0, stdout_text: str = "", stderr_text: str = ""
) -> MagicMock:
    """Build a MagicMock that mimics ``subprocess.Popen``.

    Used by both blocking and streaming agent test paths.  stdout/stderr
    are real ``io.StringIO`` objects so reader threads that call
    ``readline()`` in a loop hit EOF instead of spinning forever.
    """
    proc = MagicMock()
    proc.returncode = returncode
    proc.stdin = MagicMock()
    proc.stdout = io.StringIO(stdout_text)
    proc.stderr = io.StringIO(stderr_text)
    proc.wait.return_value = returncode
    proc.poll.return_value = returncode
    proc.pid = 0  # sentinel: skip real process-group manipulation in _kill_process_group
    return proc


def ok_proc(
    *_args: Any, stdout_text: str = "", stderr_text: str = "", **_kwargs: Any
) -> MagicMock:
    """Popen mock with exit code 0.  Works as a factory and ``side_effect``.

    ``_kwargs`` swallows Popen's ``stdout=``/``stderr=`` PIPE sentinels so
    the factory stays usable as ``side_effect`` for ``patch(Popen)``.
    """
    return _make_mock_proc(returncode=0, stdout_text=stdout_text, stderr_text=stderr_text)


def fail_proc(
    *_args: Any, stdout_text: str = "", stderr_text: str = "", **_kwargs: Any
) -> MagicMock:
    """Popen mock with exit code 1."""
    return _make_mock_proc(returncode=1, stdout_text=stdout_text, stderr_text=stderr_text)


def timeout_proc(
    *_args: Any,
    timeout: float = 5,
    stdout_text: str = "",
    stderr_text: str = "",
    **_kwargs: Any,
) -> MagicMock:
    """Popen mock whose wait() raises TimeoutExpired.

    The blocking path waits with ``proc.wait(timeout=...)`` and drains the
    output streams via reader threads, so we wire both here.  Uses
    ``itertools.chain`` so the exact number of ``poll()`` calls made by
    the blocking path is allowed to drift as the code is refactored
    without needing to re-count and update the side_effect list.
    """
    proc = _make_mock_proc(returncode=0, stdout_text=stdout_text, stderr_text=stderr_text)
    # First wait() raises TimeoutExpired; all subsequent calls return 0.
    # Uses itertools.chain so the exact number of wait() calls made by
    # cleanup paths is allowed to drift as the code is refactored.
    proc.wait.side_effect = itertools.chain(
        [subprocess.TimeoutExpired(cmd="agent", timeout=timeout)],
        itertools.repeat(0),
    )
    # First poll() (inside _kill_process_group during the except branch)
    # sees a live process; every later poll() sees it as already reaped,
    # so we don't try to kill/wait a second time in the cleanup paths.
    proc.poll.side_effect = itertools.chain([None], itertools.repeat(0))
    return proc


def ok_run_result(
    output: str = "",
) -> RunResult:
    """RunResult with exit code 0 (success).

    Mirrors :func:`ok_result` but for the runner module's return type.
    """
    return RunResult(returncode=0, output=output)


def make_mock_popen(
    stdout_lines: str = "",
    stderr_text: str = "",
    returncode: int = 0,
) -> MagicMock:
    """Create a MagicMock that mimics ``subprocess.Popen`` for the streaming path.

    Thin wrapper around :func:`_make_mock_proc` that accepts the
    ``stdout_lines`` parameter name used by streaming-path tests.
    """
    return _make_mock_proc(
        returncode=returncode, stdout_text=stdout_lines, stderr_text=stderr_text
    )


def drain_events(emitter: QueueEmitter) -> list[Event]:
    """Drain all events from a QueueEmitter and return them as a list."""
    events = []
    while not emitter.queue.empty():
        events.append(emitter.queue.get())
    return events


def events_of_type(events: list[Event], event_type: EventType) -> list[Event]:
    """Filter a list of events to only those matching *event_type*."""
    return [e for e in events if e.type == event_type]


def event_types(events: list[Event]) -> list[EventType]:
    """Extract the ordered list of event types from a list of events."""
    return [e.type for e in events]
