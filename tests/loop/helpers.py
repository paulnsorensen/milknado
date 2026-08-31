"""Shared constants and helpers for the vendored loop engine tests.

Import these via ``from tests.loop.helpers import MOCK_SUBPROCESS, ok_result``.
Fixtures stay in conftest.py — pytest discovers them automatically.
"""

from __future__ import annotations

import io
import itertools
import subprocess
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

from milknado.loop._events import (  # pyright: ignore[reportMissingTypeStubs]
    Event,
    EventType,
    QueueEmitter,
)
from milknado.loop._frontmatter import (  # pyright: ignore[reportMissingTypeStubs]
    RALPH_MARKER,
    serialize_frontmatter,
)
from milknado.loop._run_types import (  # pyright: ignore[reportMissingTypeStubs]
    DEFAULT_COMPLETION_SIGNAL,
    Command,
    CompletionVerdict,
    RunConfig,
    RunState,
)
from milknado.loop._runner import RunResult  # pyright: ignore[reportMissingTypeStubs]
from milknado.loop.hooks import AgentHook  # pyright: ignore[reportMissingTypeStubs]

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
    commands: list[dict[str, object]] | None = None,
    args: list[str] | None = None,
) -> Path:
    """Create a ralph directory with a proper RALPH.md for CLI-level tests.

    Returns the ralph directory path.  Uses :func:`serialize_frontmatter`
    to build the file, so the YAML is always well-formed.
    """
    ralph_dir = tmp_path / "my-ralph"
    ralph_dir.mkdir(exist_ok=True)
    frontmatter: dict[str, object] = {"agent": agent}
    if commands:
        frontmatter["commands"] = commands
    if args:
        frontmatter["args"] = args
    content = serialize_frontmatter(frontmatter, prompt)
    _ = (ralph_dir / RALPH_MARKER).write_text(content)
    return ralph_dir


def make_config(
    tmp_path: Path,
    ralph_content: str = "test prompt",
    *,
    agent: str = "omp",
    commands: list[Command] | None = None,
    args: dict[str, str] | None = None,
    max_iterations: int | None = 1,
    delay: float = 0,
    timeout: float | None = None,
    stop_on_error: bool = False,
    max_consecutive_failures: int | None = None,
    log_dir: Path | None = None,
    commit_footer: str | None = None,
    completion_signal: str = DEFAULT_COMPLETION_SIGNAL,
    stop_on_completion_signal: bool = False,
    max_turns: int | None = None,
    hooks: list[AgentHook] | None = None,
    completion_verifier: Callable[[], CompletionVerdict] | None = None,
    completion_probe: Callable[[], bool] | None = None,
) -> RunConfig:
    """Create a RunConfig pointing at a temp ralph directory.

    *ralph_content* is written to the ``RALPH.md`` file every time, so
    tests can supply custom frontmatter + body without manually creating
    the ralph directory first.
    """
    ralph_dir = tmp_path / "my-ralph"
    ralph_dir.mkdir(exist_ok=True)
    ralph_file = ralph_dir / RALPH_MARKER
    _ = ralph_file.write_text(ralph_content)

    return RunConfig(
        agent=agent,
        ralph_dir=ralph_dir,
        ralph_file=ralph_file,
        commands=commands or [],
        args=args or {},
        max_iterations=max_iterations,
        delay=delay,
        timeout=timeout,
        stop_on_error=stop_on_error,
        max_consecutive_failures=max_consecutive_failures,
        log_dir=log_dir,
        project_root=tmp_path,
        commit_footer=commit_footer,
        completion_signal=completion_signal,
        stop_on_completion_signal=stop_on_completion_signal,
        max_turns=max_turns,
        hooks=hooks or [],
        completion_verifier=completion_verifier,
        completion_probe=completion_probe,
    )


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
    *_args: object,
    stdout: str = "",
    stderr: str = "",
    **_kwargs: object,
) -> subprocess.CompletedProcess[str]:
    """Subprocess result with exit code 0.

    Works as a direct factory — ``ok_result(stdout="out\\n")`` — and as a
    ``side_effect`` callable where mock call args are silently absorbed.
    Used by runner tests that mock ``subprocess.run``.
    """
    return _make_completed_process(returncode=0, stdout=stdout, stderr=stderr)


def fail_result(
    *_args: object,
    stdout: str = "",
    stderr: str = "",
    **_kwargs: object,
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
    proc.wait.return_value = returncode  # pyright: ignore[reportAny]
    proc.poll.return_value = returncode  # pyright: ignore[reportAny]
    proc.pid = 0  # sentinel: skip real process-group manipulation in _kill_process_group
    return proc


def ok_proc(
    *_args: object, stdout_text: str = "", stderr_text: str = "", **_kwargs: object
) -> MagicMock:
    """Popen mock with exit code 0.  Works as a factory and ``side_effect``.

    ``_kwargs`` swallows Popen's ``stdout=``/``stderr=`` PIPE sentinels so
    the factory stays usable as ``side_effect`` for ``patch(Popen)``.
    """
    return _make_mock_proc(returncode=0, stdout_text=stdout_text, stderr_text=stderr_text)


def fail_proc(
    *_args: object, stdout_text: str = "", stderr_text: str = "", **_kwargs: object
) -> MagicMock:
    """Popen mock with exit code 1."""
    return _make_mock_proc(returncode=1, stdout_text=stdout_text, stderr_text=stderr_text)


def timeout_proc(
    *_args: object,
    timeout: float = 5,
    stdout_text: str = "",
    stderr_text: str = "",
    **_kwargs: object,
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
    proc.wait.side_effect = itertools.chain(  # pyright: ignore[reportAny]
        [subprocess.TimeoutExpired(cmd="agent", timeout=timeout)],
        itertools.repeat(0),
    )
    # First poll() (inside _kill_process_group during the except branch)
    # sees a live process; every later poll() sees it as already reaped,
    # so we don't try to kill/wait a second time in the cleanup paths.
    proc.poll.side_effect = itertools.chain([None], itertools.repeat(0))  # pyright: ignore[reportAny]
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


def drain_events(emitter: QueueEmitter) -> list[Event]:  # pyright: ignore[reportMissingTypeArgument, reportUnknownParameterType]
    """Drain all events from a QueueEmitter and return them as a list."""
    events = []
    while not emitter.queue.empty():
        events.append(emitter.queue.get())  # pyright: ignore[reportUnknownMemberType]
    return events  # pyright: ignore[reportUnknownVariableType]


def events_of_type(events: list[Event], event_type: EventType) -> list[Event]:  # pyright: ignore[reportMissingTypeArgument, reportUnknownParameterType]
    """Filter a list of events to only those matching *event_type*."""
    return [e for e in events if e.type == event_type]  # pyright: ignore[reportUnknownVariableType]


def event_types(events: list[Event]) -> list[EventType]:  # pyright: ignore[reportMissingTypeArgument, reportUnknownParameterType]
    """Extract the ordered list of event types from a list of events."""
    return [e.type for e in events]  # pyright: ignore[reportUnknownVariableType]
