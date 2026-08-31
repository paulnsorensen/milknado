"""Execute shell commands with timeout and output capture.

Used by the engine to run configured commands.  The working directory
is set by the caller — typically the project root or the ralph directory
for ``./`` commands.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from milknado.loop._output import SUBPROCESS_TEXT_KWARGS, ProcessResult, collect_output


@dataclass(slots=True)
class RunResult(ProcessResult):
    """Result of running a command or script."""

    output: str = ""


def run_command(
    *,
    command: str,
    cwd: Path,
    timeout: float,
    env: dict[str, str] | None = None,
) -> RunResult:
    """Run a shell command and return the result.

    The *command* string is split with :func:`shlex.split` and executed
    directly (no shell).  Pipes, redirections, and ``&&`` chaining are
    not supported — use a script for complex logic.

    When *env* is set, the given variables are merged on top of the
    current process environment so scripts keep ``PATH`` etc.  When
    ``None`` (default), ``subprocess.run`` inherits the parent env.

    On timeout, returns ``returncode=None`` and ``timed_out=True``.
    Raises ``FileNotFoundError`` if the command binary does not exist.
    """
    cmd = shlex.split(command)
    if not cmd:
        raise ValueError(f"Command string produced no tokens after parsing: {command!r}")

    merged_env = {**os.environ, **env} if env else None

    try:
        result = cast(
            subprocess.CompletedProcess[str],
            subprocess.run(  # pyright: ignore[reportCallIssue]
                cmd,
                capture_output=True,
                **SUBPROCESS_TEXT_KWARGS,  # pyright: ignore[reportArgumentType]
                cwd=cwd,
                timeout=timeout,
                env=merged_env,
            ),
        )
        return RunResult(
            returncode=result.returncode,
            output=collect_output(result.stdout, result.stderr),
        )
    except subprocess.TimeoutExpired as exc:
        return RunResult(
            returncode=None,
            output=collect_output(exc.stdout, exc.stderr),
            timed_out=True,
        )
