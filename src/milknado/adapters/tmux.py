"""Tmux adapter — the opt-in run substrate: one session per project, one window per run.

Pure infrastructure (subprocess calls to the tmux binary), mirroring GitAdapter:
dispatch policy (fail-closed preflight, lifecycle reconciliation, attach
preconditions) lives in ``domains/dispatch/tmux_run.py``. Every target uses
tmux's ``=`` exact-match prefix — a bare name falls back to prefix/glob
matching (man tmux, TARGET SPECIFICATIONS), which could resolve to the wrong
window.
"""

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
from pathlib import Path

from milknado.domains.dispatch import RunWindow

_TMUX_TIMEOUT_SECS = 30


class TmuxDispatchError(RuntimeError):
    """A tmux operation needed to dispatch or manage a run window failed."""


def session_name_for(project_root: Path) -> str:
    """One tmux session per project: ``milknado-<sanitized root dirname>``.

    Sanitized because tmux rejects ``.`` and ``:`` in session names (they are
    target-syntax delimiters). The ``milknado-`` prefix keeps the session out
    of any same-named session the user already runs.
    """
    sanitized = re.sub(r"[^A-Za-z0-9_-]+", "-", project_root.name).strip("-")
    return f"milknado-{sanitized or 'project'}"


class TmuxAdapter:
    def __init__(self, project_root: Path, socket_path: Path | None = None) -> None:
        self._session = session_name_for(project_root)
        # A private socket isolates tests from the user's real tmux server;
        # production uses the default server (socket_path=None).
        self._socket = socket_path

    @property
    def session_name(self) -> str:
        return self._session

    def _run(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        cmd = ["tmux"]
        if self._socket is not None:
            cmd += ["-S", str(self._socket)]
        try:
            return subprocess.run(
                [*cmd, *args],
                capture_output=True,
                text=True,
                check=False,
                timeout=_TMUX_TIMEOUT_SECS,
            )
        except subprocess.TimeoutExpired as exc:
            # TimeoutExpired is not an OSError: without this translation a
            # wedged tmux server would escape the dispatchers' claim-release
            # guards and strand a node RUNNING with no terminal run write.
            raise TmuxDispatchError(
                f"tmux {args[0]} timed out after {_TMUX_TIMEOUT_SECS}s"
            ) from exc

    def available(self) -> bool:
        return shutil.which("tmux") is not None

    def ensure_session(self) -> None:
        """Create the project session if absent (``new-session -d`` auto-starts
        the server headlessly — no TTY required)."""
        if self._run(["has-session", "-t", f"={self._session}"]).returncode != 0:
            result = self._run(["new-session", "-d", "-s", self._session])
            if result.returncode != 0:
                raise TmuxDispatchError(
                    f"tmux could not start session {self._session!r}: {result.stderr.strip()}"
                )
        # Window commands run through the session's default-shell, and the run
        # wrapper is POSIX sh — a zsh default-shell breaks it (zsh's `=word`
        # expansion rejects the exact-match kill target, observed on tmux 3.7).
        # This session is milknado-dedicated, so pinning it to /bin/sh is safe.
        # (Target has a trailing colon: set-option rejects a bare `=name`
        # exact-match session target but accepts the session-qualified form.)
        result = self._run(["set-option", "-t", f"={self._session}:", "default-shell", "/bin/sh"])
        if result.returncode != 0:
            raise TmuxDispatchError(
                f"tmux could not configure session {self._session!r}: {result.stderr.strip()}"
            )

    def window_exists(self, run_id: str) -> bool:
        """Exact-name check for the run's window; no session/server means no window."""
        result = self._run(["list-windows", "-t", f"={self._session}", "-F", "#{window_name}"])
        if result.returncode != 0:
            return False
        return run_id in result.stdout.splitlines()

    def target_for(self, run_id: str) -> str:
        return f"={self._session}:={run_id}"

    def open_run_window(self, window: RunWindow) -> int:
        """Create the run's named window; returns the pane pid.

        The pane process is the run's process-group leader, so pid-liveness and
        group signalling keep working exactly as for a detached run. A window
        already carrying this run_id is a hard error, never reused.
        """
        self.ensure_session()
        if self.window_exists(window.run_id):
            raise TmuxDispatchError(
                f"tmux window {window.run_id!r} already exists in session {self._session!r}"
            )
        args = [
            "new-window",
            "-d",
            "-P",
            "-F",
            "#{pane_pid}",
            "-t",
            f"={self._session}:",
            "-n",
            window.run_id,
            "-c",
            str(window.cwd),
            self._wrapped_command(window),
        ]
        result = self._run(args)
        if result.returncode != 0:
            raise TmuxDispatchError(
                f"tmux could not open a window for run {window.run_id!r}: {result.stderr.strip()}"
            )
        try:
            pane_pid = int(result.stdout.strip())
        except ValueError as exc:
            raise TmuxDispatchError(
                f"tmux returned an unparseable pane pid: {result.stdout!r}"
            ) from exc
        if pane_pid <= 0:
            raise TmuxDispatchError(f"tmux returned an invalid pane pid: {result.stdout!r}")
        return pane_pid

    def kill_window(self, run_id: str) -> bool:
        """Kill the run's window; an absent window is a no-op returning False.

        The internal existence check is the caller's too — returning whether a
        window was killed keeps reconcile at one exact-match query per run row.
        """
        if not self.window_exists(run_id):
            return False
        result = self._run(["kill-window", "-t", self.target_for(run_id)])
        if result.returncode != 0:
            raise TmuxDispatchError(
                f"tmux could not kill window {run_id!r}: {result.stderr.strip()}"
            )
        return True

    def _wrapped_command(self, window: RunWindow) -> str:
        """POSIX-sh wrapper enforcing the window lifecycle contract.

        - ``remain-on-exit on`` is set from inside the pane, before the runner
          starts, so a failed run's window survives as an inspectable dead pane
          with no create/set race (in-pane ``set-option`` needs the explicit
          ``$TMUX_PANE`` target — without it the option lands on the session's
          current window, empirically verified on tmux 3.7).
        - ``env -i`` gives the runner exactly ``window.env`` — parity with the
          detached path's replacement ``env=`` — because a tmux pane otherwise
          inherits the *server's* environment (which may carry user secrets the
          worker-env allowlist exists to strip), and ``-e`` can only add vars.
        - ``tee`` keeps output live in the pane AND appended to the run log the
          poll tools tail; the runner's own exit code (not tee's) lands in the
          rc file.
        - A successful run kills its own window; a failed run leaves it behind.
        """
        assignments = (shlex.quote(f"{k}={v}") for k, v in window.env.items())
        runner = " ".join(["env", "-i", *assignments, shlex.join(window.argv)])
        if window.brief_path is not None:
            runner += f" < {shlex.quote(str(window.brief_path))}"
        rc = shlex.quote(str(window.exit_code_path))
        log = shlex.quote(str(window.log_path))
        # Explicitly single-quoted: shlex.quote leaves `=…:=…` bare (POSIX-safe),
        # but an interactive shell reading the scrollback — or any non-sh shell
        # that ever sees this line — must not expand the leading `=`.
        kill_target = f"'{self.target_for(window.run_id)}'"
        return (
            'tmux set-option -w -t "$TMUX_PANE" remain-on-exit on; '
            f"{{ {runner} 2>&1; echo $? > {rc}; }} | tee -a {log}; "
            f'rc=$(cat {rc} 2>/dev/null || echo 1); [ -n "$rc" ] || rc=1; '
            f'if [ "$rc" -eq 0 ]; then tmux kill-window -t {kill_target}; fi; '
            'exit "$rc"'
        )
