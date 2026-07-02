"""TmuxAdapter: naming, wrapper contract, error branches (stubbed tmux), and
one integration test driving a real headless tmux server on a private socket."""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

import pytest

from milknado.adapters.tmux import (
    RunWindow,
    TmuxAdapter,
    TmuxDispatchError,
    session_name_for,
)

RUN_ID = "node-1-20260101T000000Z-deadbeef"


def _window(tmp_path: Path, run_id: str = RUN_ID, **overrides) -> RunWindow:
    defaults = dict(
        run_id=run_id,
        argv=("sh", "-c", "true"),
        cwd=tmp_path,
        log_path=tmp_path / f"{run_id}.log",
        exit_code_path=tmp_path / f"{run_id}.rc",
    )
    defaults.update(overrides)
    return RunWindow(**defaults)


# --- naming ------------------------------------------------------------------


def test_session_name_sanitizes_target_syntax_delimiters() -> None:
    assert session_name_for(Path("/home/u/my.proj: x")) == "milknado-my-proj-x"


def test_session_name_falls_back_when_dirname_sanitizes_away() -> None:
    assert session_name_for(Path("/")) == "milknado-project"


# --- wrapper contract ---------------------------------------------------------


def test_wrapped_command_encodes_the_window_lifecycle(tmp_path: Path) -> None:
    adapter = TmuxAdapter(tmp_path)
    cmd = adapter._wrapped_command(_window(tmp_path))
    # remain-on-exit set from inside the pane, before the runner, targeting the
    # pane explicitly (without -t the option lands on the wrong window).
    assert 'tmux set-option -w -t "$TMUX_PANE" remain-on-exit on' in cmd
    # Live output in the pane AND appended to the poll-tailed log.
    assert f"| tee -a {tmp_path / f'{RUN_ID}.log'}" in cmd
    # Runner's own exit code (not tee's) recorded for the pane waiter.
    assert f"echo $? > {tmp_path / f'{RUN_ID}.rc'}" in cmd
    # Success self-cleans via the exact-match target, single-quoted so no
    # shell ever `=word`-expands it; failure preserves the window.
    assert f"then tmux kill-window -t '={adapter.session_name}:={RUN_ID}'; fi" in cmd
    assert "< " not in cmd  # no brief redirect unless staged


def test_wrapped_command_redirects_staged_brief_to_stdin(tmp_path: Path) -> None:
    adapter = TmuxAdapter(tmp_path)
    brief = tmp_path / f"{RUN_ID}.brief"
    cmd = adapter._wrapped_command(_window(tmp_path, brief_path=brief))
    assert f"< {brief} 2>&1" in cmd


# --- availability and error branches (stubbed tmux binary) --------------------


def test_available_reflects_binary_presence(tmp_path: Path, monkeypatch) -> None:
    adapter = TmuxAdapter(tmp_path)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert adapter.available() is False
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/tmux")
    assert adapter.available() is True


def _stub_run(responses: dict[str, subprocess.CompletedProcess]):
    """Map a tmux subcommand name to a canned CompletedProcess."""

    def _run(self, args: list[str]) -> subprocess.CompletedProcess:
        return responses[args[0]]

    return _run


def _completed(rc: int, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess([], rc, stdout, stderr)


def test_ensure_session_raises_when_server_cannot_start(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        TmuxAdapter,
        "_run",
        _stub_run({"has-session": _completed(1), "new-session": _completed(1, stderr="boom")}),
    )
    with pytest.raises(TmuxDispatchError, match="could not start session .*boom"):
        TmuxAdapter(tmp_path).ensure_session()


def test_ensure_session_raises_when_default_shell_pin_fails(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        TmuxAdapter,
        "_run",
        _stub_run({"has-session": _completed(0), "set-option": _completed(1, stderr="nope")}),
    )
    with pytest.raises(TmuxDispatchError, match="could not configure session .*nope"):
        TmuxAdapter(tmp_path).ensure_session()


def test_window_exists_is_exact_not_prefix(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        TmuxAdapter,
        "_run",
        _stub_run({"list-windows": _completed(0, stdout=f"{RUN_ID}-longer\nzsh\n")}),
    )
    adapter = TmuxAdapter(tmp_path)
    assert adapter.window_exists(RUN_ID) is False
    assert adapter.window_exists(f"{RUN_ID}-longer") is True


def test_window_exists_false_when_session_absent(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        TmuxAdapter, "_run", _stub_run({"list-windows": _completed(1, stderr="no server")})
    )
    assert TmuxAdapter(tmp_path).window_exists(RUN_ID) is False


def test_open_run_window_collision_is_a_hard_error(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        TmuxAdapter,
        "_run",
        _stub_run(
            {
                "has-session": _completed(0),
                "set-option": _completed(0),
                "list-windows": _completed(0, f"{RUN_ID}\n"),
            }
        ),
    )
    with pytest.raises(TmuxDispatchError, match="already exists"):
        TmuxAdapter(tmp_path).open_run_window(_window(tmp_path))


def test_open_run_window_surfaces_tmux_failure(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        TmuxAdapter,
        "_run",
        _stub_run(
            {
                "has-session": _completed(0),
                "set-option": _completed(0),
                "list-windows": _completed(0, ""),
                "new-window": _completed(1, stderr="create failed"),
            }
        ),
    )
    with pytest.raises(TmuxDispatchError, match="could not open a window .*create failed"):
        TmuxAdapter(tmp_path).open_run_window(_window(tmp_path))


def test_open_run_window_rejects_unparseable_pane_pid(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        TmuxAdapter,
        "_run",
        _stub_run(
            {
                "has-session": _completed(0),
                "set-option": _completed(0),
                "list-windows": _completed(0, ""),
                "new-window": _completed(0, stdout="not-a-pid\n"),
            }
        ),
    )
    with pytest.raises(TmuxDispatchError, match="unparseable pane pid"):
        TmuxAdapter(tmp_path).open_run_window(_window(tmp_path))


def test_kill_window_surfaces_tmux_failure(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        TmuxAdapter,
        "_run",
        _stub_run(
            {
                "list-windows": _completed(0, f"{RUN_ID}\n"),
                "kill-window": _completed(1, stderr="denied"),
            }
        ),
    )
    with pytest.raises(TmuxDispatchError, match="could not kill window .*denied"):
        TmuxAdapter(tmp_path).kill_window(RUN_ID)


# --- the one real-tmux integration test ---------------------------------------


def _wait_until(condition, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.1)
    raise AssertionError("condition not met within timeout")


@pytest.mark.skipif(shutil.which("tmux") is None, reason="tmux binary not available")
def test_real_tmux_window_lifecycle(tmp_path: Path) -> None:
    """Drives a real headless tmux server (private socket, no TTY) through the
    full window contract: env injection, log tee, brief redirect, success
    self-clean, failure preservation, collision, and exact-match kill."""
    socket = tmp_path / "tmux-test.sock"
    adapter = TmuxAdapter(tmp_path, socket_path=socket)
    rdir = tmp_path / "runs"
    rdir.mkdir()
    try:
        adapter.ensure_session()
        adapter.ensure_session()  # idempotent — reuses, never duplicates

        # Success: output (with injected env) lands in the log, exit code in
        # the rc file, and the window cleans itself up.
        ok = "node-1-20260101T000000Z-0000aaaa"
        pane_pid = adapter.open_run_window(
            RunWindow(
                run_id=ok,
                argv=("sh", "-c", 'echo "out $MILKNADO_RUN_ID"'),
                cwd=tmp_path,
                log_path=rdir / f"{ok}.log",
                exit_code_path=rdir / f"{ok}.rc",
                env={"MILKNADO_RUN_ID": ok},
            )
        )
        assert pane_pid > 0
        _wait_until(lambda: not adapter.window_exists(ok))
        assert f"out {ok}" in (rdir / f"{ok}.log").read_text()
        assert (rdir / f"{ok}.rc").read_text().strip() == "0"

        # Brief redirect: worker reads the staged brief on stdin.
        br = "node-2-20260101T000000Z-0000bbbb"
        brief = rdir / f"{br}.brief"
        brief.write_text("brief-payload")
        adapter.open_run_window(
            RunWindow(
                run_id=br,
                argv=("cat",),
                cwd=tmp_path,
                log_path=rdir / f"{br}.log",
                exit_code_path=rdir / f"{br}.rc",
                brief_path=brief,
            )
        )
        _wait_until(lambda: not adapter.window_exists(br))
        assert "brief-payload" in (rdir / f"{br}.log").read_text()

        # Failure: the window survives as an inspectable dead pane.
        bad = "node-3-20260101T000000Z-0000cccc"
        adapter.open_run_window(
            RunWindow(
                run_id=bad,
                argv=("sh", "-c", "echo boom; exit 7"),
                cwd=tmp_path,
                log_path=rdir / f"{bad}.log",
                exit_code_path=rdir / f"{bad}.rc",
            )
        )
        _wait_until(lambda: (rdir / f"{bad}.rc").exists())
        time.sleep(0.5)  # give the wrapper time to reach its exit
        assert adapter.window_exists(bad)
        assert (rdir / f"{bad}.rc").read_text().strip() == "7"
        assert "boom" in (rdir / f"{bad}.log").read_text()

        # Collision on a still-open window name is a hard error, never reuse.
        with pytest.raises(TmuxDispatchError, match="already exists"):
            adapter.open_run_window(
                RunWindow(
                    run_id=bad,
                    argv=("sh", "-c", "true"),
                    cwd=tmp_path,
                    log_path=rdir / f"{bad}.log",
                    exit_code_path=rdir / f"{bad}.rc",
                )
            )

        # Exact-match kill removes it; killing an absent window is a no-op.
        adapter.kill_window(bad)
        assert not adapter.window_exists(bad)
        adapter.kill_window(bad)

        # Killing a LIVE window kills the run's process group — a kill-window
        # never orphans the run's process handling (pid-liveness sees the death).
        from milknado.domains.common import pid_alive

        live = "node-4-20260101T000000Z-0000dddd"
        live_pid = adapter.open_run_window(
            RunWindow(
                run_id=live,
                argv=("sh", "-c", "sleep 30"),
                cwd=tmp_path,
                log_path=rdir / f"{live}.log",
                exit_code_path=rdir / f"{live}.rc",
            )
        )
        _wait_until(lambda: pid_alive(live_pid))
        adapter.kill_window(live)
        _wait_until(lambda: not pid_alive(live_pid))
    finally:
        subprocess.run(
            ["tmux", "-S", str(socket), "kill-server"], capture_output=True, check=False
        )
