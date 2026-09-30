from __future__ import annotations

import os
import shlex
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from threading import Thread
from typing import cast

import psutil
import pytest

from milknado.app._shutdown import STOP_TIMEOUT_SECONDS
from milknado.domains.common import FlavorOverride, Gate, MilknadoConfig, save_config
from tests.execution_session_fixtures import build_graph, init_repo
from tests.worker_fixtures import install_worker_command

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals and PTY required")
_SIGHUP = getattr(signal, "SIGHUP", signal.SIGTERM)

_PTY_OUTPUT: dict[int, bytearray] = {}


def _drain_pty(fd: int) -> None:
    output = _PTY_OUTPUT[fd]
    while True:
        try:
            chunk = os.read(fd, 65536)
        except OSError:
            return
        if not chunk:
            return
        output.extend(chunk)
        if len(output) > 65536:
            del output[:-65536]


_WORKER = """
import json
import os
import time
from pathlib import Path
import psutil

identity = f"{os.getpid()}:{psutil.Process().create_time()}"
marker = Path(os.environ["SHUTDOWN_WORKER_PID"])
temporary = marker.with_name(f"{marker.name}.{os.getpid()}.tmp")
temporary.write_text(identity, encoding="utf-8")
temporary.replace(marker)
for line in __import__("sys").stdin:
    if json.loads(line).get("type") == "user":
        print(json.dumps({"type": "system", "subtype": "init",
                          "session_id": "shutdown-fixture", "model": "fixture"}), flush=True)
        break
while True:
    time.sleep(0.05)
"""


def _wait_for(path: Path, timeout: float = 12.0) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert path.exists(), f"missing fixture marker: {path}"


def _project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    repo = init_repo(tmp_path)
    graph = build_graph(repo)
    db = graph.db_path
    graph.close()
    worker_source = tmp_path / "worker.py"
    _ = worker_source.write_text(_WORKER, encoding="utf-8")
    command = install_worker_command(
        tmp_path / "bin",
        monkeypatch,
        agent="claude",
        script=f'exec {shlex.quote(sys.executable)} {shlex.quote(str(worker_source))} "$@"\n',
    )
    save_config(
        MilknadoConfig(
            execution_agent=command,
            flavors={"implement": FlavorOverride(review=False)},
            quality_gates=(Gate(command="true"),),
            worktree_pattern="milknado-wt-{node_id}-{slug}",
            concurrency_limit=1,
            project_root=repo,
            db_path=db,
        ),
        repo / "milknado.toml",
    )
    return repo, db, tmp_path / "worker.pid"


def _start(
    repo: Path, pid_file: Path, *, interactive: bool, injection: Path | None = None
) -> tuple[subprocess.Popen[bytes], int | None]:
    env = dict(os.environ)
    env["SHUTDOWN_WORKER_PID"] = str(pid_file)
    env["PYTHONPATH"] = (
        str(Path(__file__).resolve().parents[1] / "src") + os.pathsep + env.get("PYTHONPATH", "")
    )
    if injection is not None:
        env["PYTHONPATH"] = str(injection) + os.pathsep + env["PYTHONPATH"]
    argv = [
        sys.executable,
        "-c",
        "from milknado.cli import app; app()",
        "run",
        "--project-root",
        str(repo),
    ]
    if interactive:
        master, slave = os.openpty()
        try:
            proc = subprocess.Popen(
                argv,
                cwd=repo,
                env=env,
                stdin=slave,
                stdout=slave,
                stderr=slave,
                start_new_session=True,
                close_fds=True,
            )
        finally:
            os.close(slave)
        _PTY_OUTPUT[master] = bytearray()
        Thread(target=_drain_pty, args=(master,), daemon=True).start()
        return proc, master
    return subprocess.Popen(
        argv,
        cwd=repo,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    ), None


def _output(proc: subprocess.Popen[bytes], master: int | None) -> str:
    if master is None:
        assert proc.stdout is not None
        return cast(bytes, proc.stdout.read()).decode(errors="replace")
    return bytes(_PTY_OUTPUT[master]).decode(errors="replace")


def _wait_exit(proc: subprocess.Popen[bytes], master: int | None) -> int:
    try:
        return proc.wait(timeout=8.5)
    except subprocess.TimeoutExpired:
        output = _output(proc, master) if master is not None else "<headless output pending>"
        pytest.fail(f"CLI exceeded shutdown deadline; output={output[-2000:]!r}")


def _worker_state(db: Path, pid: int) -> tuple[bool, bool]:
    with sqlite3.connect(db) as conn:
        rows = cast(
            list[tuple[str | None]],
            conn.execute("SELECT ended_at FROM run_workers WHERE pid = ?", (pid,)).fetchall(),
        )
        open_record = any(ended is None for (ended,) in rows)
        node_running = (
            conn.execute("SELECT 1 FROM nodes WHERE status = 'running'").fetchone() is not None
        )
    return open_record, node_running


def _worker_identity(path: Path) -> tuple[int, float]:
    pid, token = path.read_text().split(":", 1)
    return int(pid), float(token)


def _owned_cleanup(proc: subprocess.Popen[bytes], pid_file: Path, master: int | None) -> None:
    if proc.poll() is None:
        proc.kill()
    try:
        _ = proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        pass
    if pid_file.exists():
        pid, token = _worker_identity(pid_file)
        child = psutil.Process(pid) if psutil.pid_exists(pid) else None
        if (
            child is not None
            and child.create_time() == token
            and child.status() != psutil.STATUS_ZOMBIE
        ):
            child.kill()
            _ = child.wait(timeout=3)
    if master is not None:
        os.close(master)
        _ = _PTY_OUTPUT.pop(master, None)
    if proc.stdout is not None:
        proc.stdout.close()


@pytest.mark.parametrize("interactive", [False, True], ids=["headless", "controller-pty"])
@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM, _SIGHUP])
def test_cli_signal_stops_real_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interactive: bool, signum: int
) -> None:
    repo, db, pid_file = _project(tmp_path, monkeypatch)
    proc, master = _start(repo, pid_file, interactive=interactive)
    try:
        _wait_for(pid_file)
        pid, _ = _worker_identity(pid_file)
        started = time.monotonic()
        os.kill(proc.pid, signum)
        assert _wait_exit(proc, master) == 128 + signum, _output(proc, master)
        assert time.monotonic() - started < 8.25
        alive = psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
        open_record, node_running = _worker_state(db, pid)
        assert not alive or (open_record and node_running), (
            f"worker={pid} alive={alive} open={open_record} running={node_running}"
        )
    finally:
        _owned_cleanup(proc, pid_file, master)


@pytest.mark.parametrize("interactive", [False, True], ids=["headless", "controller-pty"])
def test_repeated_interrupt_keeps_first_deadline_and_worker_ownership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interactive: bool
) -> None:
    repo, db, pid_file = _project(tmp_path, monkeypatch)
    proc, master = _start(repo, pid_file, interactive=interactive)
    try:
        _wait_for(pid_file)
        pid, _ = _worker_identity(pid_file)
        started = time.monotonic()
        os.kill(proc.pid, signal.SIGINT)
        time.sleep(0.1)
        if proc.poll() is None:
            os.kill(proc.pid, signal.SIGINT)
        assert _wait_exit(proc, master) in (128 + signal.SIGINT, -signal.SIGINT), _output(
            proc, master
        )
        assert time.monotonic() - started < 8.25
        alive = psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
        open_record, node_running = _worker_state(db, pid)
        assert not alive or (open_record and node_running)
    finally:
        _owned_cleanup(proc, pid_file, master)


_SPAWN_BARRIER = """
import os
import subprocess
import time
from pathlib import Path

from milknado.app import _shutdown

_first_signal_at = None
_original_record = _shutdown.ShutdownIntent.record
_original_bounded_stop = _shutdown.bounded_stop


def _record(self, signum, frame):
    global _first_signal_at
    _original_record(self, signum, frame)
    if _first_signal_at is None:
        _first_signal_at = self.started_at


def _timed_bounded_stop(stop, deadline):
    result = _original_bounded_stop(stop, deadline)
    completed_at = time.monotonic()
    timing = Path(os.environ["SHUTDOWN_CLEANUP_TIMING"])
    if _first_signal_at is not None and not timing.exists():
        timing.write_text(f"{_first_signal_at!r} {deadline!r} {completed_at!r}")
    return result


_shutdown.ShutdownIntent.record = _record
_shutdown.bounded_stop = _timed_bounded_stop
_original_popen = subprocess.Popen

class _BlockedPopen(_original_popen):
    def __init__(self, argv, *args, **kwargs):
        if isinstance(argv, (tuple, list)) and "milknado.loop._exec_gate" in argv:
            Path(os.environ["SHUTDOWN_SPAWN_MARKER"]).touch()
            deadline = time.monotonic() + 20
            while (
                not Path(os.environ["SHUTDOWN_SPAWN_RELEASE"]).exists()
                and time.monotonic() < deadline
            ):
                time.sleep(0.02)
        super().__init__(argv, *args, **kwargs)

subprocess.Popen = _BlockedPopen
"""


@pytest.mark.parametrize("interactive", [False, True], ids=["headless", "controller-pty"])
@pytest.mark.parametrize("signum", [signal.SIGTERM, _SIGHUP])
def test_cli_exits_while_worker_popen_is_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interactive: bool, signum: int
) -> None:
    repo, db, pid_file = _project(tmp_path, monkeypatch)
    marker = tmp_path / "popen-entered"
    release = tmp_path / "popen-release"
    timing = tmp_path / "cleanup-timing"
    _ = (tmp_path / "sitecustomize.py").write_text(_SPAWN_BARRIER, encoding="utf-8")
    monkeypatch.setenv("SHUTDOWN_SPAWN_MARKER", str(marker))
    monkeypatch.setenv("SHUTDOWN_SPAWN_RELEASE", str(release))
    monkeypatch.setenv("SHUTDOWN_CLEANUP_TIMING", str(timing))
    proc, master = _start(repo, pid_file, interactive=interactive, injection=tmp_path)
    try:
        _wait_for(marker)
        os.kill(proc.pid, signum)
        assert _wait_exit(proc, master) == 128 + signum, _output(proc, master)
        assert timing.exists(), "bounded cleanup did not report completion"
        signal_at, deadline, completed_at = map(float, timing.read_text().split())
        assert deadline == signal_at + STOP_TIMEOUT_SECONDS
        assert completed_at - signal_at < STOP_TIMEOUT_SECONDS + 0.25
        assert not pid_file.exists(), "worker command ran after shutdown intent"
        with sqlite3.connect(db) as conn:
            rows = conn.execute("SELECT status FROM nodes WHERE status = 'running'").fetchall()
        assert rows, "unresolved launch released graph ownership"
    finally:
        _owned_cleanup(proc, pid_file, master)
        release.touch()
