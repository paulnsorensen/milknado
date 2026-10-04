"""Exercise force-quit keys against a live controller and fixture worker."""

from __future__ import annotations

import asyncio
import os
import shlex
import signal
import sqlite3
import sys
from contextlib import closing
from pathlib import Path
from typing import cast

import psutil
import pytest
from rich.text import Text
from textual.widgets import Static

from milknado.app.run import build_execution_controller
from milknado.app.run_tui import ExecutionApp
from milknado.domains.common import FlavorOverride, Gate, MilknadoConfig
from tests.execution_session_fixtures import build_graph, init_repo
from tests.worker_fixtures import install_worker_command

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX worker lifecycle evidence required")

_WORKER = """\
import json
import os
import sys
import time
from pathlib import Path

import psutil

for raw in sys.stdin:
    if json.loads(raw).get("type") == "user":
        identity = f"{os.getpid()}:{psutil.Process().create_time()}"
        Path(os.environ["QUIT_WORKED_MARKER"]).write_text(identity, encoding="utf-8")
        print(json.dumps({"type": "system", "subtype": "init",
                          "session_id": "quit-fixture", "model": "fixture"}), flush=True)
        break
while True:
    time.sleep(0.05)
"""


def _worker_alive(pid: int, token: float) -> bool:
    if not psutil.pid_exists(pid):
        return False
    worker = psutil.Process(pid)
    return worker.create_time() == token and worker.status() != psutil.STATUS_ZOMBIE


async def _wait_for(path: Path) -> tuple[int, float]:
    async with asyncio.timeout(10):
        while not path.exists():
            await asyncio.sleep(0.02)
    pid, token = path.read_text(encoding="utf-8").split(":", 1)
    return int(pid), float(token)


def _worker_end(db: Path, pid: int) -> str | None:
    with closing(sqlite3.connect(db)) as conn:
        row = cast(
            tuple[str | None] | None,
            conn.execute("SELECT ended_at FROM run_workers WHERE pid = ?", (pid,)).fetchone(),
        )
    assert row is not None
    return row[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("quit_key", ["q", "ctrl+c", "ctrl+q"])
async def test_run_quit_confirms_then_stops_live_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, quit_key: str
) -> None:
    repo = init_repo(tmp_path)
    graph = build_graph(repo)
    marker = tmp_path / "worker-worked"
    source = tmp_path / "worker.py"
    _ = source.write_text(_WORKER, encoding="utf-8")
    monkeypatch.setenv("QUIT_WORKED_MARKER", str(marker))
    command = install_worker_command(
        tmp_path / "bin",
        monkeypatch,
        agent="claude",
        script=f'exec {shlex.quote(sys.executable)} {shlex.quote(str(source))} "$@"\n',
    )
    config = MilknadoConfig(
        execution_agent=command,
        flavors={"implement": FlavorOverride(review=False)},
        quality_gates=(Gate(command="true"),),
        worktree_pattern="milknado-wt-{node_id}-{slug}",
        concurrency_limit=1,
        project_root=repo,
        db_path=graph.db_path,
    )
    controller = build_execution_controller(graph, config, repo)
    app = ExecutionApp(controller, feature_branch="feature")
    pid: int | None = None
    token = 0.0
    try:
        async with app.run_test(size=(120, 36)) as pilot:
            pid, token = await _wait_for(marker)
            assert _worker_alive(pid, token)
            assert _worker_end(graph.db_path, pid) is None
            await pilot.press(quit_key)
            await pilot.pause()
            confirmation = app.screen.query_one("#confirmation-overlay", Static)
            assert "Quit and force stop 1 active run?" in cast(Text, confirmation.render()).plain
            assert _worker_alive(pid, token)
            deadline = asyncio.get_running_loop().time() + 8.0
            async with asyncio.timeout_at(deadline):
                await pilot.press("y")
                while app.cleanup_confirmed is None:
                    await asyncio.sleep(0.02)
            assert app.cleanup_confirmed is True
            assert not _worker_alive(pid, token)
            assert _worker_end(graph.db_path, pid) is not None
    finally:
        if pid is not None and _worker_alive(pid, token):
            os.kill(pid, signal.SIGKILL)
        graph.close()
