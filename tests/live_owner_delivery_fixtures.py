from __future__ import annotations

import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.worker_fixtures import install_worker_command

_OWNER_SOURCE = r"""
import os
import sys
import tempfile
from pathlib import Path
from milknado.app.run import build_execution_controller
from milknado.domains.common import FlavorOverride, Gate, MilknadoConfig
from milknado.domains.graph import MikadoGraph

repo, db, worker, ready = map(Path, sys.argv[1:5])
config = MilknadoConfig(
    execution_agent=str(worker),
    flavors={"implement": FlavorOverride(review=False)},
    quality_gates=(Gate(command="true"),),
    worktree_pattern="milknado-wt-{node_id}-{slug}",
    concurrency_limit=1,
    project_root=repo,
    db_path=db,
)
graph = MikadoGraph(db)
controller = build_execution_controller(graph, config, repo)
def observe(snapshot):
    for run in snapshot.active_runs:
        if ready.exists():
            return
        if any(event.text.startswith("Claude session started") for event in run.session.events):
            with tempfile.NamedTemporaryFile(
                dir=ready.parent, prefix=f"{ready.name}.", delete=False
            ) as marker:
                marker.write(run.run_id.encode())
                marker_path = marker.name
            os.replace(marker_path, ready)
            return
controller.subscribe(observe)
try:
    controller.run(feature_branch="feature")
finally:
    graph.close()
"""

_VENDOR_SOURCE = r"""
import json
import os
import sys
import time
from pathlib import Path

mode, wire, marker, barrier, pid_file = sys.argv[1:6]
Path(pid_file).write_text(f"{os.getpid()}:{os.getpgid(0)}", encoding="utf-8")
count = 0
for raw in sys.stdin:
    frame = json.loads(raw)
    if frame.get("type") != "user":
        continue
    text = frame.get("message", {}).get("content", "")
    count += 1
    if count == 1:
        print(json.dumps({
            "type": "system", "subtype": "init",
            "session_id": "live-owner", "model": "fixture"
        }), flush=True)
        continue
    with Path(wire).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"text": text}) + "\n")
    Path(marker).touch()
    if mode == "after" and count == 2:
        while True:
            time.sleep(0.01)
    print(json.dumps({
        "type": "user", "message": {"role": "user", "content": text}
    }), flush=True)
    if text == "wire barrier":
        Path(barrier).touch()
"""


def worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, label: str
) -> tuple[str, Path, Path, Path]:
    source = tmp_path / f"vendor-{label}.py"
    _ = source.write_text(_VENDOR_SOURCE, encoding="utf-8")
    wire = tmp_path / f"{label}.wire"
    marker = tmp_path / f"{label}.marker"
    barrier = tmp_path / f"{label}.barrier"
    command = install_worker_command(
        tmp_path / f"bin-{label}",
        monkeypatch,
        agent="claude",
        script=f'exec {shlex.quote(sys.executable)} {shlex.quote(str(source))} "$@"\n',
    )
    pid_file = tmp_path / f"{label}.pid"
    return f"{command} {mode} {wire} {marker} {barrier} {pid_file}", wire, marker, barrier


def owner(repo: Path, db: Path, worker: str, ready: Path) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "-c", _OWNER_SOURCE, str(repo), str(db), worker, str(ready)],
        start_new_session=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def wait(path: Path) -> None:
    deadline = time.monotonic() + 10
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert path.exists(), f"missing marker: {path}"


def kill_owner(owner: subprocess.Popen[str]) -> None:
    os.killpg(owner.pid, signal.SIGKILL)
    _ = owner.wait(timeout=10)
    assert owner.returncode == -signal.SIGKILL
    if owner.stdout is not None:
        owner.stdout.close()
    if owner.stderr is not None:
        owner.stderr.close()


def kill_vendor(pid_file: Path) -> None:
    if not pid_file.exists():
        return
    pid, process_group = (int(value) for value in pid_file.read_text().split(":"))
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        pid_file.unlink(missing_ok=True)
        return
    try:
        os.killpg(process_group, signal.SIGKILL)
    except PermissionError:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pid_file.unlink(missing_ok=True)
            return
        time.sleep(0.01)
    pytest.fail(f"vendor process {pid} survived cleanup")
