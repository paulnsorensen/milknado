from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psutil
import pytest

from milknado.adapters._loop_worker_evidence import LoopWorkerEvidence
from milknado.domains.common import HelperIdentity, SessionInput, WorkerOwner
from milknado.domains.graph import MikadoGraph
from milknado.loop._agent import (
    AgentRunSpec,
    _ResolvedAgentRun,
    _run_agent_blocking,
    _run_agent_streaming,
)
from milknado.loop._process_lifecycle import (
    ProtectedWorker,
    ProtectionContext,
    SpawnOptions,
    spawn_protected,
)
from milknado.loop.sessions import SessionChannel, run_session

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="POSIX helper uses passed file descriptors"
)


def _until(predicate, timeout: float = 8) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def _context(tmp_path: Path) -> tuple[MikadoGraph, ProtectionContext]:
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("worker")
    graph.runs.start("run-1", node.id, "worker.log", "2026-01-01T00:00:00+00:00", None)
    supervisor = psutil.Process()
    owner = WorkerOwner("run-1", supervisor.pid, supervisor.create_time(), "run-1", node.id)
    return graph, ProtectionContext(LoopWorkerEvidence(graph.db_path), owner, graph.db_path)


def _replacement_ready(graph: MikadoGraph, worker: ProtectedWorker, generation: int) -> bool:
    record = graph.runs.get_worker(worker.identity.invocation_id)
    return record is not None and record.ready_generation == generation


@pytest.mark.parametrize("streaming", [False, True], ids=["generic-blocking", "generic-streaming"])
def test_generic_worker_preserves_input_output_and_result_during_replacement(
    tmp_path: Path,
    streaming: bool,
) -> None:
    graph, context = _context(tmp_path)
    stage, release = tmp_path / "stage", tmp_path / "release"
    program = """
import os
import sys
import time
from pathlib import Path
value = sys.stdin.readline().strip()
Path(sys.argv[1]).write_text(f'{os.getpid()}:{value}')
print(f'before:{value}', flush=True)
while not Path(sys.argv[2]).exists():
    time.sleep(.02)
print('{"type":"result","result":"after"}', flush=True)
raise SystemExit(11)
"""
    owned: list[ProtectedWorker] = []

    def launch(options: SpawnOptions) -> ProtectedWorker:
        worker = spawn_protected(options, context)
        owned.append(worker)
        return worker

    run = _ResolvedAgentRun(
        [sys.executable, "-c", program, str(stage), str(release)],
        "request\n",
        timeout=10,
        log_dir=tmp_path,
        iteration=1,
        spawn_worker=launch,
        capture_result_text=True,
    )
    execute = _run_agent_streaming if streaming else _run_agent_blocking
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(execute, run)
            assert _until(stage.exists)
            worker = owned[0]
            before = graph.runs.get_worker(worker.identity.invocation_id)
            assert before is not None and before.helper_pid is not None
            assert stage.read_text() == f"{before.pid}:request"
            os.kill(before.helper_pid, signal.SIGKILL)
            assert _until(lambda: _replacement_ready(graph, worker, 1))
            after = graph.runs.get_worker(worker.identity.invocation_id)
            assert after is not None and after.pid == before.pid
            release.touch()
            result = future.result(timeout=10)
        assert result.returncode == 11
        assert result.result_text == "after"
        assert "before:request" in result.captured_stdout
        assert graph.runs.live_workers(run_id="run-1") == ()
    finally:
        release.touch()
        for worker in owned:
            if worker.process.poll() is None:
                worker.shutdown(time.monotonic() + 3)
        graph.close()


_NATIVE = """
import json
import os
import sys
import time
from pathlib import Path
stage, release = map(Path, sys.argv[1:3])
for _ in range(2):
    if not sys.stdin.readline():
        raise SystemExit(2)
stage.write_text(str(os.getpid()))
print(json.dumps({'type':'assistant','session_id':'sid','message':
    {'role':'assistant','content':[{'type':'text','text':'before'}]}}), flush=True)
while not release.exists():
    time.sleep(.02)
print(json.dumps({'type':'result','subtype':'success','result':'native-before',
                  'session_id':'sid'}), flush=True)
follow_up = json.loads(sys.stdin.readline())
stage.with_name('follow-up').write_text(follow_up['message']['content'])
print(json.dumps({'type':'result','subtype':'success','result':'native-after',
                  'session_id':'sid'}), flush=True)
raise SystemExit(13)
"""


def test_native_session_preserves_worker_and_result_during_replacement(tmp_path: Path) -> None:
    graph, context = _context(tmp_path)
    stage, release = tmp_path / "stage", tmp_path / "release"
    script = tmp_path / "fixture.py"
    script.write_text(_NATIVE)
    executable = tmp_path / "claude"
    executable.symlink_to(sys.executable)
    owned: list[ProtectedWorker] = []

    def launch(options: SpawnOptions) -> ProtectedWorker:
        worker = spawn_protected(options, context)
        owned.append(worker)
        return worker

    spec = AgentRunSpec(
        cmd=[str(executable), str(script), str(stage), str(release)],
        prompt="request",
        timeout=10,
        log_dir=None,
        iteration=1,
        capture_result_text=True,
        cwd=tmp_path,
        spawn_worker=launch,
    )
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            channel = SessionChannel()
            future = pool.submit(run_session, spec, channel)
            assert _until(stage.exists)
            worker = owned[0]
            before = graph.runs.get_worker(worker.identity.invocation_id)
            assert before is not None and before.helper_pid is not None
            assert stage.read_text() == str(before.pid)
            os.kill(before.helper_pid, signal.SIGKILL)
            assert _until(lambda: _replacement_ready(graph, worker, 1))
            after = graph.runs.get_worker(worker.identity.invocation_id)
            assert after is not None and after.pid == before.pid
            assert channel.submit(SessionInput(action="follow_up", text="after takeover"))
            release.touch()
            result = future.result(timeout=10)
        assert result.returncode == 13
        assert result.result_text == "native-after"
        assert stage.with_name("follow-up").read_text() == "after takeover"
        assert graph.runs.live_workers(run_id="run-1") == ()
    finally:
        release.touch()
        for worker in owned:
            if worker.process.poll() is None:
                worker.shutdown(time.monotonic() + 3)
        graph.close()


def test_stale_ready_generation_cannot_replace_current_helper(tmp_path: Path) -> None:
    graph, context = _context(tmp_path)
    worker = spawn_protected(
        SpawnOptions(
            (sys.executable, "-c", "import time; time.sleep(60)"),
            tmp_path,
            None,
            False,
            subprocess.DEVNULL,
            subprocess.PIPE,
            subprocess.PIPE,
        ),
        context,
    )
    try:
        first = graph.runs.get_worker(worker.identity.invocation_id)
        assert first is not None and first.helper_pid is not None
        os.kill(first.helper_pid, signal.SIGKILL)
        assert _until(lambda: _replacement_ready(graph, worker, 1))
        with pytest.raises(RuntimeError, match="stale helper generation"):
            graph.runs.record_helper(
                HelperIdentity(
                    worker.identity.invocation_id,
                    1,
                    first.helper_pid,
                    first.helper_start_token,
                )
            )
        after = graph.runs.get_worker(worker.identity.invocation_id)
        assert after is not None and after.ready_generation == 1
        assert after.helper_pid != first.helper_pid
        assert first.helper_start_token is not None
        assert not graph.runs.ready_helper(
            HelperIdentity(
                worker.identity.invocation_id, 0, first.helper_pid, first.helper_start_token
            ),
            after.snapshot_seq,
        )
        assert worker.process.poll() is None
    finally:
        worker.shutdown(time.monotonic() + 3)
        graph.close()


def test_replacement_retains_observed_setsid_descendant(tmp_path: Path) -> None:
    graph, context = _context(tmp_path)
    marker = tmp_path / "child"
    program = (
        "import subprocess,sys,time; from pathlib import Path; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'], "
        "start_new_session=True); Path(sys.argv[1]).write_text(str(child.pid)); time.sleep(60)"
    )
    worker = spawn_protected(
        SpawnOptions(
            (sys.executable, "-c", program, str(marker)),
            tmp_path,
            None,
            False,
            subprocess.DEVNULL,
            subprocess.PIPE,
            subprocess.PIPE,
        ),
        context,
    )
    child_pid = 0
    try:
        assert _until(marker.exists)
        child_pid = int(marker.read_text())
        assert _until(
            lambda: (
                (record := graph.runs.get_worker(worker.identity.invocation_id)) is not None
                and any(item[0] == child_pid for item in record.descendants)
            )
        )
        before = graph.runs.get_worker(worker.identity.invocation_id)
        assert before is not None and before.helper_pid is not None
        os.kill(before.helper_pid, signal.SIGKILL)
        assert _until(lambda: _replacement_ready(graph, worker, 1))
        after = graph.runs.get_worker(worker.identity.invocation_id)
        assert after is not None and any(item[0] == child_pid for item in after.descendants)
        assert worker.shutdown(time.monotonic() + 4)
        assert (
            not psutil.pid_exists(child_pid)
            or psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE
        )
        assert graph.runs.live_workers(run_id="run-1") == ()
    finally:
        if child_pid and psutil.pid_exists(child_pid):
            child = psutil.Process(child_pid)
            if child.status() != psutil.STATUS_ZOMBIE:
                child.kill()
        if worker.process.poll() is None:
            worker.shutdown(time.monotonic() + 3)
        graph.close()
