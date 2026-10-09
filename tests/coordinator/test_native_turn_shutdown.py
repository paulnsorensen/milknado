from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from threading import Event, Thread

import pytest

from milknado.adapters.coordinator_turns import NativeCoordinatorTurns
from milknado.domains.common import MilknadoConfig
from milknado.domains.coordinator import TurnPreflightError
from milknado.domains.coordinator.control_services import TurnRuntimeHooks, TurnRuntimeRequest
from milknado.domains.graph import MikadoGraph
from milknado.loop._agent import AgentResult
from milknado.loop._process_contract import WorkerHandle
from milknado.loop._process_gate import SpawnOptions
from milknado.loop._process_registry import WorkerRegistry
from milknado.loop.sessions import RuntimeRequest, RuntimeResult


def _adapter(root: Path, graph: MikadoGraph) -> NativeCoordinatorTurns:
    return NativeCoordinatorTurns(
        root,
        MilknadoConfig(
            project_root=root,
            db_path=graph.db_path,
            agent_family="codex",
            execution_agent="codex exec",
        ),
        graph,
    )


def _request(turn_id: str = "turn") -> TurnRuntimeRequest:
    return TurnRuntimeRequest(
        "codex",
        "Work",
        None,
        None,
        TurnRuntimeHooks(turn_id, lambda _identity: None, lambda _event: None),
    )


def _sleeping_worker(request: RuntimeRequest, root: Path) -> WorkerHandle:
    spawn = request.spec.spawn_worker
    assert spawn is not None
    return spawn(
        SpawnOptions(
            (sys.executable, "-c", "import time; time.sleep(30)"),
            root,
            None,
            False,
            subprocess.DEVNULL,
            subprocess.PIPE,
            subprocess.PIPE,
        )
    )


def _start_blocked_native_turn(
    tmp_path: Path, adapter: NativeCoordinatorTurns, monkeypatch: pytest.MonkeyPatch
) -> tuple[Thread, Event, Event, list[WorkerHandle], list[BaseException]]:
    spawned, callback_waiting, release = Event(), Event(), Event()
    workers: list[WorkerHandle] = []
    failures: list[BaseException] = []

    def execute(request: RuntimeRequest) -> RuntimeResult:
        worker = _sleeping_worker(request, tmp_path)
        workers.append(worker)
        spawned.set()
        assert request.spec.force_stop_event is not None
        assert request.spec.force_stop_event.wait(5)
        callback_waiting.set()
        assert release.wait(5)
        assert worker.cleanup()
        return RuntimeResult(AgentResult(0))

    monkeypatch.setattr("milknado.adapters.coordinator_turns.start_or_resume", execute)

    def run() -> None:
        try:
            _ = adapter.run(_request())
        except BaseException as error:
            failures.append(error)

    turn = Thread(target=run)
    turn.start()
    assert spawned.wait(5)
    return turn, callback_waiting, release, workers, failures


@pytest.mark.skipif(os.name == "nt", reason="protected worker requires POSIX")
def test_shutdown_stops_protected_worker_and_waits_for_runtime_callback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    adapter = _adapter(tmp_path, graph)
    turn, callback_waiting, release, workers, failures = _start_blocked_native_turn(
        tmp_path, adapter, monkeypatch
    )

    def stop() -> None:
        try:
            adapter.shutdown(timeout=5)
        except BaseException as error:
            failures.append(error)

    shutdown = Thread(target=stop)
    shutdown.start()
    assert callback_waiting.wait(5)
    assert turn.is_alive() and shutdown.is_alive()
    with pytest.raises(TurnPreflightError, match="shutting down"):
        _ = adapter.run(_request("late"))
    release.set()
    turn.join(5)
    shutdown.join(5)
    assert not turn.is_alive() and not shutdown.is_alive()
    assert failures == []
    assert workers[0].process.poll() is not None
    adapter.shutdown(timeout=0.01)
    graph.close()


def test_shutdown_rejects_unconfirmed_pending_protected_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    workers = WorkerRegistry()
    monkeypatch.setattr("milknado.adapters.coordinator_turns.WorkerRegistry", lambda: workers)
    adapter = _adapter(tmp_path, graph)
    pending = workers.reserve()
    with pytest.raises(RuntimeError, match="native worker shutdown is unconfirmed"):
        adapter.shutdown(timeout=0.01)
    pending.close()
    graph.close()


def test_shutdown_blocks_launch_admitted_before_preparation_finishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    adapter = _adapter(tmp_path, graph)
    preparing, release = Event(), Event()
    failures: list[BaseException] = []

    def prepare(_request: TurnRuntimeRequest) -> tuple[Path, list[str]]:
        preparing.set()
        assert release.wait(5)
        return tmp_path, ["codex", "exec"]

    monkeypatch.setattr(adapter, "_prepare", prepare)

    def run() -> None:
        try:
            _ = adapter.run(_request())
        except BaseException as error:
            failures.append(error)

    turn = Thread(target=run)
    turn.start()
    assert preparing.wait(5)
    with pytest.raises(RuntimeError, match="native worker shutdown is unconfirmed"):
        adapter.shutdown(timeout=0.01)
    release.set()
    turn.join(5)
    assert not turn.is_alive()
    assert len(failures) == 1
    assert isinstance(failures[0], TurnPreflightError)
    assert str(failures[0]) == "native turn is shutting down"
    adapter.shutdown(timeout=0.01)
    graph.close()
