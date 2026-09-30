"""Real loop-worker contexts use one durable owner without synthetic graph rows."""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

from milknado.adapters.loop import LoopAdapter
from milknado.domains.execution import PreservedWorkerRun
from milknado.domains.graph import (
    MikadoGraph, NodeWorkers, WorkerEvidenceStore, default_worker_db_path,
)
from milknado.loop._process_lifecycle import SpawnOptions


def _options(tmp_path: Path) -> SpawnOptions:
    return SpawnOptions(
        (sys.executable, "-c", "print('worker-ok', flush=True)"),
        tmp_path, None, True, subprocess.DEVNULL, subprocess.PIPE, subprocess.PIPE,
    )


def _finish(worker) -> None:
    assert worker.process.stdout is not None
    assert worker.process.stdout.read().strip() == "worker-ok"
    assert worker.process.wait(timeout=5) == 0
    assert worker.finish(timeout=5)


@pytest.mark.skipif(os.name == "nt", reason="POSIX lifeline requires passed file descriptors")
def test_graphless_adapter_creates_stable_store_only_at_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    db_path = default_worker_db_path()
    adapter = LoopAdapter()
    assert adapter.get_run("absent") is None
    assert not db_path.exists()
    worker = adapter._launch_worker(_options(tmp_path), "runtime-review", None)
    _finish(worker)
    with WorkerEvidenceStore(db_path) as store:
        record = store.get(worker.identity.invocation_id)
    assert record is not None and record.ended_at is not None
    assert record.runtime_run_id == "runtime-review"
    assert record.graph_run_id is None and record.node_id is None


@pytest.mark.skipif(os.name == "nt", reason="POSIX lifeline requires passed file descriptors")
def test_node_run_resolves_real_graph_association_at_launch(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("worker")
    loop_file = tmp_path / "loop.md"
    loop_file.write_text("Run fixture", encoding="utf-8")
    adapter = LoopAdapter(graph=graph)
    run = adapter.create_run(
        sys.executable, tmp_path, loop_file, None, run_id="run-1"
    )
    spawn = run.config.spawn_worker
    assert spawn is not None
    try:
        with pytest.raises(RuntimeError, match="running graph run"):
            spawn(_options(tmp_path))
        graph.runs.start("run-1", node.id, "worker.log", "2026-01-01T00:00:00+00:00", None)
        worker = spawn(_options(tmp_path))
        _finish(worker)
        record = graph.runs.get_worker(worker.identity.invocation_id)
        assert record is not None and record.ended_at is not None
        assert record.runtime_run_id == "run-1"
        assert (record.graph_run_id, record.node_id) == ("run-1", node.id)
    finally:
        graph.close()


@pytest.mark.skipif(os.name == "nt", reason="POSIX lifeline requires passed file descriptors")
def test_graph_backed_review_has_no_synthetic_run(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    try:
        adapter = LoopAdapter(graph=graph)
        worker = adapter._launch_worker(_options(tmp_path), "review-1", None)
        _finish(worker)
        record = graph.runs.get_worker(worker.identity.invocation_id)
        assert record is not None and record.ended_at is not None
        assert (record.graph_run_id, record.node_id) == (None, None)
    finally:
        graph.close()

@pytest.mark.skipif(os.name == "nt", reason="Windows retains its existing launch backend")
def test_raw_run_manager_rejects_missing_durable_context(tmp_path: Path) -> None:
    from milknado.loop import RunManager
    from tests.loop.helpers import make_config

    manager = RunManager()
    run = manager.create_run(make_config(tmp_path))
    with pytest.raises(RuntimeError, match="durable worker context"):
        manager.start_run(run.state.run_id)
    assert run.thread is None

@pytest.mark.skipif(os.name == "nt", reason="POSIX lifeline requires passed file descriptors")
def test_graph_reviewer_uses_real_node_association(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("milknado.loop.engine.validate_worker_argv", lambda _cmd: None)
    graph = MikadoGraph(tmp_path / "graph.db")
    node = graph.add_node("worker")
    graph.runs.start("graph-worker", node.id, "worker.log", "2026-01-01T00:00:00+00:00", None)
    script = tmp_path / "reviewer.py"
    script.write_text(
        "print('<verdict>approve</verdict>')\n"
        "print('<promise>MILKNADO_NODE_REVIEW_COMPLETE</promise>')\n",
        encoding="utf-8",
    )
    try:
        adapter = LoopAdapter(graph=graph)
        verdict = adapter.run_node_review(
            f"{sys.executable} {script}", "review", tmp_path, tmp_path,
            timeout_seconds=5, graph_run_id="graph-worker",
        )
        assert verdict.approved
        with WorkerEvidenceStore(graph.db_path) as store:
            records = store.live_workers(NodeWorkers(node.id))
        assert records == ()
        with sqlite3.connect(graph.db_path) as conn:
            row = conn.execute(
                "SELECT invocation_id FROM run_workers WHERE graph_run_id = ?",
                ("graph-worker",),
            ).fetchone()
        assert row is not None
        with WorkerEvidenceStore(graph.db_path) as store:
            record = store.get(row[0])
        assert record is not None and record.ended_at is not None
        assert record.graph_run_id == "graph-worker" and record.node_id == node.id
        assert record.runtime_run_id != record.graph_run_id
    finally:
        graph.close()


@pytest.mark.skipif(os.name == "nt", reason="POSIX lifeline requires passed file descriptors")
def test_unconfirmed_reviewer_retains_real_node_owner_and_unrelated_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("milknado.loop.engine.validate_worker_argv", lambda _cmd: None)
    graph = MikadoGraph(tmp_path / "graph.db")
    owner = graph.add_node("review owner")
    other = graph.add_node("unrelated worker")
    for run_id, node in (("owner-run", owner), ("other-run", other)):
        graph.runs.start(run_id, node.id, "worker.log", "2026-01-01T00:00:00+00:00", None)
    script = tmp_path / "reviewer.py"
    script.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    adapter = LoopAdapter(graph=graph)
    try:
        with monkeypatch.context() as patch:
            patch.setattr("milknado.loop.manager.RunManager.stop_and_join", lambda *_a, **_k: False)
            with pytest.raises(PreservedWorkerRun) as failure:
                adapter.run_node_review(
                    f"{sys.executable} {script}", "review", tmp_path, tmp_path,
                    timeout_seconds=1, graph_run_id="owner-run",
                )
        assert failure.value.run_id == "owner-run"
        records = graph.runs.live_workers(node_id=owner.id)
        assert len(records) == 1 and records[0].graph_run_id == "owner-run"
        assert records[0].ended_at is None
        run = graph.runs.get("owner-run")
        assert run is not None and run["status"] == "running"
        worker = adapter._launch_worker(
            SpawnOptions(
                (sys.executable, "-c", "import time; time.sleep(30)"),
                tmp_path, None, True, subprocess.DEVNULL, subprocess.PIPE, subprocess.PIPE,
            ),
            "other-local", "other-run",
        )
        try:
            assert adapter.stop_run_workers("owner-run", time.monotonic() + 4)
            assert worker.process.poll() is None
        finally:
            assert worker.shutdown(time.monotonic() + 4)
    finally:
        _ = adapter.stop_active_workers(time.monotonic() + 4)
        graph.close()
