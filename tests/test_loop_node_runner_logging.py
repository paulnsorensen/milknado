"""Loop runner logs correlate terminal events with the dispatch run ID."""

from __future__ import annotations

from pathlib import Path
from typing import NoReturn

import pytest

from milknado.domains.common import RunResult


@pytest.mark.parametrize("case", [(True, False, 0), (False, False, 2), (True, True, 1)])
def test_main_logs_terminal_event_with_run_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: tuple[bool, bool, int]
) -> None:
    finish_result, fail_run, expected_rc = case
    import milknado.adapters as adapters
    import milknado.app.project as project
    import milknado.app.worker_recovery as worker_recovery
    import milknado.domains.execution as execution
    from milknado.domains.execution import NodeLoopOutcome
    from milknado.mcp import _loop_node_runner

    messages: list[tuple[str, tuple[object, ...]]] = []

    def _log_info(message: str, *args: object) -> None:
        messages.append((message, args))

    monkeypatch.setattr(
        _loop_node_runner._logger,  # pyright: ignore[reportPrivateUsage]
        "info",
        _log_info,
    )

    class _Cfg:
        execution_agent: str = "claude"
        quality_gates: tuple[str, ...] = ()
        worktree_pattern: str = "wt-{node}"
        flavors: dict[str, object] = {}
        worker_brief_prepend: str = "Detached worker instruction."
        agent_family: str = "claude"
        worker_agent_type: str = "milknado:milknado-worker"
        loop_mode: str = "redispatch"
        max_iterations: int = 8
        host_worker_limit: int = 6
        max_turns: int = 60
        commit_footer: str | None = None

    class _Graph:
        def __init__(self) -> None:
            self.closed: bool = False
            self.finish_result: bool = True
            self.finished: RunResult | None = None
            self.runs: _Graph = self

        def get_node(self, _node_id: int) -> None:
            return None

        def finish(self, _run_id: str, result: RunResult) -> None:
            self.finished = result
            if not self.finish_result:
                from milknado.domains.graph import RunFenceLostError

                raise RunFenceLostError("runs.finish lost its running-row fence")

        def set_pid(self, *_args: object) -> None:
            pass

        def deposit_message(self, *_args: object, **_kwargs: object) -> int:
            return 1

        def close(self) -> None:
            self.closed = True

    class _Git:
        def __init__(self, _root: object) -> None: ...

        def current_branch(self) -> str:
            return "main"

    class _StubLoop:
        def bind_shutdown_intent(self, _requested: object) -> None:
            pass

        def poll_progress_events(self) -> list[object]:
            return []

    graph = _Graph()

    def _open_graph(_root: Path) -> tuple[_Graph, _Cfg]:
        return graph, _Cfg()

    def _make_git(_root: object) -> _Git:
        return _Git(_root)

    def _make_loop(*_args: object, **_kwargs: object) -> _StubLoop:
        return _StubLoop()

    class _StubExecutor:
        def use_host_capacity(self, _pool: object) -> None:
            return None

    def _make_executor(**_kwargs: object) -> _StubExecutor:
        return _StubExecutor()

    captured_configs: list[dict[str, object]] = []

    def _make_execution_config(**kwargs: object) -> object:
        captured_configs.append(kwargs)
        return object()

    confirmed: list[bool] = []

    class _StubRunLoop:
        def __init__(self, **_kwargs: object) -> None: ...

        def run_node(self, *_args: object, **_kwargs: object) -> NodeLoopOutcome:
            if fail_run:
                raise RuntimeError("worker wait failed")
            return NodeLoopOutcome(node_id=1, success=True)

        def confirm_preserved_stop(self, outcome: NodeLoopOutcome) -> NodeLoopOutcome:
            confirmed.append(outcome.ownership_preserved)
            return outcome

    recovered: list[_Graph] = []
    monkeypatch.setattr(worker_recovery, "reconcile_loop_workers", recovered.append)
    monkeypatch.setattr(project, "open_graph", _open_graph)
    monkeypatch.setattr(adapters, "GitAdapter", _make_git)
    monkeypatch.setattr(adapters, "LoopAdapter", _make_loop)
    monkeypatch.setattr(execution, "Executor", _make_executor)
    monkeypatch.setattr(execution, "ExecutionConfig", _make_execution_config)
    monkeypatch.setattr(execution, "RunLoop", _StubRunLoop)

    graph.finish_result = finish_result
    run_id = "node-1-20260101T000000Z-abcd"
    rc = _loop_node_runner.main(
        [
            "--node-id",
            "1",
            "--project-root",
            str(tmp_path),
            "--run-id",
            run_id,
            "--target-branch",
            "main",
            "--base-oid",
            "base",
        ]
    )

    assert rc == expected_rc
    assert recovered == [graph]
    assert confirmed == [fail_run]
    assert captured_configs[0]["brief_prepend"] == "Detached worker instruction."
    assert list((tmp_path / ".milknado").glob("run-*.log")) == []
    assert expected_rc != 0 or any(
        "loop runner terminal" in message and run_id in args for message, args in messages
    )
    assert expected_rc != 1 or (graph.finished is not None and graph.finished.status == "failed")


def test_finish_run_writes_terminal_error_sidecar_on_fence_loss(tmp_path: Path) -> None:
    from milknado.domains.common import RunResult
    from milknado.mcp import _loop_node_runner

    class Graph:
        def __init__(self) -> None:
            self.runs: Graph = self

        def finish(self, *_args: object) -> None:
            from milknado.domains.graph import RunFenceLostError

            raise RunFenceLostError("runs.finish lost its running-row fence")

    result = RunResult(
        status="failed",
        exit_code=-1,
        timed_out=False,
        ended_at="2026-01-01T00:00:00+00:00",
    )
    assert (
        _loop_node_runner._finish_run(  # pyright: ignore[reportPrivateUsage]
            Graph(), tmp_path, "run-1", result
        )
        is False
    )
    sidecar = tmp_path / ".milknado" / "runs" / "run-1.terminal-error"
    assert "runs.finish lost its running-row fence" in sidecar.read_text(encoding="utf-8")


def test_finish_run_records_exception_when_graph_write_raises(tmp_path: Path) -> None:
    from milknado.domains.common import RunResult
    from milknado.mcp import _loop_node_runner

    class Graph:
        def __init__(self) -> None:
            self.runs: Graph = self

        def finish(self, *_args: object) -> None:
            raise RuntimeError("database unavailable")

    result = RunResult(
        status="failed",
        exit_code=1,
        timed_out=False,
        ended_at="2026-01-01T00:00:00+00:00",
    )
    assert (
        _loop_node_runner._finish_run(  # pyright: ignore[reportPrivateUsage]
            Graph(), tmp_path, "run-raise", result
        )
        is False
    )
    assert "database unavailable" in (
        tmp_path / ".milknado" / "runs" / "run-raise.terminal-error"
    ).read_text(encoding="utf-8")


def test_finish_run_logs_sidecar_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from milknado.domains.common import RunResult
    from milknado.mcp import _loop_node_runner

    class Graph:
        def __init__(self) -> None:
            self.runs: Graph = self

        def finish(self, *_args: object) -> None:
            from milknado.domains.graph import RunFenceLostError

            raise RunFenceLostError("runs.finish lost its running-row fence")

    class Sidecar:
        def write_text(self, *_args: object, **_kwargs: object) -> NoReturn:
            raise OSError("read-only")

    class RunDirectory:
        def joinpath(self, _name: str) -> Sidecar:
            return Sidecar()

    def _runs_dir(_root: Path) -> RunDirectory:
        return RunDirectory()

    monkeypatch.setattr(_loop_node_runner, "runs_dir", _runs_dir)
    result = RunResult(
        status="failed",
        exit_code=1,
        timed_out=False,
        ended_at="2026-01-01T00:00:00+00:00",
    )
    assert (
        _loop_node_runner._finish_run(  # pyright: ignore[reportPrivateUsage]
            Graph(), tmp_path, "run-sidecar", result
        )
        is False
    )
