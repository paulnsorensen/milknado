"""Ralph runner logs correlate terminal events with the dispatch run ID."""

from __future__ import annotations

from pathlib import Path
from typing import NoReturn

import pytest


def test_main_logs_terminal_event_with_run_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import milknado.adapters as adapters
    import milknado.domains.execution as execution
    import milknado.mcp._core as mcp_core
    from milknado.domains.execution.headless import HeadlessOutcome
    from milknado.mcp import _ralph_node_runner

    messages: list[tuple[str, tuple[object, ...]]] = []

    def _log_info(message: str, *args: object) -> None:
        messages.append((message, args))

    monkeypatch.setattr(
        _ralph_node_runner._logger,  # pyright: ignore[reportPrivateUsage]
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
        max_turns: int = 60
        commit_footer: str | None = None

    class _Graph:
        def __init__(self) -> None:
            self.closed: bool = False
            self.finish_result: bool = True
            self.finished: dict[str, object] | None = None

        def get_node(self, _node_id: int) -> None:
            return None

        def finish_run(self, run_id: str, result: object) -> bool:
            self.finished = {"run_id": run_id, "result": result}
            return self.finish_result

        def set_run_pid(self, *_args: object) -> None:
            pass

        def set_pid(self, *_args: object) -> None:
            pass

        def deposit_run_message(self, *_args: object, **_kwargs: object) -> int:
            return 1

        def close(self) -> None:
            self.closed = True

    class _Git:
        def __init__(self, _root: object) -> None: ...

        def current_branch(self) -> str:
            return "main"

    class _StubRalph:
        def poll_progress_events(self) -> list[object]:
            return []

    graph = _Graph()

    def _open_graph(_root: Path) -> tuple[_Graph, _Cfg]:
        return graph, _Cfg()

    def _make_git(_root: object) -> _Git:
        return _Git(_root)

    def _make_ralph(*_args: object, **_kwargs: object) -> _StubRalph:
        return _StubRalph()

    def _make_executor(**_kwargs: object) -> object:
        return object()

    captured_configs: list[dict[str, object]] = []

    def _make_execution_config(**kwargs: object) -> object:
        captured_configs.append(kwargs)
        return object()

    def _run_node_to_completion(*_args: object, **_kwargs: object) -> HeadlessOutcome:
        return HeadlessOutcome(node_id=1, success=True, detail=None)

    monkeypatch.setattr(mcp_core, "open_graph", _open_graph)
    monkeypatch.setattr(adapters, "GitAdapter", _make_git)
    monkeypatch.setattr(adapters, "LoopAdapter", _make_ralph)
    monkeypatch.setattr(execution, "Executor", _make_executor)
    monkeypatch.setattr(execution, "ExecutionConfig", _make_execution_config)
    monkeypatch.setattr(execution, "run_node_to_completion", _run_node_to_completion)

    run_id = "node-1-20260101T000000Z-abcd"
    rc = _ralph_node_runner.main(
        [
            "--node-id",
            "1",
            "--project-root",
            str(tmp_path),
            "--run-id",
            run_id,
            "--timeout",
            "30",
            "--target-branch",
            "main",
            "--base-oid",
            "base",
        ]
    )

    assert rc == 0
    assert any("ralph runner terminal" in message and run_id in args for message, args in messages)
    assert captured_configs[0]["brief_prepend"] == "Detached worker instruction."
    assert list((tmp_path / ".milknado").glob("run-*.log")) == []

    graph.finish_result = False
    assert (
        _ralph_node_runner.main(
            [
                "--node-id",
                "1",
                "--project-root",
                str(tmp_path),
                "--run-id",
                "node-1-20260101T000000Z-beef",
                "--timeout",
                "30",
                "--target-branch",
                "main",
                "--base-oid",
                "base",
            ]
        )
        == 2
    )


def test_finish_run_writes_terminal_error_sidecar_on_fence_loss(tmp_path: Path) -> None:
    from milknado.domains.common import RunResult
    from milknado.mcp import _ralph_node_runner

    class Graph:
        def finish_run(self, *_args: object) -> bool:
            return False

    result = RunResult(
        status="failed",
        exit_code=-1,
        timed_out=False,
        ended_at="2026-01-01T00:00:00+00:00",
    )
    assert (
        _ralph_node_runner._finish_run(  # pyright: ignore[reportPrivateUsage]
            Graph(), tmp_path, "run-1", result
        )
        is False
    )
    sidecar = tmp_path / ".milknado" / "runs" / "run-1.terminal-error"
    assert "finish_run lost its running-row fence" in sidecar.read_text(encoding="utf-8")


def test_finish_run_records_exception_when_graph_write_raises(tmp_path: Path) -> None:
    from milknado.domains.common import RunResult
    from milknado.mcp import _ralph_node_runner

    class Graph:
        def finish_run(self, *_args: object) -> bool:
            raise RuntimeError("database unavailable")

    result = RunResult(
        status="failed",
        exit_code=1,
        timed_out=False,
        ended_at="2026-01-01T00:00:00+00:00",
    )
    assert (
        _ralph_node_runner._finish_run(  # pyright: ignore[reportPrivateUsage]
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
    from milknado.mcp import _ralph_node_runner

    class Graph:
        def finish_run(self, *_args: object) -> bool:
            return False

    class Sidecar:
        def write_text(self, *_args: object, **_kwargs: object) -> NoReturn:
            raise OSError("read-only")

    class RunDirectory:
        def joinpath(self, _name: str) -> Sidecar:
            return Sidecar()

    def _runs_dir(_root: Path) -> RunDirectory:
        return RunDirectory()

    monkeypatch.setattr(_ralph_node_runner, "runs_dir", _runs_dir)
    result = RunResult(
        status="failed",
        exit_code=1,
        timed_out=False,
        ended_at="2026-01-01T00:00:00+00:00",
    )
    assert (
        _ralph_node_runner._finish_run(  # pyright: ignore[reportPrivateUsage]
            Graph(), tmp_path, "run-sidecar", result
        )
        is False
    )
