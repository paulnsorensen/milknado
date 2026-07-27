"""Ralph runner logs correlate terminal events with the dispatch run ID."""

from __future__ import annotations

from pathlib import Path

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
    monkeypatch.setattr(
        _ralph_node_runner._logger,
        "info",
        lambda message, *args: messages.append((message, args)),
    )

    class _Cfg:
        execution_agent = "claude"
        quality_gates = ()
        worktree_pattern = "wt-{node}"
        flavors: dict = {}
        worker_brief_prepend = "Detached worker instruction."
        agent_family = "claude"
        worker_agent_type = "milknado:milknado-worker"
        loop_mode = "redispatch"
        max_iterations = 8
        max_turns = 60
        commit_footer = None

    class _Graph:
        def __init__(self) -> None:
            self.closed = False
            self.finish_result = True
            self.finished: dict | None = None

        def get_node(self, node_id: int) -> None:
            return None

        def finish_run(self, run_id: str, result) -> bool:
            self.finished = {"run_id": run_id, "result": result}
            return self.finish_result

        def set_run_pid(self, *_args) -> None:
            pass

        def set_pid(self, *_args) -> None:
            pass

        def deposit_run_message(self, *a, **k) -> int:
            return 1

        def close(self) -> None:
            self.closed = True

    class _Git:
        def __init__(self, _root: object) -> None: ...

        def current_branch(self) -> str:
            return "main"

    class _StubRalph:
        def poll_progress_events(self) -> list:
            return []

    graph = _Graph()
    monkeypatch.setattr(mcp_core, "open_graph", lambda _root: (graph, _Cfg()))
    monkeypatch.setattr(adapters, "GitAdapter", _Git)
    monkeypatch.setattr(adapters, "LoopAdapter", lambda *a, **k: _StubRalph())
    captured_configs: list[dict] = []
    monkeypatch.setattr(execution, "Executor", lambda **k: object())
    monkeypatch.setattr(
        execution,
        "ExecutionConfig",
        lambda **kwargs: captured_configs.append(kwargs) or object(),
    )
    monkeypatch.setattr(
        execution,
        "run_node_to_completion",
        lambda *a, **k: HeadlessOutcome(node_id=1, success=True, detail=None),
    )

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
        def finish_run(self, *_args) -> bool:
            return False

    result = RunResult(
        status="failed",
        exit_code=-1,
        timed_out=False,
        ended_at="2026-01-01T00:00:00+00:00",
    )
    assert _ralph_node_runner._finish_run(Graph(), tmp_path, "run-1", result) is False
    sidecar = tmp_path / ".milknado" / "runs" / "run-1.terminal-error"
    assert "finish_run lost its running-row fence" in sidecar.read_text(encoding="utf-8")


def test_finish_run_records_exception_when_graph_write_raises(tmp_path: Path) -> None:
    from milknado.domains.common import RunResult
    from milknado.mcp import _ralph_node_runner

    class Graph:
        def finish_run(self, *_args) -> bool:
            raise RuntimeError("database unavailable")

    result = RunResult(
        status="failed",
        exit_code=1,
        timed_out=False,
        ended_at="2026-01-01T00:00:00+00:00",
    )
    assert _ralph_node_runner._finish_run(Graph(), tmp_path, "run-raise", result) is False
    assert "database unavailable" in (
        tmp_path / ".milknado" / "runs" / "run-raise.terminal-error"
    ).read_text(encoding="utf-8")


def test_finish_run_logs_sidecar_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from milknado.domains.common import RunResult
    from milknado.mcp import _ralph_node_runner

    class Graph:
        def finish_run(self, *_args) -> bool:
            return False

    class Sidecar:
        def write_text(self, *_args, **_kwargs) -> None:
            raise OSError("read-only")

    class RunDirectory:
        def joinpath(self, _name: str) -> Sidecar:
            return Sidecar()

    monkeypatch.setattr(_ralph_node_runner, "runs_dir", lambda _root: RunDirectory())
    result = RunResult(
        status="failed",
        exit_code=1,
        timed_out=False,
        ended_at="2026-01-01T00:00:00+00:00",
    )
    assert _ralph_node_runner._finish_run(Graph(), tmp_path, "run-sidecar", result) is False
