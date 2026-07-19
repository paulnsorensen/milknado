"""Ralph runner logs correlate terminal events with the dispatch run ID."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_main_logs_terminal_event_with_run_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import milknado._mcp_core as mcp_core
    import milknado.adapters as adapters
    import milknado.domains.execution as execution
    from milknado import _ralph_node_runner
    from milknado.domains.execution.headless import HeadlessOutcome

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
        worker_brief_prepend = None
        agent_family = "claude"
        worker_agent_type = "milknado:milknado-worker"
        loop_mode = "redispatch"
        max_iterations = 8
        max_turns = 60
        commit_footer = None

    class _Graph:
        def __init__(self) -> None:
            self.closed = False
            self.finished: dict | None = None

        def get_node(self, node_id: int) -> None:  # noqa: ARG002
            return None

        def finish_run(self, run_id: str, result) -> None:  # noqa: ANN001
            self.finished = {"run_id": run_id, "result": result}

        def deposit_run_message(self, *a, **k) -> int:  # noqa: ANN002, ANN003
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
    monkeypatch.setattr(adapters, "CrgAdapter", lambda *a, **k: object())
    monkeypatch.setattr(execution, "Executor", lambda **k: object())
    monkeypatch.setattr(execution, "ExecutionConfig", lambda **k: object())
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
    assert list((tmp_path / ".milknado").glob("run-*.log")) == []
