from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TypedDict, cast

import pytest

from milknado.mcp._core import GraphSummaryResponse, NodeSummary
from milknado.mcp.node import milknado_todo_claim
from milknado.mcp.server import milknado_graph_summary, open_graph
from milknado.mcp.todo import milknado_todo_next
from milknado.mcp.todo_mutate import (
    milknado_todo_add,
    milknado_todo_set_status,
    milknado_track_follow_up,
)


class _ClaimPayload(TypedDict):
    node_id: int
    run_id: str


def _call(tool: object, **kwargs: object) -> object:
    fn = cast(Callable[..., object], getattr(tool, "fn", tool))
    return fn(**kwargs)


def _add(description: str, root: str, **options: object) -> NodeSummary:
    return cast(
        NodeSummary,
        _call(milknado_todo_add, description=description, project_root=root, **options),
    )


def _set_status(node_id: int, status: str, root: str) -> NodeSummary:
    return cast(
        NodeSummary,
        _call(milknado_todo_set_status, node_id=node_id, status=status, project_root=root),
    )


def _next(root: str) -> NodeSummary | None:
    return cast(NodeSummary | None, _call(milknado_todo_next, project_root=root))


def _ids_by_status(root: str, status: str) -> set[int]:
    response = cast(
        GraphSummaryResponse,
        _call(milknado_graph_summary, project_root=root, status=status),
    )
    return {node["id"] for node in response["nodes"]}


def _write_claim_config(root: Path) -> None:
    _ = (root / "milknado.toml").write_text(
        '[milknado]\nagent_family = "claude"\nquality_gates = ["true"]\n',
        encoding="utf-8",
    )


def test_blocked_prerequisite_suppresses_only_dependent_work(tmp_path: Path) -> None:
    root = str(tmp_path)
    goal = _add("goal", root, kind="goal")
    prerequisite = _add("operator-held prerequisite", root, parent_id=goal["id"])
    dependent = _add(
        "dependent work",
        root,
        parent_id=goal["id"],
        prereqs=[prerequisite["id"]],
    )
    sibling = _add("independent sibling", root, parent_id=goal["id"])

    _ = _set_status(prerequisite["id"], "blocked", root)

    ready = _next(root)
    assert ready is not None
    assert ready["id"] == sibling["id"]
    assert dependent["id"] in _ids_by_status(root, "pending")

    _ = _set_status(sibling["id"], "done", root)
    assert _next(root) is None


def test_operator_status_release_restores_prerequisite_readiness(tmp_path: Path) -> None:
    root = str(tmp_path)
    goal = _add("goal", root, kind="goal")
    prerequisite = _add("held prerequisite", root, parent_id=goal["id"])
    dependent = _add(
        "dependent work",
        root,
        parent_id=goal["id"],
        prereqs=[prerequisite["id"]],
    )

    _ = _set_status(prerequisite["id"], "blocked", root)
    assert _next(root) is None

    _ = _set_status(prerequisite["id"], "pending", root)
    ready = _next(root)
    assert ready is not None
    assert ready["id"] == prerequisite["id"]

    _ = _set_status(prerequisite["id"], "done", root)
    ready = _next(root)
    assert ready is not None
    assert ready["id"] == dependent["id"]


def test_blocked_claim_is_scheduler_hold_and_second_claim_is_fenced(tmp_path: Path) -> None:
    """BLOCKED is a scheduler hold, not a dedicated human-approval lock."""
    _write_claim_config(tmp_path)
    root = str(tmp_path)
    task = _add("held task", root)
    _ = _set_status(task["id"], "blocked", root)

    payload = cast(
        _ClaimPayload,
        _call(milknado_todo_claim, node_id=task["id"], worktree=False, project_root=root),
    )
    assert payload["node_id"] == task["id"]
    assert task["id"] in _ids_by_status(root, "in_progress")
    graph, _config = open_graph(tmp_path)
    try:
        held = graph.get_node(task["id"])
        assert held is not None
        assert held.run_id == payload["run_id"]
    finally:
        graph.close()

    with pytest.raises(ValueError, match="already running"):
        _ = _call(milknado_todo_claim, node_id=task["id"], worktree=False, project_root=root)


def test_worker_follow_up_is_sibling_and_ready_after_worker_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = str(tmp_path)
    goal = _add("goal", root, kind="goal")
    worker = _add("current worker", root, parent_id=goal["id"])
    monkeypatch.setenv("MILKNADO_NODE_ID", str(worker["id"]))
    monkeypatch.setenv("MILKNADO_RUN_ID", "worker-run")
    monkeypatch.setenv("MILKNADO_PROJECT_ROOT", root)

    follow_up = cast(
        NodeSummary,
        _call(milknado_track_follow_up, description="discovered follow-up", project_root=root),
    )

    graph, _config = open_graph(tmp_path)
    try:
        worker_node = graph.get_node(worker["id"])
        follow_up_node = graph.get_node(follow_up["id"])
        assert worker_node is not None
        assert follow_up_node is not None
        assert worker_node.parent_id == goal["id"]
        assert follow_up_node.parent_id == worker_node.parent_id
    finally:
        graph.close()

    _ = _set_status(worker["id"], "done", root)
    ready = _next(root)
    assert ready is not None
    assert ready["id"] == follow_up["id"]
