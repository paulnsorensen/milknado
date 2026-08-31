from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

from milknado.mcp._core import NodeSummary
from milknado.mcp.todo import milknado_todo_next
from milknado.mcp.todo_mutate import milknado_todo_add, milknado_todo_set_status


def _call(tool: object, **kwargs: object) -> NodeSummary | None:
    fn = cast(Callable[..., NodeSummary | None], getattr(tool, "fn", tool))
    return fn(**kwargs)


def test_todo_next_waits_for_named_prerequisite(tmp_path: Path) -> None:
    root = str(tmp_path)
    goal = _call(milknado_todo_add, description="goal", kind="goal", project_root=root)
    assert goal is not None
    prerequisite = _call(
        milknado_todo_add,
        description="first",
        parent_id=goal["id"],
        project_root=root,
    )
    assert prerequisite is not None
    dependent = _call(
        milknado_todo_add,
        description="second",
        parent_id=goal["id"],
        prereqs=[prerequisite["id"]],
        project_root=root,
    )
    assert dependent is not None

    first = _call(milknado_todo_next, project_root=root)
    assert first is not None
    assert first["id"] == prerequisite["id"]
    _ = _call(
        milknado_todo_set_status,
        node_id=prerequisite["id"],
        status="done",
        project_root=root,
    )
    second = _call(milknado_todo_next, project_root=root)
    assert second is not None
    assert second["id"] == dependent["id"]
