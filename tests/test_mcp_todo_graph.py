from __future__ import annotations

from pathlib import Path

from milknado.mcp_todo import milknado_todo_next
from milknado.mcp_todo_mutate import milknado_todo_add, milknado_todo_set_status


def _call(tool, **kwargs):
    return getattr(tool, "fn", tool)(**kwargs)


def test_todo_next_waits_for_named_prerequisite(tmp_path: Path) -> None:
    root = str(tmp_path)
    goal = _call(milknado_todo_add, description="goal", kind="goal", project_root=root)
    prerequisite = _call(
        milknado_todo_add,
        description="first",
        parent_id=goal["id"],
        project_root=root,
    )
    dependent = _call(
        milknado_todo_add,
        description="second",
        parent_id=goal["id"],
        prereqs=[prerequisite["id"]],
        project_root=root,
    )

    first = _call(milknado_todo_next, project_root=root)
    assert first["id"] == prerequisite["id"]
    _call(
        milknado_todo_set_status,
        node_id=prerequisite["id"],
        status="done",
        project_root=root,
    )
    second = _call(milknado_todo_next, project_root=root)
    assert second["id"] == dependent["id"]
