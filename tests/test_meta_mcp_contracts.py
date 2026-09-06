from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest
from fastmcp import Client
from fastmcp.client.client import CallToolResult
from mcp.types import Tool

from milknado.domains.common import NodeStatus, RunResult
from milknado.domains.common.types import NodeKind
from milknado.mcp._core import RunDict, mcp
from milknado.project import open_project


def _register_tools() -> None:
    from milknado.mcp import github, node, ralph, rebalance, run, server, todo, todo_mutate, wiki

    _ = (github, node, ralph, rebalance, run, server, todo, todo_mutate, wiki)


_register_tools()


def _call(
    name: str, arguments: dict[str, object], *, raise_on_error: bool = True
) -> CallToolResult:
    async def invoke() -> CallToolResult:
        async with Client(mcp) as client:
            return await client.call_tool(name, arguments, raise_on_error=raise_on_error)

    return asyncio.run(invoke())


def _registered_tools() -> dict[str, Tool]:
    async def list_tools() -> list[Tool]:
        async with Client(mcp) as client:
            return await client.list_tools()

    return {tool.name: tool for tool in asyncio.run(list_tools())}


def _payload(result: CallToolResult) -> dict[str, object]:
    assert result.structured_content is not None
    return cast(dict[str, object], result.structured_content)


def _rows(result: CallToolResult) -> list[dict[str, object]]:
    return cast(list[dict[str, object]], _payload(result)["result"])


def test_worker_identity_is_cleared_before_each_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inner = tmp_path / "test_inner.py"
    _ = inner.write_text(
        "import os\n"
        + "def test_clean():\n"
        + "    assert [k for k in os.environ if k.startswith('MILKNADO_')] == []\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(inner),
            "-q",
            "-p",
            "no:cacheprovider",
            "-p",
            "tests.conftest",
        ],
        cwd=Path(__file__).parent.parent,
        env={
            **os.environ,
            "MILKNADO_NODE_ID": "999",
            "MILKNADO_RUN_ID": "leak",
            "MILKNADO_PROJECT_ROOT": str(tmp_path / "disposable"),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    monkeypatch.setenv("MILKNADO_NODE_ID", "explicit-node")
    assert os.environ["MILKNADO_NODE_ID"] == "explicit-node"


def test_registered_schemas_match_domain_literals() -> None:
    tools = _registered_tools()
    assert {"milknado_todo_add", "milknado_todo_set_status", "milknado_todo_tree"} <= tools.keys()
    assert {"milknado_get_node", "milknado_graph_summary", "milknado_run_list"} <= tools.keys()

    add_tool = tools["milknado_todo_add"]
    add_properties = cast(dict[str, dict[str, object]], add_tool.inputSchema["properties"])
    assert add_properties["kind"]["enum"] == [kind.value for kind in NodeKind]

    status_tool = tools["milknado_todo_set_status"]
    status_properties = cast(dict[str, dict[str, object]], status_tool.inputSchema["properties"])
    assert status_properties["status"]["enum"] == ["pending", "in_progress", "blocked", "done"]


@pytest.mark.parametrize(
    ("parent_kind", "child_kind"),
    [
        (None, "roadmap"),
        (None, "goal"),
        (None, "task"),
        ("roadmap", "goal"),
        ("goal", "goal"),
        ("goal", "task"),
        ("task", "task"),
    ],
)
def test_node_kind_hierarchy_through_transport(
    tmp_path: Path, parent_kind: str | None, child_kind: str
) -> None:
    root = str(tmp_path)
    parent_id: int | None = None
    if parent_kind is not None:
        parent = _payload(
            _call(
                "milknado_todo_add",
                {"description": "parent", "kind": parent_kind, "project_root": root},
            )
        )
        parent_id = cast(int, parent["id"])

    child = _payload(
        _call(
            "milknado_todo_add",
            {
                "description": "child",
                "kind": child_kind,
                "parent_id": parent_id,
                "project_root": root,
            },
        )
    )
    assert child["kind"] == child_kind
    assert child["status"] == NodeStatus.PENDING.value


@pytest.mark.parametrize(
    ("parent_kind", "child_kind"),
    [("roadmap", "task"), ("goal", "roadmap"), ("task", "goal"), ("task", "roadmap")],
)
def test_invalid_node_kind_hierarchy_is_atomic(
    tmp_path: Path, parent_kind: str, child_kind: str
) -> None:
    root = str(tmp_path)
    parent = _payload(
        _call(
            "milknado_todo_add",
            {"description": "parent", "kind": parent_kind, "project_root": root},
        )
    )
    before = _rows(_call("milknado_todo_tree", {"project_root": root}))

    rejected = _call(
        "milknado_todo_add",
        {
            "description": "invalid child",
            "kind": child_kind,
            "parent_id": parent["id"],
            "project_root": root,
        },
        raise_on_error=False,
    )
    assert rejected.is_error
    assert getattr(rejected.content[0], "text", None) == (
        f"Error calling tool 'milknado_todo_add': kind={NodeKind(child_kind)!r} "
        f"is not a valid child of kind={NodeKind(parent_kind)!r}"
    )
    assert _rows(_call("milknado_todo_tree", {"project_root": root})) == before


def _write_custom_flavor(root: Path) -> None:
    _ = (root / "milknado.toml").write_text(
        """[milknado.flavor.custom]
quality_gates = []
""",
        encoding="utf-8",
    )


def test_task_flavors_default_and_use_project_registry(tmp_path: Path) -> None:
    _write_custom_flavor(tmp_path)
    root = str(tmp_path)

    default = _payload(
        _call("milknado_todo_add", {"description": "default", "project_root": root})
    )
    custom = _payload(
        _call(
            "milknado_todo_add",
            {"description": "custom", "flavor": "custom", "project_root": root},
        )
    )

    assert default["flavor"] == "implement"
    assert custom["flavor"] == "custom"


@pytest.mark.parametrize(
    "case",
    [
        (
            "bad goal",
            "goal",
            "custom",
            "Error calling tool 'milknado_todo_add': flavor must be None for kind=goal; "
            + "got 'custom'",
        ),
        (
            "bad task",
            "task",
            "unknown",
            "Error calling tool 'milknado_todo_add': invalid flavor 'unknown'; expected one of "
            + "['custom', 'implement', 'plate', 'prototype', 'research', "
            + "'review', 'spec', 'spike']",
        ),
    ],
)
def test_flavor_validation_is_task_only_and_atomic(
    tmp_path: Path, case: tuple[str, str, str, str]
) -> None:
    description, kind, flavor, expected = case
    _write_custom_flavor(tmp_path)
    root = str(tmp_path)
    goal = _payload(
        _call(
            "milknado_todo_add",
            {"description": "goal", "kind": "goal", "project_root": root},
        )
    )
    before = _rows(_call("milknado_todo_tree", {"project_root": root}))

    rejected = _call(
        "milknado_todo_add",
        {
            "description": description,
            "kind": kind,
            "flavor": flavor,
            "parent_id": goal["id"],
            "project_root": root,
        },
        raise_on_error=False,
    )
    assert rejected.is_error
    assert getattr(rejected.content[0], "text", None) == expected
    assert _rows(_call("milknado_todo_tree", {"project_root": root})) == before


def _seed_status_project(root: Path) -> dict[str, int]:
    project = open_project(root)
    graph = project.graph
    try:
        nodes = {
            name: graph.add_node(name)
            for name in ("pending", "running", "blocked", "failed", "done")
        }
        graph.mark_running(nodes["running"].id, run_id="run-running")
        graph.mark_blocked(nodes["blocked"].id)
        graph.mark_failed(nodes["failed"].id)
        graph.mark_running(nodes["done"].id, run_id="run-done")
        graph.mark_done(nodes["done"].id)

        started_at = datetime.now(UTC).isoformat()
        graph.start_run("run-running", nodes["running"].id, "running.log", started_at, 10)
        graph.start_run("run-failed", nodes["failed"].id, "failed.log", started_at, 10)
        _ = graph.finish_run(
            "run-failed",
            RunResult(
                status=NodeStatus.FAILED.value,
                exit_code=1,
                timed_out=False,
                ended_at=datetime.now(UTC).isoformat(),
                error="boom",
            ),
        )
    finally:
        graph.close()
    return {name: node.id for name, node in nodes.items()}


def test_read_transport_preserves_every_domain_node_status(tmp_path: Path) -> None:
    _ = _seed_status_project(tmp_path)
    root = str(tmp_path)

    tree = _rows(_call("milknado_todo_tree", {"project_root": root}))
    summary = cast(
        list[dict[str, object]],
        _payload(_call("milknado_graph_summary", {"project_root": root}))["nodes"],
    )
    assert {row["status"] for row in tree} == {status.value for status in NodeStatus}
    assert {row["status"] for row in summary} == {status.value for status in NodeStatus}

    node_id = next(row["id"] for row in tree if row["description"] == "running")
    node = _payload(_call("milknado_get_node", {"node_id": node_id, "project_root": root}))
    assert node["status"] == "running"


def test_in_progress_maps_to_running_and_failed_is_not_a_mutation(
    tmp_path: Path,
) -> None:
    root = str(tmp_path)
    task = _payload(_call("milknado_todo_add", {"description": "task", "project_root": root}))
    updated = _payload(
        _call(
            "milknado_todo_set_status",
            {"node_id": task["id"], "status": "in_progress", "project_root": root},
        )
    )
    assert updated["status"] == NodeStatus.RUNNING.value

    project = open_project(tmp_path)
    try:
        node = project.graph.get_node(cast(int, task["id"]))
        assert node is not None
        assert node.status is NodeStatus.RUNNING
    finally:
        project.graph.close()

    before = _rows(_call("milknado_todo_tree", {"project_root": root}))
    rejected = _call(
        "milknado_todo_set_status",
        {"node_id": task["id"], "status": "failed", "project_root": root},
        raise_on_error=False,
    )
    assert rejected.is_error
    assert getattr(rejected.content[0], "text", None) == (
        "1 validation error for call[milknado_todo_set_status]\n"
        "status\n"
        "  Input should be 'pending', 'in_progress', 'blocked' or 'done' "
        "[type=literal_error, input_value='failed', input_type=str]\n"
        "    For further information visit https://errors.pydantic.dev/2.13/v/literal_error"
    )
    assert _rows(_call("milknado_todo_tree", {"project_root": root})) == before


def test_run_list_uses_canonical_run_dict_schema(tmp_path: Path) -> None:
    nodes = _seed_status_project(tmp_path)
    root = str(tmp_path)

    records = _rows(_call("milknado_run_list", {"project_root": root}))
    assert {record["run_id"] for record in records} == {"run-running", "run-failed"}
    for record in records:
        assert set(record) == set(RunDict.__annotations__)
        assert record["timed_out"] is False
        assert all(
            record[field] is None
            for field in ("detail", "rebased", "pid", "summary", "result", "worktree_preserved")
        )

    by_run_id = {record["run_id"]: record for record in records}
    assert by_run_id["run-running"]["status"] == NodeStatus.RUNNING.value
    assert by_run_id["run-running"]["exit_code"] is None
    assert by_run_id["run-running"]["node_id"] == nodes["running"]
    assert by_run_id["run-running"]["log_path"] == "running.log"
    failed = by_run_id["run-failed"]
    assert failed["status"] == NodeStatus.FAILED.value
    assert failed["node_id"] == nodes["failed"]
    assert failed["log_path"] == "failed.log"
    assert failed["error"] == "boom"
    assert failed["exit_code"] == 1
