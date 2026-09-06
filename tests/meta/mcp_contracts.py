from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from fastmcp import Client
from fastmcp.client.client import CallToolResult
from mcp.types import Tool

from milknado.domains.common import NodeStatus, RunResult
from milknado.mcp._core import mcp
from milknado.project import open_project


def _register_tools() -> None:
    from milknado.mcp import github, node, ralph, rebalance, run, server, todo, todo_mutate, wiki

    _ = (github, node, ralph, rebalance, run, server, todo, todo_mutate, wiki)


_register_tools()


def call_tool(
    name: str, arguments: dict[str, object], *, raise_on_error: bool = True
) -> CallToolResult:
    async def invoke() -> CallToolResult:
        async with Client(mcp) as client:
            return await client.call_tool(name, arguments, raise_on_error=raise_on_error)

    return asyncio.run(invoke())


def registered_tools() -> dict[str, Tool]:
    async def list_tools() -> list[Tool]:
        async with Client(mcp) as client:
            return await client.list_tools()

    return {tool.name: tool for tool in asyncio.run(list_tools())}


def payload(result: CallToolResult) -> dict[str, object]:
    assert result.structured_content is not None
    return cast(dict[str, object], result.structured_content)


def rows(result: CallToolResult) -> list[dict[str, object]]:
    return cast(list[dict[str, object]], payload(result)["result"])


def add_node(root: str, **values: object) -> dict[str, object]:
    return payload(call_tool("milknado_todo_add", {**values, "project_root": root}))


def write_custom_flavor(root: Path) -> None:
    _ = (root / "milknado.toml").write_text(
        """[milknado.flavor.custom]
quality_gates = []
""",
        encoding="utf-8",
    )


def seed_status_project(root: Path) -> dict[str, int]:
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
