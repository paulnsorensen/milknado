"""Isolated inline retries run in an unoccupied worktree and keep failed work."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

from milknado.domains.common import NodeStatus, WorktreeMode
from milknado.mcp._core import NodeSummary, RunDict
from milknado.mcp.run import milknado_run_inline, milknado_run_inline_start
from milknado.mcp.server import open_graph
from milknado.mcp.todo_mutate import milknado_todo_add
from tests.test_run_inline_worktree import (
    _call,  # pyright: ignore[reportPrivateUsage]
    _init_repo,  # pyright: ignore[reportPrivateUsage]
    _wait_for_terminal,  # pyright: ignore[reportPrivateUsage]
)


def _fail_once_then_pass(root: Path, worker_stub: Callable[[str], str]) -> str:
    marker = root.parent / "first-attempt-ran"
    script = (
        f"if [ -e {marker} ]; then echo second > second.txt; exit 0; fi; "
        f"touch {marker}; echo wip > unlanded.txt; exit 1"
    )
    return worker_stub(f"sh -c '{script}'")


def _run_attempt(root: Path, node_id: int, worker_cmd: str, asynchronous: bool) -> RunDict:
    arguments = {
        "node_id": node_id,
        "worker_cmd": worker_cmd,
        "worktree": WorktreeMode.ISOLATE,
        "merge_back": False,
        "project_root": str(root),
    }
    if asynchronous:
        started = cast(RunDict, _call(milknado_run_inline_start, **arguments))
        return _wait_for_terminal(cast(str, started["run_id"]), str(root))
    return cast(RunDict, _call(milknado_run_inline, **arguments))


@pytest.mark.parametrize("asynchronous", [False, True])
def test_isolated_retry_uses_unoccupied_worktree_and_keeps_failed_work(
    tmp_path: Path, worker_stub: Callable[[str], str], asynchronous: bool
) -> None:
    root = tmp_path / "repo"
    _init_repo(root)
    _ = (root / "milknado.toml").write_text("[milknado]\nmax_node_attempts = 2\n")
    task = cast(
        NodeSummary,
        _call(milknado_todo_add, description="retry", kind="task", project_root=str(root)),
    )
    node_id = int(task["id"])
    worker_cmd = _fail_once_then_pass(root, worker_stub)

    first = _run_attempt(root, node_id, worker_cmd, asynchronous)
    assert first["exit_code"] == 1
    first_worktree = next(root.glob("milknado-*-retry"))
    assert (first_worktree / "unlanded.txt").read_text().strip() == "wip"

    second = _run_attempt(root, node_id, worker_cmd, asynchronous)

    assert second["exit_code"] == 0
    assert (first_worktree / "unlanded.txt").read_text().strip() == "wip"
    assert not (first_worktree / "second.txt").exists()
    retry_worktrees = [path for path in root.glob("milknado-*") if path != first_worktree]
    assert len(retry_worktrees) == 1
    assert (retry_worktrees[0] / "second.txt").read_text().strip() == "second"
    graph, _ = open_graph(root)
    try:
        node = graph.get_node(node_id)
        assert node is not None and node.status is NodeStatus.DONE
        assert graph.task_attempts.failed_count(node_id) == 1
    finally:
        graph.close()
