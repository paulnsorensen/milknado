from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from milknado.mcp_ralph import milknado_ralph_run_poll, milknado_ralph_run_start
from milknado.mcp_server import open_graph
from milknado.mcp_todo import milknado_todo_add, milknado_todo_tree

# A stub runner standing in for `python -m milknado._ralph_node_runner`: it honors
# the same argv the MCP tool appends and writes the requested terminal state,
# letting us exercise the start->spawn->poll plumbing without a real ralph loop.
_STUB_RUNNER = """
import argparse, json
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("--node-id", type=int)
p.add_argument("--project-root")
p.add_argument("--run-id")
p.add_argument("--state-path")
p.add_argument("--timeout")
a = p.parse_args()
sp = Path(a.state_path)
state = json.loads(sp.read_text())
state.update(status="{status}", rebased={rebased}, detail={detail})
state["ended_at"] = "2026-01-01T00:00:00+00:00"
sp.write_text(json.dumps(state))
"""


def _call(tool, **kwargs):
    fn = getattr(tool, "fn", tool)
    return fn(**kwargs)


def _stub_runner_cmd(tmp_path: Path, *, status: str, rebased: bool, detail: str = "None") -> str:
    script = tmp_path / f"stub_runner_{status}.py"
    script.write_text(_STUB_RUNNER.format(status=status, rebased=rebased, detail=detail))
    return f"{sys.executable} {script}"


def _wait_for_terminal(run_id: str, root: str, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = _call(milknado_ralph_run_poll, run_id=run_id, project_root=root)
        if last["status"] in ("done", "failed"):
            return last
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} did not finish; last={last}")


def test_start_returns_run_id_and_writes_running_state(tmp_path: Path) -> None:
    root = str(tmp_path)
    task = _call(milknado_todo_add, description="ralph-node", kind="task", project_root=root)
    # A do-nothing runner so the state stays "running" right after start.
    noop = f"{sys.executable} -c pass"
    started = _call(
        milknado_ralph_run_start,
        node_id=task["id"],
        runner_cmd=noop,
        project_root=root,
    )
    assert started["status"] == "running"
    assert started["run_id"].startswith(f"node-{task['id']}-")
    assert Path(started["state_path"]).exists()


def test_poll_reads_done_after_runner_finishes(tmp_path: Path) -> None:
    root = str(tmp_path)
    task = _call(milknado_todo_add, description="ralph-done", kind="task", project_root=root)
    started = _call(
        milknado_ralph_run_start,
        node_id=task["id"],
        runner_cmd=_stub_runner_cmd(tmp_path, status="done", rebased=True),
        project_root=root,
    )
    final = _wait_for_terminal(started["run_id"], root)
    assert final["status"] == "done"
    assert final["rebased"] is True


def test_poll_reads_failed_with_detail(tmp_path: Path) -> None:
    root = str(tmp_path)
    task = _call(milknado_todo_add, description="ralph-fail", kind="task", project_root=root)
    started = _call(
        milknado_ralph_run_start,
        node_id=task["id"],
        runner_cmd=_stub_runner_cmd(tmp_path, status="failed", rebased=False, detail="'boom'"),
        project_root=root,
    )
    final = _wait_for_terminal(started["run_id"], root)
    assert final["status"] == "failed"
    assert final["rebased"] is False
    assert final["detail"] == "boom"


def test_start_refuses_when_node_already_running(tmp_path: Path) -> None:
    root = str(tmp_path)
    task = _call(milknado_todo_add, description="busy", kind="task", project_root=root)
    graph, _cfg = open_graph(tmp_path)
    try:
        graph.mark_running(task["id"])
    finally:
        graph.close()
    with pytest.raises(ValueError, match="already running"):
        _call(
            milknado_ralph_run_start,
            node_id=task["id"],
            runner_cmd=f"{sys.executable} -c pass",
            project_root=root,
        )


def test_start_unknown_node_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not found"):
        _call(milknado_ralph_run_start, node_id=99, project_root=str(tmp_path))


def test_poll_unknown_run_id_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not found"):
        _call(
            milknado_ralph_run_poll,
            run_id="node-1-20260101T000000Z-abcd",
            project_root=str(tmp_path),
        )


def test_poll_rejects_malformed_run_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="invalid run_id"):
        _call(milknado_ralph_run_poll, run_id="../etc/passwd", project_root=str(tmp_path))


def test_node_status_unaffected_by_start_until_runner_acts(tmp_path: Path) -> None:
    """start does not pre-mark RUNNING; the detached runner owns node status.
    With a no-op runner the node stays pending."""
    root = str(tmp_path)
    task = _call(milknado_todo_add, description="pend", kind="task", project_root=root)
    _call(
        milknado_ralph_run_start,
        node_id=task["id"],
        runner_cmd=f"{sys.executable} -c pass",
        project_root=root,
    )
    tree = _call(milknado_todo_tree, project_root=root)
    assert tree[0]["status"] == "pending"
