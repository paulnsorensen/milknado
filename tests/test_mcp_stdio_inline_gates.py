"""Inline dispatch must run the project's quality gates over real MCP stdio (#498).

The installed ``milknado-mcp`` console script runs as a child process. A fake claude
worker exits 0 with a completion result frame and writes a file. A failing gate must
keep the node out of DONE and keep the change off the target branch.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import time
from pathlib import Path

import pytest
from fastmcp import Client
from fastmcp.client.transports import StdioTransport

from tests.meta.mcp_contracts import payload
from tests.test_mcp_stdio_smoke import TIMEOUT_S, server_command, server_env
from tests.worker_fixtures import install_worker_command

_WORKER_SCRIPT = (
    "cat >/dev/null\n"
    "printf 'worker change\\n' > worker-change.txt\n"
    f"echo '{json.dumps({'type': 'result', 'subtype': 'success', 'result': 'done'})}'\n"
)
_FILE = "worker-change.txt"


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def _project(root: Path, gate_command: str) -> None:
    root.mkdir()
    _ = _git(root, "init", "-b", "work")
    _ = _git(root, "config", "user.email", "t@t.com")
    _ = _git(root, "config", "user.name", "T")
    _ = (root / ".gitignore").write_text(".milknado/\nmilknado-*/\n")
    _ = (root / "milknado.toml").write_text(
        f"[milknado]\nquality_gates = [{json.dumps(gate_command)}]\n"
    )
    _ = _git(root, "add", ".")
    _ = _git(root, "commit", "-m", "init")


async def _poll_terminal(
    client: Client[StdioTransport], run_id: str, root: Path
) -> dict[str, object]:
    deadline = time.monotonic() + TIMEOUT_S
    while time.monotonic() < deadline:
        run = payload(
            await client.call_tool(
                "milknado_run_inline_poll", {"run_id": run_id, "project_root": str(root)}
            )
        )
        if run["status"] != "running":
            return run
        await asyncio.sleep(0.2)
    raise AssertionError(f"run {run_id} never reached a terminal status")


async def _dispatch(
    transport: StdioTransport, root: Path, worker: str, blocking: bool
) -> tuple[dict[str, object], dict[str, object]]:
    async with Client(transport, timeout=TIMEOUT_S, init_timeout=TIMEOUT_S) as client:
        added = payload(await client.call_tool("milknado_todo_add", {"description": "gate me"}))
        args: dict[str, object] = {
            "node_id": added["id"],
            "worker_cmd": worker,
            "timeout_seconds": 20,
            "project_root": str(root),
        }
        tool = "milknado_run_inline" if blocking else "milknado_run_inline_start"
        run = payload(await client.call_tool(tool, args))
        if not blocking:
            run = await _poll_terminal(client, str(run["run_id"]), root)
        node = payload(
            await client.call_tool(
                "milknado_get_node", {"node_id": added["id"], "project_root": str(root)}
            )
        )
    return run, node


def _run_inline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gate_command: str, blocking: bool
) -> tuple[dict[str, object], dict[str, object], Path]:
    root = tmp_path / "project"
    _project(root, gate_command)
    worker = install_worker_command(
        tmp_path / "bin", monkeypatch, agent="claude", script=_WORKER_SCRIPT
    )
    transport = StdioTransport(
        server_command(), [], env=server_env(root), cwd=str(tmp_path), keep_alive=False
    )
    run, node = asyncio.run(
        asyncio.wait_for(_dispatch(transport, root, worker, blocking), 3 * TIMEOUT_S)
    )
    return run, node, root


@pytest.mark.parametrize("blocking", [True, False], ids=["inline", "start-poll"])
def test_failing_gate_blocks_done_and_merge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, blocking: bool
) -> None:
    run, node, root = _run_inline(tmp_path, monkeypatch, "echo gate-boom; exit 1", blocking)

    assert node["status"] != "done", (run, node)
    assert run["status"] == "failed", run
    assert run["exit_code"] == 0
    assert "quality gate failed" in str(run["error"]), run
    assert "gate-boom" in str(run["error"]), run
    assert not (root / _FILE).exists()
    assert "init" == _git(root, "log", "-1", "--format=%s", "work")


@pytest.mark.parametrize("blocking", [True, False], ids=["inline", "start-poll"])
def test_passing_gate_marks_done_and_lands_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, blocking: bool
) -> None:
    run, node, root = _run_inline(tmp_path, monkeypatch, "true", blocking)

    assert node["status"] == "done", (run, node)
    assert run["status"] == "done", run
    assert run["rebased"] is True
    assert (root / _FILE).read_text() == "worker change\n"
