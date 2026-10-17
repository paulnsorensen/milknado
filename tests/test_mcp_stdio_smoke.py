"""Smoke-test the installed ``milknado-mcp`` console script over real stdio.

Every other MCP test calls tools in process. These tests start the shipped entry
point as a child process, as every install channel does, and talk MCP over its
stdin and stdout. They use no POSIX-only API, so no platform skip applies.

The fastmcp client tolerates non-JSON stdout lines, so only the raw handshake test
catches a server that writes to stdout. Keep both tests.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import psutil
from fastmcp import Client
from fastmcp.client.transports import StdioTransport
from mcp.types import LATEST_PROTOCOL_VERSION, JSONRPCMessage, JSONRPCResponse

from milknado.mcp._core import mcp as milknado_server
from tests.meta.mcp_contracts import payload, registered_tools

TIMEOUT_S = 30.0


@dataclass(frozen=True)
class _Session:
    tool_names: set[str]
    added: dict[str, object]
    read: dict[str, object]
    servers: list[psutil.Process]


def server_command() -> str:
    scripts = Path(sys.executable).parent
    command = shutil.which("milknado-mcp", path=str(scripts))
    assert command is not None, f"milknado-mcp console script is missing from {scripts}"
    return command


def server_env(project: Path) -> dict[str, str]:
    # The SDK forwards only a safe subset of the parent environment by default. Pass
    # the full test environment so XDG isolation and the agent-CLI guard reach the child.
    # FastMCP checks PyPI for updates at startup; turn that off to keep the test offline.
    return {
        **os.environ,
        "MILKNADO_PROJECT_ROOT": str(project),
        "FASTMCP_CHECK_FOR_UPDATES": "off",
    }


def _server_processes(command: str) -> list[psutil.Process]:
    found: list[psutil.Process] = []
    for child in psutil.Process().children(recursive=True):
        with contextlib.suppress(psutil.NoSuchProcess):
            if command in child.cmdline():
                found.append(child)
    return found


async def _add_and_read_node(transport: StdioTransport, command: str) -> _Session:
    async with Client(transport, timeout=TIMEOUT_S, init_timeout=TIMEOUT_S) as client:
        servers = _server_processes(command)
        names = {tool.name for tool in await client.list_tools()}
        added = payload(await client.call_tool("milknado_todo_add", {"description": "smoke"}))
        read = payload(await client.call_tool("milknado_get_node", {"node_id": added["id"]}))
    return _Session(names, added, read, servers)


def test_stdio_server_lists_tools_and_round_trips_a_node(tmp_path: Path) -> None:
    project, elsewhere = tmp_path / "project", tmp_path / "elsewhere"
    project.mkdir()
    elsewhere.mkdir()
    command = server_command()
    transport = StdioTransport(
        command, [], env=server_env(project), cwd=str(elsewhere), keep_alive=False
    )

    session = asyncio.run(asyncio.wait_for(_add_and_read_node(transport, command), 2 * TIMEOUT_S))

    assert session.tool_names == set(registered_tools())
    assert session.added["description"] == "smoke"
    assert session.read == {**session.added, "parent_id": None, "prerequisite_ids": []}
    assert (project / ".milknado" / "milknado.db").is_file()
    assert not (elsewhere / ".milknado").exists()
    assert session.servers, "no milknado-mcp child process ran during the session"
    _, alive = psutil.wait_procs(session.servers, timeout=TIMEOUT_S)
    assert not alive, f"milknado-mcp outlived its client: {alive}"


def test_stdio_server_answers_handshake_and_exits_on_stdin_close(tmp_path: Path) -> None:
    initialize = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": LATEST_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "stdio-smoke", "version": "0"},
        },
    }
    initialized = {"jsonrpc": "2.0", "method": "notifications/initialized"}

    done = subprocess.run(
        [server_command()],
        input="".join(f"{json.dumps(message)}\n" for message in (initialize, initialized)),
        capture_output=True,
        text=True,
        env=server_env(tmp_path),
        cwd=tmp_path,
        timeout=TIMEOUT_S,
        check=False,
    )

    assert done.returncode == 0, done.stderr
    messages = [JSONRPCMessage.model_validate_json(line).root for line in done.stdout.splitlines()]
    assert len(messages) == 1, done.stdout
    response = messages[0]
    assert isinstance(response, JSONRPCResponse)
    assert response.id == 1
    assert response.result["protocolVersion"] == LATEST_PROTOCOL_VERSION
    assert response.result["serverInfo"]["name"] == milknado_server.name
