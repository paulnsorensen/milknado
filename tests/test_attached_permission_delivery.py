from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from threading import Event, Thread
from typing import cast

import pytest

from milknado.app.run import ExecutionSnapshot, build_execution_controller
from milknado.domains.common import FlavorOverride, Gate, MilknadoConfig
from milknado.domains.graph import MikadoGraph
from tests.execution_session_fixtures import build_graph as _build_graph
from tests.execution_session_fixtures import init_repo as _init_repo
from tests.worker_fixtures import install_worker_command

_WORKERS = {
    "claude": r"""
import json, os, sys
from pathlib import Path
for raw in sys.stdin:
    payload = json.loads(raw)
    if payload.get("type") != "user":
        continue
    print(
        json.dumps({"type": "system", "subtype": "init", "session_id": "claude-session"}),
        flush=True,
    )
    print(json.dumps({"type": "control_request", "request_id": "permission-write",
        "request": {"subtype": "can_use_tool", "tool_name": "Write",
        "input": {"file_path": "allowed.txt", "content": "approved"}}}), flush=True)
    for response in sys.stdin:
        value = json.loads(response)
        if value.get("type") != "control_response":
            continue
        request = value["response"]
        with Path(os.environ["VENDOR_PATH"]).open("a", encoding="utf-8") as output:
            output.write(
                json.dumps({
                    "request_id": request["request_id"],
                    "behavior": request["response"]["behavior"],
                })
                + "\n"
            )
        Path("guidance.txt").write_text("approved", encoding="utf-8")
        print(json.dumps({"type": "result", "subtype": "success",
            "result": "<promise>MILKNADO_NODE_COMPLETE</promise>"}), flush=True)
        break
    break
""",
    "omp": r"""
import json, os
from pathlib import Path
for raw in __import__("sys").stdin:
    payload = json.loads(raw)
    if payload.get("type") != "prompt":
        continue
    print(json.dumps({"type": "response", "id": payload["id"], "command": "prompt",
        "success": True}), flush=True)
    print(json.dumps({"type": "extension_ui_request", "id": "permission-1",
        "method": "confirm", "title": "Apply edit?"}), flush=True)
    for response in __import__("sys").stdin:
        value = json.loads(response)
        if value.get("type") != "extension_ui_response":
            continue
        with Path(os.environ["VENDOR_PATH"]).open("a", encoding="utf-8") as output:
            output.write(
                json.dumps({"request_id": value["id"], "confirmed": value["confirmed"]})
                + "\n"
            )
        Path("guidance.txt").write_text("approved", encoding="utf-8")
        print(json.dumps({"type": "agent_end", "isTerminal": True,
            "messages": [{
                "role": "assistant",
                "content": [{
                    "type": "text",
                    "text": "<promise>MILKNADO_NODE_COMPLETE</promise>",
                }],
            }]}),
              flush=True)
        break
    break
""",
    "codex": r"""
import json, os
from pathlib import Path
for raw in __import__("sys").stdin:
    value = json.loads(raw)
    method = value.get("method")
    with open(os.environ["VENDOR_PATH"] + ".log", "a", encoding="utf-8") as log:
        log.write(f"{method!r}\n")
    if method == "initialize":
        print(json.dumps({"id": value["id"], "result": {"userAgent": "codex"}}), flush=True)
    elif method == "thread/start":
        print(json.dumps({"id": value["id"], "result": {"thread": {"id": "thread-1",
            "sessionId": "session-1"}}}), flush=True)
    elif method == "turn/start":
        print(json.dumps({"id": value["id"], "result": {"turn": {"id": "turn-1",
            "status": "inProgress"}}}), flush=True)
        print(json.dumps({"id": 42, "method": "item/commandExecution/requestApproval",
            "params": {
                "threadId": "thread-1", "turnId": "turn-1", "command": "write file"
            }
        }), flush=True)
    elif "id" in value and value["id"] == 42 and "result" in value:
        with Path(os.environ["VENDOR_PATH"]).open("a", encoding="utf-8") as output:
            output.write(json.dumps({"request_id": value["id"], "result": value["result"]}) + "\n")
        Path("guidance.txt").write_text("approved", encoding="utf-8")
        print(
            json.dumps({"method": "serverRequest/resolved", "params": {"requestId": 42}}),
            flush=True,
        )
        print(json.dumps({"method": "turn/completed",
            "params": {"turn": {
                "id": "turn-1", "status": "completed", "items": [{
                    "type": "agentMessage",
                    "text": "<promise>MILKNADO_NODE_COMPLETE</promise>",
                }]
            }}
        }),
              flush=True)
        break
""",
}

_ATTACHED = r"""
import sys
from pathlib import Path
from milknado.app.watch import AttachedWatchSource, WatchSnapshotSource, graph_command_admitter
from milknado.domains.common import SessionInput
from milknado.domains.graph import MikadoGraph
repo, db, run_id, permission_id, command_id, action = sys.argv[1:]
graph = MikadoGraph(Path(db))
try:
    source = AttachedWatchSource(
        WatchSnapshotSource(Path(repo), Path(db)), graph_command_admitter(graph)
    )
    command = SessionInput(action=action, request_id=permission_id, command_id=command_id)
    first = source.session_input(run_id, command)
    second = source.session_input(run_id, command)
    stale = source.session_input(
        run_id, SessionInput(action=action, request_id="99/" + permission_id.split("/", 1)[-1],
                             command_id=command_id + "-stale")
    )
    print(f"{int(first)} {int(second)} {int(stale)}", flush=True)
    raise SystemExit(0 if first and second and not stale else 1)
finally:
    graph.close()
"""


def _admit(args: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", _ATTACHED, *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )


def _worker_for_family(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, family: str, source: Path
) -> str:
    if family == "codex":
        bindir = tmp_path / "codex-bin"
        bindir.mkdir()
        shim = bindir / "codex"
        _ = shim.write_text(
            f'#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(source))} "$@"\n',
            encoding="utf-8",
        )
        shim.chmod(0o755)
        monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")
        return "codex"
    return install_worker_command(
        tmp_path / f"{family}-bin",
        monkeypatch,
        agent=family,
        script=f'exec {shlex.quote(sys.executable)} {shlex.quote(str(source))} "$@"\n',
    )


def _controller(owner_graph: MikadoGraph, repo: Path, worker: str):
    config = MilknadoConfig(
        execution_agent=worker,
        flavors={"implement": FlavorOverride(review=False)},
        quality_gates=(Gate(command="true"),),
        worktree_pattern="milknado-wt-{node_id}-{slug}",
        concurrency_limit=1,
        project_root=repo,
        db_path=repo / ".milknado" / "graph.db",
    )
    return build_execution_controller(owner_graph, config, repo)


def _observer(ready: Event, delivered: Event, run_ids: list[str]):
    def observe(snapshot: ExecutionSnapshot) -> None:
        runs = (*snapshot.active_runs, *snapshot.terminal_runs)
        for run in snapshot.active_runs:
            if run.session.permissions:
                run_ids.append(run.run_id)
                ready.set()
        if any(
            event.kind == "user"
            and event.action in {"approve", "deny"}
            and event.state == "delivered"
            and event.event_id.endswith("command-1")
            for run in runs
            for event in run.session.events
        ):
            delivered.set()

    return observe


def _wait_for_receipt(graph: MikadoGraph, vendor_path: Path, family: str, decision: str):
    deadline = time.monotonic() + 10
    while not vendor_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert vendor_path.exists()
    records = [
        cast(dict[str, object], json.loads(line))
        for line in vendor_path.read_text(encoding="utf-8").splitlines()
    ]
    expected_request = {"claude": "permission-write", "omp": "permission-1", "codex": 42}[family]
    assert len(records) == 1
    assert records[0]["request_id"] == expected_request
    expected = {
        "claude": "allow" if decision == "approve" else "deny",
        "omp": decision == "approve",
        "codex": "accept" if decision == "approve" else "decline",
    }[family]
    field = {"claude": "behavior", "omp": "confirmed", "codex": "result"}[family]
    value = records[0][field]
    if family == "codex":
        assert isinstance(value, dict)
        assert value["decision"] == expected
    else:
        assert value == expected
    deadline = time.monotonic() + 10
    while (history := graph.commands.history("command-1")) and time.monotonic() < deadline:
        if [receipt.status for receipt in history] == ["queued", "submitted", "delivered"]:
            break
        time.sleep(0.01)
    assert [receipt.status for receipt in history] == ["queued", "submitted", "delivered"]
    assert len(vendor_path.read_text(encoding="utf-8").splitlines()) == 1


@pytest.mark.parametrize(
    ("family", "decision"),
    (
        (family, decision)
        for family in ("claude", "codex", "omp")
        for decision in ("approve", "deny")
    ),
)
def test_attached_permission_reaches_existing_owner_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, family: str, decision: str
) -> None:
    repo = _init_repo(tmp_path)
    owner_graph = _build_graph(repo)
    vendor_path = tmp_path / "vendor.txt"
    source = tmp_path / f"{family}.py"
    _ = source.write_text(_WORKERS[family], encoding="utf-8")
    monkeypatch.setenv("VENDOR_PATH", str(vendor_path))
    controller = _controller(
        owner_graph, repo, _worker_for_family(tmp_path, monkeypatch, family, source)
    )
    ready, delivered = Event(), Event()
    run_ids: list[str] = []
    unsubscribe = controller.subscribe(_observer(ready, delivered, run_ids))
    result: list[object] = []
    runner = Thread(target=lambda: result.append(controller.run(feature_branch="feature")))
    runner.start()
    graph = MikadoGraph(repo / ".milknado" / "graph.db")
    try:
        assert ready.wait(10)
        run_id = run_ids[0]
        permission_id = controller.snapshot().active_runs[0].session.permissions[0].event_id
        child = _admit(
            (
                str(repo),
                str(repo / ".milknado" / "graph.db"),
                run_id,
                permission_id,
                "command-1",
                decision,
            )
        )
        assert child.returncode == 0, child.stderr
        assert child.stdout.strip() == "1 1 0"
        assert delivered.wait(10)
        _wait_for_receipt(graph, vendor_path, family, decision)
        runner.join(10)
        assert not runner.is_alive()
        assert len(result) == 1
    finally:
        unsubscribe()
        if runner.is_alive():
            controller.stop_scheduling()
            for run in controller.snapshot().active_runs:
                _ = controller.force_stop(run.run_id)
            runner.join(10)
        graph.close()
        owner_graph.close()
