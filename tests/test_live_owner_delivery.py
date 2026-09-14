from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import cast

import pytest

from milknado.app.watch import graph_command_admitter
from milknado.domains.common import SessionInput
from milknado.domains.dispatch import reconcile_orphaned_runs
from milknado.domains.graph import MikadoGraph
from milknado.domains.graph.commands import GraphCommand, OwnerCapabilities
from tests.attached_owner_delivery_fixtures import AttachedCommand, admit_from_process
from tests.execution_session_fixtures import build_graph, init_repo
from tests.live_owner_delivery_fixtures import kill_owner, kill_vendor, owner, wait, worker


def _history(graph: MikadoGraph, command_id: str) -> list[str]:
    return [receipt.status for receipt in graph.commands.history(command_id)]


@dataclass(frozen=True)
class _ReplacementCase:
    tmp_path: Path
    monkeypatch: pytest.MonkeyPatch
    graph: MikadoGraph
    repo: Path
    db: Path
    mode: str
    run_id: str
    old_capabilities: OwnerCapabilities
    expected: list[str]
    old_command: GraphCommand


def _wait_history(graph: MikadoGraph, command_id: str, expected: list[str]) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if _history(graph, command_id) == expected:
            return
        time.sleep(0.01)
    assert _history(graph, command_id) == expected


def _replacement_assertions(
    case: _ReplacementCase, wire: Path, marker: Path, barrier: Path
) -> None:
    wait(case.tmp_path / "new.ready")
    run_id = (case.tmp_path / "new.ready").read_text(encoding="utf-8")
    capabilities = case.graph.commands.capabilities(run_id)
    assert capabilities is not None
    assert run_id != case.run_id
    assert capabilities.owner_incarnation != case.old_capabilities.owner_incarnation
    assert capabilities.invocation_id != case.old_capabilities.invocation_id
    stale = graph_command_admitter(case.graph)(
        run_id,
        SessionInput(
            action="follow_up",
            text="stale displayed fence",
            owner_incarnation=case.old_capabilities.owner_incarnation,
            invocation_id=case.old_capabilities.invocation_id,
        ),
    )
    assert stale is False
    retry = admit_from_process(
        AttachedCommand(case.repo, case.db, run_id, f"old-{case.mode}", "old stable command")
    )
    assert retry.returncode != 0
    fresh = admit_from_process(
        AttachedCommand(case.repo, case.db, run_id, f"new-{case.mode}", "new replacement command")
    )
    assert fresh.returncode == 0, fresh.stderr
    barrier_result = admit_from_process(
        AttachedCommand(case.repo, case.db, run_id, f"barrier-{case.mode}", "wire barrier")
    )
    assert barrier_result.returncode == 0, barrier_result.stderr
    wait(marker)
    wait(barrier)
    _wait_history(
        case.graph,
        f"new-{case.mode}",
        ["queued", "submitted", "delivered"],
    )
    _wait_history(case.graph, f"barrier-{case.mode}", ["queued", "submitted", "delivered"])
    frames = [cast(dict[str, str], json.loads(line)) for line in wire.read_text().splitlines()]
    assert [frame["text"] for frame in frames] == ["new replacement command", "wire barrier"]
    assert _history(case.graph, f"old-{case.mode}") == case.expected
    expiry = datetime.fromisoformat(case.old_command.expires_at) + timedelta(seconds=1)
    _ = case.graph.commands.expire(now=expiry.isoformat())
    assert _history(case.graph, f"old-{case.mode}") == [*case.expected, "expired"]
    stored_old = case.graph.commands.command(f"old-{case.mode}")
    assert stored_old is not None
    assert stored_old.expires_at == case.old_command.expires_at
    assert _history(case.graph, f"new-{case.mode}")[-1] == "delivered"


def _run_replacement(case: _ReplacementCase) -> subprocess.Popen[str]:
    worker_command, wire, marker, barrier = worker(
        case.tmp_path, case.monkeypatch, "normal", "new"
    )
    ready = case.tmp_path / "new.ready"
    replacement = owner(case.repo, case.db, worker_command, ready)
    try:
        _replacement_assertions(case, wire, marker, barrier)
    finally:
        try:
            if replacement.poll() is None:
                kill_owner(replacement)
        finally:
            kill_vendor(case.tmp_path / "new.pid")
    return replacement


@dataclass(frozen=True)
class _OldCase:
    repo: Path
    db: Path
    graph: MikadoGraph
    owner: subprocess.Popen[str]
    ready: Path
    wire: Path
    marker: Path
    mode: str


def _admit_old(case: _OldCase) -> tuple[str, OwnerCapabilities, list[str], GraphCommand]:
    wait(case.ready)
    run_id = case.ready.read_text(encoding="utf-8")
    capabilities = case.graph.commands.capabilities(run_id)
    assert capabilities is not None
    if case.mode == "before":
        os.kill(case.owner.pid, signal.SIGSTOP)
        stopped_pid, stopped_status = os.waitpid(case.owner.pid, os.WUNTRACED)
        assert stopped_pid == case.owner.pid
        assert os.WIFSTOPPED(stopped_status)
        assert os.WSTOPSIG(stopped_status) == signal.SIGSTOP
    admitted = admit_from_process(
        AttachedCommand(case.repo, case.db, run_id, f"old-{case.mode}", "old stable command")
    )
    assert admitted.returncode == 0, admitted.stderr
    if case.mode == "after":
        wait(case.marker)
    kill_owner(case.owner)
    expected, command = _recover_task(case.graph, case.mode, f"old-{case.mode}")
    assert _history(case.graph, f"old-{case.mode}") == expected
    if case.mode == "before":
        assert not case.wire.exists()
    else:
        frames = [
            cast(dict[str, str], json.loads(line)) for line in case.wire.read_text().splitlines()
        ]
        assert [frame["text"] for frame in frames] == ["old stable command"]
    return run_id, capabilities, expected, command


def _cleanup_processes(
    owner: subprocess.Popen[str],
    replacement: subprocess.Popen[str] | None,
    old_pid: Path,
    new_pid: Path,
) -> None:
    try:
        if owner.poll() is None:
            kill_owner(owner)
    finally:
        try:
            if replacement is not None and replacement.poll() is None:
                kill_owner(replacement)
        finally:
            try:
                kill_vendor(old_pid)
            finally:
                kill_vendor(new_pid)


def _recover_task(
    graph: MikadoGraph, mode: str, command_id: str
) -> tuple[list[str], GraphCommand]:
    _ = reconcile_orphaned_runs(graph)
    command = graph.commands.command(command_id)
    assert command is not None
    graph.mark_pending(command.node_id)
    expected = ["queued"] if mode == "before" else ["queued", "submitted"]
    return expected, command


@pytest.mark.parametrize("mode", ("before", "after"))
def test_process_owner_failure_fences_and_replaces_without_old_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    repo = init_repo(tmp_path)
    graph = build_graph(repo)
    graph.close()
    db = repo / ".milknado" / "graph.db"
    worker_command, old_wire, old_marker, _ = worker(tmp_path, monkeypatch, mode, "old")
    old_pid = tmp_path / "old.pid"
    ready = tmp_path / "old.ready"
    owner_process = owner(repo, db, worker_command, ready)
    replacement = None
    graph = MikadoGraph(db)
    try:
        run_id, old_capabilities, expected, old_command = _admit_old(
            _OldCase(
                repo=repo,
                db=db,
                graph=graph,
                owner=owner_process,
                ready=ready,
                wire=old_wire,
                marker=old_marker,
                mode=mode,
            )
        )

        replacement = _run_replacement(
            _ReplacementCase(
                tmp_path=tmp_path,
                monkeypatch=monkeypatch,
                graph=graph,
                repo=repo,
                db=db,
                mode=mode,
                run_id=run_id,
                old_capabilities=old_capabilities,
                expected=expected,
                old_command=old_command,
            )
        )
    finally:
        try:
            _cleanup_processes(owner_process, replacement, old_pid, tmp_path / "new.pid")
        finally:
            graph.close()
