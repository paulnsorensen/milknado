from __future__ import annotations

import os
import select
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from typer.testing import CliRunner

from milknado.app.controller_capability import (
    CONTROLLER_MASTER_ENV,
    consume_controller_capability,
    register_controller_master,
)
from milknado.cli import app
from milknado.domains.common import NodeKind, NodeSpec
from milknado.domains.dispatch import build_worker_env
from milknado.domains.graph import (
    GoalReviewDecision,
    GoalReviewRequest,
    MikadoGraph,
)
from milknado.loop._agent import _build_spawn_env
from milknado.loop.sessions._process import start_process
from milknado.loop.sessions._protocol import SessionProtocol

cli_runner = CliRunner()


def _pending_review(project_root: Path) -> int:
    db_path = project_root / ".milknado" / "milknado.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    graph = MikadoGraph(db_path)
    try:
        goal = graph.add_node("project", spec=NodeSpec(kind=NodeKind.GOAL))
        review = graph.request_goal_review(
            GoalReviewRequest(
                goal_id=goal.id,
                goal_revision="revision-1",
                evidence="The controller must inspect this change.",
                proposed_change="Change the top-level goal.",
                reviewer="worker-1",
            )
        )
        return review.review_id
    finally:
        graph.close()


def _invoke_human(
    project_root: Path, review_id: int, decision: str, monkeypatch: pytest.MonkeyPatch
):
    import milknado.cli.graph as cli_graph

    monkeypatch.setattr(
        cli_graph,
        "sys",
        SimpleNamespace(
            stdin=SimpleNamespace(isatty=lambda: True),
            stdout=SimpleNamespace(isatty=lambda: True),
        ),
    )
    monkeypatch.setattr(cli_graph.typer, "confirm", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(cli_graph.getpass, "getuser", lambda: "human-controller")
    return cli_runner.invoke(
        app,
        [
            "graph",
            "review",
            str(review_id),
            decision,
            "--project-root",
            str(project_root),
        ],
    )


def test_goal_review_cli_rejects_non_tty_without_controller_capability(
    tmp_path: Path,
) -> None:
    review_id = _pending_review(tmp_path)
    result = cli_runner.invoke(
        app,
        ["graph", "review", str(review_id), "accepted", "--project-root", str(tmp_path)],
    )

    assert result.exit_code != 0
    assert "interactive terminal" in result.output


@pytest.mark.parametrize("decision", ["accepted", "rejected"])
def test_controller_capability_authorizes_one_exact_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, decision: str
) -> None:
    review_id = _pending_review(tmp_path)
    monkeypatch.setenv(CONTROLLER_MASTER_ENV, "external-controller-master")
    register_controller_master(tmp_path)

    first = _invoke_human(tmp_path, review_id, decision, monkeypatch)
    second = _invoke_human(tmp_path, review_id, decision, monkeypatch)

    assert first.exit_code == 0, first.output
    assert f"{decision} by human-controller" in first.output
    assert second.exit_code != 0
    assert "controller capability" in second.output

    graph = MikadoGraph(tmp_path / ".milknado" / "milknado.db")
    try:
        record = graph.get_goal_review(review_id)
        assert record is not None
        assert record.decision is GoalReviewDecision(decision)
        assert record.decided_by == "human-controller"
    finally:
        graph.close()

    ledger = tmp_path / ".milknado" / "controller-capability.db"
    assert b"external-controller-master" not in ledger.read_bytes()
    with sqlite3.connect(ledger) as conn:
        consumed = conn.execute("SELECT COUNT(*) FROM consumed_capability").fetchone()
    assert consumed == (1,)


def test_worker_environment_strips_controller_master(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CONTROLLER_MASTER_ENV, "external-controller-master")
    worker_env = build_worker_env({CONTROLLER_MASTER_ENV: "worker-spoof"})

    assert CONTROLLER_MASTER_ENV not in worker_env


@pytest.mark.skipif(sys.platform == "win32", reason="PTY worker test requires POSIX")
def test_worker_cannot_self_approve_from_a_pty(  # noqa: PLR0915 - real PTY boundary
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    review_id = _pending_review(tmp_path)
    monkeypatch.setenv(CONTROLLER_MASTER_ENV, "external-controller-master")
    register_controller_master(tmp_path)
    env = build_worker_env(
        {
            "MILKNADO_NODE_ID": "1",
            "MILKNADO_RUN_ID": "run-1",
            "MILKNADO_PROJECT_ROOT": str(tmp_path),
        }
    )
    src_root = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = src_root + os.pathsep + env.get("PYTHONPATH", "")
    env["TERM"] = "xterm"
    master_fd, slave_fd = os.openpty()
    proc: subprocess.Popen[bytes] | None = None
    output = bytearray()
    try:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "from milknado.cli import app; app()",
                "graph",
                "review",
                str(review_id),
                "accepted",
                "--project-root",
                str(tmp_path),
            ],
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            cwd=tmp_path,
            env=env,
            close_fds=True,
        )
        os.close(slave_fd)
        slave_fd = -1
        _ = os.write(master_fd, b"y\n")
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            ready, _, _ = select.select([master_fd], [], [], 0.05)
            if ready:
                try:
                    output.extend(os.read(master_fd, 4096))
                except OSError:
                    break
            elif proc.poll() is not None:
                break
        _ = proc.wait(timeout=10)
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()
            _ = proc.wait()
        if slave_fd >= 0:
            os.close(slave_fd)
        os.close(master_fd)

    assert proc is not None
    assert proc.returncode != 0
    assert b"controller capability" in bytes(output), (
        f"returncode={proc.returncode}, output={bytes(output)!r}"
    )
    graph = MikadoGraph(tmp_path / ".milknado" / "milknado.db")
    try:
        record = graph.get_goal_review(review_id)
        assert record is not None
        assert record.decision is GoalReviewDecision.PENDING
    finally:
        graph.close()


def test_loop_agent_environment_strips_controller_master(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(CONTROLLER_MASTER_ENV, "external-controller-master")

    worker_env = _build_spawn_env(None)

    assert worker_env is not None
    assert CONTROLLER_MASTER_ENV not in worker_env


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-group test")
def test_session_environment_strips_controller_master(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(CONTROLLER_MASTER_ENV, "external-controller-master")
    protocol = cast(
        SessionProtocol,
        SimpleNamespace(
            command=(
                sys.executable,
                "-c",
                "import os; print(os.environ.get('MILKNADO_CONTROLLER_MASTER', ''))",
            )
        ),
    )

    proc = start_process(protocol, tmp_path)
    stdout, _ = proc.communicate(timeout=10)

    assert stdout == b"\n"


def test_controller_capability_rejects_missing_and_wrong_master(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert not consume_controller_capability(tmp_path, 1, "accepted")
    monkeypatch.setenv(CONTROLLER_MASTER_ENV, "registered-master")
    register_controller_master(tmp_path)
    monkeypatch.setenv(CONTROLLER_MASTER_ENV, "different-master")

    assert not consume_controller_capability(tmp_path, 1, "accepted")


def test_controller_registration_repairs_ledger_permissions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(CONTROLLER_MASTER_ENV, "registered-master")
    register_controller_master(tmp_path)
    ledger = tmp_path / ".milknado" / "controller-capability.db"
    ledger.chmod(0o644)

    register_controller_master(tmp_path)

    assert ledger.stat().st_mode & 0o777 == 0o600


def test_controller_capability_rejects_corrupt_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(CONTROLLER_MASTER_ENV, "registered-master")
    ledger = tmp_path / ".milknado" / "controller-capability.db"
    ledger.parent.mkdir()
    ledger.write_text("not sqlite", encoding="utf-8")

    assert not consume_controller_capability(tmp_path, 1, "accepted")
