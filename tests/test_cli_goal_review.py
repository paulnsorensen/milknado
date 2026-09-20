from __future__ import annotations

import getpass
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
import typer
from typer.testing import CliRunner

from milknado.cli import app
from milknado.domains.common import (
    CONTROLLER_MASTER_ENV,
    WORKER_CONTEXT_ENV,
    NodeKind,
    NodeSpec,
)
from milknado.domains.dispatch import build_worker_env
from milknado.domains.graph import (
    GoalReviewDecision,
    GoalReviewRequest,
    MikadoGraph,
)
from milknado.loop._agent import _build_spawn_env  # pyright: ignore[reportPrivateUsage]
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


def _register_controller(project_root: Path) -> None:
    (project_root / ".milknado").mkdir(exist_ok=True)
    graph = MikadoGraph(project_root / ".milknado" / "milknado.db")
    try:
        graph.register_controller_master()
    finally:
        graph.close()


def _confirm(*_args: object, **_kwargs: object) -> bool:
    return True


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
    monkeypatch.setattr(typer, "confirm", _confirm)
    monkeypatch.setattr(getpass, "getuser", lambda: "human-controller")
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
    _register_controller(tmp_path)

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

    db_path = tmp_path / ".milknado" / "milknado.db"
    assert b"external-controller-master" not in db_path.read_bytes()
    with sqlite3.connect(db_path) as conn:
        consumed = cast(
            tuple[int] | None,
            conn.execute("SELECT COUNT(*) FROM consumed_controller_capabilities").fetchone(),
        )
    assert consumed == (1,)


def test_worker_environment_strips_controller_master(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CONTROLLER_MASTER_ENV, "external-controller-master")
    worker_env = build_worker_env({CONTROLLER_MASTER_ENV: "worker-spoof"})

    assert CONTROLLER_MASTER_ENV not in worker_env
    assert worker_env[WORKER_CONTEXT_ENV] == "1"


@pytest.mark.skipif(sys.platform == "win32", reason="PTY worker test requires POSIX")
def test_worker_cannot_self_approve_from_a_pty(  # noqa: PLR0915 - real PTY boundary
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    review_id = _pending_review(tmp_path)
    monkeypatch.setenv(CONTROLLER_MASTER_ENV, "external-controller-master")
    _register_controller(tmp_path)
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
    assert b"worker context" in bytes(output), (
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
    assert worker_env[WORKER_CONTEXT_ENV] == "1"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-group test")
def test_session_environment_strips_controller_master(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(CONTROLLER_MASTER_ENV, "external-controller-master")
    protocol = cast(
        SessionProtocol,
        cast(
            object,
            SimpleNamespace(
                command=(
                    sys.executable,
                    "-c",
                    "import os; print(os.environ.get('MILKNADO_CONTROLLER_MASTER', '')); "
                    + "print(os.environ.get('MILKNADO_WORKER_CONTEXT', ''))",
                )
            ),
        ),
    )
    proc = start_process(protocol, tmp_path)
    stdout, _ = proc.communicate(timeout=10)

    assert stdout == b"\n1\n"


def test_controller_registration_reuses_managed_master_and_rejects_wrong_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(CONTROLLER_MASTER_ENV)
    _register_controller(tmp_path)
    monkeypatch.delenv(CONTROLLER_MASTER_ENV, raising=False)
    _register_controller(tmp_path)
    monkeypatch.setenv(CONTROLLER_MASTER_ENV, "different-master")
    with pytest.raises(RuntimeError, match="different controller master"):
        _register_controller(tmp_path)


def test_goal_review_cli_reports_controller_storage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    review_id = _pending_review(tmp_path)
    monkeypatch.setenv("XDG_STATE_HOME", "relative-state")

    result = _invoke_human(tmp_path, review_id, "accepted", monkeypatch)

    assert result.exit_code == 1
    assert "XDG_STATE_HOME must be an absolute path" in result.output
