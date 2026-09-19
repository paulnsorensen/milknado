"""Playwright browser test harness: local server + fixture snapshot source."""

from __future__ import annotations

import hashlib
import socket
import subprocess
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from threading import Thread
from typing import cast

import pytest
import uvicorn
from starlette.applications import Starlette

from milknado.app.run_source import ExecutionSnapshot, NodeSnapshotRequest
from milknado.domains.common import MikadoNode, NodeKind, SessionInput
from milknado.domains.graph import (
    GoalReviewDecisionRequest,
    GoalReviewRecord,
    GraphSnapshot,
    NodeDetailResponse,
)
from milknado.web import LaunchToken, WebCommands, create_app

REPO_ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = REPO_ROOT / "web"
COMMITTED_STATIC_DIR = REPO_ROOT / "src" / "milknado" / "web" / "static"
FIXTURE_NODE_DESCRIPTION = "Tracer fixture node"
BROWSER_TOKEN = "browser-test-token"


def build_fixture_snapshot(goal: str = "Tracer fixture goal") -> ExecutionSnapshot:
    """Return a deterministic snapshot with one visible root goal node."""
    node = MikadoNode(id=1, description=FIXTURE_NODE_DESCRIPTION, kind=NodeKind.GOAL)
    graph = GraphSnapshot(nodes=(node,), edges=(), root_ids=(1,))
    return ExecutionSnapshot(
        goal=goal,
        active_runs=(),
        terminal_runs=(),
        completed=0,
        failed=0,
        stopped=0,
        available=1,
        event_lines=(),
        listener_errors=(),
        graph=graph,
        node=None,
    )


class BrowserSnapshotSource:
    """Publishable snapshot source for browser tests (supports a second snapshot)."""

    def __init__(self, snapshot: ExecutionSnapshot | None = None) -> None:
        self._snapshot: ExecutionSnapshot = snapshot or build_fixture_snapshot()
        self._listeners: list[Callable[[ExecutionSnapshot], None]] = []

    def snapshot(self) -> ExecutionSnapshot:
        return self._snapshot

    def subscribe(self, listener: Callable[[ExecutionSnapshot], None]) -> Callable[[], None]:
        self._listeners.append(listener)
        return lambda: self._listeners.remove(listener)

    def publish(self, snapshot: ExecutionSnapshot) -> None:
        self._snapshot = snapshot
        for listener in tuple(self._listeners):
            listener(snapshot)

    def node_snapshot(self, request: NodeSnapshotRequest) -> NodeDetailResponse:
        raise NotImplementedError(request)


@dataclass
class RecordingCommands:
    """Owner/observer command recorder for later curds' interaction assertions."""

    session_input_calls: list[tuple[str, SessionInput]] = field(default_factory=list)
    cancel_calls: list[str] = field(default_factory=list)
    force_stop_calls: list[str] = field(default_factory=list)
    stop_scheduling_calls: int = 0
    review_decision_calls: list[tuple[GoalReviewDecisionRequest, str]] = field(
        default_factory=list
    )

    def session_input(self, run_id: str, request: SessionInput) -> SessionInput | None:
        self.session_input_calls.append((run_id, request))
        return request

    def cancel(self, run_id: str) -> dict[str, object]:
        self.cancel_calls.append(run_id)
        return {"run_id": run_id, "status": "cancelled", "terminal": True}

    def force_stop(self, run_id: str) -> dict[str, object]:
        self.force_stop_calls.append(run_id)
        return {"run_id": run_id}

    def stop_scheduling(self) -> None:
        self.stop_scheduling_calls += 1

    def review_decision(
        self, request: GoalReviewDecisionRequest, *, decided_by: str
    ) -> GoalReviewRecord:
        self.review_decision_calls.append((request, decided_by))
        return GoalReviewRecord(
            review_id=request.review_id,
            goal_id=0,
            goal_revision="",
            evidence="",
            proposed_change="",
            decision=request.decision,
            affected_node_ids=None,
            reviewer="",
            assessed_at="",
            decided_at=request.decided_at,
            decided_by=decided_by,
        )


def owner_web_commands() -> tuple[WebCommands, RecordingCommands]:
    """WebCommands wired to a recorder, exercising the owner surface end to end."""
    recorder = RecordingCommands()
    commands = WebCommands(
        session_input=recorder.session_input,
        cancel=recorder.cancel,
        force_stop=recorder.force_stop,
        stop_scheduling=recorder.stop_scheduling,
        review_decision=recorder.review_decision,
    )
    return commands, recorder


def observer_web_commands() -> tuple[WebCommands, RecordingCommands]:
    """WebCommands with no owner powers, for observer-mode assertions."""
    return WebCommands(), RecordingCommands()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        address = cast("tuple[str, int]", sock.getsockname())
        return address[1]


@dataclass
class BrowserServer:
    """A background uvicorn server that can stop and restart on the same port."""

    app: Starlette
    login: LaunchToken
    port: int = field(default_factory=_free_port)
    _server: uvicorn.Server | None = field(default=None, init=False, repr=False)
    _thread: Thread | None = field(default=None, init=False, repr=False)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def login_url(self) -> str:
        return f"{self.base_url}/auth?token={self.login.value}"

    def start(self) -> None:
        config = uvicorn.Config(self.app, host="127.0.0.1", port=self.port, log_level="warning")
        server = uvicorn.Server(config)
        thread = Thread(target=server.run, daemon=True)
        thread.start()
        deadline = time.monotonic() + 5.0
        while not server.started:
            if time.monotonic() > deadline:
                raise TimeoutError("browser test server did not start within 5s")
            time.sleep(0.01)
        self._server = server
        self._thread = thread

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._server = None
        self._thread = None

    def restart(self, app: Starlette | None = None) -> None:
        self.stop()
        if app is not None:
            self.app = app
        self.start()


@pytest.fixture
def browser_source() -> BrowserSnapshotSource:
    return BrowserSnapshotSource()


@pytest.fixture
def browser_server(browser_source: BrowserSnapshotSource) -> Iterator[BrowserServer]:
    login = LaunchToken(BROWSER_TOKEN)
    commands, _ = owner_web_commands()
    app = create_app(browser_source, commands, login)
    server = BrowserServer(app=app, login=login)
    server.start()
    yield server
    server.stop()


def require_web_node_modules() -> None:
    if not (WEB_DIR / "node_modules").is_dir():
        raise RuntimeError(
            "web/node_modules is missing; run `just install` to install web dependencies."
        )


def build_web_to(out_dir: Path) -> None:
    """Build the web app to `out_dir` using the committed source, byte-stable."""
    require_web_node_modules()
    result = subprocess.run(
        ["npx", "vite", "build", "--outDir", str(out_dir), "--emptyOutDir"],
        cwd=WEB_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"web build failed:\n{result.stdout}\n{result.stderr}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def diff_build_trees(committed: Path, candidate: Path) -> list[str]:
    """Return the relative paths that differ (missing or content-changed) between builds."""
    committed_files = {p.relative_to(committed) for p in committed.rglob("*") if p.is_file()}
    candidate_files = {p.relative_to(candidate) for p in candidate.rglob("*") if p.is_file()}
    differing: list[str] = []
    for relative in sorted(committed_files | candidate_files, key=str):
        committed_file = committed / relative
        candidate_file = candidate / relative
        if not committed_file.is_file() or not candidate_file.is_file():
            differing.append(str(relative))
            continue
        if _sha256(committed_file) != _sha256(candidate_file):
            differing.append(str(relative))
    return differing
