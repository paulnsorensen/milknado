"""Browser harness for the metaharness view: a real graph.db, a real git worktree
for the run diff, and the production polled snapshot reader stack."""

from __future__ import annotations

import dataclasses
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path

import pytest
from playwright.sync_api import Locator, Page, expect

from milknado.adapters import ChangedFile
from milknado.app.watch import WatchSnapshotSource
from milknado.domains.common import NodeKind, NodeSpec, RunResult, SessionContext, SessionEvent
from milknado.domains.coordinator import CoordinatorControl, CoordinatorServices
from milknado.domains.graph import (
    GoalReviewDecisionRequest,
    GoalReviewRecord,
    GoalReviewRequest,
    MikadoGraph,
)
from milknado.web import LaunchToken, PolledSnapshotSource, create_app
from milknado.web.commands import MetaharnessInfo, WebCommands
from milknado.web.hosts import HostDependencies, observer_commands
from tests.browser.conftest import BROWSER_TOKEN, BrowserServer, open_app

POLL_INTERVAL = 0.05
PERMISSION_ID = "perm-push"


@dataclass
class FakeScheduling:
    paused: bool = False
    calls: list[str] = field(default_factory=list)

    def set_scheduling_paused(self, paused: bool) -> None:
        self.calls.append("pause" if paused else "resume")
        self.paused = paused

    def scheduling_paused(self) -> bool:
        return self.paused


@dataclass(frozen=True)
class SeededIds:
    goal: int
    review_goal: int
    annotation: int
    fingerprint: int
    drawer: int
    failed: int
    stuck: int
    review_id: int
    annotation_run: str
    fingerprint_run: str


@dataclass
class FakeCancel:
    run_ids: list[str] = field(default_factory=list)

    def __call__(self, run_id: str) -> dict[str, object]:
        self.run_ids.append(run_id)
        return {"run_id": run_id, "status": "cancelled"}


@dataclass
class MetaharnessServer:
    server: BrowserServer
    graph: MikadoGraph
    ids: SeededIds
    scheduling: FakeScheduling
    cancels: FakeCancel

    @property
    def url(self) -> str:
        return self.server.login_url


def open_view(page: Page, server: MetaharnessServer) -> None:
    page.set_viewport_size({"width": 1440, "height": 900})
    open_app(page, server.url, page.get_by_role("button", name="Metaharness"))
    page.get_by_role("button", name="Metaharness").click()
    expect(page.locator('[data-view="metaharness"]')).to_be_visible()


def compact_node(page: Page, node_id: int) -> Locator:
    return page.locator(f'[data-mh-node="{node_id}"][data-mode="compact"]')


def _ts(minutes_ago: float) -> str:
    return (datetime.now(UTC) - timedelta(minutes=minutes_ago)).isoformat()


def _git(cwd: Path, *args: str) -> str:
    done = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
    return done.stdout.strip()


def _worktree(root: Path) -> tuple[str, Path]:
    _ = _git(root, "init", "-q", "-b", "main")
    _ = _git(root, "config", "user.email", "browser@example.com")
    _ = _git(root, "config", "user.name", "Browser")
    _ = (root / "queue.js").write_text("export const queue = [];\n")
    _ = (root / ".gitignore").write_text(".milknado/\n")
    _ = _git(root, "add", "-A")
    _ = _git(root, "commit", "-q", "-m", "base")
    worktree = root / ".milknado" / "worktrees" / "milknado-annotation"
    _ = _git(root, "worktree", "add", "-q", "-b", "annotation", str(worktree))
    _ = (worktree / "queue.js").write_text("export const queue = await store.load();\n")
    return _git(root, "rev-parse", "HEAD"), worktree


def _task(graph: MikadoGraph, text: str, parent: int, flavor: str = "implement") -> int:
    return graph.add_node(text, parent, NodeSpec(kind=NodeKind.TASK, flavor=flavor)).id


def _goal(graph: MikadoGraph, text: str, parent: int | None) -> int:
    return graph.add_node(text, parent, NodeSpec(kind=NodeKind.GOAL)).id


def _live_run(graph: MikadoGraph, node_id: int, run_id: str, context: SessionContext) -> None:
    graph.mark_running(node_id)
    graph.runs.start(run_id, node_id, f"/tmp/{run_id}.log", _ts(30), 3600)
    graph.sessions.start(run_id, context)


def _seed_annotation(graph: MikadoGraph, node_id: int, context: SessionContext) -> str:
    run_id = "run-annotation"
    _live_run(graph, node_id, run_id, context)
    events = (
        SessionEvent(kind="assistant", text="Queued prompts now persist.", event_id="e0"),
        SessionEvent(kind="tool", text="just check-llm \u00b7 PASS", event_id="e1"),
    )
    for event in events:
        graph.sessions.append(run_id, event)
    permission = SessionEvent(
        kind="permission",
        text="git push origin annotation",
        event_id=PERMISSION_ID,
        state="requested",
    )
    graph.sessions.append(run_id, permission)
    _ = graph.commands.publish_capabilities(
        run_id,
        node_id,
        "invoke-1",
        "owner-1",
        ("steer", "approve", "deny"),
        (PERMISSION_ID,),
        ((PERMISSION_ID, "git push origin annotation"),),
    )
    _ = graph.runs.insert_review(node_id, "approve", "Name the poll interval constant.", _ts(5))
    return run_id


def _seed_failure(graph: MikadoGraph, node_id: int) -> None:
    graph.mark_running(node_id)
    graph.runs.start("run-strip", node_id, "/tmp/run-strip.log", _ts(90), 3600)
    result = RunResult("failed", 1, False, _ts(60), error="coverage 88% < 95%")
    graph.runs.finish("run-strip", result)
    graph.mark_failed(node_id)


def _seed(graph: MikadoGraph, root: Path) -> SeededIds:
    base, worktree = _worktree(root)
    goal = _goal(graph, "Tern browser UI", None)
    review_goal = _goal(graph, "Review integration", goal)
    annotation = _task(graph, "Annotation queued prompt", goal)
    inbox = _task(graph, "Layout issues inbox", goal)
    fingerprint = _task(graph, "Issue fingerprinting", inbox)
    drawer = _task(graph, "Inbox drawer UI", review_goal, "prototype")
    failed = _task(graph, "Session stats strip", goal)
    stuck = _task(graph, "Claude Code adapter", goal)
    graph.mark_running(goal)
    graph.mark_running(review_goal)
    context = SessionContext(family="claude", cwd=str(worktree), base_oid=base)
    annotation_run = _seed_annotation(graph, annotation, context)
    graph.mark_blocked(inbox)
    _live_run(graph, fingerprint, "run-fingerprint", context)
    _ = graph.commands.publish_capabilities(
        "run-fingerprint", fingerprint, "invoke-2", "owner-2", ("steer",)
    )
    _seed_failure(graph, failed)
    graph.mark_blocked(stuck)
    review = graph.request_goal_review(
        GoalReviewRequest(
            review_goal, "rev-1", "Schema changes.", "Hold the drawer.", reviewer="coordinator"
        )
    )
    graph.register_controller_master()
    return SeededIds(
        goal,
        review_goal,
        annotation,
        fingerprint,
        drawer,
        failed,
        stuck,
        review.review_id,
        annotation_run,
        "run-fingerprint",
    )


def _decide(
    graph: MikadoGraph, request: GoalReviewDecisionRequest, *, decided_by: str
) -> GoalReviewRecord:
    return graph.decide_goal_review(request, decided_by=decided_by)


def _app_commands(
    graph: MikadoGraph, root: Path, scheduling: FakeScheduling, cancels: FakeCancel
) -> WebCommands:
    metaharness = MetaharnessInfo(
        project_root=root, concurrency=4, flavor_gates={"implement": ("just check-llm",)}
    )
    dependencies = HostDependencies(
        graph=graph,
        flavor_registry=frozenset({"implement", "prototype"}),
        project_root=root,
        review_decision=partial(_decide, graph),
        git=_RootGit(root),
        metaharness=metaharness,
        coordinator=CoordinatorControl(graph, root, CoordinatorServices()),
    )
    return dataclasses.replace(
        observer_commands(dependencies=dependencies), scheduling=scheduling, cancel=cancels
    )


class _RootGit:
    def __init__(self, root: Path) -> None:
        from milknado.adapters import GitAdapter

        self._git: GitAdapter = GitAdapter(root)

    def changes(self, context: SessionContext) -> tuple[ChangedFile, ...]:
        return self._git.session_changes(context)

    def diff(self, context: SessionContext, path: str) -> str:
        return self._git.session_diff(context, path)


@pytest.fixture
def metaharness_server(tmp_path: Path) -> Iterator[MetaharnessServer]:
    db_path = tmp_path / ".milknado" / "milknado.db"
    db_path.parent.mkdir()
    graph = MikadoGraph(db_path)
    try:
        ids = _seed(graph, tmp_path)
        scheduling = FakeScheduling()
        cancels = FakeCancel()
        source = PolledSnapshotSource(
            WatchSnapshotSource(tmp_path, db_path), interval=POLL_INTERVAL
        )
        source.start()
        login = LaunchToken(BROWSER_TOKEN)
        commands = _app_commands(graph, tmp_path, scheduling, cancels)
        server = BrowserServer(app=create_app(source, commands, login), login=login)
        server.start()
        try:
            yield MetaharnessServer(server, graph, ids, scheduling, cancels)
        finally:
            server.stop()
            source.close()
    finally:
        graph.close()
