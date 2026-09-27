"""Capture the dashboard screens with a seeded graph.db and Playwright.

Usage: uv run python capture_web.py <out-dir>
The fixture mirrors the Milknado Web canvas: one goal, five sub-goals, tasks in
every state, two active runs, a permission request and a pending goal review.
Run output, session transcripts and changed files are synthetic; no worker runs.
"""

from __future__ import annotations

import dataclasses
import re
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from playwright.sync_api import sync_playwright  # noqa: E402

from milknado.adapters import ChangedFile  # noqa: E402
from milknado.app.run_source import (  # noqa: E402
    ActiveRunSnapshot,
    ExecutionRunStatus,
    NodeSnapshotRequest,
    RunActionAvailability,
)
from milknado.app.watch import WatchSnapshotSource  # noqa: E402
from milknado.domains.common import NodeKind, NodeSpec, SessionContext, SessionView  # noqa: E402
from milknado.domains.common.session import SessionEvent  # noqa: E402
from milknado.domains.graph import (  # noqa: E402
    GoalReviewRequest,
    MikadoGraph,
    NodeSessionSnapshot,
    SnapshotPage,
)
from milknado.domains.graph.commands import OwnerCapabilities  # noqa: E402
from milknado.web import LaunchToken, PolledSnapshotSource, WebCommands, create_app  # noqa: E402
from milknado.web.commands import GraphEditCommands  # noqa: E402
from tests.browser.conftest import BROWSER_TOKEN, BrowserServer  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "docs" / "web-ui"
RUN_8 = "node-8-20260919T140211Z-7c1e"
RUN_9 = "node-9-20260919T134702Z-a3f0"

EVENTS = (
    "13:19:50  node 14 · dispatched · Kestrel",
    "13:47:02  node 9 · dispatched · Osprey",
    "13:50:57  node 14 · failed · attempt 3/3",
    "14:01:07  node 9 · permission requested · perm_41",
    "14:02:11  node 8 · dispatched · Falcon",
    "14:04:19  node 8 · gate PASS · diff coverage 96.4%",
)

SESSION_8 = (
    ("status", "turn 4 started · brief unchanged"),
    ("tool", "tilth_read src/milknado/app/session_commands.py"),
    ("assistant", "edit: queue session input in FIFO order"),
    ("tool", "just check-llm"),
    ("error", "typecheck: 1 error · src/milknado/app/session_commands.py:188"),
    ("user", "steer: keep request_id exact. Do not infer it from the text."),
    ("assistant", "PASS · diff coverage 96.4%"),
    ("status", "result deposited → reviewer"),
)


def seed(graph: MikadoGraph) -> dict[str, int]:
    goal = graph.add_node("Interactive run steering", spec=NodeSpec(kind=NodeKind.GOAL))
    ids = {"goal": goal.id}

    def sub(title: str) -> int:
        return graph.add_node(title, goal.id, NodeSpec(kind=NodeKind.GOAL)).id

    def task(
        title: str, parent: int, flavor: str = "implement", prereqs: tuple[int, ...] = ()
    ) -> int:
        return graph.add_node(title, parent, NodeSpec(flavor=flavor, prereqs=prereqs)).id

    def done(node_id: int) -> None:
        graph.mark_running(node_id)
        graph.mark_done(node_id)

    g2 = sub("Graph steering workspace")
    for title in (
        "Graph tree panel",
        "Node inspector",
        "Detail pagination",
        "Compact layout route",
    ):
        done(task(title, g2))
    g7 = sub("Session commands")
    n8 = task("Structured session input", g7)
    n9 = task("Permission request relay", g7)
    n10 = task("Interrupt and follow-up actions", g7, prereqs=(n8,))
    n11 = task("Owner-fenced command inbox", g7, "spike")
    g12 = sub("Follow-up provenance")
    n13 = task("Track follow-up nodes", g12)
    n14 = task("Review receipt store", g12)
    task("Harvest outcome block", g12, "spec")
    g16 = sub("Web interface")
    n17 = task("Snapshot JSON endpoint", g16, "spike")
    task("Event stream", g16, prereqs=(n17,))
    task("Graph screen", g16, "prototype", prereqs=(n17,))
    g20 = sub("Steering evidence")
    for title, flavor in (
        ("Deterministic capture tapes", "implement"),
        ("Before and after pairs", "implement"),
        ("Fixture limits note", "research"),
    ):
        done(task(title, g20, flavor))
    graph.mark_running(n8, run_id=RUN_8)
    graph.mark_running(n9, run_id=RUN_9)
    graph.mark_blocked(n10)
    done(n11)
    done(n13)
    graph.mark_running(n14)
    graph.mark_failed(n14)
    graph.mark_running(g2)
    graph.mark_done(g2)
    graph.mark_running(g7)
    graph.mark_running(g12)
    graph.mark_running(g20)
    graph.mark_done(g20)
    graph.request_goal_review(
        GoalReviewRequest(
            goal.id,
            "4f2a",
            "The snapshot source gives data that a browser can read. "
            "The spike in node 17 reads one snapshot as JSON.",
            "Add a web interface to the goal. "
            "The interface gives the same functions as the terminal interface.",
            reviewer="reviewer",
            affected_node_ids=(n17,),
        )
    )
    graph.register_controller_master()
    ids.update({"n8": n8, "n9": n9, "n14": n14})
    return ids


def active_run(run_id: str, node_id: int, title: str, family: str) -> ActiveRunSnapshot:
    elapsed = 1380.0 if family == "claude" else 840.0
    return ActiveRunSnapshot(
        run_id=run_id,
        node_id=node_id,
        description=title,
        status=ExecutionRunStatus.RUNNING,
        progress=None,
        stop_requested=False,
        actions=RunActionAvailability(),
        output=(),
        pending_guidance=None,
        elapsed_seconds=elapsed,
        progress_pct=None,
        eta_seconds=None,
        attempt=1,
        max_attempts=3,
        stalled=False,
        session=SessionView(context=SessionContext(family=family, cwd="/tmp/session")),
    )


class DemoSource:
    """The polled watch source plus synthetic runs, events and one session transcript."""

    def __init__(self, inner: PolledSnapshotSource, ids: dict[str, int]) -> None:
        self.inner = inner
        self.ids = ids

    def _decorate(self, snapshot):  # noqa: ANN001, ANN202
        return dataclasses.replace(
            snapshot,
            active_runs=(
                active_run(RUN_8, self.ids["n8"], "Falcon", "claude"),
                active_run(RUN_9, self.ids["n9"], "Osprey", "codex"),
            ),
            event_lines=EVENTS,
        )

    def snapshot(self):  # noqa: ANN201
        return self._decorate(self.inner.snapshot())

    def subscribe(self, listener):  # noqa: ANN001, ANN201
        return self.inner.subscribe(lambda snapshot: listener(self._decorate(snapshot)))

    def node_snapshot(self, request: NodeSnapshotRequest):  # noqa: ANN201
        response = self.inner.node_snapshot(request)
        if request.node_id != self.ids["n8"] or response.detail is None:
            return response
        events = tuple(SessionEvent(kind=kind, text=text) for kind, text in SESSION_8)
        page = SnapshotPage(items=events, offset=0, limit=50, total=len(events), has_more=False)
        sessions = SnapshotPage(
            items=(
                NodeSessionSnapshot(
                    run_id=RUN_8, session=None, state="loaded", event_history=page
                ),
            ),
            offset=0,
            limit=50,
            total=1,
            has_more=False,
        )
        run = {
            "run_id": RUN_8,
            "node_id": request.node_id,
            "status": "running",
            "pid": 48213,
            "log_path": "",
            "started_at": "2026-09-19 14:02:11",
            "ended_at": None,
            "timed_out": False,
            "exit_code": None,
            "error": None,
            "timeout_seconds": None,
            "detail": None,
            "rebased": None,
        }
        runs = SnapshotPage(items=(run,), offset=0, limit=50, total=1, has_more=False)
        detail = dataclasses.replace(response.detail, sessions=sessions, runs=runs)
        return dataclasses.replace(response, detail=detail)


class FakeGit:
    def changes(self, context: SessionContext) -> tuple[ChangedFile, ...]:
        del context
        return (
            ChangedFile(
                path="src/milknado/app/session_commands.py",
                status="M",
                added=42,
                removed=9,
                old_path=None,
            ),
            ChangedFile(
                path="src/milknado/app/session_view.py",
                status="M",
                added=6,
                removed=2,
                old_path=None,
            ),
            ChangedFile(
                path="tests/app/test_session_input_queue.py",
                status="A",
                added=88,
                removed=0,
                old_path=None,
            ),
        )

    def diff(self, context: SessionContext, path: str) -> str:
        del context
        return (
            f"--- a/{path}\n+++ b/{path}\n"
            "@@ -176,9 +176,14 @@ def _send_session_input(self) -> None:\n"
            "     command = self._build_session_input()\n"
            "-    self._submit_session_input(command)\n"
            "+    self._input_queue.append(command)\n"
            "+    if len(self._input_queue) == 1:\n"
            "+        self._submit_session_input(command)\n"
            "@@ -201,6 +206,11 @@ def _on_session_input_done(self, result) -> None:\n"
            "+    _ = self._input_queue.popleft()\n"
            "+    if self._input_queue:\n"
            "+        self._submit_session_input(self._input_queue[0])\n"
            '     self.notify("Session input queued.")\n'
        )


def build_commands(graph: MikadoGraph, root: Path, ids: dict[str, int]) -> WebCommands:
    def decide(request, *, decided_by):  # noqa: ANN001, ANN202
        return graph.decide_goal_review(request, decided_by=decided_by)

    return WebCommands(
        session_input=lambda _run_id, request: request,
        cancel=lambda run_id: {"run_id": run_id, "status": "cancelled", "terminal": True},
        force_stop=lambda run_id: {"run_id": run_id},
        stop_scheduling=lambda: None,
        graph_edits=GraphEditCommands(
            graph, frozenset({"implement", "spec", "spike", "prototype", "research"}), root
        ),
        review_decision=decide,
        git=FakeGit(),
        owner_capabilities=OwnerCapabilities(
            run_id=RUN_8,
            node_id=ids["n8"],
            invocation_id="inv-1",
            owner_incarnation="1",
            actions=("steer", "follow_up", "interrupt", "approve", "deny"),
            permission_ids=("perm_41",),
            published_at="2026-09-19T14:04:20Z",
        ),
    )


def shot(page, out: Path, name: str, settle: int = 300) -> None:  # noqa: ANN001
    page.wait_for_timeout(settle)
    page.screenshot(path=str(out / name))


def capture_main(page, out: Path) -> None:  # noqa: ANN001
    page.get_by_role("button", name=re.compile("Structured session input")).wait_for()
    page.get_by_role("button", name="Dark", exact=True).click()
    shot(page, out, "00-main-empty.png", 600)
    page.get_by_role("button", name=re.compile("Structured session input")).click()
    page.get_by_text(RUN_8).first.wait_for()
    shot(page, out, "01-main-dark.png", 400)
    page.get_by_role("button", name="Light", exact=True).click()
    shot(page, out, "02-main-light.png")
    page.get_by_role("button", name="Dark", exact=True).click()
    page.get_by_role("button", name="Details", exact=True).click()
    shot(page, out, "03-node-detail.png")
    page.get_by_role("button", name="Changes", exact=True).click()
    page.get_by_text("session_view.py", exact=False).first.wait_for()
    page.get_by_text("src/milknado/app/session_commands.py").click()
    page.get_by_text("-    self._submit_session_input(command)").wait_for()
    shot(page, out, "04-changes-diff.png")
    page.get_by_role("button", name="Session", exact=True).click()


def capture_overlays(page, out: Path) -> None:  # noqa: ANN001
    page.get_by_role("button", name="Force stop", exact=True).click()
    page.get_by_role("alertdialog").wait_for()
    shot(page, out, "05-run-confirm.png", 400)
    page.get_by_role("button", name="Dismiss").click()
    page.get_by_role("button", name="Open", exact=True).click()
    page.get_by_text("Waits for a person").wait_for()
    shot(page, out, "06-goal-review.png")
    page.get_by_role("button", name="Close the review").click()
    page.get_by_role("button", name="Keyboard shortcuts").click()
    page.get_by_role("dialog", name="Keyboard shortcuts").wait_for()
    shot(page, out, "07-help.png", 400)
    page.get_by_role("button", name="Close", exact=True).click()
    page.get_by_role("button", name=re.compile("^Events")).click()
    shot(page, out, "10-events-dock.png")
    page.get_by_role("button", name="Add node", exact=True).click()
    page.get_by_role("dialog", name="Add node").wait_for()
    shot(page, out, "11-add-node.png", 400)


def capture(server: BrowserServer, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900}, color_scheme="dark")
        page.goto(server.login_url)
        capture_main(page, out)
        capture_overlays(page, out)
        page.close()

        narrow = browser.new_page(viewport={"width": 400, "height": 800}, color_scheme="dark")
        narrow.goto(server.login_url)
        narrow.get_by_role("button", name="Open node").wait_for()
        narrow.wait_for_timeout(400)
        narrow.screenshot(path=str(out / "08-narrow-list.png"))
        narrow.get_by_role("treeitem", name=re.compile("Structured session input")).click()
        narrow.get_by_role("button", name="Open node").click()
        narrow.get_by_text(RUN_8).first.wait_for()
        narrow.wait_for_timeout(300)
        narrow.screenshot(path=str(out / "09-narrow-detail.png"))
        browser.close()


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        graph = MikadoGraph(root / "graph.db")
        try:
            ids = seed(graph)
            watch = WatchSnapshotSource(root, root / "graph.db")
            polled = PolledSnapshotSource(watch, interval=0.1)
            polled.start()
            login = LaunchToken(BROWSER_TOKEN)
            app = create_app(DemoSource(polled, ids), build_commands(graph, root, ids), login)
            server = BrowserServer(app=app, login=login)
            server.start()
            try:
                if "--serve" in sys.argv:
                    print(server.login_url, flush=True)
                    import time

                    while True:
                        time.sleep(3600)
                capture(server, OUT)
                print(f"captured to {OUT}")
            finally:
                server.stop()
                polled.close()
        finally:
            graph.close()


if __name__ == "__main__":
    main()
