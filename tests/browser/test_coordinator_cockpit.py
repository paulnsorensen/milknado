"""Browser proof of coordinator intake, graph selection, and durable node evidence."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from playwright.sync_api import Page, expect

from milknado.app.watch import WatchSnapshotSource
from milknado.domains.batching import BatchPlan
from milknado.domains.coordinator import CoordinatorControl
from milknado.domains.coordinator.control_services import CoordinatorServices
from milknado.domains.coordinator.persistence import link_entity
from milknado.domains.graph import MikadoGraph
from milknado.domains.planning import Planner, PlanProposal, PlanResult, decode_manifest
from milknado.web import LaunchToken, PolledSnapshotSource, WebCommands, create_app
from tests.browser.conftest import BROWSER_TOKEN, BrowserServer

pytestmark = pytest.mark.browser


def test_coordinator_cockpit_uses_real_api_and_graph(page: Page, tmp_path: Path) -> None:
    db_path = tmp_path / "graph.db"
    graph = MikadoGraph(db_path)
    watch = WatchSnapshotSource(tmp_path, db_path)
    source = PolledSnapshotSource(watch, interval=0.05)
    source.start()
    login = LaunchToken(BROWSER_TOKEN)
    control = CoordinatorControl(graph, tmp_path)
    server = BrowserServer(create_app(source, WebCommands(coordinator=control), login), login)
    server.start()
    try:
        _ = page.goto(server.login_url)
        page.get_by_label("Goal", exact=True).fill("Browser cockpit goal")
        page.get_by_role("button", name="Start goal").click()
        expect(page.locator(".mk-coordinator-cockpit header strong")).to_have_text(
            "Browser cockpit goal"
        )
        node = page.locator("button.mk-node", has_text="Browser cockpit goal")
        expect(node).to_be_visible()
        node.click()

        row = cast(
            tuple[str, int],
            graph.group_connection.execute(
                "SELECT id, goal_id FROM coordinator_sessions"
            ).fetchone(),
        )
        session_id, goal_id = row
        graph.runs.start("browser-run", goal_id, "/l", "2026-01-01T00:00:00+00:00", 600)
        graph.runs.record_verification("browser-run", True, "2026-01-01T00:01:00+00:00")
        link_entity(graph.group_connection, session_id, "run", "browser-run")
        verification = page.get_by_role("region", name="Completion verification")
        expect(verification).to_contain_text("accepted")

        page.get_by_role("button", name="Propose plan").click()
        expect(page.get_by_text("Planner is not connected.", exact=True)).to_be_visible()
        _ = cast(object, page.evaluate("localStorage.removeItem('milknado.coordinator.session')"))
        _ = page.reload()
        page.get_by_role("button", name="Browser cockpit goal · claude").click()
        expect(page.locator(".mk-coordinator-cockpit header strong")).to_have_text(
            "Browser cockpit goal"
        )
        page.locator("button.mk-node", has_text="Browser cockpit goal").click()
        expect(page.get_by_role("region", name="Completion verification")).to_contain_text(
            "accepted"
        )
    finally:
        server.stop()
        source.close()
        graph.close()


def test_browser_plan_review_controls_apply_only_after_approval(
    page: Page, tmp_path: Path
) -> None:
    class PlannerStub:
        count: int = 0

        def propose(self, goal: str, project_root: Path, *, target_goal_id: int) -> PlanProposal:
            assert (goal, project_root) == ("Review goal", tmp_path)
            assert target_goal_id > 0
            self.count += 1
            manifest = decode_manifest(
                {
                    "manifest_version": "milknado.plan.v2",
                    "goal": goal,
                    "goal_summary": goal,
                    "changes": [
                        {
                            "id": f"task-{self.count}",
                            "path": f"src/task-{self.count}.py",
                            "description": f"Task {self.count}",
                        }
                    ],
                }
            )
            return PlanProposal(manifest, tmp_path / "context.md")

        def prepare_proposal(self, proposal: PlanProposal, project_root: Path) -> BatchPlan:
            _ = (proposal, project_root)
            return BatchPlan((), (), "OPTIMAL")

        def apply_proposal(
            self,
            proposal: PlanProposal,
            *,
            target_goal_id: int,
            prepared_plan: BatchPlan,
        ) -> PlanResult:
            _ = prepared_plan
            _ = graph.add_node(proposal.manifest.changes[0].description, target_goal_id)
            return PlanResult(True, 0, proposal.context_path, nodes_created=1)

    db_path = tmp_path / "graph.db"
    graph = MikadoGraph(db_path)
    watch = WatchSnapshotSource(tmp_path, db_path)
    source = PolledSnapshotSource(watch, interval=0.05)
    source.start()
    login = LaunchToken(BROWSER_TOKEN)
    planner = cast(Planner, cast(object, PlannerStub()))
    control = CoordinatorControl(graph, tmp_path, CoordinatorServices(planner=planner))
    server = BrowserServer(create_app(source, WebCommands(coordinator=control), login), login)
    server.start()
    try:
        _ = page.goto(server.login_url)
        page.get_by_label("Goal", exact=True).fill("Review goal")
        page.get_by_role("button", name="Start goal").click()
        expect(page.locator(".mk-coordinator-cockpit header strong")).to_have_text("Review goal")
        row = cast(
            tuple[str, int],
            graph.group_connection.execute(
                "SELECT id, goal_id FROM coordinator_sessions"
            ).fetchone(),
        )
        session_id, goal_id = row
        page.get_by_role("button", name="Propose plan").click()
        expect(page.get_by_role("region", name="Plan proposals")).to_contain_text("src/task-1.py")
        assert graph.get_children(goal_id) == []
        page.get_by_role("button", name="Approve", exact=False).click()
        expect(page.get_by_role("region", name="Plan proposals")).to_contain_text("applied")
        assert [node.description for node in graph.get_children(goal_id)] == ["Task 1"]
        page.get_by_role("button", name="Propose plan").click()
        expect(page.get_by_role("region", name="Plan proposals")).to_contain_text("src/task-2.py")
        page.get_by_role("button", name="Reject", exact=False).click()
        expect(page.get_by_role("region", name="Plan proposals")).to_contain_text("rejected")
        assert [node.description for node in graph.get_children(goal_id)] == ["Task 1"]
        assert control.read_coordinator_snapshot(session_id, 0).proposals[-1].status == "rejected"
    finally:
        server.stop()
        source.close()
        graph.close()
