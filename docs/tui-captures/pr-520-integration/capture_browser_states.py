"""State-specific browser interactions for the PR 520 offline capture fixture."""

from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse

from playwright.sync_api import ConsoleMessage, Page, Response, expect

from milknado.domains.graph import MikadoGraph

GOAL = "Review goal"
PLAN_STATES = frozenset({"pending", "accepted", "rejected", "stale", "applying"})
PROPOSAL_FOCUS_STATES = PLAN_STATES


@dataclass
class BrowserErrors:
    expect_coordinator_409: bool
    failures: list[str] = field(default_factory=list)
    coordinator_console: list[str] = field(default_factory=list)
    coordinator_responses: list[str] = field(default_factory=list)

    def on_console(self, message: ConsoleMessage) -> None:
        if message.type != "error":
            return
        location = message.location.get("url", "")
        known = "Failed to load resource: the server responded with a status of 409 (Conflict)"
        if (
            self.expect_coordinator_409
            and message.text == known
            and urlparse(location).path == "/api/coordinators"
        ):
            self.coordinator_console.append(location)
            return
        self.failures.append(f"console: {message.text} at {location}")

    def on_response(self, response: Response) -> None:
        if response.status >= 500:
            self.failures.append(f"HTTP {response.status}: {response.url}")
        if (
            self.expect_coordinator_409
            and response.status == 409
            and urlparse(response.url).path == "/api/coordinators"
        ):
            self.coordinator_responses.append(response.url)

    def assert_expected(self) -> None:
        if self.expect_coordinator_409:
            assert len(self.coordinator_console) == 1, self.coordinator_console
            assert self.coordinator_console == self.coordinator_responses
        else:
            assert not self.coordinator_console and not self.coordinator_responses
        assert not self.failures, "; ".join(self.failures)


def expect_goal_node(page: Page) -> None:
    viewport = page.viewport_size
    if viewport is not None and viewport["width"] < 600:
        expect(page.get_by_role("treeitem", name=GOAL)).to_be_visible()
    else:
        expect(page.locator("button.mk-node", has_text=GOAL)).to_be_visible()


def _proposal_id(graph: MikadoGraph) -> str:
    row = graph.group_connection.execute("SELECT id FROM coordinator_plan_proposals").fetchone()
    assert row is not None
    return str(row[0])


def _coordinator_unavailable(page: Page) -> None:
    expect(page.locator(".mk-coordinator-cockpit")).to_have_count(0)
    response = page.request.get(urljoin(page.url, "/api/coordinators"))
    assert response.status == 409
    assert response.json() == {"error": "Coordinator is unavailable."}
    snapshot = page.request.get(urljoin(page.url, "/api/snapshot"))
    assert snapshot.ok
    capability = snapshot.json()["capabilities"]["coordinator"]
    assert capability == {"available": False, "reason": "Coordinator is unavailable."}


def _plan_state(page: Page, graph: MikadoGraph, state: str) -> None:
    region = page.get_by_role("region", name="Plan proposals")
    expect(region).to_contain_text("src/task-1.py")
    if state == "pending":
        expect(region).to_contain_text("pending")
        return
    if state == "applying":
        from milknado.domains.coordinator.plans import transition_proposal

        _ = transition_proposal(graph.group_connection, _proposal_id(graph), "pending", "applying")
        expect(region).to_contain_text(
            "Apply incomplete. Manual recovery is required.", timeout=10000
        )
        return
    if state == "stale":
        row = graph.group_connection.execute("SELECT goal_id FROM coordinator_sessions").fetchone()
        assert row is not None
        _ = graph.add_node("Concurrent task", int(row[0]))
    decision = "Approve" if state in {"accepted", "stale"} else "Reject"
    page.get_by_role("button", name=decision, exact=False).click()
    expected = {"accepted": "applied", "rejected": "rejected", "stale": "stale"}[state]
    expect(region).to_contain_text(expected)
    if state == "stale":
        expect(region).to_contain_text("Graph changed. Request a new proposal.")


def after_state(page: Page, graph: MikadoGraph, state: str) -> None:
    if state == "intake":
        expect(page.get_by_label("Goal", exact=True)).to_be_visible()
        return
    if state == "coordinator-unavailable":
        _coordinator_unavailable(page)
        return
    page.get_by_label("Goal", exact=True).fill(GOAL)
    page.get_by_role("button", name="Start goal").click()
    expect(page.locator(".mk-coordinator-cockpit header strong")).to_have_text(GOAL)
    expect_goal_node(page)
    if state == "recovery-unavailable":
        page.get_by_role("button", name="Recover").click()
        expect(page.locator(".mk-coordinator-cockpit").get_by_role("status")).to_contain_text(
            "Recovery runtime is not connected."
        )
        return
    page.get_by_role("button", name="Propose plan").click()
    if state == "planner-unavailable":
        expect(page.get_by_text("Planner is not connected.", exact=True)).to_be_visible()
        return
    _plan_state(page, graph, state)
