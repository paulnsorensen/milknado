"""AC-12: a pending review is listed, its sidecar shows evidence and the
proposed change, and a decision stores in the graph and clears the rail."""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect

from milknado.domains.graph import GoalReviewDecision
from tests.browser.conftest import open_app, wait_until
from tests.browser.graph_db import GraphDbServer, graph_db_server

_ = graph_db_server

pytestmark = pytest.mark.browser


def test_pending_review_lists_and_accepts(page: Page, graph_db_server: GraphDbServer) -> None:
    open_app(page, graph_db_server.login_url, page.get_by_text("evidence for the change"))

    page.get_by_role("button", name="Open").click()

    expect(page.get_by_text("proposed change text")).to_be_visible()

    page.get_by_role("button", name="Accept change").click()

    wait_until(
        lambda: (
            (review := graph_db_server.graph.get_goal_review(graph_db_server.review_id))
            is not None
            and review.decision is GoalReviewDecision.ACCEPTED
        )
    )
    expect(page.get_by_text("No goal reviews are pending.")).to_be_visible()
