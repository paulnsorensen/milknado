"""The metaharness view against a real graph.db: every decision, message, and rail
action reaches the milknado backend."""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect

from milknado.domains.common import NodeStatus
from milknado.domains.graph import GoalReviewDecision
from tests.browser.conftest import wait_until
from tests.browser.metaharness_db import MetaharnessServer, metaharness_server
from tests.browser.metaharness_db import compact_node as _node
from tests.browser.metaharness_db import open_view as _open

_ = metaharness_server

pytestmark = pytest.mark.browser


def _send(page: Page, text: str) -> None:
    composer = page.get_by_role("textbox", name="Message")
    composer.fill(text)
    composer.press("Enter")


def _description(server: MetaharnessServer, node_id: int) -> str:
    node = server.graph.get_node(node_id)
    return node.description if node is not None else ""


def test_view_shows_graph_thread_and_details(
    page: Page, metaharness_server: MetaharnessServer
) -> None:
    ids = metaharness_server.ids
    _open(page, metaharness_server)

    expect(_node(page, ids.annotation)).to_have_class("mh-node is-need")
    expect(_node(page, ids.fingerprint)).to_contain_text(f"w{ids.fingerprint}")
    expect(page.locator(".mh-needs")).to_contain_text("Needs you3")
    _node(page, ids.annotation).click()
    expect(page.get_by_role("dialog", name="Worker asks")).to_contain_text(
        "git push origin annotation"
    )
    page.keyboard.press("Escape")

    log = page.get_by_role("log", name="Conversation")
    expect(log).to_contain_text("Queued prompts now persist.")
    expect(log).to_contain_text("just check-llm · PASS")
    expect(log).to_contain_text("Name the poll interval constant.")
    page.get_by_role("button", name="Details").click()
    details = page.get_by_role("complementary")
    expect(details).to_contain_text("just check-llm")
    expect(details).to_contain_text("milknado-annotation")
    expect(details).to_contain_text("Unblocks")
    expect(details.get_by_role("button", name="Tern browser UI")).to_be_visible()


def test_goal_review_accept_reaches_the_graph(
    page: Page, metaharness_server: MetaharnessServer
) -> None:
    ids = metaharness_server.ids
    _open(page, metaharness_server)
    _node(page, ids.review_goal).click()
    popover = page.get_by_role("dialog", name="Goal review")
    expect(popover).to_contain_text("Hold the drawer.")
    popover.get_by_role("button", name="Accept change").click()

    def accepted() -> bool:
        review = metaharness_server.graph.get_goal_review(ids.review_id)
        return review is not None and review.decision is GoalReviewDecision.ACCEPTED

    wait_until(accepted)
    expect(page.get_by_role("status")).to_contain_text("Accepted the goal change")


def test_retry_with_note_requeues_the_failed_node(
    page: Page, metaharness_server: MetaharnessServer
) -> None:
    ids = metaharness_server.ids
    _open(page, metaharness_server)
    _node(page, ids.failed).click()
    popover = page.get_by_role("dialog", name="Gate failed")
    expect(popover).to_contain_text("coverage 88% < 95%")
    popover.get_by_role("button", name="Retry with note").click()
    expect(page.get_by_text("retries the node with your note")).to_be_visible()
    _send(page, "Cover the empty state first.")

    def requeued() -> bool:
        node = metaharness_server.graph.get_node(ids.failed)
        return node is not None and node.status is NodeStatus.PENDING

    wait_until(requeued)
    assert "Operator note: Cover the empty state first." in _description(
        metaharness_server, ids.failed
    )


def test_skip_marks_the_failed_node_done(
    page: Page, metaharness_server: MetaharnessServer
) -> None:
    ids = metaharness_server.ids
    _open(page, metaharness_server)
    _node(page, ids.failed).click()
    page.get_by_role("dialog", name="Gate failed").get_by_role("button", name="Skip").click()

    def skipped() -> bool:
        node = metaharness_server.graph.get_node(ids.failed)
        return node is not None and node.status is NodeStatus.DONE

    wait_until(skipped)
    assert "Operator note: Skipped by the operator." in _description(
        metaharness_server, ids.failed
    )


def test_message_to_idle_node_saves_to_the_brief(
    page: Page, metaharness_server: MetaharnessServer
) -> None:
    ids = metaharness_server.ids
    _open(page, metaharness_server)
    _node(page, ids.drawer).click()
    expect(page.get_by_text("saved to the brief")).to_be_visible()
    _send(page, "Build on the final issue schema.")

    wait_until(
        lambda: (
            "Operator note: Build on the final issue schema."
            in _description(metaharness_server, ids.drawer)
        )
    )
    expect(page.get_by_role("log", name="Conversation")).to_contain_text(
        "Build on the final issue schema."
    )


def test_message_to_running_worker_steers_it(
    page: Page, metaharness_server: MetaharnessServer
) -> None:
    ids = metaharness_server.ids
    _open(page, metaharness_server)
    _node(page, ids.fingerprint).click()
    expect(page.get_by_text("delivered on next poll")).to_be_visible()
    _send(page, "Bucket viewports by breakpoint.")

    wait_until(
        lambda: any(
            command.text == "Bucket viewports by breakpoint."
            for command in metaharness_server.graph.commands.pending(ids.fingerprint_run)
        )
    )


def test_permission_approve_admits_the_command(
    page: Page, metaharness_server: MetaharnessServer
) -> None:
    ids = metaharness_server.ids
    _open(page, metaharness_server)
    _node(page, ids.annotation).click()
    page.get_by_role("dialog", name="Worker asks").get_by_role("button", name="Approve").click()

    wait_until(
        lambda: any(
            command.action == "approve"
            for command in metaharness_server.graph.commands.pending(ids.annotation_run)
        )
    )


def test_diff_takeover_shows_the_worktree_change(
    page: Page, metaharness_server: MetaharnessServer
) -> None:
    ids = metaharness_server.ids
    _open(page, metaharness_server)
    _node(page, ids.annotation).click()
    page.keyboard.press("Escape")
    page.locator(".mh-chat-actions").get_by_role("button", name="Diff").click()

    expect(page.get_by_text("queue.js")).to_be_visible()
    expect(page.get_by_text("export const queue = await store.load();")).to_be_visible()
    expect(page.get_by_role("button", name="Approve")).to_be_visible()
    page.get_by_role("button", name="Back to chat").click()
    expect(page.get_by_role("log", name="Conversation")).to_be_visible()


def test_rail_pauses_loop_and_queues_issue_fix(
    page: Page, metaharness_server: MetaharnessServer
) -> None:
    _open(page, metaharness_server)
    rail = page.get_by_role("navigation", name="Metaharness menu")
    page.mouse.move(700, 450)
    page.mouse.move(2, 400)
    rail.get_by_role("button", name="Pause run loop").click()
    wait_until(lambda: metaharness_server.scheduling.calls == ["pause"])
    expect(page.locator(".mh-loop")).to_contain_text("paused")

    page.mouse.move(700, 450)
    page.mouse.move(2, 400)
    rail.get_by_role("button", name="Issues").click()
    issues = page.get_by_role("complementary")
    expect(issues).to_contain_text("Claude Code adapter is blocked but has no open prerequisites.")
    issues.get_by_role("button", name="Queue fix").click()
    wait_until(
        lambda: any(
            node.description.startswith("Fix: Claude Code adapter")
            for node in metaharness_server.graph.get_all_nodes()
        )
    )

    page.mouse.move(700, 450)
    page.mouse.move(2, 400)
    rail.get_by_role("button", name="Stats").click()
    expect(page.get_by_role("complementary")).to_contain_text("2/4")


def test_expanded_graph_projects_and_classic_switch(
    page: Page, metaharness_server: MetaharnessServer
) -> None:
    _open(page, metaharness_server)
    page.keyboard.press("g")
    expect(page.get_by_role("dialog", name="Expanded graph")).to_contain_text("Inbox drawer UI")
    page.keyboard.press("Escape")
    expect(page.get_by_role("dialog", name="Expanded graph")).to_be_hidden()

    page.locator(".mh-crumbs").get_by_role("button", name="Projects").click()
    expect(page.locator(".mh-project")).to_contain_text("Tern browser UI")
    page.mouse.move(700, 450)
    page.mouse.move(2, 400)
    page.get_by_role("button", name="Classic view").click()
    expect(page.locator('[data-view="metaharness"]')).to_be_hidden()


def test_long_goal_title_keeps_panes_inside_the_viewport(
    page: Page, metaharness_server: MetaharnessServer
) -> None:
    ids = metaharness_server.ids
    long_title = "Burn down every open issue into reviewable stacked PRs " * 6
    _ = metaharness_server.graph.edit_node(
        ids.goal, long_title.strip(), None, None, None, None, frozenset({"implement"})
    )
    _open(page, metaharness_server)
    page.locator(".mh-crumbs").get_by_role("button", name="Projects").click()
    page.locator(".mh-project").first.click()

    expect(_node(page, ids.goal)).to_be_in_viewport()
    actions = page.locator(".mh-chat-actions").bounding_box()
    assert actions is not None
    assert actions["x"] + actions["width"] <= 1440
