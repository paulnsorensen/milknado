"""Graph editing in the metaharness view against a real graph.db: drag-to-move,
add, edit, archive, run cancel, and new goals all reach the milknado backend."""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect

from milknado.domains.common import NodeKind
from tests.browser.conftest import wait_until
from tests.browser.metaharness_db import (
    MetaharnessServer,
    compact_node,
    metaharness_server,
    open_view,
)

_ = metaharness_server

pytestmark = pytest.mark.browser


def _parent(server: MetaharnessServer, node_id: int) -> int | None:
    node = server.graph.get_node(node_id)
    assert node is not None
    return node.parent_id


def test_drag_moves_a_node_under_another_and_to_the_top_level(
    page: Page, metaharness_server: MetaharnessServer
) -> None:
    ids = metaharness_server.ids
    open_view(page, metaharness_server)

    compact_node(page, ids.failed).drag_to(compact_node(page, ids.stuck))
    wait_until(lambda: _parent(metaharness_server, ids.failed) == ids.stuck)
    expect(page.get_by_role("status")).to_contain_text(
        "Session stats strip is now a prerequisite of Claude Code adapter"
    )

    projects = page.locator(".mh-crumbs").get_by_role("button", name="Projects")
    compact_node(page, ids.stuck).drag_to(projects)
    wait_until(lambda: _parent(metaharness_server, ids.stuck) is None)


def test_card_plus_and_shortcut_open_add_with_the_parent_preset(
    page: Page, metaharness_server: MetaharnessServer
) -> None:
    ids = metaharness_server.ids
    open_view(page, metaharness_server)
    compact_node(page, ids.annotation).hover()
    page.get_by_role("button", name="Add a prerequisite to Annotation queued prompt").click()

    dialog = page.get_by_role("dialog", name="Add node")
    expect(dialog.get_by_label("Parent")).to_have_value(str(ids.annotation))
    dialog.get_by_label("Description").fill("Write queue fixtures")
    dialog.get_by_role("button", name="Add node").click()
    wait_until(
        lambda: any(
            node.description == "Write queue fixtures" and node.parent_id == ids.annotation
            for node in metaharness_server.graph.get_all_nodes()
        )
    )

    page.keyboard.press("Escape")
    compact_node(page, ids.drawer).click()
    page.keyboard.press("a")
    expect(dialog.get_by_label("Parent")).to_have_value(str(ids.drawer))


def test_details_actions_edit_then_archive_the_node(
    page: Page, metaharness_server: MetaharnessServer
) -> None:
    graph = metaharness_server.graph
    node_id = graph.add_node("Retire old queue", metaharness_server.ids.goal).id
    open_view(page, metaharness_server)
    compact_node(page, node_id).click()
    page.get_by_role("button", name="Details").click()
    details = page.get_by_role("complementary")

    details.get_by_role("button", name="Edit").click()
    edit = page.get_by_role("dialog", name="Edit node")
    edit.get_by_label("Edit description").fill("Retire the in-memory queue")
    edit.get_by_role("button", name="Save changes").click()
    wait_until(lambda: _description(metaharness_server, node_id) == "Retire the in-memory queue")

    graph.mark_running(node_id)
    graph.mark_done(node_id)
    details.get_by_role("button", name="Archive").click()
    page.get_by_role("alertdialog", name="Archive node").get_by_role(
        "button", name="Archive node"
    ).click()
    wait_until(lambda: _archived(metaharness_server, node_id))


def test_cancel_run_asks_first_then_cancels_that_run(
    page: Page, metaharness_server: MetaharnessServer
) -> None:
    ids = metaharness_server.ids
    open_view(page, metaharness_server)
    compact_node(page, ids.fingerprint).click()
    page.locator(".mh-chat-actions").get_by_role("button", name="Cancel run").click()

    confirm = page.get_by_role("alertdialog", name="Cancel the run on Issue fingerprinting?")
    confirm.get_by_role("button", name="Keep the run").click()
    assert metaharness_server.cancels.run_ids == []

    page.locator(".mh-chat-actions").get_by_role("button", name="Cancel run").click()
    confirm.get_by_role("button", name="Cancel run").click()
    wait_until(lambda: metaharness_server.cancels.run_ids == [ids.fingerprint_run])


def test_new_goal_starts_a_coordinator_goal(
    page: Page, metaharness_server: MetaharnessServer
) -> None:
    open_view(page, metaharness_server)
    page.locator(".mh-crumbs").get_by_role("button", name="Projects").click()
    page.get_by_role("button", name="New goal").first.click()

    dialog = page.get_by_role("dialog", name="New goal")
    dialog.get_by_label("Goal").fill("Ship the review inbox")
    dialog.get_by_role("button", name="codex").click()
    dialog.get_by_role("button", name="Start goal").click()

    wait_until(
        lambda: any(
            node.description == "Ship the review inbox"
            and node.kind is NodeKind.GOAL
            and node.parent_id is None
            for node in metaharness_server.graph.get_all_nodes()
        )
    )
    expect(page.locator(".mh-project", has_text="Ship the review inbox")).to_be_visible()


def _description(server: MetaharnessServer, node_id: int) -> str:
    node = server.graph.get_node(node_id)
    return node.description if node is not None else ""


def _archived(server: MetaharnessServer, node_id: int) -> bool:
    node = server.graph.get_node(node_id)
    return node is not None and node.archived_at is not None
