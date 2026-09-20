"""AC-11: add, edit, move and archive each mutate the DB and the page."""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect

from tests.browser.conftest import open_app, wait_until
from tests.browser.graph_db import GraphDbServer, graph_db_server

_ = graph_db_server

pytestmark = pytest.mark.browser


def test_add_node_mutates_db_and_page(page: Page, graph_db_server: GraphDbServer) -> None:
    open_app(
        page, graph_db_server.login_url, page.get_by_role("button", name="Add node", exact=True)
    )

    page.get_by_role("button", name="Add node", exact=True).click()
    page.get_by_label("Description").fill("Newly added node")
    page.get_by_role("dialog", name="Add node").get_by_role("button", name="Add node").click()

    wait_until(
        lambda: any(
            node.description == "Newly added node"
            for node in graph_db_server.graph.get_all_nodes()
        )
    )
    expect(page.get_by_role("button", name="Newly added node")).to_be_visible()


def test_edit_node_mutates_db_and_page(page: Page, graph_db_server: GraphDbServer) -> None:
    open_app(page, graph_db_server.login_url, page.get_by_role("button", name="Edit target"))

    page.get_by_role("button", name="Edit target").click()
    page.get_by_role("button", name="Edit node", exact=True).click()
    page.get_by_label("Edit description").fill("Edited target")
    page.get_by_role("button", name="Save changes").click()

    wait_until(
        lambda: (
            (node := graph_db_server.graph.get_node(graph_db_server.edit_node_id)) is not None
            and node.description == "Edited target"
        )
    )
    expect(page.locator("button.mk-node", has_text="Edited target")).to_be_visible()


def test_move_node_mutates_db_and_page(page: Page, graph_db_server: GraphDbServer) -> None:
    open_app(page, graph_db_server.login_url, page.get_by_role("button", name="Move target"))

    page.get_by_role("button", name="Move target").click()
    page.get_by_role("button", name="Move node", exact=True).click()
    _ = page.get_by_label("New parent").select_option(label="Target parent")
    page.get_by_role("dialog", name="Move node").get_by_role("button", name="Move node").click()

    wait_until(
        lambda: (
            (node := graph_db_server.graph.get_node(graph_db_server.move_node_id)) is not None
            and node.parent_id == graph_db_server.target_parent_id
        )
    )
    expect(page.locator("button.mk-node", has_text="Move target")).to_be_visible()


def test_archive_node_mutates_db_and_page(page: Page, graph_db_server: GraphDbServer) -> None:
    open_app(page, graph_db_server.login_url, page.get_by_role("button", name="Archive target"))

    page.get_by_role("button", name="Archive target").click()
    page.get_by_role("button", name="Archive node", exact=True).click()
    page.get_by_role("alertdialog", name="Archive node").get_by_role(
        "button", name="Archive node"
    ).click()

    wait_until(
        lambda: (
            (node := graph_db_server.graph.get_node(graph_db_server.archive_node_id)) is not None
            and node.archived_at is not None
        )
    )
    expect(page.locator("button.mk-node", has_text="Archive target")).not_to_be_visible()
