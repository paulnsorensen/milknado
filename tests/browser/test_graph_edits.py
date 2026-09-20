"""AC-11: add, edit, move and archive each mutate the DB and the page."""

from __future__ import annotations

import time
from collections.abc import Callable

import pytest
from playwright.sync_api import Page, expect

from tests.browser.graph_db import GraphDbServer, graph_db_server

_ = graph_db_server

pytestmark = pytest.mark.browser


def _wait_for(predicate: Callable[[], bool]) -> None:
    deadline = time.monotonic() + 5.0
    while not predicate():
        if time.monotonic() > deadline:
            raise TimeoutError("condition was not met within 5s")
        time.sleep(0.05)


def test_add_node_mutates_db_and_page(page: Page, graph_db_server: GraphDbServer) -> None:
    _ = page.goto(graph_db_server.login_url)
    page.wait_for_load_state("networkidle")

    page.get_by_role("button", name="Add node", exact=True).click()
    page.get_by_label("Description").fill("Newly added node")
    page.get_by_role("dialog", name="Add node").get_by_role("button", name="Add node").click()

    _wait_for(
        lambda: any(
            node.description == "Newly added node"
            for node in graph_db_server.graph.get_all_nodes()
        )
    )
    expect(page.get_by_role("button", name="Newly added node")).to_be_visible()


def test_edit_node_mutates_db_and_page(page: Page, graph_db_server: GraphDbServer) -> None:
    _ = page.goto(graph_db_server.login_url)
    page.wait_for_load_state("networkidle")

    page.get_by_role("button", name="Edit target").click()
    page.get_by_role("button", name="Edit node", exact=True).click()
    page.get_by_label("Edit description").fill("Edited target")
    page.get_by_role("button", name="Save changes").click()

    _wait_for(
        lambda: (
            graph_db_server.graph.get_node(graph_db_server.edit_node_id).description
            == "Edited target"
        )
    )
    expect(page.locator("button.mk-node", has_text="Edited target")).to_be_visible()


def test_move_node_mutates_db_and_page(page: Page, graph_db_server: GraphDbServer) -> None:
    _ = page.goto(graph_db_server.login_url)
    page.wait_for_load_state("networkidle")

    page.get_by_role("button", name="Move target").click()
    page.get_by_role("button", name="Move node", exact=True).click()
    page.get_by_label("New parent").select_option(label="Target parent")
    page.get_by_role("dialog", name="Move node").get_by_role("button", name="Move node").click()

    _wait_for(
        lambda: (
            graph_db_server.graph.get_node(graph_db_server.move_node_id).parent_id
            == graph_db_server.target_parent_id
        )
    )
    expect(page.locator("button.mk-node", has_text="Move target")).to_be_visible()


def test_archive_node_mutates_db_and_page(page: Page, graph_db_server: GraphDbServer) -> None:
    _ = page.goto(graph_db_server.login_url)
    page.wait_for_load_state("networkidle")

    page.get_by_role("button", name="Archive target").click()
    page.get_by_role("button", name="Archive node", exact=True).click()
    page.get_by_role("alertdialog", name="Archive node").get_by_role(
        "button", name="Archive node"
    ).click()

    _wait_for(
        lambda: (
            graph_db_server.graph.get_node(graph_db_server.archive_node_id).archived_at is not None
        )
    )
    expect(page.locator("button.mk-node", has_text="Archive target")).not_to_be_visible()
