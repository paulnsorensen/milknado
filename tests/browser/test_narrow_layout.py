"""Narrow and responsive dashboard layouts at operator viewport widths."""

import re
from collections.abc import Iterator
from dataclasses import replace
from typing import Any

import pytest
from playwright.sync_api import FloatRect, Page, ViewportSize, expect

from milknado.app.run_source import (
    ActiveRunSnapshot,
    ExecutionRunStatus,
    NodeSnapshotRequest,
    RunActionAvailability,
)
from milknado.domains.common import MikadoNode, NodeKind
from milknado.domains.graph import (
    NodeDetailResponse,
    NodeDetailSnapshot,
    OwnerCapabilities,
    SnapshotPage,
    SnapshotValue,
)
from milknado.web import LaunchToken, WebCommands, create_app
from tests.browser.conftest import (
    BROWSER_TOKEN,
    BrowserServer,
    BrowserSnapshotSource,
    RecordingCommands,
    open_app,
    owner_web_commands,
    wait_until,
)

pytestmark = pytest.mark.browser

NARROW_VIEWPORT: ViewportSize = {"width": 390, "height": 844}
ROSTER_VIEWPORT: ViewportSize = {"width": 800, "height": 800}
RESPONSIVE_VIEWPORTS: tuple[ViewportSize, ...] = (
    {"width": 401, "height": 844},
    {"width": 639, "height": 844},
    {"width": 640, "height": 844},
    {"width": 1179, "height": 844},
)
NODE_DESCRIPTION = "Tracer fixture node"
LONG_GOAL_DESCRIPTION = (
    "This intentionally long goal description exposes responsive header growth without "
    + "changing the fixture's selected node. "
) * 5

LONG_AGENT_DESCRIPTION = (
    "Deterministic fixture only. Do not inspect the repository, edit files, "
    "or report success with a fake receipt."
)
MIN_TOUCH_TARGET = 44


def _empty_page() -> SnapshotPage[Any]:  # pyright: ignore[reportExplicitAny]
    return SnapshotPage(items=(), offset=0, limit=50, total=0, has_more=False)


def _detail_response(request: NodeSnapshotRequest) -> NodeDetailResponse:
    return NodeDetailResponse(
        node_id=request.node_id,
        request_generation=request.request_generation,
        detail=NodeDetailSnapshot(
            node=MikadoNode(id=request.node_id, description=NODE_DESCRIPTION, kind=NodeKind.GOAL),
            description=NODE_DESCRIPTION,
            parent=None,
            children=_empty_page(),
            ancestors=_empty_page(),
            prerequisite_ids=_empty_page(),
            dependent_ids=_empty_page(),
            reverse_dependents=_empty_page(),
            owned_files=_empty_page(),
            runs=_empty_page(),
            reviews=_empty_page(),
            sessions=_empty_page(),
            receipts=_empty_page(),
            goal_claim=SnapshotValue(value=None, state="not_stored"),
            artifacts=_empty_page(),
        ),
    )


@pytest.fixture
def narrow_server() -> Iterator[BrowserServer]:
    login = LaunchToken(BROWSER_TOKEN)
    source = BrowserSnapshotSource()
    source.node_snapshot = _detail_response  # type: ignore[method-assign]
    commands = WebCommands()
    app = create_app(source, commands, login)
    server = BrowserServer(app=app, login=login)
    server.start()
    yield server
    server.stop()


@pytest.fixture
def owner_narrow_source() -> BrowserSnapshotSource:
    source = BrowserSnapshotSource()
    source.node_snapshot = _detail_response  # type: ignore[method-assign]
    return source


@pytest.fixture
def owner_narrow_server(
    owner_narrow_source: BrowserSnapshotSource,
) -> Iterator[tuple[BrowserServer, RecordingCommands]]:
    login = LaunchToken(BROWSER_TOKEN)
    commands, recorder = owner_web_commands()
    commands = replace(
        commands,
        owner_capabilities=OwnerCapabilities(
            run_id="narrow-run",
            node_id=1,
            invocation_id="narrow-invocation",
            owner_incarnation="narrow-owner",
            actions=("steer", "follow_up", "interrupt"),
            permission_ids=(),
            published_at="2026-09-28T00:00:00Z",
        ),
    )
    app = create_app(owner_narrow_source, commands, login)
    server = BrowserServer(app=app, login=login)
    server.start()
    yield server, recorder
    server.stop()


def _has_no_horizontal_scroll(page: Page) -> bool:
    overflow = page.evaluate(  # pyright: ignore[reportAny]
        "() => document.scrollingElement.scrollWidth <= document.scrollingElement.clientWidth"
    )
    return bool(overflow)  # pyright: ignore[reportAny]


def _assert_box_horizontally_inside_viewport(box: FloatRect, viewport: ViewportSize) -> None:
    assert box["x"] >= 0
    assert box["x"] + box["width"] <= viewport["width"]


def _assert_box_inside_viewport(box: FloatRect, viewport: ViewportSize) -> None:
    _assert_box_horizontally_inside_viewport(box, viewport)
    assert box["y"] >= 0
    assert box["y"] + box["height"] <= viewport["height"]


def _assert_regions_fit(page: Page, viewport: ViewportSize) -> None:
    for region in ("rail", "main", "sidecar"):
        locator = page.locator(f"[data-region='{region}']")
        box = locator.bounding_box()
        if box is None:
            continue
        _assert_box_horizontally_inside_viewport(box, viewport)
        assert locator.evaluate("(element) => element.scrollWidth <= element.clientWidth") is True


def _assert_visible_controls_fit(page: Page, viewport: ViewportSize) -> None:
    selector = ", ".join(
        (
            ".mk-shell button:visible",
            ".mk-shell [role='tab']:visible",
            ".mk-shell select:visible",
            ".mk-shell input:visible",
        )
    )
    controls = page.locator(selector)
    for index in range(controls.count()):
        control = controls.nth(index)
        control.scroll_into_view_if_needed()
        expect(control).to_be_visible()
        box = control.bounding_box()
        assert box is not None
        _assert_box_horizontally_inside_viewport(box, viewport)


@pytest.mark.parametrize(
    "viewport",
    RESPONSIVE_VIEWPORTS,
    ids=("401px", "639px", "640px", "1179px"),
)
def test_responsive_layout_keeps_operator_controls_inside_viewport(
    page: Page,
    owner_narrow_server: tuple[BrowserServer, RecordingCommands],
    viewport: ViewportSize,
) -> None:
    server, _ = owner_narrow_server
    page.set_viewport_size(viewport)
    node = page.locator(".mk-node").first
    open_app(page, server.login_url, node)

    stop_scheduling = page.get_by_role("button", name="Stop scheduling", exact=True)
    expect(stop_scheduling).to_be_visible()
    stop_scheduling.click()
    dialog = page.get_by_role("alertdialog")
    expect(dialog).to_be_visible()
    dialog_box = dialog.bounding_box()
    assert dialog_box is not None
    _assert_box_inside_viewport(dialog_box, viewport)
    dialog.get_by_role("button", name="Keep running", exact=True).click()

    node.click()
    expect(page.get_by_role("button", name="Cancel run", exact=True)).to_be_visible()
    _assert_regions_fit(page, viewport)
    _assert_visible_controls_fit(page, viewport)


def test_long_goal_title_is_clamped_at_401px(
    page: Page,
    owner_narrow_server: tuple[BrowserServer, RecordingCommands],
    owner_narrow_source: BrowserSnapshotSource,
) -> None:
    snapshot = owner_narrow_source.snapshot()
    assert snapshot.graph is not None
    root = replace(snapshot.graph.nodes[0], description=LONG_GOAL_DESCRIPTION)
    owner_narrow_source.publish(replace(snapshot, graph=replace(snapshot.graph, nodes=(root,))))

    viewport: ViewportSize = {"width": 401, "height": 844}
    page.set_viewport_size(viewport)
    server, _ = owner_narrow_server
    node = page.locator(".mk-node").first
    open_app(page, server.login_url, node)

    title = page.locator(".mk-goal-title h1")
    expect(title).to_be_visible()
    title_box = title.bounding_box()
    assert title_box is not None
    assert title_box["height"] <= 2 * 38

    stop_scheduling = page.get_by_role("button", name="Stop scheduling", exact=True)
    expect(stop_scheduling).to_be_visible()
    stop_box = stop_scheduling.bounding_box()
    assert stop_box is not None
    _assert_box_inside_viewport(stop_box, viewport)


def test_events_dock_keeps_graph_canvas_visible_at_401px(
    page: Page,
    owner_narrow_server: tuple[BrowserServer, RecordingCommands],
) -> None:
    viewport: ViewportSize = {"width": 401, "height": 844}
    page.set_viewport_size(viewport)
    server, _ = owner_narrow_server
    node = page.locator(".mk-node").first
    open_app(page, server.login_url, node)

    page.get_by_role("button", name=re.compile(r"^Events")).click()
    expect(page.get_by_role("button", name=re.compile(r"^Hide events"))).to_be_visible()

    canvas_box = page.locator("[data-region='canvas']").bounding_box()
    assert canvas_box is not None
    assert canvas_box["height"] >= 240
    expect(node).to_be_visible()
    node.click()
    expect(page.get_by_role("button", name="Cancel run", exact=True)).to_be_visible()


def test_responsive_toast_stays_inside_640px_viewport(
    page: Page,
    owner_narrow_server: tuple[BrowserServer, RecordingCommands],
) -> None:
    server, _ = owner_narrow_server
    viewport: ViewportSize = {"width": 640, "height": 844}
    page.set_viewport_size(viewport)
    _ = page.route(
        "**/api/scheduling/stop",
        lambda route: route.fulfill(
            status=409,
            content_type="application/json",
            body='{"reason":"Scheduling was already stopped."}',
        ),
    )
    stop_scheduling = page.get_by_role("button", name="Stop scheduling", exact=True)
    open_app(page, server.login_url, stop_scheduling)
    stop_scheduling.click()
    page.get_by_role("alertdialog").get_by_role("button", name="Stop runs", exact=True).click()

    toast = page.locator(".mk-toast")
    expect(toast).to_be_visible()
    expect(toast).to_contain_text("Scheduling was already stopped.")
    expect(toast.get_by_role("button", name="Dismiss", exact=True)).to_be_visible()
    toast_host = page.locator("[data-region='toast']")
    expect(toast_host).to_have_css("position", "fixed")
    toast_box = toast.bounding_box()
    assert toast_box is not None
    _assert_box_inside_viewport(toast_box, viewport)


def test_narrow_list_view_shows_outline_with_no_horizontal_scroll(
    page: Page, narrow_server: BrowserServer
) -> None:
    page.set_viewport_size(NARROW_VIEWPORT)
    open_app(page, narrow_server.login_url, page.get_by_role("treeitem", name=NODE_DESCRIPTION))

    expect(page.get_by_label("Jump to node")).to_be_visible()
    assert _has_no_horizontal_scroll(page)


def test_narrow_list_navigation_opens_the_selected_node(
    page: Page, narrow_server: BrowserServer
) -> None:
    page.set_viewport_size(NARROW_VIEWPORT)
    open_app(page, narrow_server.login_url, page.get_by_role("treeitem", name=NODE_DESCRIPTION))

    page.get_by_role("treeitem", name=NODE_DESCRIPTION).click()
    expect(page.get_by_text("node 1", exact=True)).to_be_visible()
    page.get_by_role("button", name="Open node").click()

    expect(page.get_by_role("button", name="Back to list")).to_be_visible()


def test_narrow_list_jump_to_node_selects_the_requested_node(
    page: Page, narrow_server: BrowserServer
) -> None:
    page.set_viewport_size(NARROW_VIEWPORT)
    open_app(page, narrow_server.login_url, page.get_by_role("treeitem", name=NODE_DESCRIPTION))

    jump = page.get_by_label("Jump to node")
    jump.fill("1")
    jump.press("Enter")

    expect(page.get_by_text("node 1", exact=True)).to_be_visible()


def test_narrow_detail_view_is_full_width_with_44px_controls_and_no_horizontal_scroll(
    page: Page, narrow_server: BrowserServer
) -> None:
    page.set_viewport_size(NARROW_VIEWPORT)
    open_app(page, narrow_server.login_url, page.get_by_role("treeitem", name=NODE_DESCRIPTION))

    page.get_by_role("treeitem", name=NODE_DESCRIPTION).click()
    page.get_by_role("button", name="Open node").click()

    back_button = page.get_by_role("button", name="Back to list")
    expect(back_button).to_be_visible()
    box = back_button.bounding_box()
    assert box is not None
    assert box["width"] >= MIN_TOUCH_TARGET
    assert box["height"] >= MIN_TOUCH_TARGET

    assert _has_no_horizontal_scroll(page)


def test_narrow_detail_back_link_returns_to_the_list(
    page: Page, narrow_server: BrowserServer
) -> None:
    page.set_viewport_size(NARROW_VIEWPORT)
    open_app(page, narrow_server.login_url, page.get_by_role("treeitem", name=NODE_DESCRIPTION))

    page.get_by_role("treeitem", name=NODE_DESCRIPTION).click()
    page.get_by_role("button", name="Open node").click()
    page.get_by_role("button", name="Back to list").click()

    expect(page.get_by_label("Jump to node")).to_be_visible()


def test_narrow_owner_footer_controls_confirm_commands(
    page: Page, owner_narrow_server: tuple[BrowserServer, RecordingCommands]
) -> None:
    server, recorder = owner_narrow_server
    page.set_viewport_size(NARROW_VIEWPORT)
    open_app(page, server.login_url, page.get_by_role("treeitem", name=NODE_DESCRIPTION))

    page.get_by_role("treeitem", name=NODE_DESCRIPTION).click()
    page.get_by_role("button", name="Open node").click()

    expect(page.get_by_role("button", name="Cancel run")).to_be_enabled()
    expect(page.get_by_role("button", name="Force stop")).to_be_enabled()
    for label in ("Steer", "Follow up", "Interrupt", "Send"):
        control = page.get_by_role("button", name=label)
        expect(control).to_be_visible()
        box = control.bounding_box()
        assert box is not None
        assert box["width"] >= MIN_TOUCH_TARGET
        assert box["height"] >= MIN_TOUCH_TARGET
    segment_box = page.locator(".mk-seg").bounding_box()
    send_box = page.get_by_role("button", name="Send").bounding_box()
    assert segment_box is not None
    assert send_box is not None
    assert segment_box["width"] < NARROW_VIEWPORT["width"] - 32
    assert send_box["width"] < NARROW_VIEWPORT["width"] - 32
    footer = page.locator('[data-region="run-controls"]')
    footer_box = footer.bounding_box()
    assert footer_box is not None
    assert footer_box["y"] + footer_box["height"] >= page.evaluate("window.innerHeight") - 1

    guidance = page.get_by_label("Session guidance")
    guidance.fill("Pause before the next turn")
    page.get_by_role("button", name="Interrupt").click()
    assert recorder.session_input_calls == []
    expect(page.get_by_role("button", name="Interrupt")).to_have_attribute("aria-pressed", "true")
    page.get_by_role("button", name="Send").click()

    wait_until(lambda: len(recorder.session_input_calls) == 1)
    run_id, request = recorder.session_input_calls[0]
    assert run_id == "narrow-run"
    assert request.action == "interrupt"
    assert request.text == ""

    page.get_by_role("button", name="Cancel run").click()
    page.get_by_role("alertdialog").get_by_role("button", name="Cancel run", exact=True).click()
    page.get_by_role("button", name="Force stop").click()
    page.get_by_role("alertdialog").get_by_role("button", name="Force stop", exact=True).click()

    wait_until(lambda: recorder.cancel_calls == ["narrow-run"])
    wait_until(lambda: recorder.force_stop_calls == ["narrow-run"])


def test_narrow_detail_tabs_use_aria_roles_and_roving_focus(
    page: Page, narrow_server: BrowserServer
) -> None:
    page.set_viewport_size(NARROW_VIEWPORT)
    open_app(page, narrow_server.login_url, page.get_by_role("treeitem", name=NODE_DESCRIPTION))

    page.get_by_role("treeitem", name=NODE_DESCRIPTION).click()
    page.get_by_role("button", name="Open node").click()

    tablist = page.get_by_role("tablist", name="Node detail")
    expect(tablist).to_be_visible()
    expect(tablist.get_by_role("tab")).to_have_count(3)

    tabs = {
        label: tablist.get_by_role("tab", name=label)
        for label in ("Session", "Changes", "Details")
    }
    for label, slug in (("Session", "session"), ("Changes", "changes"), ("Details", "details")):
        tab = tabs[label]
        panel = page.locator(f"#node-detail-panel-{slug}")
        expect(tab).to_have_attribute("aria-controls", f"node-detail-panel-{slug}")
        expect(panel).to_have_attribute("role", "tabpanel")
        expect(panel).to_have_attribute("aria-labelledby", f"node-detail-tab-{slug}")

    session = tabs["Session"]
    changes = tabs["Changes"]
    details = tabs["Details"]
    expect(session).to_have_attribute("aria-selected", "true")
    expect(session).to_have_attribute("tabindex", "0")

    session.press("ArrowRight")
    expect(changes).to_be_focused()
    expect(changes).to_have_attribute("aria-selected", "true")
    changes.press("ArrowRight")
    expect(details).to_be_focused()
    details.press("ArrowRight")
    expect(session).to_be_focused()


def test_long_agent_description_is_clamped_in_the_roster(
    page: Page, browser_server: BrowserServer, browser_source: BrowserSnapshotSource
) -> None:
    page.set_viewport_size(ROSTER_VIEWPORT)
    browser_source.publish(
        replace(
            browser_source.snapshot(),
            active_runs=(
                ActiveRunSnapshot(
                    run_id="run-long-description",
                    node_id=1,
                    description=LONG_AGENT_DESCRIPTION,
                    status=ExecutionRunStatus.RUNNING,
                    progress=None,
                    stop_requested=False,
                    actions=RunActionAvailability(),
                    output=(),
                    pending_guidance=None,
                    elapsed_seconds=0,
                    progress_pct=None,
                    eta_seconds=None,
                    attempt=None,
                    max_attempts=None,
                    stalled=False,
                ),
            ),
            available=0,
        )
    )
    agent_name = page.locator(".mk-agent-roster .mk-agent-name b").first
    open_app(page, browser_server.login_url, agent_name)

    expect(agent_name).to_be_visible()
    expect(agent_name).to_have_text(LONG_AGENT_DESCRIPTION)
    expect(agent_name).to_have_css("display", "block")
    expect(agent_name).to_have_css("overflow", "hidden")
    expect(agent_name).to_have_css("text-overflow", "ellipsis")
    expect(agent_name).to_have_css("white-space", "nowrap")

    agent_row = agent_name.locator("..").locator("..")
    agent_box = agent_name.bounding_box()
    row_box = agent_row.bounding_box()
    assert agent_box is not None
    assert row_box is not None
    assert agent_box["height"] <= 16
    assert row_box["height"] <= 32
