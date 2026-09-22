from __future__ import annotations

from dataclasses import replace

import pytest
from textual.widgets import Static

from milknado.app.run import ExecutionRunStatus, TerminalRunSnapshot
from milknado.domains.graph import GraphSnapshot, NodeDetailResponse
from tests.graph_navigation_fixtures import run_app, source


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["run", "watch"])
async def test_snapshot_refresh_keeps_selected_node_without_a_run(kind: str) -> None:
    source_value = source()
    app = run_app(source_value, kind)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.select_node(2)
        assert (app.selected_node_id, app.selected_run_id) == (2, None)

        app.show_snapshot(source_value.current)
        await pilot.pause()

        assert (app.selected_node_id, app.selected_run_id) == (2, None)


@pytest.mark.asyncio
async def test_snapshot_refresh_prefers_active_run_for_selected_node() -> None:
    source_value = source()
    app = run_app(source_value, "watch")
    replacement = replace(
        source_value.current.active_runs[0],
        run_id="run-2",
        progress="replacement",
    )
    refreshed = replace(source_value.current, active_runs=(replacement,))

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.show_snapshot(refreshed)
        await pilot.pause()

        assert (app.selected_node_id, app.selected_run_id) == (1, "run-2")
        assert app.selected_run() is replacement


@pytest.mark.asyncio
async def test_graph_j_and_k_move_nodes_without_runs_and_sync_selection() -> None:
    source_value = source()
    app = run_app(source_value, "watch")

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert app.screen.focused is app.query_one("#graph-tree")

        await pilot.press("j")
        await pilot.pause()

        assert (app.selected_node_id, app.selected_run_id) == (2, None)

        await pilot.press("k")
        await pilot.pause()

        assert (app.selected_node_id, app.selected_run_id) == (1, "run-1")


@pytest.mark.asyncio
async def test_graph_none_keeps_legacy_run_selection() -> None:
    source_value = source()
    app = run_app(source_value, "watch")
    snapshot = replace(source_value.current, graph=None)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.show_snapshot(snapshot)
        await pilot.pause()

        assert app.selected_run_id == "run-1"
        assert app.selected_node_id == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["run", "watch"])
async def test_empty_graph_clears_selection_and_stale_detail(kind: str) -> None:
    source_value = source()
    app = run_app(source_value, kind)
    empty = replace(source_value.current, graph=GraphSnapshot((), (), ()))

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.node_detail = NodeDetailResponse(1, app.node_request_generation, None)
        app.show_snapshot(empty)
        await pilot.pause()
        assert app.screen.focused is app.query_one("#runs")
        assert app.selected_node_id is None
        assert app.selected_run_id is None
        assert app.node_detail is None
        await pilot.press("e")
        await pilot.pause()
        focused = app.screen.focused
        await pilot.press("j")
        await pilot.pause()

        assert focused is app.screen.focused
        assert (app.selected_node_id, app.selected_run_id) == (1, "run-1")
        assert app.node_detail is None


@pytest.mark.asyncio
async def test_compact_help_advertises_graph_only_open_and_back() -> None:
    source_value = source()
    app = run_app(source_value, "watch")
    graph_only = replace(source_value.current, active_runs=())

    async with app.run_test(size=(80, 24)) as pilot:
        app.show_snapshot(graph_only)
        await pilot.pause()
        await pilot.press("h")
        await pilot.pause()
        body = app.screen.query_one("#help-overlay", Static).render()
        assert "enter open" in str(body).lower()

        await pilot.press("escape", "enter")
        await pilot.pause()
        assert app.route == "detail"
        app.action_help()
        await pilot.pause()
        body = app.screen.query_one("#help-overlay", Static).render()
        assert "escape back" in str(body).lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["run", "watch"])
async def test_graph_refresh_does_not_steal_focus_from_modal(kind: str) -> None:
    source_value = source()
    app = run_app(source_value, kind)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.action_help()
        await pilot.pause()
        modal = app.screen
        focused = modal.focused
        assert modal.is_modal
        assert focused is not None

        app.show_snapshot(source_value.current)
        await pilot.pause()

        assert app.screen is modal
        assert app.screen.focused is focused


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["run", "watch"])
async def test_refresh_drops_disappeared_run_without_switching_node(kind: str) -> None:
    source_value = source()
    other_run = replace(
        source_value.current.active_runs[0],
        run_id="run-2",
        node_id=2,
        description="tree child",
    )
    refreshed = replace(source_value.current, active_runs=(other_run,))
    app = run_app(source_value, kind)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert (app.selected_node_id, app.selected_run_id) == (1, "run-1")
        app.show_snapshot(refreshed)
        await pilot.pause()

        assert (app.selected_node_id, app.selected_run_id) == (1, None)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["run", "watch"])
async def test_active_replacement_wins_over_retained_terminal_run(kind: str) -> None:
    source_value = source()
    terminal = TerminalRunSnapshot(
        run_id="run-old",
        node_id=1,
        description="root node",
        status=ExecutionRunStatus.COMPLETED,
        output=(),
        pending_guidance=None,
        duration_seconds=2.0,
    )
    replacement = replace(source_value.current.active_runs[0], run_id="run-new")
    refreshed = replace(
        source_value.current,
        active_runs=(replacement,),
        terminal_runs=(terminal,),
    )
    app = run_app(source_value, kind)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.show_snapshot(replace(source_value.current, active_runs=(), terminal_runs=(terminal,)))
        await pilot.pause()
        assert (app.selected_node_id, app.selected_run_id) == (1, "run-old")

        app.show_snapshot(refreshed)
        await pilot.pause()

        assert (app.selected_node_id, app.selected_run_id) == (1, "run-new")


@pytest.mark.asyncio
async def test_graph_transition_focuses_visible_tree() -> None:
    source_value = source()
    app = run_app(source_value, "watch")
    graphless = replace(source_value.current, graph=None)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.show_snapshot(graphless)
        await pilot.pause()
        assert app.screen.focused is app.query_one("#runs")

        app.show_snapshot(source_value.current)
        await pilot.pause()
        assert app.screen.focused is app.query_one("#graph-tree")


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["run", "watch"])
async def test_disappeared_selected_node_moves_to_remaining_graph_node(kind: str) -> None:
    source_value = source()
    app = run_app(source_value, kind)
    selected_graph = replace(source_value.current, active_runs=())
    graph = source_value.current.graph
    assert graph is not None
    remaining_graph = replace(graph, nodes=(graph.nodes[0],), edges=(), root_ids=(1,))
    refreshed = replace(
        selected_graph,
        graph=remaining_graph,
        active_runs=source_value.current.active_runs,
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.select_node(2)
        app.show_snapshot(selected_graph)
        await pilot.pause()
        app.show_snapshot(refreshed)
        await pilot.pause()

        assert (app.selected_node_id, app.selected_run_id) == (1, "run-1")


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["run", "watch"])
async def test_graphless_refresh_replaces_missing_run_and_clears_empty(kind: str) -> None:
    source_value = source()
    app = run_app(source_value, kind)
    replacement = replace(source_value.current.active_runs[0], run_id="run-2", node_id=2)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.show_snapshot(replace(source_value.current, graph=None, active_runs=(replacement,)))
        await pilot.pause()
        assert (app.selected_node_id, app.selected_run_id) == (2, "run-2")

        app.show_snapshot(replace(source_value.current, graph=None, active_runs=()))
        await pilot.pause()
        assert (app.selected_node_id, app.selected_run_id) == (None, None)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["run", "watch"])
async def test_empty_graph_navigates_retained_runs(kind: str) -> None:
    source_value = source()
    terminal = TerminalRunSnapshot("run", 0, "run", ExecutionRunStatus.COMPLETED, (), None, 1.0)
    retained = (
        replace(terminal, run_id="run-a", node_id=1),
        replace(terminal, run_id="run-b", node_id=2),
    )
    empty = replace(
        source_value.current,
        graph=GraphSnapshot((), (), ()),
        active_runs=(),
        terminal_runs=retained,
    )
    app = run_app(source_value, kind)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.show_snapshot(empty)
        await pilot.pause()
        app.set_focus(app.query_one("#runs"))
        await pilot.press("j")
        await pilot.pause()
        assert app.selected_run_id == "run-a"
        await pilot.press("k")
        await pilot.pause()
        assert app.selected_run_id == "run-b"
