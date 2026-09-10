from __future__ import annotations

from collections.abc import Callable
from typing import cast
from xml.etree import ElementTree

import pytest
from rich.console import RenderableType
from textual.widgets import DataTable, Static

from milknado.app.run import (
    ActiveRunSnapshot,
    ExecutionRunStatus,
    ExecutionSnapshot,
    RunActionAvailability,
)
from milknado.app.run_panels import RunListPanel
from milknado.app.run_view_app import ExecutionSnapshotApp


class _Source:
    def __init__(self, snapshot: ExecutionSnapshot) -> None:
        self.current: ExecutionSnapshot = snapshot
        self.listener: Callable[[ExecutionSnapshot], None] | None = None

    def snapshot(self) -> ExecutionSnapshot:
        return self.current

    def subscribe(self, listener: Callable[[ExecutionSnapshot], None]) -> Callable[[], None]:
        self.listener = listener
        return lambda: None

    def emit(self, snapshot: ExecutionSnapshot) -> None:
        self.current = snapshot
        assert self.listener is not None
        self.listener(snapshot)


def _run(run_id: str, node_id: int, description: str) -> ActiveRunSnapshot:
    return ActiveRunSnapshot(
        run_id=run_id,
        node_id=node_id,
        description=description,
        status=ExecutionRunStatus.RUNNING,
        progress="Implementing responsive layout",
        stop_requested=False,
        actions=RunActionAvailability(),
        output=(),
        pending_guidance=None,
        elapsed_seconds=12.0,
        progress_pct=100.0,
        eta_seconds=20.0,
        attempt=2,
        max_attempts=3,
        stalled=False,
    )


def _snapshot(runs: tuple[ActiveRunSnapshot, ...]) -> ExecutionSnapshot:
    return ExecutionSnapshot("Responsive runs", runs, (), 0, 0, 0, len(runs), ())


def _labels(table: DataTable[RenderableType]) -> tuple[str, ...]:
    return tuple(column.label.plain for column in table.columns.values())


def _table(app: ExecutionSnapshotApp) -> DataTable[RenderableType]:
    return cast(
        DataTable[RenderableType],
        app.query_one("#run-panel", RunListPanel).query_one("#runs", DataTable),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(40, 15), (80, 24), (120, 40)])
async def test_run_list_columns_match_terminal_width(size: tuple[int, int]) -> None:
    runs = tuple(
        _run(f"run-{index}", index + 11, "Repair the responsive execution dashboard layout")
        for index in range(1, 51)
    )
    source = _Source(_snapshot(runs))
    app = ExecutionSnapshotApp(source)

    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        table = _table(app)
        assert table.show_vertical_scrollbar
        assert not table.show_horizontal_scrollbar
        svg = ElementTree.fromstring(app.export_screenshot())
        rendered = " ".join(
            node.text or "" for node in svg.iter("{http://www.w3.org/2000/svg}text")
        ).replace("\xa0", " ")
        assert "running 2/3" in rendered


@pytest.mark.asyncio
async def test_run_list_preserves_cursor_and_focus_across_resize() -> None:
    runs = (
        _run("run-1", 12, "First run"),
        _run("run-2", 13, "Second run"),
    )
    source = _Source(_snapshot(runs))
    app = ExecutionSnapshotApp(source)

    async with app.run_test(size=(40, 15)) as pilot:
        table = _table(app)
        _ = table.focus()
        table.move_cursor(row=1)
        assert table.has_focus
        assert table.cursor_row == 1

        await pilot.resize_terminal(80, 24)
        assert table.has_focus
        assert table.cursor_row == 1
        assert table.get_row_index("run-2") == 1
        assert table.virtual_size.width <= table.size.width
        assert _labels(table) == ("Node", "Description", "Status", "Progress", "Elapsed")
        await pilot.resize_terminal(40, 15)
        assert _labels(table) == ("Node", "Description", "Status")
        assert table.has_focus and table.cursor_row == 1
        assert not table.show_horizontal_scrollbar
        source.emit(_snapshot(tuple(reversed(runs))))
        await pilot.pause()
        assert table.has_focus
        assert table.get_row_at(table.cursor_row)[0] == "13"
        assert app.selected_run_id == "run-1"


@pytest.mark.asyncio
async def test_empty_run_list_explains_how_to_start_and_populates() -> None:
    source = _Source(_snapshot(()))
    app = ExecutionSnapshotApp(source)

    async with app.run_test(size=(40, 15)) as pilot:
        await pilot.pause()
        empty = app.query_one("#empty", Static)
        empty_text = str(empty.render())
        assert empty.has_class("visible")
        assert "milknado run" in empty_text
        assert "read-only" in empty_text.casefold()
        assert "watch" in empty_text

        source.emit(_snapshot((_run("run-3", 14, "Newly observed run"),)))
        await pilot.pause()
        assert not empty.has_class("visible")
        assert _table(app).get_row_index("run-3") == 0
