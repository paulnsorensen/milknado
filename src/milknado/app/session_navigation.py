"""Responsive node-list navigation shared by execution and watch views."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, TypeVar, cast

from textual import on
from textual.message_pump import MessagePump
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import DataTable, Input, TabbedContent, Tree

from milknado.app.action_availability import ActionAvailabilityContext, action_availability
from milknado.app.run import ActiveRunSnapshot, ExecutionSnapshot, TerminalRunSnapshot
from milknado.app.run_view import session_view

if TYPE_CHECKING:
    from textual.events import Key

RunSnapshot = ActiveRunSnapshot | TerminalRunSnapshot
_WidgetT = TypeVar("_WidgetT", bound=Widget)


class _NavigationHost(Protocol):
    snapshot: ExecutionSnapshot
    selected_run_id: str | None
    selected_node_id: int | None
    compact: bool
    minimum: bool
    route: str
    read_only: bool
    auto_follow: bool
    screen: Screen[object]

    def query_one(self, _selector: str, _expect_type: type[_WidgetT], /) -> _WidgetT: ...

    def call_after_refresh(self, callback: Callable[..., object], *args: object) -> object: ...

    def set_focus(self, _widget: Widget | None, /) -> object: ...

    def select_node(self, node_id: int) -> None: ...

    def refresh_view(self) -> None: ...

    def set_layout(self, compact: bool) -> None: ...

    def exit(self) -> object: ...


def _navigation_host(value: object) -> _NavigationHost:
    return cast(_NavigationHost, value)


class RunNavigationMixin(metaclass=type(MessagePump)):
    """Selection, compact routing, and focus behavior for snapshot apps."""

    def _runs(self) -> tuple[RunSnapshot, ...]:
        host = _navigation_host(self)
        return (*host.snapshot.active_runs, *reversed(host.snapshot.terminal_runs))

    def selected_run(self) -> RunSnapshot | None:
        selected = _navigation_host(self).selected_run_id
        return next((run for run in self._runs() if run.run_id == selected), None)

    def _selected_active_run(self) -> ActiveRunSnapshot | None:
        run = self.selected_run()
        return run if isinstance(run, ActiveRunSnapshot) else None

    def _run_index(self) -> int:
        selected = _navigation_host(self).selected_run_id
        runs = self._runs()
        ids = [run.run_id for run in runs]
        return ids.index(selected) if selected in ids else 0

    @on(DataTable.RowSelected, "#runs")
    def select_row(self, event: DataTable.RowSelected) -> None:
        host = _navigation_host(self)
        host.selected_run_id = str(event.row_key.value)
        run = host.selected_run_id
        selected = next((item for item in self._runs() if item.run_id == run), None)
        host.selected_node_id = selected.node_id if selected is not None else None
        if host.compact:
            self.action_open_detail()
        else:
            host.refresh_view()

    @on(Tree.NodeHighlighted, "#graph-tree")
    def select_tree_node(self, event: Tree.NodeHighlighted[object]) -> None:
        host = _navigation_host(self)
        tree = cast(Tree[object], host.query_one("#graph-tree", Tree))
        if event.node is not tree.cursor_node:
            return
        node_id = getattr(event.node.data, "node_id", None)
        if isinstance(node_id, int):
            _navigation_host(self).select_node(node_id)

    @on(Tree.NodeSelected, "#graph-tree")
    def open_tree_node(self, _event: Tree.NodeSelected[object]) -> None:
        if _navigation_host(self).compact:
            self.action_open_detail()

    def _editor_focused(self) -> bool:
        focused = _navigation_host(self).screen.focused
        return focused is not None and focused.id in {
            "session-input",
            "session-action",
            "session-permission",
            "session-submit",
            "guidance",
        }

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Hide controls that cannot affect the selected run."""
        del parameters
        host = _navigation_host(self)
        selected = self.selected_run()
        available = action_availability(
            action,
            ActionAvailabilityContext(
                selected=selected,
                session=session_view(selected),
                compact=host.compact,
                minimum=host.minimum,
                read_only=host.read_only,
                auto_follow=host.auto_follow,
                node_selected=host.selected_node_id is not None,
                route=host.route,
            ),
        )
        return True if available is None else available

    def action_previous_run(self) -> None:  # noqa: V105 - Textual binding action
        self._move_selection(-1)

    def action_next_run(self) -> None:  # noqa: V105 - Textual binding action
        self._move_selection(1)

    def _move_selection(self, offset: int) -> None:
        host = _navigation_host(self)
        if host.minimum:
            return
        if self._editor_focused():
            return
        graph = host.snapshot.graph
        if graph is not None and graph.nodes:
            node_ids = tuple(node.id for node in graph.nodes)
            current_index = (
                node_ids.index(host.selected_node_id) if host.selected_node_id in node_ids else 0
            )
            host.select_node(node_ids[(current_index + offset) % len(node_ids)])
            return
        runs = self._runs()
        if not runs:
            return
        selected = runs[(self._run_index() + offset) % len(runs)]
        host.selected_run_id = selected.run_id
        host.selected_node_id = selected.node_id
        host.refresh_view()

    def action_open_detail(self) -> None:
        host = _navigation_host(self)
        if host.minimum:
            return
        if self._editor_focused():
            return
        if host.compact and (self.selected_run() is not None or host.selected_node_id is not None):
            _ = host.set_focus(None)
            host.route = "detail"
            host.set_layout(True)
            host.refresh_view()

    def action_focus_session(self) -> None:
        host = _navigation_host(self)
        session = session_view(self.selected_run())
        if host.minimum:
            return
        if host.read_only or not session.active or not session.actions:
            return
        self.action_open_detail()
        host.query_one("#run-tabs", TabbedContent).active = "session"
        _ = host.call_after_refresh(host.query_one("#session-input", Input).focus)

    def action_focus_changes(self) -> None:  # noqa: V105 - Textual binding action
        host = _navigation_host(self)
        if host.minimum:
            return
        if session_view(self.selected_run()).context is None:
            return
        self.action_open_detail()
        host.query_one("#run-tabs", TabbedContent).active = "changes"
        _ = host.call_after_refresh(host.query_one("#changes-files", Widget).focus)

    async def action_back(self) -> None:
        host = _navigation_host(self)
        if host.minimum:
            return
        _ = host.set_focus(None)
        if host.compact and host.route == "detail":
            host.route = "list"
            host.set_layout(True)
            host.refresh_view()
        graph = host.snapshot.graph
        selector = "#graph-tree" if graph is not None and graph.nodes else "#runs"
        _ = host.call_after_refresh(host.query_one(selector, Widget).focus)

    def action_focus_events(self) -> None:  # noqa: V105 - Textual binding action
        host = _navigation_host(self)
        if host.minimum:
            return
        if host.compact and host.route == "detail":
            host.route = "list"
            host.set_layout(True)
            host.refresh_view()
        _ = host.call_after_refresh(host.query_one("#events", Widget).focus)

    def action_resume_output(self) -> None:  # noqa: V105 - Textual binding action
        host = _navigation_host(self)
        if host.minimum:
            return
        host.auto_follow = True
        host.refresh_view()

    def action_quit_all(self) -> None:  # noqa: V105 - Textual binding action
        _ = _navigation_host(self).exit()

    def pause_auto_follow(self) -> None:
        host = _navigation_host(self)
        if host.auto_follow:
            host.auto_follow = False
            host.refresh_view()

    def on_key(self, event: Key) -> None:  # noqa: V105 - Textual event handler
        host = _navigation_host(self)
        if (
            not host.screen.is_modal
            and event.key in {"home", "end", "pageup", "pagedown"}
            and not host.query_one("#events", Widget).has_focus
            and not host.query_one("#details-panel", Widget).has_focus
        ):
            self.pause_auto_follow()
