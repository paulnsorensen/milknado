"""Selection and bounded node-detail loading shared by run and watch."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, cast

from textual import work
from textual.screen import Screen
from textual.widget import Widget

from milknado.app.graph_pagination import related_pages
from milknado.app.run import ActiveRunSnapshot, TerminalRunSnapshot
from milknado.app.run_source import ExecutionSnapshotSource, NodeSnapshotRequest
from milknado.domains.common import MikadoNode
from milknado.domains.graph import NodeDetailResponse, SnapshotPage

if TYPE_CHECKING:
    from milknado.app.run import ExecutionSnapshot


class _GraphHost(Protocol):
    source: ExecutionSnapshotSource
    snapshot: ExecutionSnapshot
    selected_node_id: int | None
    selected_run_id: str | None
    detail_page: int
    session_event_page: int
    node_detail: NodeDetailResponse | None
    node_request_generation: int
    screen: Screen[object]

    def runs(self) -> tuple[ActiveRunSnapshot | TerminalRunSnapshot, ...]: ...

    def refresh_view(self) -> None: ...

    def call_from_thread(self, callback: Callable[..., object], *args: object) -> object: ...

    def notify(self, message: str) -> object: ...

    def query_one(self, *args: object) -> Widget: ...

    def set_focus(self, _widget: Widget | None) -> object: ...


class GraphSelectionMixin:
    """Keep graph selection and detail requests independent from presentation layout."""

    def _graph_host(self) -> _GraphHost:
        return cast(_GraphHost, cast(object, self))

    def focus_initial_selector(self) -> None:
        host = self._graph_host()
        if host.screen.is_modal:
            return
        selector = "#graph-tree" if host.snapshot.graph is not None else "#runs"
        _ = host.set_focus(host.query_one(selector))

    def selected_node(self) -> MikadoNode | None:
        host = self._graph_host()
        graph = host.snapshot.graph
        if graph is None or host.selected_node_id is None:
            return None
        return next((node for node in graph.nodes if node.id == host.selected_node_id), None)

    def reconcile_graph_selection(
        self,
        runs: tuple[ActiveRunSnapshot | TerminalRunSnapshot, ...],
        selected_run_id: str | None,
    ) -> tuple[int | None, str | None]:
        host = self._graph_host()
        graph = host.snapshot.graph
        if graph is None:
            selected = next((run for run in runs if run.run_id == selected_run_id), None)
            if selected is not None:
                return selected.node_id, selected.run_id
            first = runs[0] if runs else None
            return (first.node_id, first.run_id) if first is not None else (None, None)
        node_ids = {node.id for node in graph.nodes}
        node_id = host.selected_node_id
        if node_id not in node_ids:
            selected_run = next((run for run in runs if run.run_id == selected_run_id), None)
            node_id = (
                selected_run.node_id
                if selected_run is not None and selected_run.node_id in node_ids
                else graph.root_ids[0]
                if graph.root_ids
                else next(iter(node_ids), None)
            )
        matching = [run for run in runs if run.node_id == node_id]
        active = next((run for run in matching if isinstance(run, ActiveRunSnapshot)), None)
        selected = next((run for run in matching if run.run_id == selected_run_id), None)
        run_id = (
            active.run_id
            if active is not None
            else selected.run_id
            if selected is not None
            else matching[0].run_id
            if matching
            else None
        )
        return node_id, run_id

    def select_node(self, node_id: int) -> None:
        host = self._graph_host()
        if host.selected_node_id == node_id:
            return
        host.selected_node_id = node_id
        runs = host.runs()
        run = next((item for item in runs if item.node_id == node_id), None)
        host.selected_run_id = run.run_id if run is not None else None
        host.detail_page = 0
        host.session_event_page = 0
        self._request_node_detail()
        if host.screen.is_mounted:
            host.refresh_view()

    def _request_node_detail(self, *, clear_detail: bool = True) -> None:
        host = self._graph_host()
        if host.selected_node_id is None:
            return
        host.node_request_generation += 1
        if clear_detail:
            host.node_detail = None
        _ = self._load_node_detail(
            NodeSnapshotRequest(
                host.selected_node_id,
                host.node_request_generation,
                page=host.detail_page,
                session_event_page=host.session_event_page,
            )
        )

    def _detail_focused(self) -> bool:
        return self._graph_host().query_one("#details-panel", Widget).has_focus

    @staticmethod
    def _detail_pages(detail: NodeDetailResponse | None) -> tuple[SnapshotPage[object], ...]:
        if detail is None or detail.detail is None:
            return ()
        return related_pages(detail.detail)

    def action_previous_detail_page(self) -> None:  # noqa: V105 - Textual binding action
        host = self._graph_host()
        if self._detail_focused() and host.detail_page > 0:
            host.detail_page -= 1
            self._request_node_detail()

    def action_next_detail_page(self) -> None:  # noqa: V105 - Textual binding action
        host = self._graph_host()
        if self._detail_focused() and any(
            page.has_more for page in self._detail_pages(host.node_detail)
        ):
            host.detail_page += 1
            self._request_node_detail()

    def action_previous_history_page(self) -> None:  # noqa: V105 - Textual binding action
        host = self._graph_host()
        if self._detail_focused() and host.session_event_page > 0:
            host.session_event_page -= 1
            self._request_node_detail()

    def action_next_history_page(self) -> None:  # noqa: V105 - Textual binding action
        host = self._graph_host()
        current_detail = host.node_detail
        detail = current_detail.detail if current_detail is not None else None
        if (
            self._detail_focused()
            and detail is not None
            and any(session.event_history.has_more for session in detail.sessions.items or ())
        ):
            host.session_event_page += 1
            self._request_node_detail()

    @work(thread=True, group="node-detail", exclusive=True)
    def _load_node_detail(self, request: NodeSnapshotRequest) -> None:
        host = self._graph_host()
        try:
            response = host.source.node_snapshot(request)
        except (OSError, RuntimeError, ValueError) as error:
            _ = host.call_from_thread(host.notify, f"Node detail unavailable: {error}")
            return
        _ = host.call_from_thread(self._apply_node_detail, response)

    def _apply_node_detail(self, response: NodeDetailResponse) -> None:
        host = self._graph_host()
        if response.matches(host.selected_node_id or -1, host.node_request_generation):
            host.node_detail = response
            if host.screen.is_mounted:
                host.refresh_view()
