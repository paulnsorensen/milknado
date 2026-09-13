from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from threading import Event
from typing import TypeVar, cast

from rich.text import Text
from textual.pilot import Pilot
from textual.widgets import Static

from milknado.app.run import (
    ActiveRunSnapshot,
    ExecutionController,
    ExecutionRunStatus,
    ExecutionSnapshot,
    RunActionAvailability,
)
from milknado.app.run_source import NodeSnapshotRequest
from milknado.app.run_tui import ExecutionApp
from milknado.app.run_view_app import ExecutionSnapshotApp
from milknado.app.watch_tui import WatchApp
from milknado.domains.common import (
    MikadoEdge,
    MikadoNode,
    NodeKind,
    SessionContext,
    SessionEvent,
    SessionView,
)
from milknado.domains.execution import RunLoopResult
from milknado.domains.graph import (
    ArtifactSnapshot,
    GraphSnapshot,
    NodeDetailResponse,
    NodeDetailSnapshot,
    NodeSessionSnapshot,
    RunRecord,
    SnapshotPage,
    SnapshotValue,
)
from milknado.domains.graph._goal_claims import GoalClaim
from milknado.domains.graph._run_persistence import NodeReviewRecord
from milknado.domains.graph.commands import CommandReceipt

_T = TypeVar("_T")
_CREATED = datetime(2026, 9, 12, tzinfo=UTC)


@dataclass
class PagedSource:
    current: ExecutionSnapshot
    requests: list[NodeSnapshotRequest]
    listener: Callable[[ExecutionSnapshot], None] | None = None
    stale_next: bool = False
    detail_marker: str = ""
    block_next: bool = False
    started: Event = field(default_factory=Event)
    release: Event = field(default_factory=Event)
    response_generation_delta: int = 0
    response_node_id: int | None = None
    receipts_only: bool = False

    def snapshot(self) -> ExecutionSnapshot:
        return self.current

    def attached_watch_source(self) -> PagedSource:
        return self

    def subscribe(self, listener: Callable[[ExecutionSnapshot], None]) -> Callable[[], None]:
        self.listener = listener
        return lambda: None

    def node_snapshot(self, request: NodeSnapshotRequest) -> NodeDetailResponse:
        self.requests.append(request)
        label = "stale" if self.stale_next else ""
        detail = detail_snapshot(
            request.page,
            request.session_event_page,
            label + self.detail_marker,
            receipts_only=self.receipts_only,
        )
        if self.block_next:
            self.block_next = False
            self.started.set()
            if not self.release.wait(timeout=2):
                raise AssertionError("controlled detail source was not released")
        response_node_id = self.response_node_id or request.node_id
        response_generation = request.request_generation + self.response_generation_delta
        self.response_node_id = None
        self.response_generation_delta = 0
        if self.stale_next:
            self.stale_next = False
            response_generation -= 1
        return NodeDetailResponse(response_node_id, response_generation, detail)


def page(page_number: int, items: tuple[_T, ...]) -> SnapshotPage[_T]:
    return SnapshotPage(items, page_number, 1, 3, page_number < 2)


def detail_snapshot(
    page_number: int, session_page: int, label: str, *, receipts_only: bool = False
) -> NodeDetailSnapshot:
    node = MikadoNode(1, "root node", kind=NodeKind.GOAL, created_at=_CREATED)
    child = MikadoNode(
        100 + page_number,
        f"{label}child-page-{page_number}",
        parent_id=1,
        created_at=_CREATED,
    )
    session = NodeSessionSnapshot(
        "run-history",
        SessionView(context=SessionContext(family="codex", cwd="/repo")),
        "loaded",
        SnapshotPage(
            (SessionEvent(kind="assistant", text=f"{label}history-page-{session_page}"),),
            session_page,
            1,
            2,
            session_page == 0,
        ),
    )
    receipt = CommandReceipt(
        command_id=f"receipt-page-{page_number}",
        status="delivered" if page_number else "queued",
        node_id=1,
        run_id="run-history",
        invocation_id="invoke-1",
        owner_incarnation="owner-1",
        action="steer",
        text=f"receipt-text-page-{page_number}",
        permission_id=None,
        expires_at="2026-09-13T00:00:00+00:00",
        admitted_at="2026-09-12T00:00:00+00:00",
        recorded_at="2026-09-12T00:00:00+00:00",
    )
    related = SnapshotPage((), page_number, 1, 0, False)
    return NodeDetailSnapshot(
        node=node,
        description=node.description,
        parent=None,
        children=related if receipts_only else page(page_number, (child,)),
        ancestors=related if receipts_only else page(page_number, (node,)),
        prerequisite_ids=related if receipts_only else page(page_number, (page_number,)),
        dependent_ids=related if receipts_only else page(page_number, (page_number + 1,)),
        reverse_dependents=related if receipts_only else page(page_number, (node,)),
        owned_files=related
        if receipts_only
        else page(page_number, (f"docs/page-{page_number}.md",)),
        runs=related
        if receipts_only
        else page(
            page_number,
            (cast(RunRecord, cast(object, {"run_id": f"{label}run-page-{page_number}"})),),
        ),
        reviews=related
        if receipts_only
        else page(
            page_number,
            (
                cast(
                    NodeReviewRecord,
                    cast(object, {"verdict": f"{label}review-page-{page_number}"}),
                ),
            ),
        ),
        sessions=related if receipts_only else page(page_number, (session,)),
        receipts=SnapshotPage((receipt,), page_number, 1, 2, page_number == 0)
        if receipts_only
        else page(page_number, ()),
        goal_claim=SnapshotValue(
            cast(
                GoalClaim,
                cast(object, {"goal_id": page_number, "run_id": f"claim-page-{page_number}"}),
            ),
            "loaded",
        ),
        artifacts=related
        if receipts_only
        else page(
            page_number,
            (
                ArtifactSnapshot(
                    f"docs/artifact-{page_number}.md",
                    SnapshotValue(f"{label}artifact-page-{page_number}", "loaded"),
                ),
            ),
        ),
    )


def source() -> PagedSource:
    root = MikadoNode(1, "root node", kind=NodeKind.GOAL, created_at=_CREATED)
    child = MikadoNode(2, "tree child", parent_id=1, created_at=_CREATED)
    graph = GraphSnapshot((root, child), (MikadoEdge(1, 2),), (1,))
    run = ActiveRunSnapshot(
        run_id="run-1",
        node_id=1,
        description="root node",
        status=ExecutionRunStatus.RUNNING,
        progress="working",
        stop_requested=False,
        actions=RunActionAvailability(),
        output=(),
        pending_guidance=None,
        elapsed_seconds=1.0,
        progress_pct=10.0,
        eta_seconds=9.0,
        attempt=1,
        max_attempts=1,
        stalled=False,
    )
    run = replace(run, session=SessionView(actions=("steer",), active=True))
    snapshot = ExecutionSnapshot(
        "pagination",
        (run,),
        (),
        0,
        0,
        0,
        1,
        (),
        graph=graph,
    )
    return PagedSource(snapshot, [])


def navigation_text(app: ExecutionSnapshotApp) -> str:
    return cast(Text, app.query_one("#detail-navigation", Static).render()).plain


def run_app(source_value: PagedSource, kind: str) -> ExecutionSnapshotApp:
    if kind == "run":
        return ExecutionApp(cast(ExecutionController, cast(object, source_value)))
    return WatchApp(source_value, poll_interval=60.0)


def details_text(app: ExecutionSnapshotApp) -> str:
    return cast(Text, app.query_one("#brief", Static).render()).plain


async def wait_for_requests(
    pilot: Pilot[RunLoopResult | None], source_value: PagedSource, count: int
) -> None:
    for _ in range(40):
        await pilot.pause()
        if len(source_value.requests) >= count:
            await pilot.pause()
            return
    raise AssertionError(f"expected {count} requests, got {len(source_value.requests)}")
