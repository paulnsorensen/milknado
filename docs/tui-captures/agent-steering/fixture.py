from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields
from datetime import UTC, datetime
from typing import TypeVar, cast

try:
    from milknado.domains.graph import CommandReceipt
except ImportError:
    CommandReceipt = None  # type: ignore[assignment,misc]


from milknado.app.run import (
    ActiveRunSnapshot,
    ExecutionRunStatus,
    ExecutionSnapshot,
    RunActionAvailability,
    TerminalRunSnapshot,
)
from milknado.app.run_source import NodeSnapshotRequest
from milknado.domains.common import (
    MikadoEdge,
    MikadoNode,
    NodeKind,
    NodeStatus,
    SessionAction,
    SessionContext,
    SessionEvent,
    SessionView,
)
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

_T = TypeVar("_T")
_CREATED = datetime(2026, 9, 12, 12, 0, tzinfo=UTC)
_DISPATCHED = datetime(2026, 9, 12, 12, 1, tzinfo=UTC)
_FINISHED = datetime(2026, 9, 12, 12, 3, tzinfo=UTC)


PROVIDER_ACTIONS: dict[str, tuple[SessionAction, ...]] = {
    "codex": ("approve", "deny"),
    "claude": ("follow_up", "interrupt", "approve", "deny"),
    "omp": ("steer", "follow_up", "interrupt", "approve", "deny"),
}


def _page(
    items: tuple[_T, ...], *, total: int | None = None, has_more: bool = False
) -> SnapshotPage[_T]:
    return SnapshotPage(items, 0, 50, len(items) if total is None else total, has_more)


def _nodes() -> tuple[MikadoNode, ...]:
    return (
        MikadoNode(
            id=12,
            description="Steer the shared execution workspace",
            status=NodeStatus.RUNNING,
            worktree_path="/fixture/worktree",
            branch_name="feat/tui-agent-steering",
            run_id="run-12",
            pid=4242,
            created_at=_CREATED,
            dispatched_at=_DISPATCHED,
            oversized=True,
            batch_index=0,
            kind=NodeKind.GOAL,
            goal_run_id="goal-run-12",
            wiki_ref="interactive-run-steering",
            github_ref="PVTI_fixture",
            artifact_path="docs/steering.md",
        ),
        MikadoNode(
            id=13,
            description="Protect completion gates",
            status=NodeStatus.FAILED,
            parent_id=12,
            worktree_path="/fixture/worktree/task-13",
            branch_name="task/protect-completion-gates",
            run_id="run-13",
            created_at=_CREATED,
            dispatched_at=_DISPATCHED,
            completed_at=_FINISHED,
            completion_duration_seconds=83.0,
            kind=NodeKind.TASK,
            flavor="implement",
            artifact_path="docs/completion-gates.md",
        ),
        MikadoNode(
            id=14,
            description="Render standard and compact layouts",
            status=NodeStatus.DONE,
            parent_id=12,
            worktree_path="/fixture/worktree/task-14",
            branch_name="task/render-layouts",
            created_at=_CREATED,
            dispatched_at=_DISPATCHED,
            completed_at=_FINISHED,
            completion_duration_seconds=42.0,
            kind=NodeKind.TASK,
            flavor="implement",
            artifact_path="docs/layouts.md",
        ),
        MikadoNode(
            id=15,
            description="Check provider capability fences",
            status=NodeStatus.PENDING,
            parent_id=12,
            created_at=_CREATED,
            kind=NodeKind.TASK,
            flavor="review",
        ),
        MikadoNode(
            id=16,
            description="Publish matched evidence captures",
            status=NodeStatus.PENDING,
            parent_id=12,
            created_at=_CREATED,
            kind=NodeKind.TASK,
            flavor="plate",
            artifact_path="docs/tui-captures/steering.md",
        ),
    )


NODES = _nodes()
NODE_BY_ID = {node.id: node for node in NODES}
EDGES = (MikadoEdge(13, 16), MikadoEdge(14, 16))
GRAPH = GraphSnapshot(NODES, EDGES, (12,))


def _session(state: str) -> SessionView:
    permission = SessionEvent(
        kind="permission",
        text="Allow the worker to edit src/steering.py?",
        event_id="perm-1",
        state="requested",
    )
    events = (
        SessionEvent(kind="status", text="Worker started"),
        SessionEvent(kind="assistant", text="I will update the steering adapter."),
        SessionEvent(kind="tool", text="Reading src/steering.py"),
        SessionEvent(kind="user", text="Keep the change bounded.", action="follow_up"),
        permission,
    )
    if state == "error":
        events += (SessionEvent(kind="error", text="Owner stream ended unexpectedly"),)
    if state == "review":
        events += (SessionEvent(kind="status", text="Human review pending for top-level goal #12"),)
    actions = () if state == "owner-unavailable" else PROVIDER_ACTIONS["claude"]
    session_data = {
        "events": events,
        "actions": actions,
        "active": True,
        "permissions": (permission,),
    }
    session_fields = {item.name for item in fields(SessionView)}
    if "owner_incarnation" in session_fields:
        session_data["owner_incarnation"] = "claude-owner-1" if actions else ""
    if "invocation_id" in session_fields:
        session_data["invocation_id"] = "invoke-12" if actions else ""
    return SessionView(**session_data)


def _actions(state: str, read_only: bool) -> RunActionAvailability:
    if read_only:
        return RunActionAvailability(*("Observer mode is read-only.",) * 3)
    if state == "owner-unavailable":
        return RunActionAvailability(*("Live owner unavailable.",) * 3)
    if state == "error":
        return RunActionAvailability(*("Snapshot source unavailable.",) * 3)
    return RunActionAvailability()


def _detail(node: MikadoNode, state: str) -> NodeDetailSnapshot:
    parent = NODE_BY_ID.get(node.parent_id) if node.parent_id is not None else None
    children = tuple(item for item in NODES if item.parent_id == node.id)
    ancestors: list[MikadoNode] = []
    current = parent
    while current is not None:
        ancestors.append(current)
        current = NODE_BY_ID.get(current.parent_id) if current.parent_id is not None else None
    prereqs = tuple(edge.parent_id for edge in EDGES if edge.child_id == node.id)
    dependents = tuple(edge.child_id for edge in EDGES if edge.parent_id == node.id)
    run_id = node.run_id or f"run-{node.id}"
    session = _session(state) if node.id == 12 else SessionView()
    run = cast(
        RunRecord,
        cast(
            object,
            {
                "run_id": run_id,
                "node_id": node.id,
                "status": "running" if node.status is NodeStatus.RUNNING else node.status.value,
                "pid": node.pid,
                "log_path": f"/fixture/logs/{run_id}.log",
                "started_at": _DISPATCHED.isoformat(),
                "ended_at": _FINISHED.isoformat() if node.status is NodeStatus.FAILED else None,
                "timed_out": False,
                "exit_code": 1 if node.status is NodeStatus.FAILED else None,
                "error": (
                    "Quality gate rejected an incomplete change."
                    if node.status is NodeStatus.FAILED
                    else None
                ),
                "timeout_seconds": 60,
                "detail": "synthetic evidence run",
                "rebased": False,
            },
        ),
    )
    review = cast(
        NodeReviewRecord,
        cast(
            object,
            {
                "node_id": 12,
                "round": 1,
                "verdict": "pending" if state == "review" else "approved",
                "findings": "Human review targets only possible changes to the top-level goal.",
                "created_at": _FINISHED.isoformat(),
            },
        ),
    )
    receipt = (
        CommandReceipt(
            command_id="fixture-command-1",
            status="queued" if state != "owner-unavailable" else "unconfirmed",
            node_id=12,
            run_id="run-12",
            invocation_id="invoke-12",
            owner_incarnation="claude-owner-1",
            action="follow_up",
            text="Keep the change bounded.",
            permission_id=None,
            expires_at="2026-09-12T12:05:00+00:00",
            admitted_at="2026-09-12T12:01:00+00:00",
            recorded_at="2026-09-12T12:01:01+00:00",
            detail="synthetic receipt",
        )
        if CommandReceipt is not None
        else None
    )
    session_snapshot = NodeSessionSnapshot(
        run_id=run_id,
        session=session,
        state="loaded",
        event_history=_page(session.events),
    )
    goal_claim = cast(
        GoalClaim,
        cast(
            object,
            {
                "goal_id": 12,
                "run_id": "goal-run-12",
                "pid": 4242,
                "claimed_at": _DISPATCHED.isoformat(),
            },
        ),
    )
    detail = {
        "node": node,
        "description": node.description,
        "parent": parent,
        "children": _page(children),
        "ancestors": _page(tuple(ancestors)),
        "prerequisite_ids": _page(prereqs),
        "dependent_ids": _page(dependents),
        "reverse_dependents": _page(tuple(NODE_BY_ID[item] for item in dependents)),
        "owned_files": _page((node.artifact_path,) if node.artifact_path else ()),
        "runs": _page((run,)),
        "reviews": _page((review,)),
        "sessions": _page((session_snapshot,)),
        "goal_claim": SnapshotValue(
            goal_claim if node.id == 12 else None,
            "loaded" if node.id == 12 else "not_stored",
        ),
        "artifacts": _page(
            (
                ArtifactSnapshot(
                    node.artifact_path or f"docs/node-{node.id}.md",
                    SnapshotValue("Synthetic artifact content.", "loaded"),
                ),
            )
        ),
    }
    if receipt is not None and "receipts" in {item.name for item in fields(NodeDetailSnapshot)}:
        detail["receipts"] = _page((receipt,))
    return NodeDetailSnapshot(**detail)
class Source:
    def __init__(self, state: str = "main", read_only: bool = False) -> None:
        self.state: str = state
        self.read_only: bool = read_only

    def snapshot(self) -> ExecutionSnapshot:
        active = ActiveRunSnapshot(
            run_id="run-12",
            node_id=12,
            description="Steer the shared execution workspace",
            status=ExecutionRunStatus.RUNNING,
            progress="Waiting for provider input",
            stop_requested=False,
            actions=_actions(self.state, self.read_only),
            output=("Provider capability snapshot loaded.",),
            pending_guidance=None,
            elapsed_seconds=125.0,
            progress_pct=45.0,
            eta_seconds=150.0,
            attempt=1,
            max_attempts=3,
            stalled=False,
            session=_session(self.state),
        )
        failed = TerminalRunSnapshot(
            run_id="run-13",
            node_id=13,
            description="Protect completion gates",
            status=ExecutionRunStatus.FAILED,
            output=("Quality gate rejected an incomplete change.",),
            pending_guidance=None,
            duration_seconds=83.0,
            session=SessionView(),
        )
        review_event = (
            ("Human review pending for top-level goal #12",) if self.state == "review" else ()
        )
        return ExecutionSnapshot(
            goal="Milknado agent steering",
            active_runs=(active,),
            terminal_runs=(failed,),
            completed=1,
            failed=1,
            stopped=0,
            available=2,
            event_lines=(
                "Node 12 started",
                "Node 13 failed its quality gate",
                *review_event,
            ),
            listener_errors=("Snapshot source unavailable",) if self.state == "error" else (),
            graph=GRAPH,
        )

    def node_snapshot(self, request: NodeSnapshotRequest) -> NodeDetailResponse:
        node = NODE_BY_ID.get(request.node_id)
        return NodeDetailResponse(
            request.node_id,
            request.request_generation,
            _detail(node, self.state) if node is not None else None,
        )

    def subscribe(self, listener: Callable[[ExecutionSnapshot], None]) -> Callable[[], None]:
        del listener
        return lambda: None
