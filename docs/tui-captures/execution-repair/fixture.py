"""Reproduce the archived execution-repair states with synthetic worker I/O."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import cast

import msgspec

from milknado.app.run import ExecutionController, ExecutionSnapshot
from milknado.app.run_source import NodeSnapshotRequest
from milknado.app.run_tui import ExecutionApp
from milknado.app.run_view_app import ExecutionSnapshotApp
from milknado.app.watch import AttachedWatchSource
from milknado.app.watch_tui import WatchApp
from milknado.domains.common import MikadoNode, NodeKind, SessionEvent, SessionInput, SessionView
from milknado.domains.graph import GraphSnapshot, NodeDetailResponse

ROOT = Path(__file__).resolve().parents[3]


def load(name: str, path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load capture fixture: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


graph_fixture = load("audit_graph", "docs/tui-captures/agent-steering/fixture.py")
session_fixture = load("audit_session", "docs/tui-captures/structured-sessions/fixture.py")


def _graph(state: str) -> GraphSnapshot:
    graph = cast(GraphSnapshot, graph_fixture.GRAPH)
    if state != "large":
        return graph
    extra = tuple(
        MikadoNode(
            id=i,
            description=f"Audit task {i:03} with a long descriptive title",
            parent_id=12,
            kind=NodeKind.TASK,
            created_at=graph_fixture._CREATED,
        )
        for i in range(17, 77)
    )
    return replace(graph, nodes=(*graph.nodes, *extra))


def _session(view: SessionView, state: str, family: str) -> SessionView:
    actions = graph_fixture.PROVIDER_ACTIONS[family]
    assert view.context is not None
    view = replace(
        view,
        context=msgspec.structs.replace(view.context, family=family),
        actions=actions,
        owner_incarnation="audit-owner",
        invocation_id="audit-invoke",
    )
    permission = SessionEvent(
        kind="permission",
        text="Allow audit fixture edit?",
        event_id="audit-perm",
        state="requested",
    )
    view = replace(view, permissions=(permission,), events=(*view.events, permission))
    if state == "owner-unavailable":
        return replace(view, actions=(), owner_incarnation="", invocation_id="")
    if state == "legacy":
        return SessionView()
    if state == "output":
        return replace(
            view,
            events=tuple(
                SessionEvent(kind="assistant", text=f"Audit transcript line {i:03}")
                for i in range(150)
            ),
        )
    return view


def _snapshot(workspace: Path, state: str, family: str) -> ExecutionSnapshot:
    initial = replace(
        session_fixture.initial_snapshot(workspace, state), goal="Milknado isolated TUI audit"
    )
    if state == "empty":
        return initial
    graph = _graph(state)
    if state == "empty-graph":
        return replace(
            initial,
            graph=GraphSnapshot((), (), ()),
            event_lines=(),
        )
    run = replace(
        initial.active_runs[0],
        run_id="run-12",
        description=graph.nodes[0].description,
        session=_session(initial.active_runs[0].session, state, family),
    )
    return replace(
        initial,
        graph=graph,
        active_runs=(run,),
        terminal_runs=(replace(initial.terminal_runs[0], run_id="run-13"),),
        listener_errors=("Snapshot source unavailable",) if state == "error" else (),
    )


class Source(session_fixture.Source):
    def __init__(self, state: str = "main", family: str = "omp") -> None:
        self.state = state
        self.calls: list[tuple[str, str, str, str]] = []
        self.workspace = Path(tempfile.mkdtemp(prefix="milknado-audit-git-"))
        super().__init__(_snapshot(self.workspace, state, family))

    def snapshot(self, request: object = None) -> ExecutionSnapshot:
        _ = request
        return self.current

    def node_snapshot(self, request: NodeSnapshotRequest) -> NodeDetailResponse:
        return graph_fixture.Source(self.state).node_snapshot(request)

    def session_input(self, run_id: str, command: SessionInput) -> bool:
        self.calls.append((run_id, command.action, command.text, command.request_id))
        if self.state == "reject":
            return False
        return super().session_input(run_id, command)

    def queue_guidance(self, run_id: str, text: str) -> bool:
        self.calls.append((run_id, "guidance", text, ""))
        return True


def make_app(
    mode: str, state: str = "main", family: str = "omp"
) -> tuple[ExecutionSnapshotApp, Source]:
    source = Source(state, family)
    if mode == "run":
        app = ExecutionApp(cast(ExecutionController, source))
    elif mode == "attached":
        app = WatchApp(AttachedWatchSource(source, source.session_input), read_only=False)
    else:
        source.current = replace(
            source.current,
            active_runs=tuple(
                replace(run, actions=graph_fixture._actions(state, True))
                for run in source.current.active_runs
            ),
        )
        app = WatchApp(source)
    return app, source


if __name__ == "__main__":
    app, source = make_app(*sys.argv[1:])
    app.run()
    print("AUDIT_APP_EXITED")
