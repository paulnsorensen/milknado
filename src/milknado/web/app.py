# pyright: reportAny=false, reportExplicitAny=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnannotatedClassAttribute=false, reportUnnecessaryCast=false, reportUnnecessaryIsInstance=false
"""Starlette application factory for the local web adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from starlette.applications import Starlette

from milknado.app.run_source import ExecutionSnapshot, ExecutionSnapshotSource
from milknado.web.commands import WebCommands, build_capabilities
from milknado.web.guards import RequestGuards
from milknado.web.login import LaunchToken
from milknado.web.routes import discover_routes


class SnapshotSource(Protocol):
    def snapshot(self) -> ExecutionSnapshot: ...


@dataclass(frozen=True, slots=True)
class WebContext:
    source: ExecutionSnapshotSource
    commands: WebCommands
    login: LaunchToken
    capabilities: dict[str, object]


def create_app(
    source: ExecutionSnapshotSource, commands: WebCommands, login: LaunchToken
) -> Starlette:
    """Build the guarded local application from injected seams."""
    app = Starlette(routes=discover_routes())
    app.state.web = WebContext(source, commands, login, build_capabilities(commands))
    app.add_middleware(RequestGuards, login=login)
    return app
