"""Starlette application factory for the local web adapter."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from starlette.applications import Starlette

from milknado.app.run_source import ExecutionSnapshotSource
from milknado.web.commands import WebCommands
from milknado.web.guards import LOOPBACK_HOSTS, RequestGuards, allowed_host
from milknado.web.login import LaunchToken
from milknado.web.routes import discover_routes


@dataclass(frozen=True, slots=True)
class WebContext:
    source: ExecutionSnapshotSource
    commands: WebCommands
    login: LaunchToken


def create_app(
    source: ExecutionSnapshotSource,
    commands: WebCommands,
    login: LaunchToken,
    extra_hosts: Iterable[str] = (),
) -> Starlette:
    """Build the guarded local application from injected seams.

    `extra_hosts` widens the loopback Host allow-list, for example to a tailnet name.
    """
    app = Starlette(routes=discover_routes())
    app.state.web = WebContext(source, commands, login)
    hosts = (*LOOPBACK_HOSTS, *(allowed_host(host) for host in extra_hosts))
    app.add_middleware(RequestGuards, login=login, allowed_hosts=hosts)
    return app
