"""Owner and observer command capability builders."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from milknado.web.commands import WebCommands


def owner_commands(
    session_input: Callable[..., Any] | None = None,
    cancel: Callable[..., Any] | None = None,
    force_stop: Callable[..., Any] | None = None,
    stop_scheduling: Callable[..., Any] | None = None,
) -> WebCommands:
    return WebCommands(session_input, cancel, force_stop, stop_scheduling)


def observer_commands(
    session_input: Callable[..., Any] | None = None,
    cancel: Callable[..., Any] | None = None,
) -> WebCommands:
    return WebCommands(session_input=session_input, cancel=cancel)
