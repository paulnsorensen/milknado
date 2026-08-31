"""Typed options for launching planning subprocesses."""

from __future__ import annotations

from typing import TypedDict

PlanningSubprocessOptions = TypedDict(  # noqa: UP013 - vulture treats TypedDict keys as locals
    "PlanningSubprocessOptions",
    {"env": dict[str, str], "input": str, "text": bool},
)
