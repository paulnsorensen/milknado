from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

from milknado.domains.common import RunHandle
from milknado.loop import RunConfig

if TYPE_CHECKING:
    from milknado.loop._events import EventData


class RunHandleView(RunHandle, Protocol):
    config: RunConfig


def event_text(data: EventData, *keys: str) -> str:
    payload = cast(Mapping[str, object], data)
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def build_verify_prompt(
    spec_text: str,
    graph_state: object | None,
    completion_signal: str,
) -> str:
    graph_summary = str(graph_state) if graph_state is not None else "(no graph state)"
    return (
        "# Spec Verification\n\n"
        "Review whether the following spec has been fully implemented.\n\n"
        f"## Spec\n\n{spec_text}\n\n"
        f"## Graph State\n\n{graph_summary}\n\n"
        "## Instructions\n\n"
        "If the spec is fully satisfied, emit:\n"
        "<result>done</result>\n\n"
        "If there are unmet requirements, emit:\n"
        "<result>gaps</result>\n"
        "<goal_delta>\n"
        "Description of what is still missing\n"
        "</goal_delta>\n\n"
        "Then emit the completion signal to stop this run:\n"
        f"<promise>{completion_signal}</promise>\n"
    )


@dataclass(frozen=True)
class ReviewVerdict:
    approved: bool
    findings_md: str
    error: bool = False
