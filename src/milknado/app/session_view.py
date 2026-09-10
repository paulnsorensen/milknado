"""Pure Rich formatting for structured sessions."""

from __future__ import annotations

from rich.text import Text

from milknado.domains.common import SessionAction, SessionEvent, SessionView

_ACTION_LABELS: dict[SessionAction, str] = {
    "steer": "Steer current turn",
    "follow_up": "Follow up after turn",
    "interrupt": "Interrupt current turn",
    "approve": "Approve permission request",
    "deny": "Deny permission request",
}
_EVENT_LABELS = {
    "assistant": "Assistant",
    "tool": "Tool",
    "user": "You",
    "status": "Status",
    "permission": "Permission",
    "error": "Error",
}
_EVENT_STYLES = {
    "assistant": "green",
    "tool": "cyan",
    "user": "bold white",
    "status": "yellow",
    "permission": "magenta",
    "error": "bold red",
}
_ACTION_EVENT_LABELS = {
    "steer": "Steer",
    "follow_up": "Follow-up",
    "interrupt": "Interrupt",
    "approve": "Approve",
    "deny": "Deny",
}


def short_title(description: str, *, limit: int = 56) -> str:
    """Derive a stable compact node title without changing the stored brief."""
    first_line = next((line.strip() for line in description.splitlines() if line.strip()), "")
    if first_line.startswith("#"):
        first_line = first_line.lstrip("#").strip()
    title = " ".join(first_line.split()) or "Untitled node"
    if len(title) <= limit:
        return title
    return title[: max(1, limit - 1)].rstrip() + "…"


def full_brief(description: str) -> str:
    """Return the complete node description, with a visible empty fallback."""
    return description.strip() or "No brief supplied."


def action_label(action: SessionAction) -> str:
    return _ACTION_LABELS[action]


def action_options(actions: tuple[SessionAction, ...]) -> tuple[tuple[str, SessionAction], ...]:
    return tuple((action_label(action), action) for action in actions)


def _event_header(event: SessionEvent) -> str:
    label = _EVENT_LABELS[event.kind]
    action = event.action
    if event.kind == "user" and action is not None:
        return f"{label} · {_ACTION_EVENT_LABELS[action]}"
    return label


def _append_event(target: Text, event: SessionEvent) -> None:
    if target:
        _ = target.append("\n")
    _ = target.append(f"{_event_header(event)}", style=_EVENT_STYLES[event.kind])
    if event.state:
        _ = target.append(f" [{event.state}]", style="dim")
    _ = target.append(": ")
    _ = target.append(event.text)


def transcript_text(view: SessionView) -> Text:
    """Format assistant, user, tool, status, and permission events readably."""
    transcript = Text()
    for event in view.events:
        if event.kind != "error":
            _append_event(transcript, event)
    if not transcript:
        _ = transcript.append("No session transcript yet.", style="dim")
    return transcript


def error_text(view: SessionView) -> Text:
    """Format protocol/runtime errors separately from the transcript."""
    errors = Text()
    for event in view.events:
        if event.kind == "error":
            _append_event(errors, event)
    if not errors:
        _ = errors.append("No session errors.", style="dim")
    return errors


def permission_options(view: SessionView) -> tuple[tuple[str, str], ...]:
    """Return exact request-id choices; never infer a request from its text."""
    options: list[tuple[str, str]] = []
    for event in view.permissions:
        if event.event_id and event.state == "requested":
            text = " ".join(event.text.split())
            label = f"{event.event_id}: {text}" if text else event.event_id
            options.append((label, event.event_id))
    return tuple(options)


def session_state_text(view: SessionView) -> str:
    if view.active:
        return "active"
    if any(event.state == "stopped" for event in view.events):
        return "stopped"
    return "complete" if view.events else "not started"
