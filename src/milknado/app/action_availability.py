"""Pure availability rules for snapshot-view controls."""

from dataclasses import dataclass

from milknado.app.run import ActiveRunSnapshot, TerminalRunSnapshot
from milknado.domains.common import SessionView

RunSnapshot = ActiveRunSnapshot | TerminalRunSnapshot


@dataclass(frozen=True, slots=True)
class ActionAvailabilityContext:
    selected: RunSnapshot | None
    session: SessionView
    compact: bool
    minimum: bool
    read_only: bool
    auto_follow: bool
    node_selected: bool
    route: str


def action_availability(action: str, context: ActionAvailabilityContext) -> bool | None:
    """Return visibility for controls whose context changes with a snapshot."""
    selected = context.selected
    active = selected if isinstance(selected, ActiveRunSnapshot) else None
    if action == "open_detail":
        return (
            not context.minimum
            and context.compact
            and context.route == "list"
            and (selected is not None or context.node_selected)
        )
    if action == "focus_session":
        return (
            not context.minimum
            and not context.read_only
            and context.session.active
            and bool(context.session.actions)
        )
    if action == "focus_changes":
        return not context.minimum and context.session.context is not None
    if action == "resume_output":
        return not context.minimum and not context.auto_follow
    if action == "focus_guidance":
        return (
            not context.minimum
            and not context.read_only
            and (
                bool(context.session.actions)
                or active is not None
                and active.actions.can_queue_guidance
            )
        )
    if action == "cancel":
        return not context.minimum and active is not None and active.actions.can_cancel
    if action == "force":
        return not context.minimum and active is not None and active.actions.can_force_stop
    return None
