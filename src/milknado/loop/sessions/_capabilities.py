"""Session capability publication helpers."""

from collections.abc import Callable

from milknado.domains.common import SessionAction, SessionContext, SessionEvent

PermissionCapabilities = tuple[tuple[str, ...], tuple[tuple[str, str], ...]]
CapabilitySink = Callable[
    [SessionContext, tuple[SessionAction, ...], str, PermissionCapabilities], None
]
CapabilityState = tuple[
    SessionContext | None,
    tuple[SessionAction, ...],
    str,
    PermissionCapabilities,
]


def snapshot(
    context: SessionContext | None,
    actions: tuple[SessionAction, ...],
    invocation_id: str,
    permissions: tuple[SessionEvent, ...],
) -> CapabilityState:
    return (
        context,
        actions,
        invocation_id,
        (
            tuple(event.event_id for event in permissions),
            tuple((event.event_id, event.text) for event in permissions),
        ),
    )


def refresh(sink: CapabilitySink | None, state: CapabilityState) -> None:
    context, actions, invocation_id, permissions = state
    if context is not None:
        publish(sink, context, actions, invocation_id, permissions)


def publish(  # noqa: PLR0913
    sink: CapabilitySink | None,
    context: SessionContext,
    actions: tuple[SessionAction, ...],
    invocation_id: str,
    permissions: PermissionCapabilities,
) -> None:
    if sink is not None and invocation_id:
        sink(context, actions, invocation_id, permissions)
