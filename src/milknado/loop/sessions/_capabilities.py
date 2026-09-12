"""Session capability publication helpers."""

from collections.abc import Callable

from milknado.domains.common import SessionAction, SessionContext

CapabilitySink = Callable[[SessionContext, tuple[SessionAction, ...], str, tuple[str, ...]], None]
CapabilityState = tuple[SessionContext | None, tuple[SessionAction, ...], str, tuple[str, ...]]


def refresh(sink: CapabilitySink | None, state: CapabilityState) -> None:
    context, actions, invocation_id, permission_ids = state
    if context is not None:
        publish(sink, context, actions, invocation_id, permission_ids)


def publish(  # noqa: PLR0913
    sink: CapabilitySink | None,
    context: SessionContext,
    actions: tuple[SessionAction, ...],
    invocation_id: str,
    permission_ids: tuple[str, ...],
) -> None:
    if sink is not None and invocation_id:
        sink(context, actions, invocation_id, permission_ids)
