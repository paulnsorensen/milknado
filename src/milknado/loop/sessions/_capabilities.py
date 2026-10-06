"""Session capability publication helpers."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from milknado.domains.common import SessionAction, SessionContext, SessionEvent
from milknado.loop.sessions._protocol import ProviderFamily

LifecycleOperation = Literal[
    "start",
    "resume",
    "streamed_output",
    "user_input",
    "interrupt",
    "approval",
    "cancel",
    "terminal_result",
    "recovery_report",
]
CapabilitySupport = Literal["native", "runtime", "unsupported"]


@dataclass(frozen=True, slots=True)
class RuntimeCapabilities:
    floor: Mapping[LifecycleOperation, CapabilitySupport]  # noqa: F841, RUF100
    native_actions: frozenset[SessionAction]
    unsupported_actions: frozenset[SessionAction]  # noqa: F841, RUF100


_COMMON_FLOOR: dict[LifecycleOperation, CapabilitySupport] = {
    "start": "native",
    "resume": "unsupported",
    "streamed_output": "native",
    "user_input": "native",
    "interrupt": "native",
    "approval": "native",
    "cancel": "runtime",
    "terminal_result": "native",
    "recovery_report": "unsupported",
}
_ACTIONS: dict[ProviderFamily, frozenset[SessionAction]] = {
    "claude": frozenset({"follow_up", "interrupt", "approve", "deny"}),
    "codex": frozenset({"steer", "interrupt", "approve", "deny"}),
}
_ALL_ACTIONS: frozenset[SessionAction] = frozenset(
    action for actions in _ACTIONS.values() for action in actions
)


def runtime_capabilities(family: ProviderFamily) -> RuntimeCapabilities:
    """Report implemented support, including unsupported lifecycle operations."""
    if family not in _ACTIONS:
        raise ValueError(f"unsupported provider family: {family}")
    actions = _ACTIONS[family]
    return RuntimeCapabilities(
        floor=MappingProxyType(_COMMON_FLOOR),
        native_actions=actions,
        unsupported_actions=_ALL_ACTIONS - actions,
    )


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
