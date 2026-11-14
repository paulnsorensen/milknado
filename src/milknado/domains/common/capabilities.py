"""Runtime capability metadata for each structured provider family."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from milknado.domains.common.session import SessionAction

ProviderFamily = Literal["claude", "codex"]
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
    "resume": "native",
    "streamed_output": "native",
    "user_input": "native",
    "interrupt": "native",
    "approval": "native",
    "cancel": "runtime",
    "terminal_result": "native",
    "recovery_report": "runtime",
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
