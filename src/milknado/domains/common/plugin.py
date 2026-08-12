from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from milknado.domains.common.types import MikadoNode, NodeStatus


@dataclass(frozen=True)
class PluginMeta:
    name: str
    version: str
    description: str


class PluginHook(Protocol):
    @property
    def meta(self) -> PluginMeta: ...

    def on_node_status_change(
        self,
        node: MikadoNode,
        old_status: NodeStatus,  # noqa: V107
        new_status: NodeStatus,  # noqa: V107
    ) -> None: ...
