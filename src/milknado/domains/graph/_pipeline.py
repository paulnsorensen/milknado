"""Status-change middleware pipeline for MikadoGraph transitions.

Promotes the ad-hoc plugin-notification call into a composed StatusPipeline
object: registered StatusMiddleware run before/after each transition, in
registration order, with per-hook exceptions isolated (never propagated).
Locked semantics: before-hooks are observe-only (no veto); the legacy
after-only PluginHook adapts in via _PluginAsMiddleware.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from milknado.domains.common import MikadoNode, NodeStatus, PluginHook, PluginMeta

_logger = logging.getLogger(__name__)
__all__ = ["_PluginAsMiddleware"]


class StatusMiddleware(Protocol):
    @property
    def meta(self) -> PluginMeta: ...

    def before_status_change(
        self, node: MikadoNode, old: NodeStatus | None, new: NodeStatus, /
    ) -> None:
        pass

    def after_status_change(
        self, node: MikadoNode, old: NodeStatus | None, new: NodeStatus, /
    ) -> None:
        pass


class _PluginAsMiddleware:
    """Adapt a legacy after-only PluginHook into StatusMiddleware."""

    def __init__(self, hook: PluginHook) -> None:
        self._hook: PluginHook = hook

    @property
    def meta(self) -> PluginMeta:
        return self._hook.meta

    def before_status_change(
        self, _node: MikadoNode, _old: NodeStatus | None, _new: NodeStatus
    ) -> None:
        return None

    def after_status_change(
        self, node: MikadoNode, old: NodeStatus | None, new: NodeStatus
    ) -> None:
        if old is not None:
            self._hook.on_node_status_change(node, old, new)


class StatusPipeline:
    """Runs registered middleware before/after a guarded status-mutating write.

    `mutate` performs the DB write and reports success; after-hooks fire only
    when it returns True (e.g. a lost claim_node race must not notify).
    """

    def __init__(self, middleware: Sequence[StatusMiddleware]) -> None:
        self._middleware: tuple[StatusMiddleware, ...] = tuple(middleware)

    def run(
        self,
        node_getter: Callable[[int], MikadoNode | None],
        node_id: int,
        old: NodeStatus | None,
        new: NodeStatus,
        mutate: Callable[[], bool],
    ) -> bool:
        if self._middleware:
            node = node_getter(node_id)
            if node is not None:
                for mw in self._middleware:
                    self._safe(mw, mw.before_status_change, node, old, new)
        ok = mutate()
        if ok and self._middleware:
            node = node_getter(node_id)
            if node is not None:
                for mw in self._middleware:
                    self._safe(mw, mw.after_status_change, node, old, new)
        return ok

    def _safe(self, mw: StatusMiddleware, hook_fn: Callable[..., None], *args: object) -> None:
        try:
            name = mw.meta.name
        except Exception:
            name = "<unknown>"
        try:
            hook_fn(*args)
        except Exception:
            _logger.exception("Middleware %s raised in %s", name, hook_fn.__name__)
