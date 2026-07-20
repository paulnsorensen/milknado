from __future__ import annotations

import logging

import pytest

from milknado.domains.common import MikadoNode, NodeStatus, PluginMeta
from milknado.domains.graph._pipeline import (
    StatusMiddleware,
    StatusPipeline,
    _PluginAsMiddleware,
)


def _node(node_id: int = 1) -> MikadoNode:
    return MikadoNode(id=node_id, description="n")


class _RecordingMiddleware:
    def __init__(self, name: str, log: list[str]) -> None:
        self._meta = PluginMeta(name=name, version="0.1.0", description="")
        self._log = log

    @property
    def meta(self) -> PluginMeta:
        return self._meta

    def before_status_change(self, node: MikadoNode, old: NodeStatus, new: NodeStatus) -> None:
        self._log.append(f"{self._meta.name}:before")

    def after_status_change(self, node: MikadoNode, old: NodeStatus, new: NodeStatus) -> None:
        self._log.append(f"{self._meta.name}:after")


class _RaisingMiddleware:
    def __init__(self, name: str, log: list[str]) -> None:
        self._meta = PluginMeta(name=name, version="0.1.0", description="")
        self._log = log

    @property
    def meta(self) -> PluginMeta:
        return self._meta

    def before_status_change(self, node: MikadoNode, old: NodeStatus, new: NodeStatus) -> None:
        self._log.append(f"{self._meta.name}:before")
        raise RuntimeError("boom in before hook")

    def after_status_change(self, node: MikadoNode, old: NodeStatus, new: NodeStatus) -> None:
        self._log.append(f"{self._meta.name}:after")
        raise RuntimeError("boom in after hook")


class TestStatusPipelineOrdering:
    def test_before_fires_before_mutate_and_after_fires_after_in_registration_order(self) -> None:
        log: list[str] = []
        mw_a = _RecordingMiddleware("a", log)
        mw_b = _RecordingMiddleware("b", log)
        pipeline = StatusPipeline([mw_a, mw_b])

        def mutate() -> bool:
            log.append("mutate")
            return True

        ok = pipeline.run(
            node_getter=lambda _nid: _node(),
            node_id=1,
            old=NodeStatus.PENDING,
            new=NodeStatus.RUNNING,
            mutate=mutate,
        )

        assert ok is True
        # Registration order is preserved on each side, and the mutate is
        # sandwiched between the full before pass and the full after pass —
        # not interleaved per-middleware.
        assert log == ["a:before", "b:before", "mutate", "a:after", "b:after"]


class TestStatusPipelineHookErrorIsolation:
    def test_exception_in_one_hook_is_isolated_and_other_hooks_plus_mutate_still_run(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        log: list[str] = []
        raising = _RaisingMiddleware("raiser", log)
        recording = _RecordingMiddleware("recorder", log)
        pipeline = StatusPipeline([raising, recording])

        def mutate() -> bool:
            log.append("mutate")
            return True

        with caplog.at_level(logging.ERROR, logger="milknado.domains.graph._pipeline"):
            ok = pipeline.run(
                node_getter=lambda _nid: _node(),
                node_id=1,
                old=NodeStatus.PENDING,
                new=NodeStatus.RUNNING,
                mutate=mutate,
            )

        assert ok is True
        # The raiser's before/after both ran (and raised), but did not stop
        # the other middleware's hooks or the mutate from running.
        assert log == [
            "raiser:before",
            "recorder:before",
            "mutate",
            "raiser:after",
            "recorder:after",
        ]
        # Both raised exceptions were caught and logged, not swallowed silently.
        error_messages = [r.message for r in caplog.records if r.levelno == logging.ERROR]
        assert len(error_messages) == 2
        assert all("raiser" in msg for msg in error_messages)


class TestStatusPipelineGatedMutate:
    def test_after_hooks_do_not_fire_when_mutate_returns_false(self) -> None:
        log: list[str] = []
        mw = _RecordingMiddleware("a", log)
        pipeline = StatusPipeline([mw])

        ok = pipeline.run(
            node_getter=lambda _nid: _node(),
            node_id=1,
            old=NodeStatus.RUNNING,
            new=NodeStatus.DONE,
            mutate=lambda: False,
        )

        assert ok is False
        # Before-hooks still fire (observe-only), but a failed guarded mutate
        # (e.g. a lost claim_node race) must not notify middleware of a
        # transition that never happened.
        assert log == ["a:before"]


class TestPluginAsMiddleware:
    def test_before_is_noop_and_after_delegates_to_legacy_on_node_status_change(self) -> None:
        calls: list[tuple[int, NodeStatus, NodeStatus]] = []

        class LegacyPlugin:
            @property
            def meta(self) -> PluginMeta:
                return PluginMeta(name="legacy", version="0.1.0", description="")

            def on_node_status_change(
                self, node: MikadoNode, old_status: NodeStatus, new_status: NodeStatus
            ) -> None:
                calls.append((node.id, old_status, new_status))

        adapter: StatusMiddleware = _PluginAsMiddleware(LegacyPlugin())
        node = _node(node_id=7)

        # before_status_change is a no-op: the legacy PluginHook protocol has
        # no before-hook, so nothing should be recorded.
        adapter.before_status_change(node, NodeStatus.PENDING, NodeStatus.RUNNING)
        assert calls == []

        adapter.after_status_change(node, NodeStatus.PENDING, NodeStatus.RUNNING)
        assert calls == [(7, NodeStatus.PENDING, NodeStatus.RUNNING)]
