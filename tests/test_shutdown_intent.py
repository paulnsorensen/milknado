from __future__ import annotations

import signal

import pytest

from milknado.app._shutdown import ShutdownIntent


def test_first_signal_and_timestamp_remain_fixed(monkeypatch: pytest.MonkeyPatch) -> None:
    times = iter((10.0, 20.0))
    monkeypatch.setattr("milknado.app._shutdown.monotonic", lambda: next(times))
    intent = ShutdownIntent()

    intent.record(15, None)
    intent.record(2, None)

    assert intent.signum == 15
    assert intent.started_at == 10.0
    assert intent.deadline(8.0) == 18.0


def test_installed_handlers_record_intent_and_restore_prior_handler() -> None:
    prior = signal.getsignal(signal.SIGINT)
    intent = ShutdownIntent()

    with intent.installed():
        handler = signal.getsignal(signal.SIGINT)
        assert callable(handler)
        handler(signal.SIGINT, None)
        handler(signal.SIGINT, None)
        assert intent.signum == signal.SIGINT

    assert signal.getsignal(signal.SIGINT) is prior


def test_rearm_restores_intent_after_ui_replaces_handler() -> None:
    prior = signal.getsignal(signal.SIGINT)
    intent = ShutdownIntent()
    with intent.installed():
        _ = signal.signal(signal.SIGINT, signal.SIG_DFL)
        intent.rearm()
        assert signal.getsignal(signal.SIGINT) == intent.record
    assert signal.getsignal(signal.SIGINT) is prior


def test_supervision_records_cleanup_confirmation() -> None:
    from threading import Event, Thread

    from milknado.app._shutdown import ShutdownSignal, supervise

    started = Event()
    release = Event()
    intent = ShutdownIntent()

    def run() -> None:
        started.set()
        _ = release.wait(1.0)

    signaler = Thread(target=lambda: (started.wait(), intent.record(signal.SIGTERM, None)))
    signaler.start()
    try:
        with pytest.raises(ShutdownSignal):
            supervise(run, intent, lambda _deadline: True, "shutdown-test")
    finally:
        release.set()
        signaler.join(1.0)
    assert intent.cleanup_confirmed is True
