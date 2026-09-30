from __future__ import annotations

import signal

from milknado.app._shutdown import ShutdownIntent


def test_first_signal_and_timestamp_remain_fixed(monkeypatch) -> None:
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
