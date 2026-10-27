"""The pane signal handler ignores repeat signals and exits with the shell code."""

from __future__ import annotations

import signal

import pytest

from milknado.loop._tmux_exec import (
    _terminate,  # pyright: ignore[reportPrivateUsage]
)

_SIGNALS = (signal.SIGTERM, signal.SIGHUP)


@pytest.mark.parametrize("signum", _SIGNALS)
def test_first_pane_signal_exits_and_ignores_repeats(signum: signal.Signals) -> None:
    saved = {sig: signal.getsignal(sig) for sig in _SIGNALS}
    try:
        with pytest.raises(SystemExit) as raised:
            _terminate(signum, None)
        assert raised.value.code == 128 + signum
        assert all(signal.getsignal(sig) == signal.SIG_IGN for sig in _SIGNALS)
    finally:
        for sig, handler in saved.items():
            _ = signal.signal(sig, handler)
